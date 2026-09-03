"""Batched Hugging Face execution, scoring, and optional tensor capture."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .core import (
    SOFTMAX_LOG_BASE,
    SOFTMAX_LOG_UNIT,
    CaptureSpec,
    MetricSpec,
    TokenizerBinding,
)
from .dataset import TranscriptDataset, _atomic_write_text


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3.5-4B"
    revision: str | None = None
    dtype: str = "auto"
    device_map: object | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    max_gpu_memory_gib: float | None = None
    max_cpu_memory_gib: float | None = None


@dataclass(frozen=True)
class SGLangMTPConfig:
    """Native-MTP backend for continuous completion generation."""

    enabled: bool = False
    python_executable: str | Path = ".venv-sglang/bin/python"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 4
    context_length: int = 2048
    mem_fraction_static: float = 0.8
    cuda_graph_max_batch_size: int = 48
    attention_backend: str = "triton"
    sampling_backend: str = "pytorch"
    mamba_ssm_dtype: str | None = "bfloat16"
    mamba_full_memory_ratio: float = 4.0
    startup_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 3600.0

    def __post_init__(self) -> None:
        positive_integers = {
            "speculative_num_steps": self.speculative_num_steps,
            "speculative_eagle_topk": self.speculative_eagle_topk,
            "speculative_num_draft_tokens": self.speculative_num_draft_tokens,
            "context_length": self.context_length,
            "cuda_graph_max_batch_size": self.cuda_graph_max_batch_size,
        }
        invalid = [name for name, value in positive_integers.items() if value < 1]
        if invalid:
            raise ValueError(f"SGLang MTP values must be positive: {', '.join(invalid)}.")
        if not 0 < self.mem_fraction_static < 1:
            raise ValueError("mem_fraction_static must be strictly between zero and one.")
        if self.mamba_full_memory_ratio <= 0:
            raise ValueError("mamba_full_memory_ratio must be positive.")
        if self.startup_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("SGLang timeout values must be positive.")


@dataclass(frozen=True)
class ExecutionConfig:
    experiment_dir: str | Path | None = None
    run_id: str = "default"
    batch_size: int = 2
    completion_batch_size: int | None = None
    capture_batch_size: int | None = None
    score_batch_size: int | None = None
    checkpoint_every_batches: int = 25
    max_completion_tokens: int = 512
    resume: bool = True
    capture: CaptureSpec = field(default_factory=CaptureSpec)
    metrics: MetricSpec = field(default_factory=MetricSpec)
    generation_kwargs: Mapping[str, object] = field(default_factory=dict)
    completion_mtp: SGLangMTPConfig = field(default_factory=SGLangMTPConfig)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        stage_batch_sizes = {
            "completion_batch_size": self.completion_batch_size,
            "capture_batch_size": self.capture_batch_size,
            "score_batch_size": self.score_batch_size,
        }
        invalid = [name for name, value in stage_batch_sizes.items() if value is not None and value < 1]
        if invalid:
            raise ValueError(f"Stage batch sizes must be positive: {', '.join(invalid)}.")
        if self.checkpoint_every_batches < 1:
            raise ValueError("checkpoint_every_batches must be positive.")
        if self.max_completion_tokens < 1:
            raise ValueError("Generation token limits must be positive.")
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be one non-empty path component.")

    def batch_size_for(self, stage: str) -> int:
        overrides = {
            "completion": self.completion_batch_size,
            "capture": self.capture_batch_size,
            "score": self.score_batch_size,
        }
        if stage not in overrides:
            raise ValueError(f"Unknown execution stage {stage!r}.")
        return overrides[stage] or self.batch_size


def resolve_selector(selector: object, length: int) -> list[int]:
    """Resolve all/last/index/slice selectors, including negative indices."""

    if length < 0:
        raise ValueError("length must be non-negative.")
    if selector == "all":
        return list(range(length))
    if selector == "last":
        return [] if length == 0 else [length - 1]
    if isinstance(selector, int):
        raw = [selector]
    elif isinstance(selector, slice):
        return list(range(length))[selector]
    elif isinstance(selector, str) and ":" in selector:
        parts = selector.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice selector {selector!r}.")
        values = [int(value) if value else None for value in parts]
        return list(range(length))[slice(*values)]
    elif isinstance(selector, Sequence) and not isinstance(selector, (str, bytes)):
        raw = [int(index) for index in selector]
    else:
        raise ValueError(f"Unsupported selector {selector!r}.")
    result: list[int] = []
    for index in raw:
        resolved = index + length if index < 0 else index
        if not 0 <= resolved < length:
            raise IndexError(f"Index {index} is out of range for length {length}.")
        result.append(resolved)
    return result


def select_unpadded_tokens(tensor: Any, attention_mask: Any, selector: object) -> Any:
    """Remove padding first, then apply a token selector to a [batch, seq, ...] tensor."""

    import torch

    rows = []
    for row, mask in zip(tensor, attention_mask):
        unpadded = row[mask.to(dtype=torch.bool)]
        indices = resolve_selector(selector, len(unpadded))
        rows.append(unpadded[indices].detach().cpu())
    return rows


def parse_model_choice(
    text: str,
    *,
    allow_same: bool,
    x_surface: str = "X",
    y_surface: str = "Y",
    same_surface: str = "SAME",
    strict: bool = False,
    answer_prefix: str = "ANSWER:",
) -> str | None:
    """Parse model-facing surfaces and return the internal X/Y/SAME label.

    Strict mode accepts only a bare allowed value or that value preceded by the
    configured answer prefix.  It therefore rejects explanations that merely
    mention an allowed value.
    """

    surfaces = {"X": x_surface, "Y": y_surface, "SAME": same_surface}
    reverse = {surface.casefold(): choice for choice, surface in surfaces.items()}
    if len(reverse) != len(surfaces):
        raise ValueError("Choice surfaces must be distinct (ignoring case).")
    enabled = {key: value for key, value in surfaces.items() if key != "SAME" or allow_same}
    alternatives = sorted(enabled.values(), key=len, reverse=True)
    if strict:
        value_pattern = "|".join(re.escape(value) for value in alternatives)
        prefix = answer_prefix.strip()
        optional_prefix = f"(?:{re.escape(prefix)}\\s*)?" if prefix else ""
        match = re.fullmatch(
            rf"\s*{optional_prefix}({value_pattern})\s*",
            text,
            flags=re.IGNORECASE,
        )
        return reverse[match.group(1).casefold()] if match else None
    pattern = r"(?<!\w)(" + "|".join(re.escape(value) for value in alternatives) + r")(?!\w)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    return reverse[matches[-1].casefold()]


def parse_completion_contract(
    text: str,
    *,
    reasoning: bool,
    allow_same: bool,
    x_surface: str,
    y_surface: str,
    same_surface: str = "SAME",
    answer_prefix: str = "ANSWER:",
) -> dict[str, object]:
    """Parse a terminal answer while preserving exact-format diagnostics."""

    enabled = {"X": x_surface, "Y": y_surface}
    if allow_same:
        enabled["SAME"] = same_surface
    alternatives = sorted(enabled.items(), key=lambda item: len(item[1]), reverse=True)
    value_pattern = "|".join(re.escape(surface) for _, surface in alternatives)
    marker_positions = [match.start() for match in re.finditer(re.escape(answer_prefix), text)]
    marker_start = marker_positions[-1] if marker_positions else None
    marker_end = marker_start + len(answer_prefix) if marker_start is not None else None
    semantic_pattern = (
        rf"{re.escape(answer_prefix)}(?P<leading_whitespace>\s*)"
        rf"(?P<value>{value_pattern})(?P<trailing_whitespace>\s*)"
    )
    exact_pattern = rf"{re.escape(answer_prefix)} (?P<value>{value_pattern})"
    semantic_match = (
        re.search(rf"{semantic_pattern}\Z", text)
        if reasoning
        else re.fullmatch(semantic_pattern, text)
    )
    exact_match = (
        re.search(rf"{exact_pattern}\Z", text)
        if reasoning
        else re.fullmatch(exact_pattern, text)
    )
    reverse = {surface: label for label, surface in alternatives}
    choice = reverse[semantic_match.group("value")] if semantic_match else None
    if marker_start is None:
        break_reason = "missing_answer_marker"
    elif semantic_match is None:
        break_reason = "invalid_or_nonterminal_answer"
    else:
        break_reason = None
    return {
        "model_choice": choice,
        "answer_marker_present": marker_start is not None,
        "answer_marker_count": len(marker_positions),
        "answer_marker_char_start": marker_start,
        "answer_marker_char_end": marker_end,
        "answer_value_char_start": (
            semantic_match.start("value") if semantic_match is not None else None
        ),
        "answer_value_char_end": (
            semantic_match.end("value") if semantic_match is not None else None
        ),
        "answer_value_surface": (
            semantic_match.group("value") if semantic_match is not None else None
        ),
        "answer_leading_whitespace": (
            semantic_match.group("leading_whitespace")
            if semantic_match is not None
            else None
        ),
        "answer_trailing_whitespace": (
            semantic_match.group("trailing_whitespace")
            if semantic_match is not None
            else None
        ),
        "answer_leading_whitespace_character_count": (
            len(semantic_match.group("leading_whitespace"))
            if semantic_match is not None
            else None
        ),
        "answer_trailing_whitespace_character_count": (
            len(semantic_match.group("trailing_whitespace"))
            if semantic_match is not None
            else None
        ),
        "semantic_answer_compliance": semantic_match is not None,
        "exact_answer_format_compliance": exact_match is not None,
        # Backward-compatible name: compliance now means a semantically valid terminal answer.
        "strict_answer_compliance": semantic_match is not None,
        "compliance_break": break_reason,
    }


def _model_layers(model: Any) -> list[Any]:
    candidates = [
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
        ("layers",),
    ]
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
            return list(value)
        except AttributeError:
            continue
    raise AttributeError("Could not locate transformer layers on the model.")


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    if isinstance(value, Mapping):
        for candidate in value.values():
            with contextlib.suppress(TypeError):
                return _first_tensor(candidate)
    raise TypeError("Hook output does not contain a tensor.")


class _ActivationHooks:
    def __init__(self, model: Any, spec: CaptureSpec) -> None:
        self.buffers: dict[tuple[str, int], Any] = {}
        self.handles: list[Any] = []
        if not spec.streams:
            return
        layers = _model_layers(model)
        for index in resolve_selector(spec.layers, len(layers)):
            layer = layers[index]
            if "resid_pre" in spec.streams:
                self.handles.append(
                    layer.register_forward_pre_hook(self._pre_hook("resid_pre", index))
                )
            if "resid_post" in spec.streams:
                self.handles.append(
                    layer.register_forward_hook(self._post_hook("resid_post", index))
                )
            if "mlp_out" in spec.streams:
                self.handles.append(
                    layer.mlp.register_forward_hook(self._post_hook("mlp_out", index))
                )
            if "token_mixer_out" in spec.streams:
                mixer = next(
                    (
                        getattr(layer, name)
                        for name in ("self_attn", "linear_attn", "token_mixer")
                        if hasattr(layer, name)
                    ),
                    None,
                )
                if mixer is None:
                    raise AttributeError(f"Layer {index} has no recognizable token mixer.")
                self.handles.append(
                    mixer.register_forward_hook(self._post_hook("token_mixer_out", index))
                )

    def _pre_hook(self, stream: str, layer: int) -> Any:
        def hook(_module: Any, inputs: Any) -> None:
            self.buffers[(stream, layer)] = _first_tensor(inputs).detach()

        return hook

    def _post_hook(self, stream: str, layer: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.buffers[(stream, layer)] = _first_tensor(output).detach()

        return hook

    def clear(self) -> None:
        self.buffers.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _atomic_safetensors(path: Path, tensors: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".safetensors", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file({name: value.contiguous() for name, value in tensors.items()}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


class QwenRunner:
    """Execute dense or GPTQ Qwen-family causal language models in length buckets."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: Any | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_metadata: dict[str, object] = {}
        self._oom_retries = 0
        self._effective_batch_sizes: list[int] = []
        self._effective_capture_batch_sizes: list[int] = []
        self._effective_score_batch_sizes: list[int] = []
        self._generation_backend_metadata: dict[str, object] = {
            "backend": "transformers"
        }

    def _ensure_tokenizer_loaded(self) -> None:
        if self.tokenizer is not None:
            return
        from transformers import AutoProcessor

        config = self.model_config
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def _ensure_loaded(self) -> None:
        self._ensure_tokenizer_loaded()
        if self.model is not None:
            if self.device is None:
                self.device = self.model.get_input_embeddings().weight.device
            return
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            Qwen3_5ForConditionalGeneration,
        )

        config = self.model_config
        load_kwargs: dict[str, object] = {
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
            "dtype": config.dtype,
        }
        if config.device_map is not None:
            load_kwargs["device_map"] = config.device_map
        elif torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        memory: dict[object, str] = {}
        if config.max_gpu_memory_gib is not None:
            memory[0] = f"{config.max_gpu_memory_gib:g}GiB"
        if config.max_cpu_memory_gib is not None:
            memory["cpu"] = f"{config.max_cpu_memory_gib:g}GiB"
        if memory:
            load_kwargs["max_memory"] = memory
        architecture_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        model_class = (
            Qwen3_5ForConditionalGeneration
            if architecture_config.model_type == "qwen3_5"
            else AutoModelForCausalLM
        )
        model = model_class.from_pretrained(config.model_name_or_path, **load_kwargs)
        model.eval()
        self.model = model
        self.device = model.get_input_embeddings().weight.device
        self.model_metadata = {
            "architecture": type(model).__name__,
            "model_name_or_path": config.model_name_or_path,
            "revision": config.revision,
            "model_type": architecture_config.model_type,
            "quantization_config": getattr(architecture_config, "quantization_config", None),
            "completion_generation": self._generation_backend_metadata,
        }

    @staticmethod
    def _free_loopback_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def _server_log_tail(log_file: Any, *, characters: int = 12_000) -> str:
        log_file.flush()
        position = log_file.tell()
        log_file.seek(0)
        contents = log_file.read()
        log_file.seek(position)
        return str(contents)[-characters:]

    @contextlib.contextmanager
    def _sglang_mtp_server(self, config: SGLangMTPConfig):
        python_executable = Path(config.python_executable).expanduser()
        if not python_executable.is_absolute():
            python_executable = Path.cwd() / python_executable
        if not python_executable.is_file():
            raise FileNotFoundError(
                "SGLang MTP is enabled but its Python executable does not exist: "
                f"{python_executable}"
            )

        port = self._free_loopback_port()
        base_url = f"http://127.0.0.1:{port}"
        command = [
            str(python_executable),
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_config.model_name_or_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            self.model_config.dtype,
            "--context-length",
            str(config.context_length),
            "--mem-fraction-static",
            str(config.mem_fraction_static),
            "--cuda-graph-max-bs-decode",
            str(config.cuda_graph_max_batch_size),
            "--cuda-graph-backend-prefill",
            "disabled",
            "--attention-backend",
            config.attention_backend,
            "--sampling-backend",
            config.sampling_backend,
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            str(config.speculative_num_steps),
            "--speculative-eagle-topk",
            str(config.speculative_eagle_topk),
            "--speculative-num-draft-tokens",
            str(config.speculative_num_draft_tokens),
            "--log-level",
            "warning",
        ]
        if config.mamba_ssm_dtype is not None:
            command.extend(["--mamba-ssm-dtype", config.mamba_ssm_dtype])
        command.extend(
            ["--mamba-full-memory-ratio", str(config.mamba_full_memory_ratio)]
        )
        if self.model_config.revision:
            command.extend(["--revision", self.model_config.revision])
        if self.model_config.trust_remote_code:
            command.append("--trust-remote-code")

        environment = os.environ.copy()
        if self.model_config.local_files_only:
            environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

        started = time.monotonic()
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                deadline = started + config.startup_timeout_seconds
                last_error: BaseException | None = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"SGLang exited during startup with status {process.returncode}.\n"
                            f"{self._server_log_tail(server_log)}"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"{base_url}/model_info", timeout=2
                        ) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError) as error:
                        last_error = error
                    time.sleep(1)
                else:
                    raise TimeoutError(
                        "SGLang did not become ready within "
                        f"{config.startup_timeout_seconds:g} seconds; last error={last_error!r}.\n"
                        f"{self._server_log_tail(server_log)}"
                    )
                print(
                    "SGLang native MTP ready for "
                    f"{self.model_config.model_name_or_path} after "
                    f"{time.monotonic() - started:.1f}s "
                    f"(NEXTN steps={config.speculative_num_steps}, "
                    f"topk={config.speculative_eagle_topk}, "
                    f"draft_tokens={config.speculative_num_draft_tokens}).",
                    flush=True,
                )
                yield base_url
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)

    def _generate_completion_with_sglang_mtp(
        self,
        items: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, object],
        config: SGLangMTPConfig,
    ) -> dict[str, dict[str, object]]:
        allowed_generation_kwargs = {
            "frequency_penalty",
            "ignore_eos",
            "min_new_tokens",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "stop",
            "stop_token_ids",
            "top_k",
            "top_p",
        }
        unsupported = set(generation_kwargs) - allowed_generation_kwargs
        if unsupported:
            raise ValueError(
                "SGLang MTP completion generation does not support these generation_kwargs: "
                + ", ".join(sorted(unsupported))
            )

        prompt_lengths = {}
        for row_id, prompt in items:
            try:
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            except TypeError:
                token_ids = self.tokenizer.encode(prompt)
            prompt_lengths[row_id] = len(token_ids)
        longest_prompt = max(prompt_lengths.values(), default=0)
        if longest_prompt + max_new_tokens > config.context_length:
            raise ValueError(
                "SGLang context_length is too small for completion generation: "
                "longest prompt has "
                f"{longest_prompt} tokens and max_new_tokens={max_new_tokens}, but "
                f"context_length={config.context_length}."
            )

        results: dict[str, dict[str, object]] = {}
        total_accepted = 0
        total_proposed = 0
        generated_tokens = 0
        request_groups = 0
        with self._sglang_mtp_server(config) as base_url:
            sampling_params: dict[str, object] = {
                **dict(generation_kwargs),
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            }
            eos_ids = sorted(self._eos_ids(sampling_params.get("stop_token_ids")))
            if "stop_token_ids" not in sampling_params and eos_ids:
                sampling_params["stop_token_ids"] = eos_ids
            generation_started = time.monotonic()
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                payload = {
                    "text": [prompt for _, prompt in chunk],
                    "sampling_params": sampling_params,
                }
                request = urllib.request.Request(
                    f"{base_url}/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=config.request_timeout_seconds
                    ) as response:
                        outputs = json.load(response)
                except (OSError, urllib.error.URLError) as error:
                    raise RuntimeError(
                        f"SGLang MTP completion batch {request_groups + 1} failed."
                    ) from error
                if not isinstance(outputs, list) or len(outputs) != len(chunk):
                    raise RuntimeError(
                        "SGLang returned an unexpected number of generations: "
                        f"expected {len(chunk)}, got "
                        f"{len(outputs) if isinstance(outputs, list) else type(outputs).__name__}."
                    )

                for (row_id, _), output in zip(chunk, outputs):
                    output_ids = [int(token_id) for token_id in output.get("output_ids", [])]
                    meta = dict(output.get("meta_info", {}))
                    finish_reason = meta.get("finish_reason")
                    finish_type = (
                        finish_reason.get("type")
                        if isinstance(finish_reason, Mapping)
                        else finish_reason
                    )
                    matched_stop = (
                        finish_reason.get("matched")
                        if isinstance(finish_reason, Mapping)
                        else None
                    )
                    total_accepted += int(meta.get("spec_accepted_drafts", 0))
                    total_proposed += int(meta.get("spec_proposed_drafts", 0))
                    generated_tokens += len(output_ids)
                    results[row_id] = {
                        "text": str(output.get("text", "")),
                        "token_ids": output_ids,
                        "generated_token_ids": output_ids,
                        "generated_sequence_token_ids": output_ids,
                        "terminal_stop_token_id": (
                            int(matched_stop)
                            if isinstance(matched_stop, int)
                            else None
                        ),
                        "finish_reason": finish_reason,
                        "completion_length": len(output_ids),
                        "reached_eos": finish_type == "stop",
                        "hit_token_cap": finish_type == "length",
                        "effective_batch_size": len(chunk),
                        "generation_backend": "sglang_native_mtp",
                        "mtp_metrics": {
                            key: meta[key]
                            for key in (
                                "spec_accept_rate",
                                "spec_accept_length",
                                "spec_accepted_drafts",
                                "spec_proposed_drafts",
                                "spec_verify_ct",
                            )
                            if key in meta
                        },
                    }
                request_groups += 1

            acceptance = (
                total_accepted / total_proposed if total_proposed else float("nan")
            )
            print(
                f"SGLang MTP continuous completion generation: {len(items)} rows in "
                f"{request_groups} batch(es), {generated_tokens} output tokens, "
                f"draft acceptance={acceptance:.1%}, "
                f"elapsed={time.monotonic() - generation_started:.1f}s.",
                flush=True,
            )

        self._generation_backend_metadata = {
            "backend": "sglang_native_mtp",
            "algorithm": "NEXTN",
            "speculative_num_steps": config.speculative_num_steps,
            "speculative_eagle_topk": config.speculative_eagle_topk,
            "speculative_num_draft_tokens": config.speculative_num_draft_tokens,
            "context_length": config.context_length,
            "attention_backend": config.attention_backend,
            "sampling_backend": config.sampling_backend,
            "mamba_ssm_dtype": config.mamba_ssm_dtype,
            "mamba_full_memory_ratio": config.mamba_full_memory_ratio,
            "request_groups": request_groups,
            "accepted_drafts": total_accepted,
            "proposed_drafts": total_proposed,
            "acceptance_rate": (
                total_accepted / total_proposed if total_proposed else None
            ),
        }
        return results

    def _serialize(self, messages: Sequence[Mapping[str, str]]) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            rendered = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            rendered = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        if not isinstance(rendered, str):
            raise TypeError("apply_chat_template(..., tokenize=False) must return a string.")
        return rendered

    def _encode_prompts(self, prompts: Sequence[str]) -> dict[str, Any]:
        import torch

        tokenizer = self.tokenizer
        old_side = getattr(tokenizer, "padding_side", None)
        if old_side is not None:
            tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(
                list(prompts),
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            if old_side is not None:
                tokenizer.padding_side = old_side
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
            mask = encoded.get("attention_mask")
        else:
            input_ids = encoded.input_ids
            mask = getattr(encoded, "attention_mask", None)
        if not hasattr(input_ids, "to"):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if mask is None:
            pad_id = getattr(tokenizer, "pad_token_id", 0)
            mask = input_ids.ne(pad_id)
        elif not hasattr(mask, "to"):
            mask = torch.tensor(mask, dtype=torch.long)
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": mask.to(self.device),
        }

    @staticmethod
    def _normalize_token_ids(raw: object) -> set[int]:
        values = raw if isinstance(raw, (tuple, list, set)) else [raw]
        return {int(value) for value in values if value is not None}

    def _eos_ids(self, override: object | None = None) -> set[int]:
        if override is not None:
            return self._normalize_token_ids(override)
        # Qwen checkpoints can expose <|endoftext|> through generation_config while
        # the tokenizer's EOS is the earlier <|im_end|> assistant-turn delimiter.
        # Either token must end this assistant turn.
        return {
            *self._normalize_token_ids(
                getattr(
                    getattr(self.model, "generation_config", None),
                    "eos_token_id",
                    None,
                )
            ),
            *self._normalize_token_ids(getattr(self.tokenizer, "eos_token_id", None)),
        }

    def _generate_once(
        self,
        items: Sequence[tuple[str, str]],
        *,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, object],
        capture_first_token_logits: bool = False,
        allowed_token_sequences_by_row: (
            Mapping[str, Sequence[Sequence[int]]] | None
        ) = None,
    ) -> dict[str, dict[str, object]]:
        import torch

        prompts = [prompt for _, prompt in items]
        encoded = self._encode_prompts(prompts)
        kwargs = {"do_sample": False, **dict(generation_kwargs)}
        eos_ids = self._eos_ids(kwargs.get("eos_token_id"))
        if "eos_token_id" not in kwargs and eos_ids:
            kwargs["eos_token_id"] = sorted(eos_ids)
        prefix_width = encoded["input_ids"].shape[1]
        if allowed_token_sequences_by_row is not None:
            allowed_by_batch_index = [
                [
                    [int(token_id) for token_id in sequence]
                    for sequence in allowed_token_sequences_by_row[row_id]
                ]
                for row_id, _ in items
            ]
            if any(
                not sequences or any(not sequence for sequence in sequences)
                for sequences in allowed_by_batch_index
            ):
                raise ValueError(
                    "Every constrained row needs at least one non-empty token sequence."
                )
            allowed_eos_ids = sorted(eos_ids)
            if not allowed_eos_ids:
                raise ValueError("Constrained answer generation requires an EOS token ID.")

            def prefix_allowed_tokens_fn(batch_id: int, input_ids: Any) -> list[int]:
                generated_ids = [
                    int(token_id) for token_id in input_ids[prefix_width:].tolist()
                ]
                next_token_ids = {
                    sequence[len(generated_ids)]
                    for sequence in allowed_by_batch_index[batch_id]
                    if len(sequence) > len(generated_ids)
                    and sequence[: len(generated_ids)] == generated_ids
                }
                return sorted(next_token_ids) if next_token_ids else allowed_eos_ids

            kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        if capture_first_token_logits:
            kwargs.update({"return_dict_in_generate": True, "output_logits": True})
        with torch.inference_mode():
            generated = self.model.generate(**encoded, max_new_tokens=max_new_tokens, **kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        generated_logits = getattr(generated, "logits", None)
        first_token_logits = (
            generated_logits[0].detach()
            if generated_logits is not None and len(generated_logits) > 0
            else None
        )
        continuation = sequences[:, prefix_width:]
        records: dict[str, dict[str, object]] = {}
        for row_index, ((row_id, _), tokens) in enumerate(zip(items, continuation)):
            token_list = [int(token) for token in tokens.detach().cpu().tolist()]
            stop = next((index for index, token in enumerate(token_list) if token in eos_ids), None)
            effective = token_list if stop is None else token_list[:stop]
            text = self.tokenizer.decode(effective, skip_special_tokens=True)
            records[row_id] = {
                "text": text,
                "token_ids": effective,
                "generated_token_ids": effective,
                "generated_sequence_token_ids": (
                    token_list if stop is None else token_list[: stop + 1]
                ),
                "terminal_stop_token_id": (
                    token_list[stop] if stop is not None else None
                ),
                "completion_length": len(effective),
                "reached_eos": stop is not None,
                "hit_token_cap": stop is None and len(token_list) >= max_new_tokens,
                "effective_batch_size": len(items),
            }
            if first_token_logits is not None:
                records[row_id]["_first_token_logits"] = first_token_logits[row_index]
        self._effective_batch_sizes.append(len(items))
        return records

    def _generate_resilient(
        self,
        items: Sequence[tuple[str, str]],
        *,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, object],
        capture_first_token_logits: bool = False,
        allowed_token_sequences_by_row: (
            Mapping[str, Sequence[Sequence[int]]] | None
        ) = None,
    ) -> dict[str, dict[str, object]]:
        if not items:
            return {}
        try:
            return self._generate_once(
                items,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
                capture_first_token_logits=capture_first_token_logits,
                allowed_token_sequences_by_row=allowed_token_sequences_by_row,
            )
        except RuntimeError as error:
            if not _is_oom(error) or len(items) == 1:
                raise
            self._oom_retries += 1
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.empty_cache()
            midpoint = len(items) // 2
            return {
                **self._generate_resilient(
                    items[:midpoint],
                    max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                    capture_first_token_logits=capture_first_token_logits,
                    allowed_token_sequences_by_row=allowed_token_sequences_by_row,
                ),
                **self._generate_resilient(
                    items[midpoint:],
                    max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                    capture_first_token_logits=capture_first_token_logits,
                    allowed_token_sequences_by_row=allowed_token_sequences_by_row,
                ),
            }

    def _run_bucketed(
        self,
        items: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, object],
        capture_first_token_logits: bool = False,
    ) -> dict[str, dict[str, object]]:
        ordered = sorted(items, key=lambda item: len(item[1]))
        result: dict[str, dict[str, object]] = {}
        for start in range(0, len(ordered), batch_size):
            result.update(
                self._generate_resilient(
                    ordered[start : start + batch_size],
                    max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                    capture_first_token_logits=capture_first_token_logits,
                )
            )
        return result

    def _surface_ids(self, surface: str) -> list[int]:
        encoded = self.tokenizer(surface, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(value) for value in ids]

    def _supports_logits_to_keep(self) -> bool:
        with contextlib.suppress(TypeError, ValueError):
            return "logits_to_keep" in inspect.signature(self.model.forward).parameters
        return False

    def _surface_raw_logits(self, logits: Any, spec: MetricSpec) -> dict[str, object]:
        selected: dict[str, object] = {}
        for label in ("X", "Y"):
            surface = spec.surfaces[label]
            token_ids = self._surface_ids(surface)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Answer-surface logit capture requires a singleton token for {surface!r}."
                )
            values = logits[..., token_ids[0]].detach().float().cpu()
            selected[surface] = (
                float(values.item()) if values.ndim == 0 else values.tolist()
            )
        return selected

    def _prompt_token_record(self, prompt: str) -> dict[str, object]:
        """Return the exact, unpadded runtime-tokenizer representation of a prompt."""

        input_ids = self._surface_ids(prompt)
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            raw_tokens = convert(input_ids)
            tokens = [str(token) for token in raw_tokens]
        else:
            tokens = []
            for token_id in input_ids:
                try:
                    token = self.tokenizer.decode([token_id], skip_special_tokens=False)
                except TypeError:
                    token = self.tokenizer.decode([token_id])
                tokens.append(str(token))
        return {"input_ids": input_ids, "tokens": tokens}

    def _generated_token_boundary(
        self, token_ids: Sequence[int], text: str, char_end: int
    ) -> int | None:
        """Return the exact generated-token boundary for a character boundary."""

        if not 0 <= char_end <= len(text):
            return None
        for count in range(len(token_ids) + 1):
            prefix = self.tokenizer.decode(
                list(token_ids[:count]), skip_special_tokens=True
            )
            if prefix == text[:char_end]:
                return count
            if len(prefix) > char_end:
                return None
        return None

    def _score_batch(
        self, prompts: Sequence[str], metric_specs: Sequence[MetricSpec]
    ) -> list[dict[str, object]]:
        """Batched teacher-forced scores for potentially multi-token answer surfaces."""

        import torch

        if len(prompts) != len(metric_specs):
            raise ValueError("Every scoring prompt must have one resolved MetricSpec.")
        surface_ids_by_row = [
            {choice: self._surface_ids(surface) for choice, surface in spec.surfaces.items()}
            for spec in metric_specs
        ]
        records = [
            {
                "answer_surfaces": spec.surfaces,
                "answer_surface_token_ids": surface_ids,
                "sequence_log_probabilities": {},
                "sequence_log_probability_base": SOFTMAX_LOG_BASE,
                "sequence_log_probability_unit": SOFTMAX_LOG_UNIT,
            }
            for spec, surface_ids in zip(metric_specs, surface_ids_by_row)
        ]
        label_scores_by_row: list[dict[str, float]] = [{} for _ in prompts]
        jobs: list[tuple[int, str, list[int], list[int]]] = []
        for row_index, prompt in enumerate(prompts):
            prompt_ids = self._surface_ids(prompt)
            for choice, continuation in surface_ids_by_row[row_index].items():
                if continuation:
                    jobs.append((row_index, choice, prompt_ids, continuation))
        if not jobs:
            return records
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        maximum = max(len(prompt) + len(answer) for _, _, prompt, answer in jobs)
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        offsets: list[int] = []
        for _, _, prompt, answer in jobs:
            sequence = prompt + answer
            padding = maximum - len(sequence)
            rows.append([pad_id] * padding + sequence)
            masks.append([0] * padding + [1] * len(sequence))
            offsets.append(padding + len(prompt) - 1)
        input_ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            # PyTorch log_softmax is ln(softmax(logits)); scores are therefore in nats.
            log_probs = output.logits.float().log_softmax(-1)
        for job_index, ((row_index, choice, _, answer), offset) in enumerate(zip(jobs, offsets)):
            positions = torch.arange(offset, offset + len(answer), device=self.device)
            targets = torch.tensor(answer, dtype=torch.long, device=self.device)
            score = log_probs[job_index, positions, targets].sum().item()
            label_scores_by_row[row_index][choice] = float(score)
        for record, surface_ids, spec, scores in zip(
            records, surface_ids_by_row, metric_specs, label_scores_by_row
        ):
            record["sequence_log_probabilities"] = {
                spec.surfaces[label]: score for label, score in scores.items()
            }
            record["x_sequence_log_probability"] = scores["X"]
            record["y_sequence_log_probability"] = scores["Y"]
            record["same_sequence_log_probability"] = scores["SAME"]
            x_ids, y_ids = surface_ids["X"], surface_ids["Y"]
            if len(x_ids) == len(y_ids) == 1:
                # Difference of one-token log probabilities equals the raw-logit difference.
                record["x_minus_y_logit"] = float(scores["X"] - scores["Y"])  # type: ignore[index]
            else:
                record["x_minus_y_logit"] = None
        return records

    def _score_resilient(
        self, prompts: Sequence[str], metric_specs: Sequence[MetricSpec]
    ) -> list[dict[str, object]]:
        if not prompts:
            return []
        if len(prompts) != len(metric_specs):
            raise ValueError("Every scoring prompt must have one resolved MetricSpec.")
        if not all(spec.sequence_scores for spec in metric_specs):
            if any(spec.sequence_scores for spec in metric_specs):
                raise ValueError("A scoring minibatch cannot mix enabled and disabled metrics.")
            return [
                {
                    "answer_surfaces": spec.surfaces,
                    "answer_surface_token_ids": {
                        choice: self._surface_ids(surface)
                        for choice, surface in spec.surfaces.items()
                    },
                    "sequence_log_probabilities": None,
                    "sequence_log_probability_base": SOFTMAX_LOG_BASE,
                    "sequence_log_probability_unit": SOFTMAX_LOG_UNIT,
                    "x_sequence_log_probability": None,
                    "y_sequence_log_probability": None,
                    "same_sequence_log_probability": None,
                    "x_minus_y_logit": None,
                }
                for spec in metric_specs
            ]
        try:
            records = self._score_batch(prompts, metric_specs)
            self._effective_score_batch_sizes.append(len(prompts))
            return records
        except RuntimeError as error:
            if not _is_oom(error) or len(prompts) == 1:
                raise
            self._oom_retries += 1
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.empty_cache()
            midpoint = len(prompts) // 2
            return [
                *self._score_resilient(prompts[:midpoint], metric_specs[:midpoint]),
                *self._score_resilient(prompts[midpoint:], metric_specs[midpoint:]),
            ]

    def _capture_batch(
        self,
        *,
        items: Sequence[tuple[str, str]],
        boundary: str,
        spec: CaptureSpec,
        run_dir: Path,
        decode_token_ids: Mapping[str, Sequence[int]] | None = None,
        boundary_token_counts: Mapping[str, int | None] | None = None,
        metric_specs: Mapping[str, MetricSpec] | None = None,
        capture_logits: bool = True,
    ) -> dict[str, dict[str, object]]:
        """Forward once and atomically split captured tensors into per-row files."""

        if not items or not spec.enabled:
            return {}
        import torch

        prompts = [prompt for _, prompt in items]
        if decode_token_ids is not None:
            # Every generated prefix position is scored by a teacher-forced full-sequence pass.
            prompt_ids = [self._surface_ids(prompt) for prompt in prompts]
            merged = [
                prefix + list(decode_token_ids[row_id])
                for (row_id, _), prefix in zip(items, prompt_ids)
            ]
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
            width = max(map(len, merged))
            input_rows = [[pad_id] * (width - len(row)) + row for row in merged]
            masks = [[0] * (width - len(row)) + [1] * len(row) for row in merged]
            encoded = {
                "input_ids": torch.tensor(input_rows, dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor(masks, dtype=torch.long, device=self.device),
            }
        else:
            encoded = self._encode_prompts(prompts)
            prompt_ids = [
                row[mask.to(dtype=torch.bool)].detach().cpu().tolist()
                for row, mask in zip(encoded["input_ids"], encoded["attention_mask"])
            ]
        wants_logits = capture_logits and boundary in spec.logits_boundaries
        last_prompt_logits_only = (
            wants_logits
            and decode_token_ids is None
            and spec.logit_tokens == "last"
        )
        forward_kwargs: dict[str, object] = {"use_cache": False}
        if self._supports_logits_to_keep() and (not wants_logits or last_prompt_logits_only):
            forward_kwargs["logits_to_keep"] = 1
        hooks = _ActivationHooks(self.model, spec)
        try:
            with torch.inference_mode():
                output = self.model(**encoded, **forward_kwargs)
        except Exception:
            hooks.clear()
            raise
        finally:
            hooks.close()
        logits = output.logits.detach()
        per_row: dict[str, dict[str, object]] = {}
        for row_index, (row_id, _) in enumerate(items):
            mask = encoded["attention_mask"][row_index].to(dtype=torch.bool)
            activation_tensors: dict[str, Any] = {}
            logit_tensors: dict[str, Any] = {}
            surface_logits: dict[str, object] | None = None
            if wants_logits:
                if last_prompt_logits_only and logits.shape[1] == 1:
                    unpadded_logits = logits[row_index]
                else:
                    unpadded_logits = logits[row_index][mask]
                boundary_count = (
                    boundary_token_counts.get(row_id)
                    if boundary_token_counts is not None
                    else None
                )
                if boundary_token_counts is not None:
                    if boundary_count is None:
                        selected = None
                    else:
                        selected = unpadded_logits[
                            len(prompt_ids[row_index]) + boundary_count - 1
                        ]
                elif decode_token_ids is not None and spec.every_decode_position:
                    count = len(decode_token_ids[row_id])
                    start = len(prompt_ids[row_index]) - 1
                    selected = unpadded_logits[start : start + count]
                elif spec.logit_tokens == "last":
                    selected = unpadded_logits[-1]
                else:
                    indices = resolve_selector(spec.logit_tokens, len(unpadded_logits))
                    selected = unpadded_logits[indices]
                if selected is None:
                    pass
                elif spec.logits_scope == "full":
                    logit_tensors[f"{boundary}.logits"] = selected.float().cpu()
                else:
                    if metric_specs is None or row_id not in metric_specs:
                        raise ValueError(
                            "answer_surfaces logit capture requires a resolved MetricSpec."
                        )
                    surface_logits = self._surface_raw_logits(selected, metric_specs[row_id])
            for (stream, layer), tensor in hooks.buffers.items():
                unpadded = tensor[row_index][mask]
                if decode_token_ids is not None and spec.every_decode_position:
                    count = len(decode_token_ids[row_id])
                    start = len(prompt_ids[row_index])
                    selected = unpadded[start : start + count]
                else:
                    indices = resolve_selector(spec.tokens, len(unpadded))
                    selected = unpadded[indices]
                activation_tensors[f"{boundary}.{stream}.layer_{layer}"] = selected.detach().cpu()
            paths: dict[str, object] = {}
            if activation_tensors:
                path = run_dir / "activations" / f"{row_id}.safetensors"
                # Multiple boundaries merge rather than replace a prior stage.
                if path.exists():
                    from safetensors.torch import load_file

                    activation_tensors = {**load_file(path), **activation_tensors}
                _atomic_safetensors(path, activation_tensors)
                paths["activation_path"] = str(path.relative_to(run_dir))
            if logit_tensors:
                path = run_dir / "logits" / f"{row_id}.safetensors"
                if path.exists():
                    from safetensors.torch import load_file

                    logit_tensors = {**load_file(path), **logit_tensors}
                _atomic_safetensors(path, logit_tensors)
                paths["logit_path"] = str(path.relative_to(run_dir))
            if surface_logits is not None:
                paths[f"{boundary}_surface_raw_logits"] = surface_logits
            per_row[row_id] = paths
        hooks.clear()
        self._effective_capture_batch_sizes.append(len(items))
        return per_row

    def _capture_resilient(
        self,
        *,
        items: Sequence[tuple[str, str]],
        boundary: str,
        spec: CaptureSpec,
        run_dir: Path,
        decode_token_ids: Mapping[str, Sequence[int]] | None = None,
        boundary_token_counts: Mapping[str, int | None] | None = None,
        metric_specs: Mapping[str, MetricSpec] | None = None,
        capture_logits: bool = True,
    ) -> dict[str, dict[str, object]]:
        """Retry capture OOMs by recursively splitting without retaining hook buffers."""

        try:
            return self._capture_batch(
                items=items,
                boundary=boundary,
                spec=spec,
                run_dir=run_dir,
                decode_token_ids=decode_token_ids,
                boundary_token_counts=boundary_token_counts,
                metric_specs=metric_specs,
                capture_logits=capture_logits,
            )
        except RuntimeError as error:
            if not _is_oom(error) or len(items) == 1:
                raise
            self._oom_retries += 1
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.empty_cache()
            midpoint = len(items) // 2
            first_ids = (
                {row_id: decode_token_ids[row_id] for row_id, _ in items[:midpoint]}
                if decode_token_ids is not None
                else None
            )
            second_ids = (
                {row_id: decode_token_ids[row_id] for row_id, _ in items[midpoint:]}
                if decode_token_ids is not None
                else None
            )
            first_boundaries = (
                {row_id: boundary_token_counts[row_id] for row_id, _ in items[:midpoint]}
                if boundary_token_counts is not None
                else None
            )
            second_boundaries = (
                {row_id: boundary_token_counts[row_id] for row_id, _ in items[midpoint:]}
                if boundary_token_counts is not None
                else None
            )
            return {
                **self._capture_resilient(
                    items=items[:midpoint],
                    boundary=boundary,
                    spec=spec,
                    run_dir=run_dir,
                    decode_token_ids=first_ids,
                    boundary_token_counts=first_boundaries,
                    metric_specs=metric_specs,
                    capture_logits=capture_logits,
                ),
                **self._capture_resilient(
                    items=items[midpoint:],
                    boundary=boundary,
                    spec=spec,
                    run_dir=run_dir,
                    decode_token_ids=second_ids,
                    boundary_token_counts=second_boundaries,
                    metric_specs=metric_specs,
                    capture_logits=capture_logits,
                ),
            }

    def execute(self, dataset: TranscriptDataset, config: ExecutionConfig) -> TranscriptDataset:
        self._ensure_tokenizer_loaded()
        runtime_tokenizer_fingerprint = TokenizerBinding(self.tokenizer).fingerprint
        root = Path(config.experiment_dir) if config.experiment_dir else dataset.experiment_dir
        if root is None:
            raise ValueError("Set ExecutionConfig.experiment_dir or save the dataset first.")
        run_dir = root / "runs" / config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        results_path = run_dir / "results.jsonl"
        completed: dict[str, dict[str, object]] = {}
        if config.resume and results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    completed[str(row["row_id"])] = row
        pending = [row for row in dataset if str(row["row_id"]) not in completed]
        pending_by_id = {str(row["row_id"]): row for row in pending}
        self._oom_retries = 0
        self._effective_batch_sizes = []
        self._effective_capture_batch_sizes = []
        self._effective_score_batch_sizes = []
        self._generation_backend_metadata = {"backend": "transformers"}

        generation_messages: dict[str, list[dict[str, str]]] = {}
        generation_items: list[tuple[str, str]] = []
        for source in pending:
            row_id = str(source["row_id"])
            messages = [dict(message) for message in source["messages"]]  # type: ignore[union-attr]
            prompt = self._serialize(messages)
            generation_messages[row_id] = messages
            generation_items.append((row_id, prompt))
        prompt_records = {
            row_id: self._prompt_token_record(prompt) for row_id, prompt in generation_items
        }

        if config.completion_mtp.enabled and generation_items:
            completion_results = self._generate_completion_with_sglang_mtp(
                generation_items,
                batch_size=config.batch_size_for("completion"),
                max_new_tokens=config.max_completion_tokens,
                generation_kwargs=config.generation_kwargs,
                config=config.completion_mtp,
            )
            self._ensure_loaded()
            self.model_metadata["completion_generation"] = self._generation_backend_metadata
        else:
            self._ensure_loaded()
            completion_results = self._run_bucketed(
                generation_items,
                batch_size=config.batch_size_for("completion"),
                max_new_tokens=config.max_completion_tokens,
                generation_kwargs=config.generation_kwargs,
            )

        metric_specs_by_id = {
            row_id: config.metrics.resolve(x=int(source["x"]), y=int(source["y"]))
            for row_id, source in pending_by_id.items()
        }
        contract_by_id: dict[str, dict[str, object]] = {}
        boundary_token_counts: dict[str, int | None] = {}
        full_completion_by_id: dict[str, str] = {}
        for row_id, source in pending_by_id.items():
            completion = completion_results[row_id]
            generated_ids = [
                int(token_id)
                for token_id in completion.get("generated_token_ids", completion["token_ids"])
            ]
            completion["token_ids"] = generated_ids
            completion["generated_token_ids"] = generated_ids
            completion.setdefault("generated_sequence_token_ids", generated_ids)
            generated_text = str(completion["text"])
            full_completion = generated_text
            full_completion_by_id[row_id] = full_completion
            surfaces = metric_specs_by_id[row_id].surfaces
            contract = parse_completion_contract(
                full_completion,
                reasoning=bool(source["reasoning"]),
                allow_same=bool(source["allow_same"]),
                x_surface=str(surfaces["X"]),
                y_surface=str(surfaces["Y"]),
                same_surface=str(surfaces["SAME"]),
                answer_prefix=str(source.get("answer_prefix", "ANSWER:")),
            )
            value_start = contract["answer_value_char_start"]
            generated_value_start = int(value_start) if value_start is not None else None
            value_end = contract["answer_value_char_end"]
            generated_value_end = int(value_end) if value_end is not None else None
            marker_end = contract["answer_marker_char_end"]
            generated_marker_end = int(marker_end) if marker_end is not None else None
            # Both modes generate the complete terminal answer line. This consumes
            # exactly the generated marker and whitespace before the candidate.
            boundary_count = (
                self._generated_token_boundary(
                    generated_ids, generated_text, generated_value_start
                )
                if generated_value_start is not None
                else None
            )
            marker_boundary_count = (
                self._generated_token_boundary(
                    generated_ids, generated_text, generated_marker_end
                )
                if generated_marker_end is not None
                else None
            )
            value_end_boundary_count = (
                self._generated_token_boundary(
                    generated_ids, generated_text, generated_value_end
                )
                if generated_value_end is not None
                else None
            )
            contract["answer_boundary_break"] = (
                "answer_candidate_not_at_token_boundary"
                if value_start is not None and boundary_count is None
                else None
            )
            contract["answer_marker_generated_token_count"] = marker_boundary_count
            contract["answer_value_end_generated_token_count"] = value_end_boundary_count
            contract["answer_leading_whitespace_token_ids"] = (
                generated_ids[marker_boundary_count:boundary_count]
                if marker_boundary_count is not None and boundary_count is not None
                else None
            )
            contract["answer_value_generated_token_ids"] = (
                generated_ids[boundary_count:value_end_boundary_count]
                if boundary_count is not None and value_end_boundary_count is not None
                else None
            )
            contract["answer_trailing_whitespace_token_ids"] = (
                generated_ids[value_end_boundary_count:]
                if value_end_boundary_count is not None
                else None
            )
            boundary_token_counts[row_id] = boundary_count
            contract_by_id[row_id] = contract

        capture_records: dict[str, dict[str, object]] = {}
        if config.capture.enabled:
            capture_batch_size = config.batch_size_for("capture")
            for start in range(0, len(generation_items), capture_batch_size):
                capture_items = generation_items[start : start + capture_batch_size]
                decode_ids = {
                    row_id: completion_results[row_id]["generated_sequence_token_ids"]
                    for row_id, _ in capture_items
                }
                boundaries = {
                    row_id: boundary_token_counts[row_id] for row_id, _ in capture_items
                }
                captured = self._capture_resilient(
                    items=capture_items,
                    boundary="answer",
                    spec=config.capture,
                    run_dir=run_dir,
                    decode_token_ids=decode_ids,
                    boundary_token_counts=boundaries,
                    metric_specs=metric_specs_by_id,
                )
                for row_id, values in captured.items():
                    capture_records.setdefault(row_id, {}).update(values)

        score_records: dict[str, dict[str, object]] = {}
        for row_id, _ in generation_items:
            # Sequence scoring is normally disabled for this experiment. The prompt text here
            # is retained for backward compatibility; answer-boundary logits above always use
            # the exact teacher-forced token sequence.
            boundary_count = boundary_token_counts[row_id]
            prompt_ids = list(prompt_records[row_id]["input_ids"])
            generated_ids = list(completion_results[row_id]["generated_token_ids"])
            boundary_ids = (
                prompt_ids + generated_ids[:boundary_count]
                if boundary_count is not None
                else prompt_ids
            )
            boundary_prompt = self.tokenizer.decode(boundary_ids, skip_special_tokens=False)
            score_records[row_id] = self._score_resilient(
                [boundary_prompt], [metric_specs_by_id[row_id]]
            )[0]

        for row_id, prompt in generation_items:
            source = pending_by_id[row_id]
            completion = completion_results[row_id]
            contract = contract_by_id[row_id]
            score_record = score_records[row_id]
            answer_surfaces = score_record["answer_surfaces"]
            answer_surface_token_ids = score_record["answer_surface_token_ids"]
            choice = contract["model_choice"]
            candidate_1 = int(source.get("candidate_1", source["x"]))
            x_is_candidate_1 = int(source["x"]) == candidate_1
            candidate_1_label = "X" if x_is_candidate_1 else "Y"
            candidate_2_label = "Y" if x_is_candidate_1 else "X"
            candidate_1_answer_surface = answer_surfaces[candidate_1_label]
            candidate_2_answer_surface = answer_surfaces[candidate_2_label]
            candidate_1_answer_token_ids = answer_surface_token_ids[candidate_1_label]
            candidate_2_answer_token_ids = answer_surface_token_ids[candidate_2_label]
            if choice == "SAME":
                canonical_choice = "SAME"
                chosen_candidate = "SAME"
            elif choice in ("X", "Y"):
                chosen_value = int(source[str(choice).lower()])
                chosen_candidate = str(chosen_value)
                canonical_choice = "C1" if chosen_value == candidate_1 else "C2"
            else:
                canonical_choice = None
                chosen_candidate = None
            ground_truth = source.get("ground_truth_choice")
            posterior_correct = choice == ground_truth if ground_truth is not None else None
            canonical_ground_truth = source.get("canonical_ground_truth_choice")
            canonical_posterior_correct = (
                canonical_choice == canonical_ground_truth
                if canonical_ground_truth is not None
                else None
            )
            x_sequence_score = score_record.get("x_sequence_log_probability")
            y_sequence_score = score_record.get("y_sequence_log_probability")
            if x_is_candidate_1:
                candidate_1_sequence_score = x_sequence_score
                candidate_2_sequence_score = y_sequence_score
            else:
                candidate_1_sequence_score = y_sequence_score
                candidate_2_sequence_score = x_sequence_score
            positional_logit_difference = score_record.get("x_minus_y_logit")
            canonical_logit_difference = (
                float(positional_logit_difference)
                * (1.0 if x_is_candidate_1 else -1.0)
                if positional_logit_difference is not None
                else None
            )
            prompt_ids = list(prompt_records[row_id]["input_ids"])
            generated_ids = list(completion["generated_token_ids"])
            generated_sequence_ids = list(completion["generated_sequence_token_ids"])
            boundary_count = boundary_token_counts[row_id]
            teacher_forced_ids = prompt_ids + generated_sequence_ids
            reasoning_length = boundary_count if bool(source["reasoning"]) else None
            record = {
                **source,
                "generation_messages": generation_messages[row_id],
                "generation_serialized_prompt": prompt,
                "generation_input_ids": prompt_ids,
                "generation_input_tokens": prompt_records[row_id]["tokens"],
                "runtime_tokenizer_template_fingerprint": runtime_tokenizer_fingerprint,
                "completion": completion,
                "full_completion": full_completion_by_id[row_id],
                "assistant_completion": full_completion_by_id[row_id],
                "generated_token_ids": generated_ids,
                "generated_sequence_token_ids": completion["generated_sequence_token_ids"],
                "teacher_forced_input_ids": teacher_forced_ids,
                "teacher_forced_completion_start": len(prompt_ids),
                "answer_boundary_source": "generated_completion",
                "answer_boundary_generated_token_count": boundary_count,
                "answer_boundary_input_index": (
                    len(prompt_ids) + boundary_count - 1
                    if boundary_count is not None
                    else None
                ),
                "reasoning_length_tokens": reasoning_length,
                "model_choice": choice,
                "model_choice_surface": (
                    answer_surfaces[choice] if choice is not None else None  # type: ignore[index]
                ),
                "model_choice_candidate": chosen_candidate,
                "model_choice_canonical": canonical_choice,
                "parse_compliance": bool(contract["strict_answer_compliance"]),
                "semantic_answer_compliance": bool(
                    contract["semantic_answer_compliance"]
                ),
                "strict_answer_compliance": bool(contract["strict_answer_compliance"]),
                "exact_answer_format_compliance": bool(
                    contract["exact_answer_format_compliance"]
                ),
                "reasoning_compliance": (
                    bool(contract["strict_answer_compliance"])
                    if bool(source["reasoning"])
                    else None
                ),
                "no_reasoning_compliance": (
                    bool(contract["strict_answer_compliance"])
                    if not bool(source["reasoning"])
                    else None
                ),
                "compliance_break": contract["compliance_break"],
                "answer_marker_present": contract["answer_marker_present"],
                "answer_marker_count": contract["answer_marker_count"],
                "answer_marker_char_start": contract["answer_marker_char_start"],
                "answer_marker_char_end": contract["answer_marker_char_end"],
                "answer_value_char_start": contract["answer_value_char_start"],
                "answer_value_char_end": contract["answer_value_char_end"],
                "answer_value_surface": contract["answer_value_surface"],
                "answer_leading_whitespace": contract["answer_leading_whitespace"],
                "answer_trailing_whitespace": contract["answer_trailing_whitespace"],
                "answer_leading_whitespace_character_count": contract[
                    "answer_leading_whitespace_character_count"
                ],
                "answer_trailing_whitespace_character_count": contract[
                    "answer_trailing_whitespace_character_count"
                ],
                "answer_boundary_break": contract["answer_boundary_break"],
                "answer_marker_generated_token_count": contract[
                    "answer_marker_generated_token_count"
                ],
                "answer_value_end_generated_token_count": contract[
                    "answer_value_end_generated_token_count"
                ],
                "answer_leading_whitespace_token_ids": contract[
                    "answer_leading_whitespace_token_ids"
                ],
                "answer_value_generated_token_ids": contract[
                    "answer_value_generated_token_ids"
                ],
                "answer_trailing_whitespace_token_ids": contract[
                    "answer_trailing_whitespace_token_ids"
                ],
                "posterior_correct": posterior_correct,
                "canonical_posterior_correct": canonical_posterior_correct,
                **score_record,
                "candidate_1_answer_surface": candidate_1_answer_surface,
                "candidate_2_answer_surface": candidate_2_answer_surface,
                "candidate_1_answer_token_ids": candidate_1_answer_token_ids,
                "candidate_2_answer_token_ids": candidate_2_answer_token_ids,
                "candidate_1_sequence_log_probability": candidate_1_sequence_score,
                "candidate_2_sequence_log_probability": candidate_2_sequence_score,
                "candidate_1_minus_candidate_2_logit": canonical_logit_difference,
                **capture_records.get(row_id, {}),
                "generation_settings": {
                    "do_sample": False,
                    "enable_thinking": False,
                    "max_completion_tokens": config.max_completion_tokens,
                    "generation_backend": self._generation_backend_metadata,
                    **dict(config.generation_kwargs),
                },
            }
            completed[row_id] = record

        ordered_checkpoint = [
            completed[str(row["row_id"])]
            for row in dataset
            if str(row["row_id"]) in completed
        ]
        _atomic_write_text(
            results_path,
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_checkpoint
            ),
        )

        ordered_results = [
            completed[str(row["row_id"])] for row in dataset if row["row_id"] in completed
        ]
        result_dataset = TranscriptDataset(
            ordered_results,
            manifest={"schema_version": dataset.manifest.get("schema_version", "1.0")},
            experiment_dir=root,
        )
        manifest = {
            "schema_version": dataset.manifest.get("schema_version", "1.0"),
            "run_id": config.run_id,
            "row_count": len(ordered_results),
            "model": self.model_metadata or asdict(self.model_config),
            "runtime_tokenizer_template_fingerprint": runtime_tokenizer_fingerprint,
            "execution": {
                **asdict(config),
                "experiment_dir": str(root),
                "capture": asdict(config.capture),
                "metrics": asdict(config.metrics),
                "completion_mtp": asdict(config.completion_mtp),
            },
            "effective_batch_sizes": self._effective_batch_sizes,
            "effective_capture_batch_sizes": self._effective_capture_batch_sizes,
            "effective_score_batch_sizes": self._effective_score_batch_sizes,
            "oom_retries": self._oom_retries,
            "aggregates": result_dataset.summarize(),
            "results_file": "results.jsonl",
        }
        _atomic_write_text(
            run_dir / "run_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        )
        result_dataset.manifest = manifest
        return result_dataset
