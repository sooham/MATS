"""Batched Hugging Face execution, scoring, and optional tensor capture."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
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
    enforce_reasoning,
    stage_two_messages,
    strip_thinking_markers,
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
class ExecutionConfig:
    experiment_dir: str | Path | None = None
    run_id: str = "default"
    batch_size: int = 2
    max_answer_tokens: int = 8
    max_reasoning_tokens: int = 256
    resume: bool = True
    capture: CaptureSpec = field(default_factory=CaptureSpec)
    metrics: MetricSpec = field(default_factory=MetricSpec)
    generation_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.max_answer_tokens < 1 or self.max_reasoning_tokens < 1:
            raise ValueError("Generation token limits must be positive.")
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be one non-empty path component.")


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

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            if self.device is None:
                self.device = self.model.get_input_embeddings().weight.device
            return
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            Qwen3_5ForConditionalGeneration,
        )

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
        self.tokenizer = tokenizer
        self.device = model.get_input_embeddings().weight.device
        self.model_metadata = {
            "architecture": type(model).__name__,
            "model_name_or_path": config.model_name_or_path,
            "revision": config.revision,
            "model_type": architecture_config.model_type,
            "quantization_config": getattr(architecture_config, "quantization_config", None),
        }

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

    def _serialize_continued_assistant(
        self, messages: Sequence[Mapping[str, str]]
    ) -> str:
        """Serialize a final assistant prefill without closing that message."""

        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Continued serialization requires a final assistant message.")
        kwargs: dict[str, object] = {
            "tokenize": False,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "enable_thinking": False,
        }
        try:
            rendered = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        except TypeError:
            # Simple or older tokenizers may not expose Qwen's optional template kwargs.
            kwargs.pop("enable_thinking")
            try:
                rendered = self.tokenizer.apply_chat_template(list(messages), **kwargs)
            except TypeError:
                kwargs.pop("continue_final_message")
                rendered = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        if not isinstance(rendered, str):
            raise TypeError("apply_chat_template(..., tokenize=False) must return a string.")
        assistant_prefill = str(messages[-1]["content"])
        if not rendered.endswith(assistant_prefill):
            raise ValueError(
                "The tokenizer closed or altered the continued assistant prefill; candidate "
                "logits would not be measured immediately after the configured answer prefix."
            )
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

    def _eos_ids(self) -> set[int]:
        raw = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if raw is None:
            raw = getattr(self.tokenizer, "eos_token_id", None)
        values = raw if isinstance(raw, (tuple, list, set)) else [raw]
        return {int(value) for value in values if value is not None}

    def _generate_once(
        self,
        items: Sequence[tuple[str, str]],
        *,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        import torch

        prompts = [prompt for _, prompt in items]
        encoded = self._encode_prompts(prompts)
        kwargs = {"do_sample": False, **dict(generation_kwargs)}
        with torch.inference_mode():
            generated = self.model.generate(**encoded, max_new_tokens=max_new_tokens, **kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prefix_width = encoded["input_ids"].shape[1]
        continuation = sequences[:, prefix_width:]
        eos_ids = self._eos_ids()
        records: dict[str, dict[str, object]] = {}
        for (row_id, _), tokens in zip(items, continuation):
            token_list = [int(token) for token in tokens.detach().cpu().tolist()]
            stop = next((index for index, token in enumerate(token_list) if token in eos_ids), None)
            effective = token_list if stop is None else token_list[:stop]
            text = self.tokenizer.decode(effective, skip_special_tokens=True)
            records[row_id] = {
                "text": text,
                "token_ids": effective,
                "completion_length": len(effective),
                "reached_eos": stop is not None,
                "hit_token_cap": stop is None and len(token_list) >= max_new_tokens,
                "effective_batch_size": len(items),
            }
        self._effective_batch_sizes.append(len(items))
        return records

    def _generate_resilient(
        self,
        items: Sequence[tuple[str, str]],
        *,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        if not items:
            return {}
        try:
            return self._generate_once(
                items,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
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
                ),
                **self._generate_resilient(
                    items[midpoint:],
                    max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                ),
            }

    def _run_bucketed(
        self,
        items: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        ordered = sorted(items, key=lambda item: len(item[1]))
        result: dict[str, dict[str, object]] = {}
        for start in range(0, len(ordered), batch_size):
            result.update(
                self._generate_resilient(
                    ordered[start : start + batch_size],
                    max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                )
            )
        return result

    def _surface_ids(self, surface: str) -> list[int]:
        encoded = self.tokenizer(surface, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(value) for value in ids]

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
    ) -> dict[str, dict[str, str]]:
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
        hooks = _ActivationHooks(self.model, spec)
        try:
            with torch.inference_mode():
                output = self.model(**encoded, use_cache=False)
        except Exception:
            hooks.clear()
            raise
        finally:
            hooks.close()
        logits = output.logits.detach()
        per_row: dict[str, dict[str, str]] = {}
        for row_index, (row_id, _) in enumerate(items):
            mask = encoded["attention_mask"][row_index].to(dtype=torch.bool)
            activation_tensors: dict[str, Any] = {}
            logit_tensors: dict[str, Any] = {}
            if boundary in spec.logits_boundaries:
                unpadded_logits = logits[row_index][mask]
                if decode_token_ids is not None and spec.every_decode_position:
                    count = len(decode_token_ids[row_id])
                    start = len(prompt_ids[row_index]) - 1
                    selected = unpadded_logits[start : start + count]
                else:
                    selected = unpadded_logits[-1]
                logit_tensors[f"{boundary}.logits"] = selected.float().cpu()
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
            paths: dict[str, str] = {}
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
    ) -> dict[str, dict[str, str]]:
        """Retry capture OOMs by recursively splitting without retaining hook buffers."""

        try:
            return self._capture_batch(
                items=items,
                boundary=boundary,
                spec=spec,
                run_dir=run_dir,
                decode_token_ids=decode_token_ids,
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
            return {
                **self._capture_resilient(
                    items=items[:midpoint],
                    boundary=boundary,
                    spec=spec,
                    run_dir=run_dir,
                    decode_token_ids=first_ids,
                ),
                **self._capture_resilient(
                    items=items[midpoint:],
                    boundary=boundary,
                    spec=spec,
                    run_dir=run_dir,
                    decode_token_ids=second_ids,
                ),
            }

    def execute(self, dataset: TranscriptDataset, config: ExecutionConfig) -> TranscriptDataset:
        self._ensure_loaded()
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
        pending = [row for row in dataset if row["row_id"] not in completed]
        self._oom_retries = 0
        self._effective_batch_sizes = []
        self._effective_capture_batch_sizes = []
        self._effective_score_batch_sizes = []

        reasoning_rows = [row for row in pending if int(row["reasoning_budget"]) > 0]
        reasoning_items = [
            (str(row["row_id"]), self._serialize(row["messages"]))  # type: ignore[arg-type]
            for row in reasoning_rows
        ]
        reasoning_prompts_by_id = dict(reasoning_items)
        reasoning_prompt_records = {
            row_id: self._prompt_token_record(prompt) for row_id, prompt in reasoning_items
        }
        reasoning_results = self._run_bucketed(
            reasoning_items,
            batch_size=config.batch_size,
            max_new_tokens=config.max_reasoning_tokens,
            generation_kwargs=config.generation_kwargs,
        )
        for completion in reasoning_results.values():
            raw_text = str(completion["text"])
            cleaned_text = strip_thinking_markers(raw_text)
            completion["raw_text"] = raw_text
            completion["text"] = cleaned_text
            completion["thinking_markers_removed"] = cleaned_text != raw_text.strip()
        capture_paths: dict[str, dict[str, str]] = {}
        if "reasoning" in config.capture.logits_boundaries or config.capture.streams:
            for start in range(0, len(reasoning_items), config.batch_size):
                reasoning_decode_ids = (
                    {
                        row_id: reasoning_results[row_id]["token_ids"]
                        for row_id, _ in reasoning_items[start : start + config.batch_size]
                    }
                    if config.capture.every_decode_position
                    else None
                )
                captured = self._capture_resilient(
                    items=reasoning_items[start : start + config.batch_size],
                    boundary="reasoning",
                    spec=config.capture,
                    run_dir=run_dir,
                    decode_token_ids=reasoning_decode_ids,
                )
                for row_id, paths in captured.items():
                    capture_paths.setdefault(row_id, {}).update(paths)

        answer_items: list[tuple[str, str]] = []
        answer_messages: dict[str, list[dict[str, str]]] = {}
        enforced_by_id: dict[str, str] = {}
        for row in pending:
            row_id = str(row["row_id"])
            answer_prefix = str(row.get("answer_prefix", "ANSWER:"))
            if int(row["reasoning_budget"]) > 0:
                enforced = enforce_reasoning(
                    str(reasoning_results[row_id]["text"]), int(row["reasoning_budget"])
                )
                enforced_by_id[row_id] = enforced
                base_messages = stage_two_messages(
                    first_messages=row["messages"],  # type: ignore[arg-type]
                    enforced_reasoning=enforced,
                    layout=str(row["call_layout"]),  # type: ignore[arg-type]
                    answer_prefix=answer_prefix,
                )
                messages = base_messages
            else:
                base_messages = [  # type: ignore[union-attr]
                    dict(message) for message in row["messages"]
                ]
                messages = [
                    *base_messages,
                    {"role": "assistant", "content": answer_prefix},
                ]
            prompt = self._serialize_continued_assistant(messages)
            answer_messages[row_id] = messages
            answer_items.append((row_id, prompt))
        answer_prompt_records = {
            row_id: self._prompt_token_record(prompt) for row_id, prompt in answer_items
        }
        pending_by_id = {str(row["row_id"]): row for row in pending}
        ordered_answer_items = sorted(answer_items, key=lambda item: len(item[1]))
        # A finalized result checkpoint is written after each successful answer minibatch.
        for start in range(0, len(ordered_answer_items), config.batch_size):
            chunk = ordered_answer_items[start : start + config.batch_size]
            answer_results = self._generate_resilient(
                chunk,
                max_new_tokens=config.max_answer_tokens,
                generation_kwargs=config.generation_kwargs,
            )
            if "answer" in config.capture.logits_boundaries or config.capture.streams:
                decode_ids = (
                    {row_id: answer_results[row_id]["token_ids"] for row_id, _ in chunk}
                    if config.capture.every_decode_position
                    else None
                )
                captured = self._capture_resilient(
                    items=chunk,
                    boundary="answer",
                    spec=config.capture,
                    run_dir=run_dir,
                    decode_token_ids=decode_ids,
                )
                for row_id, paths in captured.items():
                    capture_paths.setdefault(row_id, {}).update(paths)
            metric_specs = [
                config.metrics.resolve(
                    x=int(pending_by_id[row_id]["x"]),
                    y=int(pending_by_id[row_id]["y"]),
                )
                for row_id, _ in chunk
            ]
            score_records = self._score_resilient(
                [prompt for _, prompt in chunk], metric_specs
            )
            scores_by_id = {row_id: record for (row_id, _), record in zip(chunk, score_records)}
            for row_id, prompt in chunk:
                source = pending_by_id[row_id]
                answer = answer_results[row_id]
                score_record = scores_by_id[row_id]
                answer_surfaces = score_record["answer_surfaces"]
                assistant_completion = str(source.get("answer_prefix", "ANSWER:")) + str(
                    answer["text"]
                )
                choice = parse_model_choice(
                    assistant_completion,
                    allow_same=bool(source["allow_same"]),
                    x_surface=str(answer_surfaces["X"]),  # type: ignore[index]
                    y_surface=str(answer_surfaces["Y"]),  # type: ignore[index]
                    same_surface=str(answer_surfaces["SAME"]),  # type: ignore[index]
                    strict=True,
                    answer_prefix=str(source.get("answer_prefix", "ANSWER:")),
                )
                candidate_1 = int(source.get("candidate_1", source["x"]))
                candidate_2 = int(source.get("candidate_2", source["y"]))
                x_is_candidate_1 = int(source["x"]) == candidate_1
                if choice == "SAME":
                    canonical_choice = "SAME"
                    chosen_candidate = "SAME"
                elif choice == "X":
                    chosen_value = int(source["x"])
                    chosen_candidate = str(chosen_value)
                    if chosen_value == candidate_1:
                        canonical_choice = "C1"
                    elif chosen_value == candidate_2:
                        canonical_choice = "C2"
                    else:
                        raise ValueError("Parsed choice is not one of the canonical candidates.")
                elif choice == "Y":
                    chosen_value = int(source["y"])
                    chosen_candidate = str(chosen_value)
                    if chosen_value == candidate_1:
                        canonical_choice = "C1"
                    elif chosen_value == candidate_2:
                        canonical_choice = "C2"
                    else:
                        raise ValueError("Parsed choice is not one of the canonical candidates.")
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
                reasoning = reasoning_results.get(row_id)
                reasoning_prompt_record = reasoning_prompt_records.get(row_id)
                answer_prompt_record = answer_prompt_records[row_id]
                record = {
                    **source,
                    "answer_messages": answer_messages[row_id],
                    "reasoning_serialized_prompt": (
                        reasoning_prompts_by_id[row_id] if reasoning is not None else None
                    ),
                    "reasoning_input_ids": (
                        reasoning_prompt_record["input_ids"] if reasoning_prompt_record else None
                    ),
                    "reasoning_input_tokens": (
                        reasoning_prompt_record["tokens"] if reasoning_prompt_record else None
                    ),
                    "answer_serialized_prompt": prompt,
                    "answer_input_ids": answer_prompt_record["input_ids"],
                    "answer_input_tokens": answer_prompt_record["tokens"],
                    "runtime_tokenizer_template_fingerprint": runtime_tokenizer_fingerprint,
                    "raw_reasoning": reasoning["raw_text"] if reasoning else None,
                    "enforced_reasoning": enforced_by_id.get(row_id),
                    "reasoning_completion": reasoning,
                    "answer_completion": answer,
                    "assistant_completion": assistant_completion,
                    "model_choice": choice,
                    "model_choice_surface": (
                        answer_surfaces[choice] if choice is not None else None  # type: ignore[index]
                    ),
                    "model_choice_candidate": chosen_candidate,
                    "model_choice_canonical": canonical_choice,
                    "parse_compliance": choice is not None,
                    "strict_answer_compliance": choice is not None,
                    "zero_reasoning_compliance": (
                        choice is not None if int(source["reasoning_budget"]) == 0 else None
                    ),
                    "posterior_correct": posterior_correct,
                    "canonical_posterior_correct": canonical_posterior_correct,
                    **score_record,
                    "candidate_1_sequence_log_probability": candidate_1_sequence_score,
                    "candidate_2_sequence_log_probability": candidate_2_sequence_score,
                    "candidate_1_minus_candidate_2_logit": canonical_logit_difference,
                    **capture_paths.get(row_id, {}),
                    "generation_settings": {
                        "do_sample": False,
                        "enable_thinking": False,
                        "max_answer_tokens": config.max_answer_tokens,
                        "max_reasoning_tokens": config.max_reasoning_tokens,
                        **dict(config.generation_kwargs),
                    },
                }
                completed[row_id] = record
            ordered = [
                completed[str(row["row_id"])] for row in dataset if row["row_id"] in completed
            ]
            _atomic_write_text(
                results_path,
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
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
