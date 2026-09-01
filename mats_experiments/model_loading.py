"""Memory-bounded Qwen3.5 loading and reproducibility metadata."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOWNLOAD_MANIFEST = ".mats_model_snapshot.json"


@dataclass(frozen=True)
class LoadedQwen:
    model: Any
    processor: Any
    input_device: Any
    metadata: dict[str, object]


def read_snapshot_metadata(model_path: Path) -> dict[str, object]:
    path = model_path / DOWNLOAD_MANIFEST
    if not path.exists():
        return {
            "model_id": model_path.name.replace("--", "/", 1),
            "requested_revision": None,
            "resolved_revision": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_device_map(device_map: dict[str, object] | None) -> dict[str, int]:
    if not device_map:
        return {"cuda:0": 1}
    counts = Counter(str(device) for device in device_map.values())
    return dict(sorted(counts.items()))


def _quantization_metadata(config: Any) -> dict[str, object] | None:
    quantization = getattr(config, "quantization_config", None)
    if quantization is None:
        return None
    if hasattr(quantization, "to_dict"):
        return dict(quantization.to_dict())
    if isinstance(quantization, dict):
        return dict(quantization)
    return {"repr": repr(quantization)}


def _checkpoint_component_sizes(model_path: Path) -> dict[str, int]:
    from safetensors import safe_open

    dtype_bytes = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "I32": 4,
        "I64": 8,
        "U8": 1,
    }
    sizes: Counter[str] = Counter()
    for shard in model_path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor_slice = handle.get_slice(key)
                element_bytes = dtype_bytes[str(tensor_slice.get_dtype())]
                tensor_bytes = element_bytes
                for dimension in tensor_slice.get_shape():
                    tensor_bytes *= dimension
                if key.startswith("model.language_model.layers."):
                    component = f"layer.{key.split('.')[3]}"
                elif key.startswith("model.language_model.embed_tokens"):
                    component = "embed"
                elif key.startswith("lm_head"):
                    component = "lm_head"
                elif key.startswith("model.language_model.norm"):
                    component = "norm"
                elif key.startswith("model.visual"):
                    component = "visual"
                elif key.startswith("mtp"):
                    component = "mtp"
                else:
                    component = "other"
                sizes[component] += tensor_bytes
    if not sizes:
        raise FileNotFoundError(f"No safetensors weights found in {model_path}.")
    return dict(sizes)


def build_gptq_device_map(
    *, model_path: Path, num_hidden_layers: int, max_gpu_memory_gib: float
) -> tuple[dict[str, str], int]:
    """Place a checkpoint prefix on GPU while retaining runtime headroom."""

    sizes = _checkpoint_component_sizes(model_path)
    gib = 2**30
    runtime_reserve = int(1.25 * gib)
    budget = int(max_gpu_memory_gib * gib) - runtime_reserve
    used = sizes.get("embed", 0) + sizes.get("lm_head", 0) + sizes.get("norm", 0)
    if used >= budget:
        raise ValueError("GPU cap is too small for embeddings, output head, and runtime reserve.")
    gpu_layers = 0
    for layer in range(num_hidden_layers):
        layer_size = sizes.get(f"layer.{layer}")
        if layer_size is None:
            raise KeyError(f"Checkpoint is missing layer {layer}.")
        if used + layer_size > budget:
            break
        used += layer_size
        gpu_layers += 1
    device_map = {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cuda:0",
        "model.language_model.norm": "cuda:0",
        "lm_head": "cuda:0",
        "mtp": "cpu",
    }
    for layer in range(num_hidden_layers):
        device_map[f"model.language_model.layers.{layer}"] = (
            "cuda:0" if layer < gpu_layers else "cpu"
        )
    return device_map, gpu_layers


def load_qwen(
    *,
    model_path: Path,
    max_gpu_memory_gib: float | None,
    max_cpu_memory_gib: float = 700,
    gptq_backend: str = "gptq_torch",
) -> LoadedQwen:
    """Load one local Qwen checkpoint, optionally spilling layers to host RAM."""

    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")
    if max_gpu_memory_gib is not None and max_gpu_memory_gib <= 0:
        raise ValueError("max_gpu_memory_gib must be positive.")
    if max_cpu_memory_gib <= 0:
        raise ValueError("max_cpu_memory_gib must be positive.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    raw_quantization = _quantization_metadata(config)
    quantization_method = (
        str(raw_quantization.get("quant_method")) if raw_quantization else None
    )
    gpu_layers: int | None = None
    if quantization_method == "gptq":
        if max_gpu_memory_gib is None:
            raise ValueError("GPTQ checkpoints require an explicit GPU memory cap.")
        from gptqmodel import BACKEND, GPTQModel

        try:
            resolved_backend = BACKEND(gptq_backend)
        except ValueError as exc:
            choices = ", ".join(backend.value for backend in BACKEND)
            raise ValueError(
                f"Unknown GPTQ backend {gptq_backend!r}; choose one of: {choices}"
            ) from exc

        text_config = getattr(config, "text_config", config)
        device_map, gpu_layers = build_gptq_device_map(
            model_path=model_path,
            num_hidden_layers=int(text_config.num_hidden_layers),
            max_gpu_memory_gib=max_gpu_memory_gib,
        )
        wrapped = GPTQModel.load(
            str(model_path),
            device_map=device_map,
            backend=resolved_backend,
            trust_remote_code=False,
        )
        model = wrapped.model
        loader_backend = f"gptqmodel.{resolved_backend.name}"
    else:
        load_kwargs: dict[str, object] = {
            "dtype": dtype,
            "local_files_only": True,
        }
        if max_gpu_memory_gib is not None:
            load_kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {
                        0: f"{max_gpu_memory_gib:g}GiB",
                        "cpu": f"{max_cpu_memory_gib:g}GiB",
                    },
                }
            )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        if max_gpu_memory_gib is None:
            model = model.to(torch.device("cuda"))
        loader_backend = "transformers"
    model.eval()

    input_device = model.get_input_embeddings().weight.device
    if input_device.type == "meta":
        raise RuntimeError("Input embeddings remained on the meta device after loading.")
    snapshot = read_snapshot_metadata(model_path)
    text_config = getattr(model.config, "text_config", model.config)
    metadata = {
        **snapshot,
        "local_path": str(model_path),
        "architecture": type(model).__name__,
        "model_type": str(model.config.model_type),
        "hidden_size": int(text_config.hidden_size),
        "num_hidden_layers": int(text_config.num_hidden_layers),
        "quantization_config": _quantization_metadata(model.config),
        "loader_backend": loader_backend,
        "gpu_resident_transformer_layers": gpu_layers,
        "load_dtype": str(dtype),
        "max_gpu_memory_gib": max_gpu_memory_gib,
        "max_cpu_memory_gib": max_cpu_memory_gib,
        "input_device": str(input_device),
        "device_map_summary": summarize_device_map(
            getattr(model, "hf_device_map", None)
        ),
    }
    return LoadedQwen(model, processor, input_device, metadata)
