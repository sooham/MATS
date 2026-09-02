"""Convenience readers for per-row activation and logit safetensors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal


def _capture_path(
    result_row: Mapping[str, object],
    run_dir: str | Path,
    kind: Literal["activation", "logit"],
) -> Path:
    field = f"{kind}_path"
    relative = result_row.get(field)
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Result row has no {field!r} capture.")
    path = Path(run_dir) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_activation_tensors(
    result_row: Mapping[str, object], run_dir: str | Path
) -> dict[str, Any]:
    """Load every captured activation tensor for one result row."""

    from safetensors.torch import load_file

    return load_file(_capture_path(result_row, run_dir, "activation"))


def load_logit_tensors(
    result_row: Mapping[str, object], run_dir: str | Path
) -> dict[str, Any]:
    """Load every captured full-vocabulary logit tensor for one result row."""

    from safetensors.torch import load_file

    return load_file(_capture_path(result_row, run_dir, "logit"))


def get_activation(
    result_row: Mapping[str, object],
    run_dir: str | Path,
    *,
    boundary: Literal["reasoning", "answer"],
    stream: Literal["resid_pre", "token_mixer_out", "mlp_out", "resid_post"],
    layer: int,
    token_index: int | None = None,
) -> Any:
    """Return a captured [token, hidden] tensor, or one selected token vector."""

    key = f"{boundary}.{stream}.layer_{layer}"
    tensors = load_activation_tensors(result_row, run_dir)
    if key not in tensors:
        raise KeyError(f"Capture {key!r} not found; available keys: {sorted(tensors)}")
    tensor = tensors[key]
    return tensor if token_index is None else tensor[token_index]


def get_logits(
    result_row: Mapping[str, object],
    run_dir: str | Path,
    *,
    boundary: Literal["reasoning", "answer"],
    position: int | None = None,
) -> Any:
    """Return full-vocabulary logits, optionally at one captured decode position."""

    key = f"{boundary}.logits"
    tensors = load_logit_tensors(result_row, run_dir)
    if key not in tensors:
        raise KeyError(f"Capture {key!r} not found; available keys: {sorted(tensors)}")
    tensor = tensors[key]
    if position is None:
        return tensor
    if tensor.ndim == 1:
        if position not in (0, -1):
            raise IndexError("A boundary-only logit capture has one implicit position.")
        return tensor
    return tensor[position]


def get_answer_surface_logits(
    result_row: Mapping[str, object],
    run_dir: str | Path,
    *,
    boundary: Literal["reasoning", "answer"] = "answer",
    position: int = -1,
) -> dict[str, float]:
    """Index a full-vocabulary capture by the row's resolved answer token IDs.

    The returned keys are the two model-facing probe surfaces (for example
    ``"2"`` and ``"7"``), not the framework's internal X/Y comparison labels.
    Multi-token X/Y surfaces require sequence log probabilities and are rejected.
    """

    surfaces = result_row.get("answer_surfaces")
    token_ids = result_row.get("answer_surface_token_ids")
    if not isinstance(surfaces, Mapping) or not isinstance(token_ids, Mapping):
        raise TypeError("Result row does not contain resolved answer surfaces and token IDs.")
    logits = get_logits(result_row, run_dir, boundary=boundary)
    if logits.ndim > 1:
        logits = logits[position]
    selected: dict[str, float] = {}
    for internal_label in ("X", "Y"):
        ids = token_ids.get(internal_label)
        surface = surfaces.get(internal_label)
        if not isinstance(ids, (list, tuple)) or not isinstance(surface, str):
            raise TypeError(f"Malformed answer surface metadata for {internal_label}.")
        if len(ids) != 1:
            raise ValueError(
                f"Answer surface {surface!r} has {len(ids)} tokens; use the stored "
                "sequence_log_probabilities instead."
            )
        selected[surface] = float(logits[int(ids[0])].item())
    return selected
