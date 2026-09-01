import json
from pathlib import Path

from mats_experiments.model_loading import (
    DOWNLOAD_MANIFEST,
    build_gptq_device_map,
    read_snapshot_metadata,
    summarize_device_map,
)


def test_read_snapshot_metadata(tmp_path: Path) -> None:
    payload = {
        "model_id": "Qwen/Qwen3.5-9B",
        "requested_revision": "abc",
        "resolved_revision": "abc",
    }
    (tmp_path / DOWNLOAD_MANIFEST).write_text(json.dumps(payload))
    assert read_snapshot_metadata(tmp_path) == payload


def test_snapshot_metadata_fallback(tmp_path: Path) -> None:
    model_path = tmp_path / "Qwen--Qwen3.5-9B"
    model_path.mkdir()
    assert read_snapshot_metadata(model_path)["model_id"] == "Qwen/Qwen3.5-9B"


def test_device_map_summary() -> None:
    assert summarize_device_map(None) == {"cuda:0": 1}
    assert summarize_device_map({"a": 0, "b": 0, "c": "cpu"}) == {
        "0": 2,
        "cpu": 1,
    }


def test_gptq_device_map_respects_calculated_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mats_experiments.model_loading._checkpoint_component_sizes",
        lambda _: {
            "embed": 2 * 2**30,
            "lm_head": 2 * 2**30,
            "norm": 1,
            **{f"layer.{index}": 2**30 for index in range(4)},
        },
    )
    device_map, gpu_layers = build_gptq_device_map(
        model_path=tmp_path,
        num_hidden_layers=4,
        max_gpu_memory_gib=6.5,
    )
    assert gpu_layers == 1
    assert device_map["model.language_model.layers.0"] == "cuda:0"
    assert device_map["model.language_model.layers.1"] == "cpu"
    assert device_map["model.visual"] == "cpu"
