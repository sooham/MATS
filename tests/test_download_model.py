from argparse import Namespace

import pytest

from scripts.download_model import (
    MODEL_SETS,
    ModelSpec,
    default_output_dir,
    requested_specs,
)


def args(**overrides: object) -> Namespace:
    values = {
        "model": None,
        "model_set": None,
        "output_dir": None,
        "revision": "main",
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_model_and_output_directory() -> None:
    assert requested_specs(args()) == (ModelSpec("Qwen/Qwen3.5-4B", "main"),)
    assert default_output_dir("Qwen/Qwen3.5-27B-GPTQ-Int4").as_posix() == (
        "models/Qwen--Qwen3.5-27B-GPTQ-Int4"
    )


def test_size_study_is_pinned_and_ordered() -> None:
    specs = requested_specs(args(model_set="qwen3.5-size-study"))
    assert specs == MODEL_SETS["qwen3.5-size-study"]
    assert [spec.model_id for spec in specs] == [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B-GPTQ-Int4",
    ]
    assert all(len(spec.revision) == 40 for spec in specs)


def test_model_set_rejects_single_model_overrides() -> None:
    with pytest.raises(ValueError, match="output-dir"):
        requested_specs(args(model_set="qwen3.5-size-study", output_dir="custom"))
    with pytest.raises(ValueError, match="revision"):
        requested_specs(args(model_set="qwen3.5-size-study", revision="main~1"))
