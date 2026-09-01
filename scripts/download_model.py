#!/usr/bin/env python3
"""Download a Hugging Face model snapshot for local, repeatable experiments.

Authentication is read from HF_TOKEN (or another variable selected with
--token-env), falling back to the local Hugging Face login cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "main"
DOWNLOAD_MANIFEST = ".mats_model_snapshot.json"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str


MODEL_SETS: dict[str, tuple[ModelSpec, ...]] = {
    "qwen3.5-size-study": (
        ModelSpec(
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ),
        ModelSpec(
            "Qwen/Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        ),
        ModelSpec(
            "Qwen/Qwen3.5-27B-GPTQ-Int4",
            "8f0c09f227ae570e79617c6d9172b59df9c16081",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="One Hugging Face repository ID.")
    source.add_argument(
        "--model-set",
        choices=sorted(MODEL_SETS),
        help="Download a pinned set of checkpoints for a repository experiment.",
    )
    source.add_argument(
        "--list-model-sets",
        action="store_true",
        help="Print the pinned model sets and exit.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Git revision, tag, or commit hash (use a commit hash for exact reproducibility).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: models/<model-id>).",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token (default: HF_TOKEN).",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers.")
    parser.add_argument("--include", action="append", help="Pattern to include; may be repeated.")
    parser.add_argument("--exclude", action="append", help="Pattern to exclude; may be repeated.")
    return parser.parse_args()


def default_output_dir(model_id: str) -> Path:
    return Path("models") / model_id.replace("/", "--")


def requested_specs(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    if args.model_set:
        if args.output_dir is not None:
            raise ValueError("--output-dir is only valid with a single --model.")
        if args.revision != DEFAULT_REVISION:
            raise ValueError("Pinned model sets do not accept --revision overrides.")
        return MODEL_SETS[args.model_set]
    return (ModelSpec(args.model or DEFAULT_MODEL, args.revision),)


def write_download_manifest(
    *, destination: Path, spec: ModelSpec, resolved_revision: str
) -> None:
    payload = {
        "model_id": spec.model_id,
        "requested_revision": spec.revision,
        "resolved_revision": resolved_revision,
    }
    (destination / DOWNLOAD_MANIFEST).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.list_model_sets:
        for name, specs in MODEL_SETS.items():
            print(name)
            for spec in specs:
                print(f"  {spec.model_id}@{spec.revision}")
        return
    token = os.environ.get(args.token_env)
    if token:
        print(f"Using the token in ${args.token_env}.")
    else:
        print("No token environment variable found; using the local Hugging Face login cache.")

    # hf_transfer uses parallel ranged requests. Do not enable the backend unless
    # its optional package is present; older Hub versions otherwise fail instead
    # of falling back cleanly.
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        transfer_available = importlib.util.find_spec("hf_transfer") is not None
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if transfer_available else "0"
        if not transfer_available:
            print("Optional hf-transfer is not installed; using the standard Hub downloader.")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required; install the project dependencies with `uv sync`."
        ) from exc

    specs = requested_specs(args)
    api = HfApi(token=token)
    for spec in specs:
        destination = args.output_dir or default_output_dir(spec.model_id)
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec.model_id}@{spec.revision} to {destination}")
        snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            token=token,
            local_dir=destination,
            max_workers=args.max_workers,
            allow_patterns=args.include,
            ignore_patterns=args.exclude,
        )
        resolved_revision = str(api.model_info(spec.model_id, revision=spec.revision).sha)
        write_download_manifest(
            destination=destination,
            spec=spec,
            resolved_revision=resolved_revision,
        )
        print(f"Model snapshot ready at {destination} ({resolved_revision})")


if __name__ == "__main__":
    main()
