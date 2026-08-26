#!/usr/bin/env python3
"""Download a Hugging Face model snapshot for local, repeatable experiments.

Authentication is read from HF_TOKEN (or another variable selected with
--token-env), falling back to the local Hugging Face login cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repository ID.")
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


def main() -> None:
    args = parse_args()
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
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required; install the project dependencies with `uv sync`."
        ) from exc

    destination = args.output_dir or default_output_dir(args.model)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model}@{args.revision} to {destination}")

    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        token=token,
        local_dir=destination,
        max_workers=args.max_workers,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
    )
    print(f"Model snapshot ready at {destination}")


if __name__ == "__main__":
    main()
