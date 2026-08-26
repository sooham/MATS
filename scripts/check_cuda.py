#!/usr/bin/env python3
"""Fail fast unless PyTorch can execute a small CUDA workload."""

from __future__ import annotations

import sys

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"Compiled CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Check `nvidia-smi`, the host driver, and that `uv sync --frozen` "
            "installed the Linux cu128 wheels from the configured PyTorch index."
        )

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"VRAM: {properties.total_memory / 2**30:.1f} GiB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    left = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    right = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    result = left @ right
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise SystemExit("CUDA matrix-multiplication smoke test produced non-finite values.")
    print("CUDA matrix-multiplication smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CUDA smoke test failed: {exc}", file=sys.stderr)
        raise
