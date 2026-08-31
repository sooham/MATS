#!/usr/bin/env bash
set -euo pipefail

notebook_path="${1:-notebooks/01_minimal_noisy_source.ipynb}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-24}"

if [[ ! -f "${notebook_path}" ]]; then
    echo "Notebook not found: ${notebook_path}" >&2
    exit 2
fi

uv run --frozen python scripts/check_cuda.py
uv run --frozen python -m ipykernel install \
    --sys-prefix \
    --name mats-cuda \
    --display-name "Python (MATS CUDA)" >/dev/null
echo "Running ${notebook_path} with INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE}"
uv run --frozen jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=mats-cuda \
    --ExecutePreprocessor.skip_cells_with_tag=interactive \
    "${notebook_path}"
