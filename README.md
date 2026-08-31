# MATS

Experiments in mechanistic interpretability of language models.

## Environment

This repository uses [uv](https://docs.astral.sh/uv/) and targets Python 3.11. The required scientific and Hugging Face packages are declared in `pyproject.toml`; they are intentionally not installed by repository setup.

If the packages are already available in your global Python installation, create a project environment that can see them:

```bash
uv venv --python 3.11 --system-site-packages
source .venv/bin/activate
```

This keeps the repository isolated while reusing globally installed packages. For a fully reproducible machine or CI environment, install from the declaration and lockfile instead:

```bash
uv sync --frozen
```

On Linux, the project pins the official PyTorch 2.11 CUDA 12.8 wheels. On macOS,
the same lockfile falls back to PyPI's native wheel. The development group also
installs JupyterLab and `nbconvert`, so both interactive and headless notebook
execution work after one sync.

The optional `download` extra enables Hugging Face's faster transfer backend:

```bash
uv sync --frozen --extra download
```

## Vast.ai CUDA setup

Choose an NVIDIA instance whose host driver supports CUDA 12.8, clone the
repository, and install the locked environment. The CUDA toolkit bundled in the
container image is not used to build PyTorch; the official wheel supplies its
CUDA runtime, while the host still needs a compatible NVIDIA driver.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv sync --frozen --extra download
uv run --frozen python scripts/check_cuda.py
```

`check_cuda.py` prints the installed PyTorch/CUDA versions and GPU details, then
runs a small matrix multiplication on the GPU. Do not start the experiment until
that check passes.

Download the model as described in the next section, then run the first notebook
headlessly with:

```bash
INFERENCE_BATCH_SIZE=24 scripts/run_notebook.sh
```

The runner registers the project interpreter as the `mats-cuda` kernel and skips
only cells explicitly tagged `interactive`; model inference and static analyses
still execute. This keeps widget-only cells from blocking headless `nbconvert`.

The batch size counts prompts and must be a multiple of six because every probe
has six counterbalanced label assignments. Start at 24 on a 24 GiB GPU; try 48
or 96 on larger GPUs, or reduce to 12 after an out-of-memory error. The notebook
also remains runnable interactively with `uv run jupyter lab`.

The model-independent controlled posterior banks can be generated without loading Qwen:

```bash
uv run --frozen python scripts/generate_controlled_posterior.py
```

This writes the elicitation controls, N=2/N=4 ladder, fixed N=8 bank, and a manifest to
`artifacts/controlled_posterior/`. Add `--include-endpoints` for the separately labeled `r=0` and
`r=1` diagnostic rows; zero-probability endpoint histories are omitted because their posterior is
undefined. Use `--n8-schedules` to change the independent question-schedule count.

Run the full Qwen behavioral gate on a CUDA machine with:

```bash
INFERENCE_BATCH_SIZE=48 scripts/run_notebook.sh notebooks/03_controlled_posterior_behavior.ipynb
```

Set `RUN_THINKING_ARM=1` to additionally generate the separately reported deliberative
one-observation capability arm.

## Download Qwen3.5-4B

After `hf auth login`, the script can use your local Hugging Face login cache. Alternatively, export a read token in the shell. Do not put the token in a file tracked by Git:

```bash
hf auth login
# or:
export HF_TOKEN="hf_..."
uv run --frozen python scripts/download_model.py
```

The default destination is `models/Qwen--Qwen3.5-4B/`, which is ignored by Git. For exact replication, replace `main` with the immutable commit hash shown on the model's Hugging Face page:

```bash
uv run --frozen python scripts/download_model.py --revision <commit-hash>
```

The script accepts `--model`, `--revision`, `--output-dir`, `--include`, and `--exclude`, so later experiments can reuse it for other repositories or smaller file subsets. It passes the token directly to `snapshot_download`, enables the faster transfer backend when available, and resumes safely through the Hugging Face cache.

## Experiments

- `notebooks/01_minimal_noisy_source.ipynb` generates scripted noisy-source transcripts, computes exact Bayesian posteriors, and evaluates Qwen3.5-4B with forced-choice candidate and half-domain probes. Its six label assignments are batched, with a CUDA-aware default and an `INFERENCE_BATCH_SIZE` override. The notebook is committed unexecuted; its model-loading and inference cells are tagged `run-when-ready`.
- `notebooks/02_fixed_transcript_reliability_sweep.ipynb` replays an exhaustive bank of identical question/answer histories under every reliability condition, separating controlled reliability sensitivity from exact natural-distribution-weighted performance.
- `notebooks/03_controlled_posterior_behavior.ipynb` implements the behavioral gate: an explicit elicitation control, a counterfactually paired N=2/N=4 ladder, and a 32-schedule N=8 fixed bank. Its primary continuous target is exact posterior log-odds; left/right/tie classification, surface heuristics, tie subtypes, role/vocabulary/number-format robustness, and schedule-clustered uncertainty are reported separately.
- `notebooks/04_raw_evidence_deliberation_probe.ipynb` tests four bounded single-call generated-response probes under the same N=8, K=3 random-memoryless game. None passes the development gate; the best reaches 51.8% and never predicts the second candidate. The notebook then verifies the selected multi-call Qwen self-computation probe, which reaches 224/224 on development repeats 0–3 and 224/224 on a frozen replication using previously unused repeats 4–7. Evaluator-derived memberships, counts, likelihoods, and posterior values are never supplied to Qwen.

Reproduce the frozen candidate-only self-computation result with:

```bash
uv run --frozen python scripts/run_candidate_self_computation.py \
  --output artifacts/raw_reasoning_probe/self_computation_v4_replication_results.jsonl \
  --manifest artifacts/raw_reasoning_probe/self_computation_v4_replication_manifest.json \
  --repeats 4,5,6,7 \
  --batch-size 24
```

The selected candidate method is explicitly multi-call: Qwen generates every
membership answer, count draft/adjudication, and final comparison. For a strict
one-message/one-continuation estimand, notebook 04's negative single-call result
is the relevant conclusion. At exactly `r=0.5`, all candidate posteriors tie, so
three-way argmax accuracy need not decrease even though the observations contain
no information; posterior effect size is the appropriate entropy-sensitive target.

Primary papers that materially affect experiment design or interpretation are maintained as an
annotated ledger in [`CITATIONS.md`](CITATIONS.md).
