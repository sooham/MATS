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

- `notebooks/01_minimal_noisy_source.ipynb` generates scripted noisy-source transcripts, computes exact Bayesian posteriors, and evaluates Qwen3.5-4B with forced-choice candidate and half-domain probes. Its final section verifies the selected honest single-call candidate probe.
- `notebooks/02_fixed_transcript_reliability_sweep.ipynb` replays an exhaustive bank of identical question/answer histories under every reliability condition, separating controlled reliability sensitivity from exact natural-distribution-weighted performance.
- `notebooks/03_controlled_posterior_behavior.ipynb` implements the behavioral gate: an explicit elicitation control, a counterfactually paired N=2/N=4 ladder, and a 32-schedule N=8 fixed bank. Its primary continuous target is exact posterior log-odds; left/right/tie classification, surface heuristics, tie subtypes, role/vocabulary/number-format robustness, and schedule-clustered uncertainty are reported separately.
- `notebooks/04_raw_evidence_deliberation_probe.ipynb` records an earlier failed single-call search and a historical multi-stage scaffold. The multi-stage result is excluded from the current estimand.
- `notebooks/05_single_call_candidate_probe.ipynb` develops the replacement candidate-number readout under the same N=8, K=3 random-memoryless game. It uses exactly one no-thinking Qwen continuation per transcript and only public rules plus raw questions/reports. The frozen method scores 52/56 on development, 47/56 on validation, and 107/112 (95.5%) on held-out repeats 2–3 with 100% parse compliance.

Reproduce the frozen single-call held-out result with:

```bash
uv run --frozen python scripts/run_single_call_candidate.py \
  --output artifacts/single_call_candidate/test_system_reason_results.jsonl \
  --manifest artifacts/single_call_candidate/test_system_reason_manifest.json \
  --repeats 2,3 --variant number_system_reason --batch-size 4 --overwrite
```

The runner hardcodes `enable_thinking=False`. Candidate presentation order is
derived from the public example ID rather than the target, and evaluator-derived
memberships, counts, likelihoods, posteriors, and corrections never enter model
messages. At exactly `r=0.5`, all candidate posteriors tie, so three-way argmax
accuracy need not decrease even though the observations contain no information;
posterior effect size is the appropriate entropy-sensitive target.

Primary papers that materially affect experiment design or interpretation are maintained as an
annotated ledger in [`CITATIONS.md`](CITATIONS.md).
