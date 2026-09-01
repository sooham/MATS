#!/usr/bin/env python3
"""Build the three executed-report notebooks for the filler-token experiments."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def notebook(cells: list) -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (MATS CUDA)",
                "language": "python",
                "name": "mats-cuda",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


COMMON_SETUP = r"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Run this notebook inside the MATS repository.")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


REPO_ROOT = find_repo_root()
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "filler_token_probe"
plt.style.use("seaborn-v0_8-whitegrid")
"""


def build_06() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
                # Exact filler-token sweep before `FINAL: `

                This is the no-thinking successor to notebook 05. The public game,
                raw reports, and candidates 2 versus 7 are unchanged. The assistant
                prefix is constructed at token-ID level as
                `chat_prefix + [filler_id] * F + tokenize("FINAL: ")`, and the next
                token is scored only over `2`, `7`, and `=`. There is no generated
                scratchpad and no reasoning-length instruction.

                The full `F = 0..100` sweep over five single-token fillers is the
                development split. Before looking at a repeat, the selection rule was
                frozen: choose the filler with highest mean candidate-only accuracy,
                then its smallest maximizer of a centered five-point moving average.
                Exactly that one condition is evaluated once on repeat 1.
                """
            ),
            code(COMMON_SETUP),
            code(
                r"""
dev_path = ARTIFACT_DIR / "development_all_fillers_results.jsonl"
dev_manifest_path = ARTIFACT_DIR / "development_all_fillers_manifest.json"
val_path = ARTIFACT_DIR / "validation_selected_results.jsonl"
val_manifest_path = ARTIFACT_DIR / "validation_selected_manifest.json"

dev = load_jsonl(dev_path)
val = load_jsonl(val_path)
dev_manifest = json.loads(dev_manifest_path.read_text())
val_manifest = json.loads(val_manifest_path.read_text())
conditions = dev_manifest["summary"]["conditions"]

fillers = sorted(dev_manifest["protocol"]["fillers"])
expected_fillers = ["comma", "newline", "period", "space", "underscore"]
assert fillers == expected_fillers
assert len(dev) == 56 * 101 * len(fillers) == 28_280
assert len(val) == 56
assert dev_manifest["protocol"]["enable_thinking"] is False
assert dev_manifest["protocol"]["generated_reasoning_tokens"] == 0
assert val_manifest["protocol"]["enable_thinking"] is False
assert dev_manifest["protocol"]["final_prefix"] == "FINAL: "
assert dev_manifest["protocol"]["final_prefix_token_ids"] == [95429, 25, 220]
assert dev_manifest["protocol"]["answer_token_ids"] == {"2": 17, "7": 22, "=": 28}
assert all(row["enable_thinking"] is False for row in dev + val)

cell_sizes = Counter((row["filler"], row["filler_count"]) for row in dev)
assert len(cell_sizes) == 5 * 101 and set(cell_sizes.values()) == {56}
for filler in fillers:
    assert {f for name, f in cell_sizes if name == filler} == set(range(101))
    for example_id in {row["example_id"] for row in dev}:
        group = sorted(
            (
                row for row in dev
                if row["filler"] == filler and row["example_id"] == example_id
            ),
            key=lambda row: row["filler_count"],
        )
        baseline = group[0]["input_token_count"]
        assert all(
            row["input_token_count"] == baseline + row["filler_count"]
            for row in group
        )

print("Token-exact contract and 505-cell coverage: PASS")
print("development rows:", len(dev), "validation rows:", len(val))
for name in fillers:
    spec = dev_manifest["protocol"]["fillers"][name]
    print(f"{name:10s} surface={spec['surface']!r:4s} token_id={spec['token_id']}")
"""
            ),
            md(
                """
                ## Development sweep and frozen selection

                Candidate-only accuracy excludes the normative tie cases; overall
                accuracy includes them. The model never chose the single-token tie
                answer in this experiment, so overall accuracy is necessarily lower.
                The horizontal line marks chance for the two non-tie candidates.
                """
            ),
            code(
                r"""
by_filler = {
    filler: sorted(
        (row for row in conditions if row["filler"] == filler),
        key=lambda row: row["filler_count"],
    )
    for filler in fillers
}
means = {
    filler: np.mean([row["candidate_only_accuracy"] for row in rows])
    for filler, rows in by_filler.items()
}
selected_filler = max(fillers, key=lambda filler: means[filler])
selected_curve = {
    row["filler_count"]: row["candidate_only_accuracy"]
    for row in by_filler[selected_filler]
}
smoothed = {
    f: np.mean([selected_curve[j] for j in range(f - 2, f + 3)])
    for f in range(2, 99)
}
best_smooth = max(smoothed.values())
selected_f = min(f for f, value in smoothed.items() if np.isclose(value, best_smooth))
assert (selected_filler, selected_f) == ("underscore", 11)

fig, ax = plt.subplots(figsize=(12, 5.4))
for filler, rows in by_filler.items():
    ax.plot(
        [row["filler_count"] for row in rows],
        [row["candidate_only_accuracy"] for row in rows],
        label=f"{filler} (mean {means[filler]:.3f})",
        alpha=0.9,
        linewidth=1.8,
    )
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="two-way chance")
ax.axvline(selected_f, color="#555555", linestyle=":", linewidth=1.5)
ax.set(xlabel="F: exact filler-token count", ylabel="Candidate-only accuracy",
       title="Development: every filler identity at every F = 0..100")
ax.set_ylim(0.25, 0.83)
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
figure_path = ARTIFACT_DIR / "exp1_filler_sweep.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()

print("Frozen selection:", selected_filler, "F =", selected_f)
print("five-point mean:", round(best_smooth, 4))
"""
            ),
            code(
                r"""
rows = by_filler[selected_filler]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].plot(
    [row["filler_count"] for row in rows],
    [row["candidate_2_accuracy"] for row in rows],
    label="target 2",
)
axes[0].plot(
    [row["filler_count"] for row in rows],
    [row["candidate_7_accuracy"] for row in rows],
    label="target 7",
)
axes[0].axvline(selected_f, color="black", linestyle=":")
axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1)
axes[0].set(
    xlabel="F",
    ylabel="Per-target accuracy",
    title="Underscore: direction-specific accuracy",
    ylim=(0.2, 0.9),
)
axes[0].legend()

order = sorted(fillers, key=means.get, reverse=True)
axes[1].bar(order, [means[name] for name in order], color="#4c78a8")
axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
axes[1].set(
    ylabel="Mean candidate-only accuracy",
    title="Filler identity matters",
    ylim=(0.4, 0.68),
)
axes[1].tick_params(axis="x", rotation=25)
fig.tight_layout()
figure_path = ARTIFACT_DIR / "exp1_selected_filler_bias.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
"""
            ),
            md("## One-shot repeat-1 validation"),
            code(
                r"""
dev_selected = next(
    row for row in conditions
    if row["filler"] == selected_filler and row["filler_count"] == selected_f
)
val_selected = val_manifest["summary"]["conditions"][0]
assert val_selected["filler"] == selected_filler
assert val_selected["filler_count"] == selected_f

print("metric                    development repeat 0    validation repeat 1")
for key in (
    "accuracy",
    "candidate_only_accuracy",
    "candidate_2_accuracy",
    "candidate_7_accuracy",
):
    print(f"{key:25s} {dev_selected[key]:20.3f} {val_selected[key]:22.3f}")
print("prediction counts         ", dev_selected["prediction_counts"],
      val_selected["prediction_counts"])
"""
            ),
            md(
                """
                ## Interpretation

                Exact filler tokens alter the immediate readout substantially:
                underscore averages 0.621 candidate-only accuracy across all 101
                lengths, versus 0.460–0.513 for the other identities. The frozen
                underscore/F=11 condition validates at 0.625 candidate-only accuracy
                (development 0.639). Its direction-specific validation recalls are
                0.600 for candidate 2 and 0.667 for candidate 7, so the development
                direction imbalance does not persist.

                This is not evidence that silent filler tokens implement a faithful
                reasoning trace. The model never predicts tie, and the sensitivity to
                filler identity is itself a warning that the suffix changes the answer
                readout. Experiment 3 tests that distinction causally.
                """
            ),
        ]
    )


def build_07() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
                # Scaling the exact filler-token probe

                This experiment evaluates every combination of
                `N ∈ {12,16,20,24,28,32,64,128}` and
                `r ∈ {0.0,0.1,...,1.0}`, with 32 deterministic replicates per cell.
                It freezes the development-selected underscore token at F=11 and
                continues to score exactly one answer token with thinking disabled.

                Multi-digit candidate strings are not single Qwen tokens, so each
                prompt assigns the two candidates to the single-token aliases `X` and
                `Y`. Within every cell, low/high presentation order and the X/Y alias
                mapping are each exactly balanced 16/16.
                """
            ),
            code(COMMON_SETUP),
            code(
                r"""
results_path = ARTIFACT_DIR / "scaled_grid_results.jsonl"
manifest_path = ARTIFACT_DIR / "scaled_grid_manifest.json"
rows = load_jsonl(results_path)
manifest = json.loads(manifest_path.read_text())

ns = [12, 16, 20, 24, 28, 32, 64, 128]
reliabilities = [round(i / 10, 1) for i in range(11)]
assert manifest["grid"]["n_values"] == ns
assert manifest["grid"]["reliabilities"] == reliabilities
assert len(rows) == 88 * 32 == 2_816
assert manifest["protocol"]["enable_thinking"] is False
assert manifest["protocol"]["generated_reasoning_tokens"] == 0
assert manifest["protocol"]["filler"] == "underscore"
assert manifest["protocol"]["filler_token_id"] == 62
assert manifest["protocol"]["filler_count"] == 11
assert all(row["enable_thinking"] is False for row in rows)

cells = defaultdict(list)
for row in rows:
    cells[(row["n"], row["reliability"])].append(row)
assert len(cells) == 88 and {len(group) for group in cells.values()} == {32}
for group in cells.values():
    assert {row["normative_role"] for row in group} <= {"first", "second", "tie"}
    assert sum(row["first_candidate"] < row["second_candidate"] for row in group) == 16
    assert sum(row["first_alias"] == "X" for row in group) == 16

print("Full 88-cell grid and within-cell counterbalancing: PASS")
print("rows:", len(rows), "single-token answers:", manifest["protocol"]["answer_token_ids"])
"""
            ),
            md("## Accuracy across N and reliability"),
            code(
                r"""
overall = np.empty((len(reliabilities), len(ns)))
candidate = np.empty_like(overall)
for i, reliability in enumerate(reliabilities):
    for j, n in enumerate(ns):
        group = cells[(n, reliability)]
        overall[i, j] = np.mean([row["correct"] for row in group])
        non_ties = [row for row in group if row["normative_answer"] != "tie"]
        candidate[i, j] = (
            np.mean([row["correct"] for row in non_ties]) if non_ties else np.nan
        )

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
for ax, matrix, title in (
    (axes[0], overall, "Overall accuracy (ties included)"),
    (axes[1], candidate, "Candidate-only accuracy"),
):
    image = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=1,
                      cmap="viridis")
    ax.set_xticks(range(len(ns)), ns)
    ax.set_yticks(range(len(reliabilities)), reliabilities)
    ax.set(xlabel="Number of questions N", ylabel="Reliability r", title=title)
    for i in range(len(reliabilities)):
        for j in range(len(ns)):
            value = matrix[i, j]
            label = "—" if np.isnan(value) else f"{value:.2f}"
            color = "white" if not np.isnan(value) and value < 0.45 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, shrink=0.82)
figure_path = ARTIFACT_DIR / "exp2_n_reliability_heatmaps.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
"""
            ),
            md("## Presentation-order controls and answer collapse"),
            code(
                r"""
role_rates = {"first": [], "second": [], "low": [], "high": []}
candidate_by_n = []
for n in ns:
    group = [row for row in rows if row["n"] == n and row["normative_role"] != "tie"]
    candidate_by_n.append(np.mean([row["correct"] for row in group]))
    for role in ("first", "second"):
        subset = [row for row in group if row["normative_role"] == role]
        role_rates[role].append(np.mean([row["correct"] for row in subset]))
    for magnitude in ("low", "high"):
        subset = [
            row for row in group
            if (
                row["normative_answer"] == min(row["candidates"])
                if magnitude == "low"
                else row["normative_answer"] == max(row["candidates"])
            )
        ]
        role_rates[magnitude].append(np.mean([row["correct"] for row in subset]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(ns, candidate_by_n, marker="o", linewidth=2, label="all non-ties")
for role, style in (("first", "--"), ("second", "--"), ("low", ":"), ("high", ":")):
    axes[0].plot(ns, role_rates[role], marker="o", linestyle=style, label=role)
axes[0].axhline(0.5, color="black", linewidth=1)
axes[0].set(xlabel="N", ylabel="Accuracy", title="No systematic presentation-order gain",
            ylim=(0.35, 0.65))
axes[0].legend(ncol=2)

prediction_counts = Counter(row["predicted_surface"] for row in rows)
target_counts = Counter(row["normative_surface"] for row in rows)
labels = ["X", "Y", "="]
x = np.arange(len(labels))
width = 0.38
axes[1].bar(x - width / 2, [target_counts[k] for k in labels], width,
            label="normative", color="#4c78a8")
axes[1].bar(x + width / 2, [prediction_counts[k] for k in labels], width,
            label="predicted", color="#f58518")
axes[1].set_xticks(x, labels)
axes[1].set(ylabel="Count", title="Immediate readout collapses to alias Y")
axes[1].legend()
fig.tight_layout()
figure_path = ARTIFACT_DIR / "exp2_position_and_surface_bias.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()

non_ties = [row for row in rows if row["normative_answer"] != "tie"]
print("overall accuracy:", f"{np.mean([row['correct'] for row in rows]):.3f}")
print("candidate-only accuracy:", f"{np.mean([row['correct'] for row in non_ties]):.3f}")
print("target counts:", dict(target_counts))
print("prediction counts:", dict(prediction_counts))
for surface in labels:
    subset = [row for row in rows if row["normative_surface"] == surface]
    print(f"recall({surface}) = {np.mean([row['correct'] for row in subset]):.3f}")
"""
            ),
            md("## Prompt-length scaling"),
            code(
                r"""
length_min = []
length_max = []
for n in ns:
    lengths = [row["input_token_count"] for row in rows if row["n"] == n]
    length_min.append(min(lengths))
    length_max.append(max(lengths))
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.fill_between(ns, length_min, length_max, alpha=0.25, color="#4c78a8")
ax.plot(ns, length_min, marker="o", label="minimum")
ax.plot(ns, length_max, marker="o", label="maximum")
ax.set(xlabel="N", ylabel="Input tokens", title="Context length grows with evidence count")
ax.legend()
fig.tight_layout()
figure_path = ARTIFACT_DIR / "exp2_prompt_lengths.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
"""
            ),
            md(
                """
                ## Interpretation

                Candidate-only accuracy stays near chance at every N (0.467–0.544
                after averaging over reliability), while overall accuracy is only
                0.280 because the model never emits the tie token. Crucially, this
                should not be read as stable candidate reasoning: 2,799 of 2,816
                predictions are the surface alias `Y`; recall is 0.010 for X, 0.999
                for Y, and 0 for tie.

                The exact counterbalance makes presentation position diagnosable.
                First/second and low/high candidate accuracies remain close within N,
                so ordinary positional bias is not the main failure. The dominant
                failure is the single-token answer interface itself—a hypothesis the
                alias-swap intervention tests in notebook 08.
                """
            ),
        ]
    )


def build_08() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
                # Mechanism probe: alias swap and layerwise logit lens

                This experiment separates semantic candidate selection from answer-token
                preference. It uses 16 non-tie N=32 examples (four each at
                r=0.1, 0.3, 0.7, 0.9), swaps the X/Y alias assignment while holding
                all evidence fixed, and crosses that intervention with underscore and
                period fillers at F = 0,1,2,4,8,11,16,32,64,100.

                At every embedding/layer depth, the ordinary final RMS norm and output
                head measure logits for X, Y, and `=` at the exact position after
                `FINAL: `. No text is generated; thinking is explicitly disabled.
                Alias-swap semantic consistency asks whether the *same numeric candidate*
                is selected after its output alias changes.
                """
            ),
            code(COMMON_SETUP),
            code(
                r"""
results_path = ARTIFACT_DIR / "mechanism_results.jsonl"
manifest_path = ARTIFACT_DIR / "mechanism_manifest.json"
rows = load_jsonl(results_path)
manifest = json.loads(manifest_path.read_text())
final_layer = max(row["layer"] for row in rows)
final = [row for row in rows if row["layer"] == final_layer]
filler_counts = [0, 1, 2, 4, 8, 11, 16, 32, 64, 100]
fillers = ["underscore", "period"]

assert len(rows) == 21_120 and len(final) == 640
assert set(row["layer"] for row in rows) == set(range(33))
assert manifest["protocol"]["enable_thinking"] is False
assert manifest["protocol"]["generated_reasoning_tokens"] == 0
assert manifest["protocol"]["answer_token_ids"] == {"X": 55, "Y": 56, "=": 28}
assert all(row["enable_thinking"] is False for row in rows)
cells = Counter(
    (row["filler"], row["filler_count"], row["alias_condition"])
    for row in final
)
assert len(cells) == 40 and set(cells.values()) == {16}

for filler in fillers:
    for f in filler_counts:
        group = [row for row in final if row["filler"] == filler and row["filler_count"] == f]
        paired = defaultdict(set)
        for row in group:
            paired[row["example_id"]].add(row["alias_condition"])
        assert len(paired) == 16 and all(value == {"original", "swapped"} for value in paired.values())

print("Causal-pair and 33-depth coverage: PASS")
print("layer records:", len(rows), "final records:", len(final))
"""
            ),
            md("## Causal alias-swap result"),
            code(
                r"""
surface_accuracy = {filler: [] for filler in fillers}
semantic_consistency = {filler: [] for filler in fillers}
surface_invariance = {filler: [] for filler in fillers}
for filler in fillers:
    for f in filler_counts:
        group = [
            row for row in final
            if row["filler"] == filler and row["filler_count"] == f
        ]
        surface_accuracy[filler].append(np.mean([row["correct"] for row in group]))
        pairs = defaultdict(dict)
        for row in group:
            pairs[row["example_id"]][row["alias_condition"]] = row
        semantic_consistency[filler].append(np.mean([
            pair["original"]["predicted_answer"] == pair["swapped"]["predicted_answer"]
            for pair in pairs.values()
        ]))
        surface_invariance[filler].append(np.mean([
            pair["original"]["predicted_surface"] == pair["swapped"]["predicted_surface"]
            for pair in pairs.values()
        ]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for filler in fillers:
    axes[0].plot(filler_counts, surface_accuracy[filler], marker="o", label=filler)
    axes[1].plot(filler_counts, semantic_consistency[filler], marker="o", label=filler)
axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
axes[0].set(xlabel="F", ylabel="Accuracy", title="Final candidate accuracy", ylim=(-0.03, 0.75))
axes[1].set(xlabel="F", ylabel="Pair consistency",
            title="Same numeric answer after X/Y swap", ylim=(-0.03, 1.03))
for ax in axes:
    ax.legend()
fig.tight_layout()
figure_path = ARTIFACT_DIR / "exp3_alias_swap.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()

for filler in fillers:
    print(filler)
    print("  mean accuracy:", round(float(np.mean(surface_accuracy[filler])), 3))
    print("  mean semantic consistency:",
          round(float(np.mean(semantic_consistency[filler])), 3))
    print("  mean surface-token invariance:",
          round(float(np.mean(surface_invariance[filler])), 3))
"""
            ),
            md("## Layerwise answer-token dynamics"),
            code(
                r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
for ax, filler in zip(axes, fillers, strict=True):
    matrix = np.empty((33, len(filler_counts)))
    for layer in range(33):
        for j, f in enumerate(filler_counts):
            group = [
                row for row in rows
                if row["layer"] == layer
                and row["filler"] == filler
                and row["filler_count"] == f
            ]
            matrix[layer, j] = np.mean([
                row["answer_logits"]["Y"] - row["answer_logits"]["X"]
                for row in group
            ])
    limit = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm",
                      vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(filler_counts)), filler_counts)
    ax.set(xlabel="F", ylabel="Embedding/layer depth",
           title=f"{filler}: mean logit(Y) − logit(X)")
    fig.colorbar(image, ax=ax, shrink=0.8)
figure_path = ARTIFACT_DIR / "exp3_layerwise_y_minus_x.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
"""
            ),
            code(
                r"""
layer_accuracy = []
layer_predictions = []
for layer in range(33):
    group = [row for row in rows if row["layer"] == layer]
    layer_accuracy.append(np.mean([row["correct"] for row in group]))
    layer_predictions.append(Counter(row["predicted_surface"] for row in group))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(range(33), layer_accuracy, marker="o", markersize=3)
axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
axes[0].set(xlabel="Embedding/layer depth", ylabel="Semantic accuracy",
            title="Logit-lens candidate accuracy", ylim=(-0.03, 0.65))

for surface, color in (("X", "#4c78a8"), ("Y", "#f58518"), ("=", "#54a24b")):
    axes[1].plot(
        range(33),
        [counts[surface] / 640 for counts in layer_predictions],
        label=surface,
        color=color,
    )
axes[1].set(xlabel="Embedding/layer depth", ylabel="Prediction fraction",
            title="Preferred answer token changes across depth", ylim=(-0.03, 1.03))
axes[1].legend()
fig.tight_layout()
figure_path = ARTIFACT_DIR / "exp3_layerwise_predictions.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()

print("final prediction counts:", dict(layer_predictions[-1]))
print("final semantic accuracy:", round(float(layer_accuracy[-1]), 3))
"""
            ),
            md(
                """
                ## Interpretation

                The intervention supports a readout-bias account. Underscore fillers
                produce `Y` on 317 of 320 final prompts. Because the same surface token
                denotes the opposite numeric candidate after an alias swap, mean
                semantic consistency is only 0.019, despite 0.497 mean accuracy.
                Period is somewhat less deterministic but still has only 0.125 mean
                semantic consistency. A robust internal candidate choice would instead
                switch output tokens and preserve the numeric answer.

                The layerwise lens also rules out a simple story in which the same
                answer preference is present at every depth: the favored restricted
                token moves from tie in early layers, through X/Y phases, to a strong Y
                preference at the final readout. A logit lens is descriptive rather
                than a faithful decoder of intermediate computations, but together
                with the causal alias swap it shows that filler-dependent accuracy is
                dominated by the output surface. These probes therefore do not support
                treating filler tokens as hidden reasoning time.
                """
            ),
        ]
    )


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "06_exact_filler_token_sweep.ipynb": build_06(),
        "07_scaled_filler_grid.ipynb": build_07(),
        "08_filler_alias_mechanism.ipynb": build_08(),
    }
    for name, document in outputs.items():
        path = NOTEBOOK_DIR / name
        nbf.write(document, path)
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
