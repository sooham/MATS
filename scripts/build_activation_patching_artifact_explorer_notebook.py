from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    REPO_ROOT
    / "notebooks"
    / "17_noisy_channel_bayesian_activation_patching_artifact_explorer.ipynb"
)


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown(
        r"""
# Noisy-channel Bayesian activation-patching artifact explorer (reasoning off)

This is the reasoning-off, single-presentation counterpart of notebook 14. It reads the 5,120 completed Qwen3.5-9B transcript rows generated for the activation-patching experiment and never changes the source artifact. The run crosses 64 question sets, all eight answer patterns, and 10 interior shared-reliability values. Every row has `reasoning=False` and presentation order `C1_C2`.

The notebook retains notebook 14's transcript inspection, error audit, endpoint audit, calibration metrics, reliability plots, conditional selection queries, and reusable SQL recipes. Position-balanced and reasoning-paired statistics are explicitly marked unavailable because the source run contains neither swapped presentations nor reasoning-on rows. All SQL tables are derived in memory from a source file opened read-only.

The semantic answer-surface contrast used throughout is

\[
z = \operatorname{logit}(C_1)-\operatorname{logit}(C_2),
\qquad q_{\mathrm{LLM}}=\sigma(z).
\]
"""
    ),
    code(
        '''
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from statistics import fmean

import matplotlib.pyplot as plt
from IPython.display import display

REPO_ROOT = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "pyproject.toml").exists()
)
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from noisy_channel_artifact_explorer import SQLResult, load_transcript_explorer

MODEL_KEY = "Qwen_Qwen3.5-9B_4c87a623"
RUN_ID = "qwen35_9b_n9_k3_r10_s64_selected_tokens_v2_reasoning_off"
RESULTS_PATH = (
    REPO_ROOT
    / "artifacts"
    / "noisy_channel_bayesian_experiment_2_activation_patching"
    / MODEL_KEY
    / "runs"
    / RUN_ID
    / "results.jsonl"
)
explorer = load_transcript_explorer(RESULTS_PATH)

assert len(explorer.transcript_rows) == 5_120
assert {bool(row["reasoning"]) for row in explorer.transcript_rows} == {False}
assert {row["presentation_order"] for row in explorer.transcript_rows} == {"C1_C2"}
assert explorer.table_names() == ("transcript_rows",)

reliability_order = [
    row["reliability_exact"]
    for row in explorer.sql("""
        SELECT reliability_exact
        FROM transcript_rows
        GROUP BY reliability_exact
        ORDER BY reliability
    """)
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 115,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

explorer.summary()
'''
    ),
    markdown(
        r"""
## Column model and query semantics

`transcript_rows` contains all presentation-level fields from notebook 14: question set, answer pattern, membership subsets, reports, agreement vectors and totals, likelihoods, posteriors, raw answer-surface logits, generated choice, correctness, and compliance. There are no derived `balanced_rows` or `reasoning_pairs` tables for this run.

Use `explorer.sql(...)` for SQLite or the immutable Python query builder. SQLite uses 0/1 for booleans and `?` placeholders for parameters. Here, every \(q\) is the probability assigned to semantic C1 (candidate value 2), while C2 is candidate value 7.
"""
    ),
    code(
        '''
display(explorer.sql("""
SELECT
    COUNT(*) AS transcript_rows,
    COUNT(DISTINCT question_set_index) AS question_sets,
    COUNT(DISTINCT answer_pattern_index) AS answer_patterns,
    COUNT(DISTINCT reliability_exact) AS reliability_values,
    COUNT(DISTINCT presentation_order) AS presentation_orders,
    SUM(answer_boundary_available) AS rows_with_boundary_logits,
    SUM(semantic_answer_compliance) AS semantically_compliant_rows
FROM transcript_rows
"""))

# Uncomment for the full scalar column inventory.
# display(explorer.columns("transcript_rows"))
'''
    ),
    code(
        '''
summary_by_reliability = explorer.sql("""
SELECT
    reliability_exact,
    reliability,
    COUNT(*) AS n,
    SUM(q_true IS NOT NULL AND ABS(q_true - 0.5) > 1e-12) AS scored_targets,
    SUM(canonical_posterior_correct IS NOT NULL) AS scored_emitted_choices,
    AVG(canonical_posterior_correct) AS emitted_choice_accuracy,
    AVG(logit_correct) AS boundary_logit_accuracy,
    AVG(semantic_answer_compliance) AS semantic_compliance,
    AVG(answer_boundary_available) AS boundary_availability,
    AVG(z) AS mean_z,
    AVG(q_true) AS mean_q_true,
    AVG(q_llm) AS mean_q_llm,
    AVG(q_llm - q_true) AS mean_calibration_error
FROM transcript_rows
GROUP BY reliability_exact, reliability
ORDER BY reliability
""")
display(summary_by_reliability)
'''
    ),
    code(
        '''
# Same direct-SQL exploration style as notebook 14.
display(explorer.sql("""
SELECT
    row_id,
    question_set_index,
    answer_pattern,
    question_subsets_json,
    observed_reports_json,
    reliability_exact,
    total_agreement_candidate_1,
    total_agreement_candidate_2,
    candidate_1_likelihood_total,
    candidate_2_likelihood_total,
    candidate_1_posterior,
    candidate_2_posterior,
    q_true,
    candidate_1_raw_logit,
    candidate_2_raw_logit,
    z,
    q_llm,
    model_choice_canonical,
    canonical_posterior_correct
FROM transcript_rows
ORDER BY ABS(z) DESC
LIMIT 12
"""))
'''
    ),
    markdown(
        """
### Inspecting an untouched source transcript

The SQL table flattens nested JSON into scalar or JSON-text columns. `raw_row` returns the untouched source dictionary, including prompts, generated token IDs, activation paths, answer-boundary metadata, raw logits, and the full completion.
"""
    ),
    code(
        '''
def inspect_transcript(row_id: str, *, include_prompt: bool = False) -> dict:
    raw = explorer.raw_row(row_id)
    selected = {
        "row_id": raw["row_id"],
        "question_set_index": raw["question_set_index"],
        "answer_pattern": raw["answer_pattern"],
        "reliabilities_exact": raw["reliabilities_exact"],
        "reasoning": raw["reasoning"],
        "presentation_order": raw["presentation_order"],
        "membership_sets": raw["membership_sets"],
        "observed_reports": raw["observed_reports"],
        "agreement_candidate_1_by_question": raw["agreement_candidate_1_by_question"],
        "agreement_candidate_2_by_question": raw["agreement_candidate_2_by_question"],
        "answer_surface_raw_logits": raw.get("answer_surface_raw_logits"),
        "model_choice_canonical": raw.get("model_choice_canonical"),
        "canonical_posterior_correct": raw.get("canonical_posterior_correct"),
        "full_completion": raw.get("full_completion"),
        "activation_path": raw.get("activation_path"),
    }
    if include_prompt:
        selected["messages"] = raw["messages"]
    return selected


sample_id = explorer.transcript_rows[0]["row_id"]
pprint(inspect_transcript(sample_id))
'''
    ),
    markdown(
        r"""
## 1. Likely incorrect reasoning-off answers

There is no reasoning-on counterpart with which to define a rescue. Instead, this section preserves notebook 14's configurable “likely incorrect” audit for the reasoning-off boundary logits. A row is scored only when the pair-normalized Bayesian target is defined and not tied. The default threshold includes every wrong boundary-logit choice; raise it to 0.75 to isolate confident errors.

The detailed table keeps the actual emitted choice and its correctness separate from the choice implied by the captured raw-logit contrast.
"""
    ),
    code(
        '''
LIKELY_INCORRECT_THRESHOLD = 0.50

likely_incorrect = explorer.sql("""
SELECT
    row_id,
    question_set_index,
    question_subsets_json AS question_membership,
    observed_reports_json AS observed_answers,
    answer_pattern,
    reliability_exact,
    total_agreement_candidate_1,
    total_agreement_candidate_2,
    q_true,
    z,
    q_llm,
    MAX(q_llm, 1.0 - q_llm) AS wrong_confidence,
    logit_choice,
    logit_correct,
    model_choice_canonical AS emitted_choice,
    canonical_posterior_correct AS emitted_choice_correct,
    full_completion
FROM transcript_rows
WHERE
    q_true IS NOT NULL
    AND ABS(q_true - 0.5) > 1e-12
    AND logit_correct = 0
    AND MAX(q_llm, 1.0 - q_llm) >= ?
ORDER BY wrong_confidence DESC, reliability, question_set_index, answer_pattern_index
""", (LIKELY_INCORRECT_THRESHOLD,))

likely_incorrect_summary = explorer.sql("""
SELECT
    COUNT(*) AS scored_rows,
    SUM(logit_correct = 0) AS incorrect_boundary_logit_choices,
    SUM(canonical_posterior_correct = 0) AS incorrect_emitted_choices,
    SUM(logit_choice != model_choice_canonical) AS logit_emission_disagreements
FROM transcript_rows
WHERE q_true IS NOT NULL AND ABS(q_true - 0.5) > 1e-12
""")
display(likely_incorrect_summary)
print(
    f"{len(likely_incorrect)} boundary-logit errors at confidence "
    f">= {LIKELY_INCORRECT_THRESHOLD:.0%}"
)
display(likely_incorrect)
'''
    ),
    code(
        '''
if likely_incorrect:
    pprint(inspect_transcript(likely_incorrect[0]["row_id"]))
'''
    ),
    markdown(
        """
## 2. Order-specific stability — unavailable for this run

Notebook 14 compares `C1_C2` with `C2_C1`. This activation-patching transcript set deliberately contains only `C1_C2`, so order gaps, sign flips after a swap, and position-balanced logits cannot be estimated from these rows.
"""
    ),
    code(
        '''
display(explorer.sql("""
SELECT presentation_order, COUNT(*) AS n
FROM transcript_rows
GROUP BY presentation_order
ORDER BY presentation_order
"""))
'''
    ),
    markdown(
        """
## 3. Reasoning-on versus reasoning-off — unavailable for this run

All 5,120 rows are reasoning off. Consequently, rescue transitions, paired sign-permutation tests, McNemar tests, dumbbells, confidence amplification, and reasoning-length comparisons have no matched reasoning-on observations and are not computed.
"""
    ),
    code(
        '''
display(explorer.sql("""
SELECT reasoning, COUNT(*) AS n, COUNT(reasoning_length_tokens) AS rows_with_reasoning_length
FROM transcript_rows
GROUP BY reasoning
ORDER BY reasoning
"""))
'''
    ),
    markdown(
        r"""
## 4. Deterministic-endpoint audit

Notebook 14 explicitly separates endpoint categories at \(r=0\) and \(r=1\). This run uses only interior reliabilities from 0.05 through 0.95, so the endpoint query should return zero rows. The range query makes that design fact auditable.
"""
    ),
    code(
        '''
display(explorer.sql("""
SELECT
    MIN(reliability) AS minimum_reliability,
    MAX(reliability) AS maximum_reliability,
    SUM(reliability IN (0.0, 1.0)) AS deterministic_endpoint_rows,
    COUNT(DISTINCT reliability_exact) AS reliability_values
FROM transcript_rows
"""))

display(explorer.sql("""
SELECT reliability_exact, endpoint_category, COUNT(*) AS n
FROM transcript_rows
WHERE reliability IN (0.0, 1.0)
GROUP BY reliability_exact, endpoint_category
ORDER BY reliability, endpoint_category
"""))
'''
    ),
    markdown(
        r"""
## 5. Calibration error, scores, regression, and reliability diagrams

For every row with candidate-defined Bayesian mass and captured logits,

\[
q_{\mathrm{true}}=
\frac{p(C_1\mid D)}{p(C_1\mid D)+p(C_2\mid D)},
\qquad q_{\mathrm{LLM}}=\sigma(z).
\]

Brier score and soft-label log loss include \(q_{true}\) values of 0 and 1. The regression of \(z\) on \(\operatorname{logit}(q_{true})\) uses only rows with \(0<q_{true}<1\); groups with no true-log-odds variance report undefined coefficients.
"""
    ),
    code(
        '''
calibration_rows = []
for source_row in explorer.transcript_rows:
    if source_row["q_true"] is None or source_row["q_llm"] is None:
        continue
    row = dict(source_row)
    q_true = float(row["q_true"])
    q_llm = float(row["q_llm"])
    clipped = min(max(q_llm, 1e-12), 1.0 - 1e-12)
    row["calibration_error"] = q_llm - q_true
    row["brier_score"] = (q_llm - q_true) ** 2
    row["log_loss"] = -(
        q_true * math.log(clipped)
        + (1.0 - q_true) * math.log(1.0 - clipped)
    )
    calibration_rows.append(row)


def ols_calibration(rows: list[dict]) -> tuple[float | None, float | None, int]:
    finite = [
        row for row in rows
        if 0.0 < row["q_true"] < 1.0 and row["z"] is not None
    ]
    if not finite:
        return None, None, 0
    x = [math.log(row["q_true"] / (1.0 - row["q_true"])) for row in finite]
    y = [row["z"] for row in finite]
    mean_x, mean_y = fmean(x), fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None, None, len(finite)
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator
    return mean_y - slope * mean_x, slope, len(finite)


def summarize_calibration(rows: list[dict], label: str) -> dict:
    intercept, slope, regression_n = ols_calibration(rows)
    return {
        "group": label,
        "n": len(rows),
        "mean_error": fmean(row["calibration_error"] for row in rows),
        "mean_absolute_error": fmean(abs(row["calibration_error"]) for row in rows),
        "brier_score": fmean(row["brier_score"] for row in rows),
        "log_loss": fmean(row["log_loss"] for row in rows),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "regression_n": regression_n,
    }


calibration_summaries = [summarize_calibration(calibration_rows, "pooled")]
for reliability_exact in reliability_order:
    selected = [
        row for row in calibration_rows
        if row["reliability_exact"] == reliability_exact
    ]
    calibration_summaries.append(
        summarize_calibration(selected, f"r={reliability_exact}")
    )

summary_columns = list(calibration_summaries[0])
display(SQLResult(
    summary_columns,
    [[row[column] for column in summary_columns] for row in calibration_summaries],
))
'''
    ),
    code(
        r'''
fig, axis = plt.subplots(figsize=(6.5, 5.5))
bin_count = 10
bins = defaultdict(list)
for row in calibration_rows:
    index = min(int(row["q_llm"] * bin_count), bin_count - 1)
    bins[index].append(row)

predicted_means = [fmean(row["q_llm"] for row in bins[index]) for index in sorted(bins)]
target_means = [fmean(row["q_true"] for row in bins[index]) for index in sorted(bins)]
counts = [len(bins[index]) for index in sorted(bins)]
axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
axis.plot(predicted_means, target_means, "o-", color="tab:blue", label="observed")
for x_value, y_value, count in zip(predicted_means, target_means, counts, strict=True):
    axis.annotate(
        str(count), (x_value, y_value), xytext=(3, 3),
        textcoords="offset points", fontsize=7,
    )
axis.set(
    xlim=(0, 1), ylim=(0, 1),
    xlabel=r"mean $q_{\mathrm{LLM}}$ in bin",
    ylabel=r"mean $q_{\mathrm{true}}$ in bin",
    title="Reasoning-off reliability diagram (labels are bin counts)",
)
axis.legend()
fig.tight_layout()
plt.show()
'''
    ),
    code(
        r'''
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].scatter(
    [row["q_true"] for row in calibration_rows],
    [row["q_llm"] for row in calibration_rows],
    s=18, alpha=0.20, color="tab:blue",
)
axes[0].plot([0, 1], [0, 1], "--", color="black")
axes[0].set(
    xlim=(0, 1), ylim=(0, 1),
    xlabel=r"$q_{\mathrm{true}}$",
    ylabel=r"$q_{\mathrm{LLM}}$",
    title="Pair-normalized probability",
)

mean_errors = []
for reliability_exact in reliability_order:
    rows = [
        row for row in calibration_rows
        if row["reliability_exact"] == reliability_exact
    ]
    mean_errors.append(fmean(row["calibration_error"] for row in rows))
axes[1].bar(range(len(reliability_order)), mean_errors, color="tab:blue")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xticks(range(len(reliability_order)), reliability_order, rotation=35)
axes[1].set(
    xlabel="reliability",
    ylabel=r"mean $q_{\mathrm{LLM}}-q_{\mathrm{true}}$",
    title="Signed calibration error",
)

pooled = calibration_summaries[0]
metric_names = ["brier_score", "log_loss"]
axes[2].bar(
    range(len(metric_names)), [pooled[name] for name in metric_names],
    color="tab:blue",
)
axes[2].set_xticks(range(len(metric_names)), ["Brier", "log loss"])
axes[2].set(title="Pooled proper scoring rules", ylabel="lower is better")

fig.suptitle("Reasoning-off calibration diagnostics")
fig.tight_layout()
plt.show()
'''
    ),
    markdown(
        """
## 6. Emitted-choice and boundary-logit behavior

The following tables keep the generated answer separate from the answer-boundary argmax. Tied Bayesian targets remain unscored instead of being assigned an arbitrary correct answer.
"""
    ),
    code(
        '''
display(explorer.sql("""
SELECT
    reliability_exact,
    model_choice_canonical AS emitted_choice,
    logit_choice,
    COUNT(*) AS n,
    SUM(canonical_posterior_correct IS NOT NULL) AS emitted_scored,
    AVG(canonical_posterior_correct) AS emitted_accuracy,
    SUM(logit_correct IS NOT NULL) AS logit_scored,
    AVG(logit_correct) AS logit_accuracy
FROM transcript_rows
GROUP BY reliability_exact, reliability, model_choice_canonical, logit_choice
ORDER BY reliability, emitted_choice, logit_choice
"""))

display(explorer.sql("""
SELECT
    CASE
        WHEN agreement_difference > 0 THEN 'agreement favors C1'
        WHEN agreement_difference < 0 THEN 'agreement favors C2'
        ELSE 'agreement tie'
    END AS agreement_direction,
    model_choice_canonical AS emitted_choice,
    COUNT(*) AS n,
    AVG(z) AS mean_z,
    AVG(q_llm) AS mean_q_llm
FROM transcript_rows
GROUP BY agreement_direction, emitted_choice
ORDER BY agreement_direction, emitted_choice
"""))
'''
    ),
    markdown(
        r"""
## 7. Conditional reasoning-off selection audits

These reproduce notebook 14's two separate exploration queries on the new transcript set. Selection means the actual generated `model_choice_canonical`. `matching_outputs` is also the number of distinct evidence scenarios here because this run has exactly one presentation per scenario.

Each detail table includes question membership, answer pattern, reliability, agreement vectors and totals, likelihoods, posteriors, \(q_{true}\), \(q_{LLM}\), calibration metrics, raw logits, logit-implied choice, emitted choice, and correctness.
"""
    ),
    code(
        '''
SELECTION_AUDIT_COLUMNS = """
    row_id,
    positional_control_pair_id,
    question_set_index,
    question_subsets_json AS question_membership,
    observed_reports_json AS observed_answers,
    answer_pattern,
    reliability_exact,
    reliability,
    presentation_order,
    candidate_1,
    candidate_2,
    candidate_1_agreements_json,
    candidate_2_agreements_json,
    total_agreement_candidate_1,
    total_agreement_candidate_2,
    agreement_difference,
    candidate_1_likelihood_total,
    candidate_2_likelihood_total,
    candidate_1_posterior,
    candidate_2_posterior,
    total_marginalized_probability,
    q_true,
    candidate_1_raw_logit,
    candidate_2_raw_logit,
    z,
    q_llm,
    q_llm - q_true AS calibration_error,
    ABS(q_llm - q_true) AS absolute_calibration_error,
    (q_llm - q_true) * (q_llm - q_true) AS brier_score,
    CASE
        WHEN q_true IS NULL OR q_llm IS NULL THEN NULL
        ELSE -(
            q_true * LN(MIN(MAX(q_llm, 1e-12), 1.0 - 1e-12))
            + (1.0 - q_true)
              * LN(1.0 - MIN(MAX(q_llm, 1e-12), 1.0 - 1e-12))
        )
    END AS log_loss,
    logit_choice,
    logit_correct,
    model_choice_canonical AS emitted_choice,
    canonical_posterior_correct AS emitted_choice_correct,
    strict_answer_compliance,
    answer_boundary_available
"""


def selection_audit(condition_sql: str) -> tuple[SQLResult, SQLResult]:
    predicate = f"reasoning = 0 AND ({condition_sql})"
    counts = explorer.sql(f"""
        SELECT
            COUNT(*) AS matching_outputs,
            COUNT(DISTINCT positional_control_pair_id)
                AS distinct_evidence_scenarios,
            COUNT(DISTINCT question_set_index) AS represented_question_sets,
            SUM(answer_boundary_available) AS outputs_with_boundary_logits
        FROM transcript_rows
        WHERE {predicate}
    """)
    details = explorer.sql(f"""
        SELECT {SELECTION_AUDIT_COLUMNS}
        FROM transcript_rows
        WHERE {predicate}
        ORDER BY
            reliability, question_set_index, answer_pattern_index, presentation_order
    """)
    return counts, details
'''
    ),
    markdown(
        r"""
### 7.1 \(r>0.5\), agreement(C1) > agreement(C2), but C2 was emitted
"""
    ),
    code(
        '''
high_reliability_c2_counts, high_reliability_c2_outputs = selection_audit(
    "reliability > 0.5 "
    "AND total_agreement_candidate_1 > total_agreement_candidate_2 "
    "AND model_choice_canonical = 'C2'"
)
display(high_reliability_c2_counts)
display(high_reliability_c2_outputs)
'''
    ),
    markdown(
        r"""
### 7.2 \(r<0.5\), agreement(C2) > agreement(C1), but C1 was emitted
"""
    ),
    code(
        '''
low_reliability_c1_counts, low_reliability_c1_outputs = selection_audit(
    "reliability < 0.5 "
    "AND total_agreement_candidate_2 > total_agreement_candidate_1 "
    "AND model_choice_canonical = 'C1'"
)
display(low_reliability_c1_counts)
display(low_reliability_c1_outputs)
'''
    ),
    markdown(
        """
## Further exploration recipes

All result objects are ordinary Python dictionaries as well as SQL rows. The source artifact remains read-only; use `inspect_transcript(row_id, include_prompt=True)` to open any exact prompt and completion.
"""
    ),
    code(
        '''
# Question schedules with the most emitted-choice errors.
display(explorer.sql("""
SELECT
    question_set_index,
    reliability_exact,
    COUNT(*) AS scored_rows,
    SUM(canonical_posterior_correct = 0) AS emitted_choice_errors,
    AVG(ABS(q_llm - q_true)) AS mean_absolute_calibration_error,
    AVG(ABS(z)) AS mean_absolute_logit_contrast
FROM transcript_rows
WHERE canonical_posterior_correct IS NOT NULL
GROUP BY question_set_index, reliability_exact, reliability
HAVING COUNT(*) >= 5
ORDER BY emitted_choice_errors DESC, mean_absolute_calibration_error DESC
LIMIT 30
"""))

# A narrow condition query; edit values or selected columns freely.
display(
    explorer.query("transcript_rows")
    .select(
        "row_id",
        "question_set_index",
        "answer_pattern",
        "question_subsets_json",
        "observed_reports_json",
        "candidate_1_posterior",
        "candidate_2_posterior",
        "q_true",
        "z",
        "q_llm",
        "model_choice_canonical",
        "canonical_posterior_correct",
    )
    .where_eq(reliability_exact="19/20", reasoning=False)
    .where("question_set_index IN (?, ?)", 0, 1)
    .sort("question_set_index", "answer_pattern_index")
    .run()
)
'''
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (MATS CUDA)",
            "language": "python",
            "name": "mats-cuda",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
)
nbf.write(notebook, OUTPUT_PATH)
print(OUTPUT_PATH)
