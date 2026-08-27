"""Analyze all completed noisy-source posterior experiments and render report figures."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ARTIFACT_ROOT = REPO_ROOT / "artifacts"
OUTPUT_DIR = REPO_ROOT / "reports" / "posterior_behavior"
PRIMARY_ARM = ("user_only", "yes_no", "decimal", False)
LEGACY_PRIMARY_ARM = ("single_user", "yes_no", "decimal", False)
CHOICES = ("left", "right", "tie")
COLORS = {"left": "#2878B5", "right": "#D55E00", "tie": "#3A923A"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arm_key(row: dict[str, Any]) -> tuple[str, str, str, bool]:
    return (
        row["transcript_format"],
        row["answer_vocabulary"],
        row["reliability_format"],
        bool(row["enable_thinking"]),
    )


def target(row: dict[str, Any]) -> str:
    return row["probe"]["normative_choice"]


def prediction(row: dict[str, Any]) -> str:
    return row["predicted_semantic_choice"]


def balanced_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    labels = sorted({target(row) for row in rows})
    if not labels:
        return math.nan
    recalls = []
    for label in labels:
        class_rows = [row for row in rows if target(row) == label]
        recalls.append(sum(prediction(row) == label for row in class_rows) / len(class_rows))
    return sum(recalls) / len(recalls)


def mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        for index in ordered[start:end]:
            result[index] = average_rank
        start = end
    return result


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else math.nan


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(ranks(left), ranks(right))


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    cluster: Callable[[dict[str, Any]], object],
    *,
    resamples: int = 2_000,
    seed: int = 20260827,
) -> tuple[float, float, float]:
    grouped: dict[object, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[cluster(row)].append(row)
    identifiers = list(grouped)
    estimate = statistic(rows)
    if len(identifiers) < 2:
        return estimate, math.nan, math.nan
    rng = random.Random(seed)
    sampled_statistics = []
    for _ in range(resamples):
        sampled_ids = rng.choices(identifiers, k=len(identifiers))
        sample = [row for identifier in sampled_ids for row in grouped[identifier]]
        sampled_statistics.append(statistic(sample))
    return estimate, percentile(sampled_statistics, 0.025), percentile(sampled_statistics, 0.975)


def complexity_key(row: dict[str, Any], examples: dict[str, dict[str, Any]]) -> str:
    stage = row["stage"]
    if stage == "one_observation":
        return "1 observation\nN=2"
    if stage == "accumulation":
        turns = len(examples[row["example_id"]]["observations"])
        return f"{turns} observations\nN=2"
    if stage == "compositional":
        return "3 observations\nN=4"
    if row["probe"]["kind"] == "candidate":
        return "3 observations\nN=8 candidate"
    return "3 observations\nN=8 partition"


COMPLEXITY_ORDER = (
    "1 observation\nN=2",
    "2 observations\nN=2",
    "3 observations\nN=2",
    "4 observations\nN=2",
    "3 observations\nN=4",
    "3 observations\nN=8 candidate",
    "3 observations\nN=8 partition",
)


def task_family(row: dict[str, Any]) -> str:
    if row["stage"] == "one_observation":
        return "One observation, N=2"
    if row["stage"] == "accumulation":
        return "Accumulation, N=2"
    if row["stage"] == "compositional":
        return "Compositional, N=4"
    if row["probe"]["kind"] == "candidate":
        return "N=8 candidate"
    return "N=8 partition"


FAMILY_ORDER = (
    "One observation, N=2",
    "Accumulation, N=2",
    "Compositional, N=4",
    "N=8 candidate",
    "N=8 partition",
)


def cluster_id(row: dict[str, Any]) -> tuple[object, ...]:
    if row["stage"] == "n8":
        return ("n8", row["schedule_id"])
    return (row["stage"], row["schedule_id"], row["world_id"])


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUTPUT_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_progression(
    primary: list[dict[str, Any]], examples: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    nonneutral = [row for row in primary if row["reliability"] != 0.5]
    grouped = {
        key: [row for row in nonneutral if complexity_key(row, examples) == key]
        for key in COMPLEXITY_ORDER
    }
    balanced = []
    intervals = []
    mapping_accuracy = []
    compliance = []
    choice_mass = []
    for key in COMPLEXITY_ORDER:
        rows = grouped[key]
        estimate, lower, upper = cluster_bootstrap(rows, balanced_accuracy, cluster_id)
        balanced.append(estimate)
        intervals.append((lower, upper))
        mapping_accuracy.append(mean(rows, "mapping_accuracy"))
        compliance.append(mean(rows, "greedy_choice_compliance"))
        choice_mass.append(mean(rows, "mean_choice_probability_mass"))

    x = list(range(len(COMPLEXITY_ORDER)))
    labels = [label.replace("\n", "\n") for label in COMPLEXITY_ORDER]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    errors = [
        [value - lower for value, (lower, _) in zip(balanced, intervals)],
        [upper - value for value, (_, upper) in zip(balanced, intervals)],
    ]
    axes[0].bar(x, balanced, color="#4C78A8", alpha=0.9, label="Aggregate balanced accuracy")
    axes[0].errorbar(x, balanced, yerr=errors, fmt="none", color="black", capsize=3)
    axes[0].plot(x, mapping_accuracy, color="#F58518", marker="o", label="Mean per-mapping accuracy")
    axes[0].axhline(1 / 3, color="gray", linestyle="--", label="Three-way chance")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Non-neutral reliability conditions (r ≠ 0.5)")
    axes[0].legend(ncol=3, loc="upper right")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

    width = 0.38
    axes[1].bar([value - width / 2 for value in x], compliance, width, color="#54A24B", label="Greedy A/B/C compliance")
    axes[1].bar([value + width / 2 for value in x], choice_mass, width, color="#E45756", label="Total A/B/C probability mass")
    axes[1].set_ylabel("Fraction / probability")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks(x, labels)
    axes[1].legend(ncol=2)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    figure.suptitle("Behavior and instruction compliance degrade with task complexity", y=1.01)
    figure.tight_layout()
    save_figure(figure, "01_complexity_progression.png")
    return {
        key.replace("\n", " / "): {
            "n": len(grouped[key]),
            "balanced_accuracy_nonneutral": balanced[index],
            "balanced_accuracy_cluster_ci": intervals[index],
            "mapping_accuracy_nonneutral": mapping_accuracy[index],
            "greedy_choice_compliance": compliance[index],
            "choice_probability_mass": choice_mass[index],
        }
        for index, key in enumerate(COMPLEXITY_ORDER)
    }


def plot_reliability(primary: list[dict[str, Any]]) -> dict[str, Any]:
    reliabilities = sorted({row["reliability"] for row in primary})
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        family_rows = [row for row in primary if task_family(row) == family]
        accuracies = []
        tie_rates = []
        family_summary = {}
        for reliability in reliabilities:
            rows = [row for row in family_rows if row["reliability"] == reliability]
            accuracies.append(balanced_accuracy(rows))
            tie_rates.append(sum(prediction(row) == "tie" for row in rows) / len(rows))
            family_summary[str(reliability)] = {
                "n": len(rows),
                "accuracy": sum(bool(row["counterbalanced_correct"]) for row in rows) / len(rows),
                "balanced_accuracy": balanced_accuracy(rows),
                "predicted_tie_rate": tie_rates[-1],
            }
        summary[family] = family_summary
        axes[0].plot(reliabilities, accuracies, marker="o", label=family)
        axes[1].plot(reliabilities, tie_rates, marker="o", label=family)
    for axis in axes:
        axis.axvline(0.5, color="black", linestyle=":", linewidth=1.2)
        axis.set_xlabel("Declared reliability r")
        axis.set_ylim(-0.02, 1.02)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].axhline(1 / 3, color="gray", linestyle="--")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Conditional forced-choice accuracy")
    axes[1].set_ylabel("Predicted tie fraction")
    axes[1].set_title("Tie becomes the default in complex prompts")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.suptitle("Reliability sweep: r=0.5 is a special-case shortcut", y=1.02)
    figure.tight_layout()
    save_figure(figure, "02_reliability_sweep.png")
    return summary


def plot_log_odds(primary: list[dict[str, Any]]) -> dict[str, Any]:
    reliabilities = (0.1, 0.3, 0.7, 0.9)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    summary: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        correlations = []
        family_summary = {}
        for reliability in reliabilities:
            rows = [
                row
                for row in primary
                if task_family(row) == family
                and row["reliability"] == reliability
                and row["probe"]["target_log_odds_state"] == "finite"
            ]
            correlation = spearman(
                [float(row["probe"]["target_log_odds"]) for row in rows],
                [float(row["behavioral_log_odds"]) for row in rows],
            )
            correlations.append(correlation)
            family_summary[str(reliability)] = correlation
        summary[family] = family_summary
        axis.plot(reliabilities, correlations, marker="o", linewidth=2, label=family)
    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(0.5, color="black", linestyle=":", linewidth=1.2)
    axis.set_xticks(reliabilities)
    axis.set_xlabel("Declared reliability r")
    axis.set_ylabel("Spearman ρ: behavioral vs exact log-odds")
    axis.set_ylim(-0.4, 0.4)
    axis.set_title("Evidence preference has the wrong sign below r=0.5")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout()
    save_figure(figure, "03_log_odds_tracking.png")
    return summary


def plot_ties(primary: list[dict[str, Any]]) -> dict[str, Any]:
    families = FAMILY_ORDER[1:]
    uniform_recall = []
    interior_recall = []
    false_tie = []
    summary: dict[str, Any] = {}
    for family in families:
        rows = [row for row in primary if task_family(row) == family]
        uniform = [row for row in rows if row["probe"]["tie_type"] == "uniform_positive"]
        interior = [row for row in rows if row["probe"]["tie_type"] == "equal_positive"]
        nonties = [row for row in rows if row["probe"]["tie_type"] is None and row["reliability"] != 0.5]
        uniform_value = sum(prediction(row) == "tie" for row in uniform) / len(uniform)
        interior_value = (
            sum(prediction(row) == "tie" for row in interior) / len(interior)
            if interior
            else math.nan
        )
        false_value = sum(prediction(row) == "tie" for row in nonties) / len(nonties)
        uniform_recall.append(uniform_value)
        interior_recall.append(interior_value)
        false_tie.append(false_value)
        summary[family] = {
            "r_half_uniform_tie_recall": uniform_value,
            "interior_equal_evidence_tie_recall": interior_value,
            "false_tie_rate_on_nonties": false_value,
            "n_uniform_ties": len(uniform),
            "n_interior_ties": len(interior),
            "n_nonties": len(nonties),
        }
    x = list(range(len(families)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar([value - width for value in x], uniform_recall, width, label="Tie recall at r=0.5", color="#4C78A8")
    axis.bar(x, interior_recall, width, label="Interior equal-evidence tie recall", color="#F58518")
    axis.bar([value + width for value in x], false_tie, width, label="False tie rate on non-ties", color="#E45756")
    axis.set_xticks(x, families)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Fraction")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_title("Tie behavior changes from aversion to near-universal default")
    axis.legend(ncol=3, loc="upper left")
    figure.tight_layout()
    save_figure(figure, "04_tie_behavior.png")
    return summary


def confusion_matrix(rows: Sequence[dict[str, Any]]) -> list[list[int]]:
    return [
        [sum(target(row) == actual and prediction(row) == predicted for row in rows) for predicted in CHOICES]
        for actual in CHOICES
    ]


def plot_n8_confusions(primary: list[dict[str, Any]]) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    summary: dict[str, Any] = {}
    image = None
    for axis, kind in zip(axes, ("candidate", "partition")):
        rows = [
            row
            for row in primary
            if row["stage"] == "n8"
            and row["probe"]["kind"] == kind
            and row["reliability"] != 0.5
        ]
        counts = confusion_matrix(rows)
        normalized = [
            [value / sum(count_row) for value in count_row] for count_row in counts
        ]
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(3), CHOICES)
        axis.set_yticks(range(3), CHOICES)
        axis.set_xlabel("Conditional predicted meaning")
        axis.set_ylabel("Exact posterior meaning")
        axis.set_title(f"N=8 {kind}")
        for row_index, count_row in enumerate(counts):
            for column_index, count in enumerate(count_row):
                fraction = normalized[row_index][column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{count}\n{fraction:.0%}",
                    ha="center",
                    va="center",
                    color="white" if fraction > 0.55 else "black",
                )
        summary[kind] = {actual: dict(zip(CHOICES, counts[index])) for index, actual in enumerate(CHOICES)}
    if image is not None:
        figure.colorbar(image, ax=axes, label="Fraction within exact class")
    figure.suptitle("At N=8 the conditional scorer predicts tie for almost everything (r ≠ 0.5)")
    save_figure(figure, "05_n8_confusions.png")
    return summary


def plot_robustness(
    all_results: list[dict[str, Any]], primary: list[dict[str, Any]]
) -> dict[str, Any]:
    base = {(row["example_id"], row["probe"]["probe_id"]): row for row in primary}
    transcript_format = primary[0]["transcript_format"]
    names = {
        (transcript_format, "symbols", "decimal", False): "KET/ZOG",
        (transcript_format, "true_false", "decimal", False): "TRUE/FALSE",
        (transcript_format, "yes_no", "fraction", False): "Fraction r",
        (transcript_format, "yes_no", "percent", False): "Percent r",
    }
    labels = []
    accuracies = []
    matched_primary = []
    agreements = []
    logit_correlations = []
    summary: dict[str, Any] = {}
    for arm, label in names.items():
        rows = [row for row in all_results if arm_key(row) == arm]
        base_rows = [base[(row["example_id"], row["probe"]["probe_id"])] for row in rows]
        accuracy = sum(bool(row["counterbalanced_correct"]) for row in rows) / len(rows)
        base_accuracy = sum(bool(row["counterbalanced_correct"]) for row in base_rows) / len(base_rows)
        agreement = sum(prediction(row) == prediction(base_row) for row, base_row in zip(rows, base_rows)) / len(rows)
        correlation = pearson(
            [float(row["behavioral_log_odds"]) for row in rows],
            [float(row["behavioral_log_odds"]) for row in base_rows],
        )
        labels.append(label)
        accuracies.append(accuracy)
        matched_primary.append(base_accuracy)
        agreements.append(agreement)
        logit_correlations.append(correlation)
        summary[label] = {
            "n": len(rows),
            "conditional_accuracy": accuracy,
            "matched_primary_accuracy": base_accuracy,
            "conditional_choice_agreement": agreement,
            "behavioral_log_odds_correlation": correlation,
            "greedy_choice_compliance": mean(rows, "greedy_choice_compliance"),
        }
    x = list(range(len(labels)))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([value - width / 2 for value in x], matched_primary, width, label="Matched primary prompt", color="#4C78A8")
    axes[0].bar([value + width / 2 for value in x], accuracies, width, label="Robustness variant", color="#F58518")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 0.65)
    axes[0].set_ylabel("Conditional aggregate accuracy")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend()
    axes[0].set_title("No surface form reliably rescues accuracy")
    axes[1].bar([value - width / 2 for value in x], agreements, width, label="Same conditional class", color="#54A24B")
    axes[1].bar([value + width / 2 for value in x], logit_correlations, width, label="Decision-logit correlation", color="#B279A2")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend()
    axes[1].set_title("Surface variants mostly preserve the same preference")
    figure.tight_layout()
    save_figure(figure, "06_surface_robustness.png")
    return summary


def predicted_distributions(
    rows: Sequence[dict[str, Any]], reliability_field: str = "reliability"
) -> dict[str, dict[str, float]]:
    result = {}
    for reliability in sorted({float(row[reliability_field]) for row in rows}):
        selected = [row for row in rows if float(row[reliability_field]) == reliability]
        result[str(reliability)] = {
            choice: sum(prediction(row) == choice for row in selected) / len(selected)
            for choice in CHOICES
        }
    return result


def plot_prior_notebooks(
    notebook1: list[dict[str, Any]], notebook2: list[dict[str, Any]]
) -> dict[str, Any]:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    summary: dict[str, Any] = {}
    for row_index, (name, rows) in enumerate((("Notebook 1", notebook1), ("Notebook 2", notebook2))):
        for column_index, kind in enumerate(("candidate", "half")):
            selected = [row for row in rows if row["probe_kind"] == kind]
            reliabilities = sorted({float(row["reliability"]) for row in selected})
            bottoms = [0.0] * len(reliabilities)
            distribution = predicted_distributions(selected)
            for choice in CHOICES:
                values = [distribution[str(reliability)][choice] for reliability in reliabilities]
                axes[row_index, column_index].bar(
                    reliabilities,
                    values,
                    bottom=bottoms,
                    width=0.07,
                    color=COLORS[choice],
                    label=choice,
                )
                bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
            axes[row_index, column_index].set_title(f"{name}: {kind}")
            axes[row_index, column_index].set_xlabel("r")
            axes[row_index, column_index].yaxis.set_major_formatter(PercentFormatter(1))
            axes[row_index, column_index].set_ylim(0, 1)
            summary[f"{name} {kind}"] = distribution
    axes[0, 0].set_ylabel("Predicted semantic fraction")
    axes[1, 0].set_ylabel("Predicted semantic fraction")
    axes[0, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.suptitle("Earlier notebooks were dominated by fixed semantic and presentation preferences")
    figure.tight_layout()
    save_figure(figure, "07_prior_notebook_response_biases.png")
    return summary


def reliability_inversion(primary: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [row for row in primary if row["probe"]["kind"] == "candidate"]
    index = {
        (row["bank_id"], row["probe"]["probe_id"], row["reliability"]): row
        for row in candidate_rows
    }
    summary: dict[str, Any] = {}
    for family in FAMILY_ORDER[:-1]:
        pairs = []
        for row in candidate_rows:
            if task_family(row) != family or row["reliability"] not in (0.1, 0.3):
                continue
            partner = index.get(
                (row["bank_id"], row["probe"]["probe_id"], 1 - row["reliability"])
            )
            if partner is not None:
                pairs.append((row, partner))
        summary[family] = {
            "n_pairs": len(pairs),
            "behavioral_logit_sign_reversal": sum(
                (left["behavioral_log_odds"] > 0) != (right["behavioral_log_odds"] > 0)
                for left, right in pairs
            )
            / len(pairs),
            "left_right_class_flip": sum(
                {prediction(left), prediction(right)} == {"left", "right"}
                for left, right in pairs
            )
            / len(pairs),
            "same_predicted_class": sum(
                prediction(left) == prediction(right) for left, right in pairs
            )
            / len(pairs),
            "mean_abs_logit_antisymmetry_error": sum(
                abs(left["behavioral_log_odds"] + right["behavioral_log_odds"])
                for left, right in pairs
            )
            / len(pairs),
        }
    return summary


def greedy_token_summary(primary: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        rows = [row for row in primary if task_family(row) == family]
        token_counts = Counter(
            mapping["greedy_token_text"] for row in rows for mapping in row["mappings"]
        )
        summary[family] = {
            "greedy_tokens": dict(token_counts.most_common()),
            "greedy_choice_compliance": mean(rows, "greedy_choice_compliance"),
            "mean_choice_probability_mass": mean(rows, "mean_choice_probability_mass"),
        }
    return summary


def natural_n8(primary: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for kind in ("candidate", "partition"):
        for reliability in sorted({row["reliability"] for row in primary}):
            rows = [
                row
                for row in primary
                if row["stage"] == "n8"
                and row["probe"]["kind"] == kind
                and row["reliability"] == reliability
            ]
            weight = sum(float(row["prior_predictive_probability"]) for row in rows)
            summary[f"{kind} r={reliability}"] = {
                "controlled_accuracy": sum(bool(row["counterbalanced_correct"]) for row in rows)
                / len(rows),
                "natural_weighted_accuracy": sum(
                    float(row["prior_predictive_probability"])
                    * bool(row["counterbalanced_correct"])
                    for row in rows
                )
                / weight,
            }
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    controlled_root = ARTIFACT_ROOT / "controlled_posterior"
    all_results = load_jsonl(controlled_root / "qwen_results.jsonl")
    elicitation = load_jsonl(controlled_root / "qwen_elicitation_results.jsonl")
    examples = {
        row["example_id"]: row
        for filename in ("ladder.jsonl", "n8_fixed_bank.jsonl")
        for row in load_jsonl(controlled_root / filename)
    }
    primary = [row for row in all_results if arm_key(row) == PRIMARY_ARM]
    if not primary:
        primary = [row for row in all_results if arm_key(row) == LEGACY_PRIMARY_ARM]
    notebook1 = load_jsonl(
        ARTIFACT_ROOT / "minimal_noisy_source" / "qwen_counterbalanced_results.jsonl"
    )
    notebook2 = load_jsonl(
        ARTIFACT_ROOT / "fixed_transcript_reliability" / "qwen_counterbalanced_results.jsonl"
    )

    summary = {
        "dataset": {
            "primary_aggregate_decisions": len(primary),
            "robustness_aggregate_decisions": len(all_results) - len(primary),
            "elicitation_controls": len(elicitation),
            "thinking_arm_present": (controlled_root / "qwen_thinking_capability_results.jsonl").exists(),
            "notebook1_decisions": len(notebook1),
            "notebook2_decisions": len(notebook2),
        },
        "elicitation": {
            "accuracy": sum(bool(row["counterbalanced_correct"]) for row in elicitation)
            / len(elicitation),
            "mapping_accuracy": mean(elicitation, "mapping_accuracy"),
            "greedy_choice_compliance": mean(elicitation, "greedy_choice_compliance"),
        },
        "complexity": plot_progression(primary, examples),
        "reliability": plot_reliability(primary),
        "log_odds": plot_log_odds(primary),
        "ties": plot_ties(primary),
        "n8_confusions": plot_n8_confusions(primary),
        "robustness": plot_robustness(all_results, primary),
        "prior_notebooks": plot_prior_notebooks(notebook1, notebook2),
        "reliability_inversion": reliability_inversion(primary),
        "greedy_tokens": greedy_token_summary(primary),
        "natural_n8": natural_n8(primary),
    }
    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
        handle.write("\n")
    print(f"Wrote {len(list(OUTPUT_DIR.glob('*.png')))} figures and analysis_summary.json")


if __name__ == "__main__":
    main()
