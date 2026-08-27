"""Small dependency-free analysis helpers for controlled posterior results."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TypeVar

Row = TypeVar("Row")


def balanced_accuracy(
    rows: Sequence[Row],
    *,
    target: Callable[[Row], str],
    prediction: Callable[[Row], str],
) -> float:
    recalls: list[float] = []
    for label in sorted({target(row) for row in rows}):
        class_rows = [row for row in rows if target(row) == label]
        recalls.append(sum(prediction(row) == label for row in class_rows) / len(class_rows))
    return sum(recalls) / len(recalls) if recalls else math.nan


def clustered_bootstrap_mean(
    rows: Sequence[Row],
    *,
    value: Callable[[Row], float],
    cluster: Callable[[Row], Hashable],
    resamples: int = 2_000,
    seed: int = 20260827,
) -> tuple[float, float, float]:
    """Return estimate and percentile CI after resampling independent clusters."""

    by_cluster: dict[Hashable, list[Row]] = defaultdict(list)
    for row in rows:
        by_cluster[cluster(row)].append(row)
    cluster_ids = list(by_cluster)
    if not cluster_ids:
        return math.nan, math.nan, math.nan
    estimate = sum(value(row) for row in rows) / len(rows)
    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    for _ in range(resamples):
        sampled_ids = rng.choices(cluster_ids, k=len(cluster_ids))
        sampled_rows = [row for cluster_id in sampled_ids for row in by_cluster[cluster_id]]
        bootstrap_values.append(
            sum(value(row) for row in sampled_rows) / len(sampled_rows)
        )
    bootstrap_values.sort()
    lower = bootstrap_values[int(0.025 * (resamples - 1))]
    upper = bootstrap_values[int(0.975 * (resamples - 1))]
    return estimate, lower, upper


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator else math.nan


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def heuristic_accuracies(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    candidate_rows = [row for row in rows if isinstance(row.get("heuristic_predictions"), dict)]
    if not candidate_rows:
        return {}
    names = sorted(candidate_rows[0]["heuristic_predictions"])  # type: ignore[arg-type]
    return {
        name: sum(
            row["heuristic_predictions"][name] == row["probe"]["normative_choice"]  # type: ignore[index]
            for row in candidate_rows
        )
        / len(candidate_rows)
        for name in names
    }
