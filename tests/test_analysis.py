from mats_experiments.analysis import (
    balanced_accuracy,
    clustered_bootstrap_mean,
    spearman_correlation,
)


def test_balanced_accuracy_does_not_follow_class_prevalence() -> None:
    rows = [
        {"target": "tie", "prediction": "tie"},
        {"target": "tie", "prediction": "tie"},
        {"target": "left", "prediction": "tie"},
    ]
    assert balanced_accuracy(
        rows,
        target=lambda row: row["target"],
        prediction=lambda row: row["prediction"],
    ) == 0.5


def test_spearman_handles_tied_ranks() -> None:
    assert spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == 1.0
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == -1.0


def test_cluster_bootstrap_reports_point_estimate() -> None:
    rows = [
        {"cluster": 0, "value": 0.0},
        {"cluster": 0, "value": 0.0},
        {"cluster": 1, "value": 1.0},
        {"cluster": 1, "value": 1.0},
    ]
    estimate, lower, upper = clustered_bootstrap_mean(
        rows,
        value=lambda row: row["value"],
        cluster=lambda row: row["cluster"],
        resamples=200,
    )
    assert estimate == 0.5
    assert lower <= estimate <= upper
