from mats_experiments.controlled_posterior import (
    LABEL_ASSIGNMENTS,
    LABELS,
    ExperimentConfig,
    build_ladder_examples,
)
from mats_experiments.qwen_scoring import aggregate_counterbalanced_scores


def test_counterbalanced_aggregation_recovers_semantics_and_log_odds() -> None:
    example = build_ladder_examples(ExperimentConfig(world_count=1))[0]
    probe = example.probes[0]
    semantic_scores = {"left": -2.0, "right": 2.0, "tie": -3.0}
    permutation_scores = []
    for assignment in LABEL_ASSIGNMENTS:
        label_scores = {
            label: semantic_scores[semantic]
            for label, semantic in zip(LABELS, assignment)
        }
        permutation_scores.append(
            {
                "label_logits": label_scores,
                "label_log_probabilities": label_scores,
                "greedy_token_id": 0,
                "greedy_token_text": max(label_scores, key=label_scores.get),
                "greedy_is_choice_label": True,
            }
        )

    result = aggregate_counterbalanced_scores(
        example=example,
        probe=probe,
        label_assignments=LABEL_ASSIGNMENTS,
        permutation_scores=permutation_scores,
        transcript_format="user_only",
        answer_vocabulary="yes_no",
        reliability_format="decimal",
        enable_thinking=False,
    )

    assert result["predicted_semantic_choice"] == "right"
    assert result["mapping_accuracy"] == 1.0
    assert result["behavioral_log_odds"] == -4.0
