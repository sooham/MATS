"""Batched, label-counterbalanced scoring for the controlled posterior tasks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .controlled_posterior import (
    LABELS,
    SEMANTIC_CHOICES,
    ControlledExample,
    ElicitationControl,
    Probe,
    SemanticChoice,
    candidate_heuristics,
    serialize_probe,
)


def label_token_ids(processor: Any) -> dict[str, int]:
    """Resolve single-token choice labels for the loaded tokenizer."""

    result: dict[str, int] = {}
    for label in LABELS:
        token_ids = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"Choice label {label!r} is not one token: {token_ids}")
        result[label] = token_ids[0]
    if len(set(result.values())) != len(result):
        raise ValueError(f"Choice labels do not have distinct token IDs: {result}")
    return result


def validate_contextual_label_tokens(
    processor: Any, choice_token_ids: Mapping[str, int], *, enable_thinking: bool = False
) -> None:
    """Check that an assistant answer begins with the same IDs scored at the prompt boundary."""

    prefix_messages = [
        {"role": "system", "content": "Reply with exactly A, B, or C."},
        {"role": "user", "content": "Choose A."},
    ]
    prefix_ids = processor.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if prefix_ids and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    for label, token_id in choice_token_ids.items():
        full_ids = processor.apply_chat_template(
            [*prefix_messages, {"role": "assistant", "content": label}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        if full_ids and isinstance(full_ids[0], list):
            full_ids = full_ids[0]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Chat template changed before the assistant response boundary.")
        continuation = full_ids[len(prefix_ids) :]
        if not continuation or continuation[0] != token_id:
            raise ValueError(
                f"Contextual token for {label!r} does not match scored token ID {token_id}."
            )


def _relative_log_scores(log_scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max(log_scores.values())
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in log_scores.values())
    )
    return {key: value - log_normalizer for key, value in log_scores.items()}


def score_labels_batch(
    *,
    model: Any,
    processor: Any,
    device: Any,
    messages_batch: Sequence[Sequence[dict[str, str]]],
    choice_token_ids: Mapping[str, int],
    enable_thinking: bool = False,
) -> list[dict[str, object]]:
    """Score A/B/C and retain the actual full-vocabulary greedy first token."""

    import torch

    if not messages_batch:
        return []
    prompt = processor.apply_chat_template(
        [list(messages) for messages in messages_batch],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
        processor_kwargs={"padding": True},
    )
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)
    ordered_choice_ids = torch.tensor(
        [choice_token_ids[label] for label in LABELS], device=device
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        next_logits = outputs.logits[:, -1, :].float()
        selected_logits = next_logits.index_select(-1, ordered_choice_ids).cpu()
        selected_log_probabilities = (
            next_logits.log_softmax(-1).index_select(-1, ordered_choice_ids).cpu()
        )
        greedy_ids = next_logits.argmax(-1).cpu().tolist()

    records: list[dict[str, object]] = []
    choice_id_set = set(choice_token_ids.values())
    for logits, log_probabilities, greedy_id in zip(
        selected_logits.tolist(), selected_log_probabilities.tolist(), greedy_ids
    ):
        records.append(
            {
                "label_logits": dict(zip(LABELS, map(float, logits))),
                "label_log_probabilities": dict(
                    zip(LABELS, map(float, log_probabilities))
                ),
                "greedy_token_id": int(greedy_id),
                "greedy_token_text": processor.tokenizer.decode([greedy_id]),
                "greedy_is_choice_label": greedy_id in choice_id_set,
            }
        )
    return records


def generate_responses_batch(
    *,
    model: Any,
    processor: Any,
    device: Any,
    messages_batch: Sequence[Sequence[dict[str, str]]],
    enable_thinking: bool,
    max_new_tokens: int = 512,
) -> list[dict[str, object]]:
    """Greedily generate full responses for the separate deliberative capability arm."""

    import torch

    if not messages_batch:
        return []
    prompt = processor.apply_chat_template(
        [list(messages) for messages in messages_batch],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
        processor_kwargs={"padding": True},
    )
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    continuation_ids = generated[:, input_ids.shape[1] :]
    texts = processor.tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
    records: list[dict[str, object]] = []
    for text in texts:
        label_matches = re.findall(r"(?<![A-Z])([ABC])(?![A-Z])", text.upper())
        records.append(
            {
                "generated_text": text,
                "parsed_label": label_matches[-1] if label_matches else None,
                "parse_success": bool(label_matches),
            }
        )
    return records


def aggregate_counterbalanced_scores(
    *,
    example: ControlledExample,
    probe: Probe,
    label_assignments: Sequence[Sequence[SemanticChoice]],
    permutation_scores: Sequence[Mapping[str, object]],
    transcript_format: str,
    answer_vocabulary: str,
    reliability_format: str,
    enable_thinking: bool,
) -> dict[str, object]:
    """Aggregate mappings without treating them as independent examples."""

    if len(label_assignments) != len(permutation_scores):
        raise ValueError("Every label assignment must have exactly one score record.")
    semantic_probability_values: dict[str, list[float]] = {
        semantic: [] for semantic in SEMANTIC_CHOICES
    }
    behavioral_log_odds: list[float] = []
    mapping_records: list[dict[str, object]] = []

    for assignment, raw_score in zip(label_assignments, permutation_scores):
        label_log_probabilities = raw_score["label_log_probabilities"]
        if not isinstance(label_log_probabilities, Mapping):
            raise TypeError("label_log_probabilities must be a mapping.")
        typed_log_probabilities = {
            str(label): float(value) for label, value in label_log_probabilities.items()
        }
        conditional_logs = _relative_log_scores(typed_log_probabilities)
        label_to_semantic = dict(zip(LABELS, assignment))
        semantic_to_label = {semantic: label for label, semantic in label_to_semantic.items()}
        semantic_probabilities = {
            semantic: math.exp(conditional_logs[semantic_to_label[semantic]])
            for semantic in SEMANTIC_CHOICES
        }
        for semantic, probability in semantic_probabilities.items():
            semantic_probability_values[semantic].append(probability)
        behavioral_log_odds.append(
            typed_log_probabilities[semantic_to_label["left"]]
            - typed_log_probabilities[semantic_to_label["right"]]
        )
        predicted_label = max(typed_log_probabilities, key=typed_log_probabilities.get)
        predicted_semantic = label_to_semantic[predicted_label]
        choice_mass = sum(math.exp(value) for value in typed_log_probabilities.values())
        mapping_records.append(
            {
                "semantic_by_label": label_to_semantic,
                "label_logits": raw_score["label_logits"],
                "label_log_probabilities": typed_log_probabilities,
                "conditional_semantic_probabilities": semantic_probabilities,
                "predicted_label": predicted_label,
                "predicted_semantic_choice": predicted_semantic,
                "normative_correct": predicted_semantic == probe.normative_choice,
                "choice_probability_mass": choice_mass,
                "greedy_token_id": raw_score["greedy_token_id"],
                "greedy_token_text": raw_score["greedy_token_text"],
                "greedy_is_choice_label": raw_score["greedy_is_choice_label"],
            }
        )

    mean_semantic_probabilities = {
        semantic: sum(values) / len(values)
        for semantic, values in semantic_probability_values.items()
    }
    predicted_semantic = max(
        mean_semantic_probabilities, key=mean_semantic_probabilities.get
    )
    return {
        "example_id": example.example_id,
        "bank_id": example.bank_id,
        "stage": example.stage,
        "schedule_id": example.schedule_id,
        "world_id": example.world_id,
        "reliability": float(example.reliability),
        "reliability_exact": (
            f"{example.reliability.numerator}/{example.reliability.denominator}"
        ),
        "prior_predictive_probability": float(example.prior_predictive_probability),
        "probe": serialize_probe(probe),
        "transcript_format": transcript_format,
        "answer_vocabulary": answer_vocabulary,
        "reliability_format": reliability_format,
        "enable_thinking": enable_thinking,
        "mean_semantic_probabilities": mean_semantic_probabilities,
        "predicted_semantic_choice": predicted_semantic,
        "counterbalanced_correct": predicted_semantic == probe.normative_choice,
        "mapping_accuracy": sum(
            bool(record["normative_correct"]) for record in mapping_records
        )
        / len(mapping_records),
        "behavioral_log_odds": sum(behavioral_log_odds) / len(behavioral_log_odds),
        "heuristic_predictions": candidate_heuristics(example.observations, probe),
        "orientation_consistency_estimand": "behavioral_log_odds",
        "mean_choice_probability_mass": sum(
            float(record["choice_probability_mass"]) for record in mapping_records
        )
        / len(mapping_records),
        "greedy_choice_compliance": sum(
            bool(record["greedy_is_choice_label"]) for record in mapping_records
        )
        / len(mapping_records),
        "mappings": mapping_records,
    }


def aggregate_elicitation_scores(
    *,
    control: ElicitationControl,
    label_assignments: Sequence[Sequence[SemanticChoice]],
    permutation_scores: Sequence[Mapping[str, object]],
    enable_thinking: bool,
) -> dict[str, object]:
    """Aggregate the non-narrative Stage-0 controls."""

    semantic_values: dict[str, list[float]] = {
        semantic: [] for semantic in SEMANTIC_CHOICES
    }
    mapping_records: list[dict[str, object]] = []
    for assignment, raw_score in zip(label_assignments, permutation_scores):
        raw_log_probabilities = raw_score["label_log_probabilities"]
        if not isinstance(raw_log_probabilities, Mapping):
            raise TypeError("label_log_probabilities must be a mapping.")
        label_logs = {
            str(label): float(value) for label, value in raw_log_probabilities.items()
        }
        conditional_logs = _relative_log_scores(label_logs)
        label_to_semantic = dict(zip(LABELS, assignment))
        semantic_to_label = {semantic: label for label, semantic in label_to_semantic.items()}
        probabilities = {
            semantic: math.exp(conditional_logs[semantic_to_label[semantic]])
            for semantic in SEMANTIC_CHOICES
        }
        for semantic, probability in probabilities.items():
            semantic_values[semantic].append(probability)
        predicted_label = max(label_logs, key=label_logs.get)
        predicted_semantic = label_to_semantic[predicted_label]
        mapping_records.append(
            {
                "semantic_by_label": label_to_semantic,
                "predicted_semantic_choice": predicted_semantic,
                "normative_correct": predicted_semantic == control.normative_choice,
                "conditional_semantic_probabilities": probabilities,
                "greedy_token_text": raw_score["greedy_token_text"],
                "greedy_is_choice_label": raw_score["greedy_is_choice_label"],
            }
        )
    mean_probabilities = {
        semantic: sum(values) / len(values) for semantic, values in semantic_values.items()
    }
    predicted = max(mean_probabilities, key=mean_probabilities.get)
    return {
        "control_id": control.control_id,
        "stage": "elicitation",
        "left_weight_exact": (
            f"{control.left_weight.numerator}/{control.left_weight.denominator}"
        ),
        "right_weight_exact": (
            f"{control.right_weight.numerator}/{control.right_weight.denominator}"
        ),
        "normative_choice": control.normative_choice,
        "enable_thinking": enable_thinking,
        "mean_semantic_probabilities": mean_probabilities,
        "predicted_semantic_choice": predicted,
        "counterbalanced_correct": predicted == control.normative_choice,
        "mapping_accuracy": sum(
            bool(record["normative_correct"]) for record in mapping_records
        )
        / len(mapping_records),
        "greedy_choice_compliance": sum(
            bool(record["greedy_is_choice_label"]) for record in mapping_records
        )
        / len(mapping_records),
        "mappings": mapping_records,
    }
