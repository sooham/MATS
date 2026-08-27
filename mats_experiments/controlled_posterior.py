"""Controlled noisy-channel posterior experiments.

The module deliberately contains no model imports. Dataset generation, exact Bayesian
targets, prompt rendering, and design invariants can therefore be tested without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Literal

Answer = Literal["YES", "NO"]
SemanticChoice = Literal["left", "right", "tie"]
ProbeKind = Literal["candidate", "partition"]
TieType = Literal["uniform_positive", "equal_positive", "both_zero"]
TranscriptFormat = Literal["single_user", "alternating"]
AnswerVocabulary = Literal["yes_no", "true_false", "symbols"]
ReliabilityFormat = Literal["decimal", "percent", "fraction"]

LABELS = ("A", "B", "C")
SEMANTIC_CHOICES: tuple[SemanticChoice, ...] = ("left", "right", "tie")
LABEL_ASSIGNMENTS = tuple(permutations(SEMANTIC_CHOICES))
INTERIOR_RELIABILITIES = tuple(Fraction(value, 10) for value in (1, 3, 5, 7, 9))
ENDPOINT_RELIABILITIES = (Fraction(0), Fraction(1))


@dataclass(frozen=True)
class Observation:
    turn: int
    subset: tuple[str, ...]
    answer: Answer


@dataclass(frozen=True)
class Probe:
    probe_id: str
    kind: ProbeKind
    left: tuple[str, ...]
    right: tuple[str, ...]
    left_probability: Fraction
    right_probability: Fraction
    normative_choice: SemanticChoice
    target_log_odds: float | None
    target_log_odds_state: Literal["finite", "positive_infinity", "negative_infinity", "undefined"]
    tie_type: TieType | None
    left_match_count: int | None = None
    right_match_count: int | None = None


@dataclass(frozen=True)
class ControlledExample:
    example_id: str
    stage: Literal["one_observation", "accumulation", "compositional", "n8"]
    bank_id: str
    schedule_id: int
    world_id: int
    candidates: tuple[str, ...]
    reliability: Fraction
    observations: tuple[Observation, ...]
    posterior: tuple[Fraction, ...]
    prior_predictive_probability: Fraction
    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class ElicitationControl:
    control_id: str
    left_weight: Fraction
    right_weight: Fraction
    normative_choice: SemanticChoice


@dataclass(frozen=True)
class ExperimentConfig:
    reliabilities: tuple[Fraction, ...] = INTERIOR_RELIABILITIES
    world_count: int = 4
    include_endpoint_diagnostics: bool = False
    base_seed: int = 20260827


@dataclass(frozen=True)
class N8Config:
    reliabilities: tuple[Fraction, ...] = INTERIOR_RELIABILITIES
    endpoint_reliabilities: tuple[Fraction, ...] = ENDPOINT_RELIABILITIES
    num_schedules: int = 32
    num_turns: int = 3
    subset_size: int = 4
    base_seed: int = 20260827


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(map(str, (base_seed, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def answer_patterns(num_turns: int) -> tuple[tuple[Answer, ...], ...]:
    return tuple(product(("YES", "NO"), repeat=num_turns))  # type: ignore[return-value]


def exact_posterior(
    candidates: Sequence[str], observations: Sequence[Observation], reliability: Fraction
) -> tuple[tuple[Fraction, ...], Fraction]:
    """Return exact posterior and prior-predictive probability for a uniform prior."""

    if not candidates:
        raise ValueError("At least one candidate is required.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate labels must be unique.")
    if not Fraction(0) <= reliability <= Fraction(1):
        raise ValueError("Reliability must be in [0, 1].")
    candidate_set = set(candidates)
    if any(not set(observation.subset) <= candidate_set for observation in observations):
        raise ValueError("Every question subset must be contained in the candidate domain.")

    weights: list[Fraction] = []
    for candidate in candidates:
        weight = Fraction(1, len(candidates))
        for observation in observations:
            candidate_predicts_yes = candidate in observation.subset
            reported_yes = observation.answer == "YES"
            weight *= reliability if candidate_predicts_yes == reported_yes else 1 - reliability
        weights.append(weight)
    evidence = sum(weights, start=Fraction(0))
    if evidence == 0:
        raise ValueError("The observation history has zero probability under this channel.")
    posterior = tuple(weight / evidence for weight in weights)
    assert sum(posterior, start=Fraction(0)) == 1
    return posterior, evidence


def candidate_match_count(candidate: str, observations: Sequence[Observation]) -> int:
    return sum(
        (candidate in observation.subset) == (observation.answer == "YES")
        for observation in observations
    )


def compare(left: Fraction, right: Fraction) -> SemanticChoice:
    if left > right:
        return "left"
    if right > left:
        return "right"
    return "tie"


def compare_counts(left: int, right: int) -> SemanticChoice:
    if left > right:
        return "left"
    if right > left:
        return "right"
    return "tie"


def candidate_heuristics(
    observations: Sequence[Observation], probe: Probe
) -> dict[str, SemanticChoice] | None:
    """Preregister simple surface heuristics for candidate probes."""

    if probe.kind != "candidate":
        return None
    left, right = probe.left[0], probe.right[0]
    left_yes = sum(
        observation.answer == "YES" and left in observation.subset
        for observation in observations
    )
    right_yes = sum(
        observation.answer == "YES" and right in observation.subset
        for observation in observations
    )
    left_mentions = sum(left in observation.subset for observation in observations)
    right_mentions = sum(right in observation.subset for observation in observations)
    distinguishing = [
        observation
        for observation in observations
        if (left in observation.subset) != (right in observation.subset)
    ]
    if distinguishing:
        latest = distinguishing[-1]
        reported_yes = latest.answer == "YES"
        latest_supports_left = (left in latest.subset) == reported_yes
        recency: SemanticChoice = "left" if latest_supports_left else "right"
    else:
        recency = "tie"
    return {
        "always_left": "left",
        "always_right": "right",
        "always_tie": "tie",
        "yes_count": compare_counts(left_yes, right_yes),
        "mention_count": compare_counts(left_mentions, right_mentions),
        "recency": recency,
        "match_count_assume_r_above_half": compare_counts(
            candidate_match_count(left, observations),
            candidate_match_count(right, observations),
        ),
    }


def log_odds(
    left: Fraction, right: Fraction
) -> tuple[
    float | None,
    Literal["finite", "positive_infinity", "negative_infinity", "undefined"],
]:
    if left == right == 0:
        return None, "undefined"
    if right == 0:
        return None, "positive_infinity"
    if left == 0:
        return None, "negative_infinity"
    return math.log(float(left / right)), "finite"


def classify_tie(
    reliability: Fraction, left: Fraction, right: Fraction
) -> TieType | None:
    if left != right:
        return None
    if left == 0:
        return "both_zero"
    if reliability == Fraction(1, 2):
        return "uniform_positive"
    return "equal_positive"


def _probability_mass(
    candidates: Sequence[str], posterior: Sequence[Fraction], members: Sequence[str]
) -> Fraction:
    posterior_by_candidate = dict(zip(candidates, posterior))
    return sum((posterior_by_candidate[member] for member in members), start=Fraction(0))


def make_candidate_probe(
    *,
    probe_id: str,
    candidates: Sequence[str],
    observations: Sequence[Observation],
    posterior: Sequence[Fraction],
    reliability: Fraction,
    left: str,
    right: str,
) -> Probe:
    posterior_by_candidate = dict(zip(candidates, posterior))
    left_probability = posterior_by_candidate[left]
    right_probability = posterior_by_candidate[right]
    target_log_odds, state = log_odds(left_probability, right_probability)
    return Probe(
        probe_id=probe_id,
        kind="candidate",
        left=(left,),
        right=(right,),
        left_probability=left_probability,
        right_probability=right_probability,
        normative_choice=compare(left_probability, right_probability),
        target_log_odds=target_log_odds,
        target_log_odds_state=state,
        tie_type=classify_tie(reliability, left_probability, right_probability),
        left_match_count=candidate_match_count(left, observations),
        right_match_count=candidate_match_count(right, observations),
    )


def make_partition_probe(
    *,
    probe_id: str,
    candidates: Sequence[str],
    posterior: Sequence[Fraction],
    reliability: Fraction,
    left: Sequence[str],
    right: Sequence[str],
) -> Probe:
    if set(left) | set(right) != set(candidates) or set(left) & set(right):
        raise ValueError("Partition probe sides must be disjoint and cover the domain.")
    left_probability = _probability_mass(candidates, posterior, left)
    right_probability = _probability_mass(candidates, posterior, right)
    target_log_odds, state = log_odds(left_probability, right_probability)
    return Probe(
        probe_id=probe_id,
        kind="partition",
        left=tuple(left),
        right=tuple(right),
        left_probability=left_probability,
        right_probability=right_probability,
        normative_choice=compare(left_probability, right_probability),
        target_log_odds=target_log_odds,
        target_log_odds_state=state,
        tie_type=classify_tie(reliability, left_probability, right_probability),
    )


def _world_labels(size: int, world_id: int, base_seed: int, tag: str) -> tuple[str, ...]:
    pool = tuple(str(index) for index in range(1, max(8, size) + 1))
    rng = random.Random(stable_seed(base_seed, "world", tag, world_id))
    return tuple(rng.sample(pool, size))


def _build_small_example(
    *,
    stage: Literal["one_observation", "accumulation", "compositional"],
    schedule_id: int,
    world_id: int,
    canonical_subsets: Sequence[Sequence[int]],
    answer_pattern: tuple[Answer, ...],
    reliability: Fraction,
    domain_size: int,
    base_seed: int,
) -> ControlledExample:
    candidates = _world_labels(domain_size, world_id, base_seed, stage)
    observations = tuple(
        Observation(
            turn=turn,
            subset=tuple(candidates[index] for index in canonical_subset),
            answer=answer,
        )
        for turn, (canonical_subset, answer) in enumerate(
            zip(canonical_subsets, answer_pattern), start=1
        )
    )
    posterior, evidence = exact_posterior(candidates, observations, reliability)
    pattern_tag = "".join(answer[0] for answer in answer_pattern)
    bank_id = f"{stage}_schedule{schedule_id}_world{world_id}_{pattern_tag}"
    pair = (candidates[0], candidates[1])
    probes = (
        make_candidate_probe(
            probe_id=f"{bank_id}_candidate_forward",
            candidates=candidates,
            observations=observations,
            posterior=posterior,
            reliability=reliability,
            left=pair[0],
            right=pair[1],
        ),
        make_candidate_probe(
            probe_id=f"{bank_id}_candidate_reverse",
            candidates=candidates,
            observations=observations,
            posterior=posterior,
            reliability=reliability,
            left=pair[1],
            right=pair[0],
        ),
    )
    return ControlledExample(
        example_id=f"{bank_id}_r{reliability.numerator}-{reliability.denominator}",
        stage=stage,
        bank_id=bank_id,
        schedule_id=schedule_id,
        world_id=world_id,
        candidates=candidates,
        reliability=reliability,
        observations=observations,
        posterior=posterior,
        prior_predictive_probability=evidence,
        probes=probes,
    )


def _append_small_example(
    examples: list[ControlledExample],
    *,
    stage: Literal["one_observation", "accumulation", "compositional"],
    schedule_id: int,
    world_id: int,
    canonical_subsets: Sequence[Sequence[int]],
    answer_pattern: tuple[Answer, ...],
    reliability: Fraction,
    domain_size: int,
    base_seed: int,
) -> None:
    try:
        examples.append(
            _build_small_example(
                stage=stage,
                schedule_id=schedule_id,
                world_id=world_id,
                canonical_subsets=canonical_subsets,
                answer_pattern=answer_pattern,
                reliability=reliability,
                domain_size=domain_size,
                base_seed=base_seed,
            )
        )
    except ValueError as error:
        if reliability in ENDPOINT_RELIABILITIES and "zero probability" in str(error):
            return
        raise


def build_ladder_examples(config: ExperimentConfig | None = None) -> list[ControlledExample]:
    """Build the N=2/N=4 behavioral ladder with exhaustive visible answer patterns."""

    config = config or ExperimentConfig()
    reliabilities = config.reliabilities + (
        ENDPOINT_RELIABILITIES if config.include_endpoint_diagnostics else ()
    )
    examples: list[ControlledExample] = []
    for world_id in range(config.world_count):
        for answer_pattern in answer_patterns(1):
            for reliability in reliabilities:
                _append_small_example(
                    examples,
                    stage="one_observation",
                    schedule_id=0,
                    world_id=world_id,
                    canonical_subsets=((0,),),
                    answer_pattern=answer_pattern,
                    reliability=reliability,
                    domain_size=2,
                    base_seed=config.base_seed,
                )

        schedule_id = 1
        for num_turns in (2, 3, 4):
            schedule_shapes = (
                tuple((0,) for _ in range(num_turns)),
                tuple((turn % 2,) for turn in range(num_turns)),
            )
            for canonical_subsets in schedule_shapes:
                for answer_pattern in answer_patterns(num_turns):
                    for reliability in reliabilities:
                        _append_small_example(
                            examples,
                            stage="accumulation",
                            schedule_id=schedule_id,
                            world_id=world_id,
                            canonical_subsets=canonical_subsets,
                            answer_pattern=answer_pattern,
                            reliability=reliability,
                            domain_size=2,
                            base_seed=config.base_seed,
                        )
                schedule_id += 1

        compositional_schedules = (
            ((0, 1), (0, 2), (0, 3)),
            ((0,), (1, 2), (0, 3)),
        )
        for local_schedule_id, canonical_subsets in enumerate(compositional_schedules):
            for answer_pattern in answer_patterns(len(canonical_subsets)):
                for reliability in reliabilities:
                    _append_small_example(
                        examples,
                        stage="compositional",
                        schedule_id=local_schedule_id,
                        world_id=world_id,
                        canonical_subsets=canonical_subsets,
                        answer_pattern=answer_pattern,
                        reliability=reliability,
                        domain_size=4,
                        base_seed=config.base_seed,
                    )
    return examples


def _n8_schedule(config: N8Config, schedule_id: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(stable_seed(config.base_seed, "n8_questions", schedule_id))
    domain = tuple(range(8))
    return tuple(
        tuple(sorted(rng.sample(domain, config.subset_size)))
        for _ in range(config.num_turns)
    )


def _n8_world(config: N8Config, schedule_id: int) -> tuple[str, ...]:
    labels = [str(index) for index in range(1, 9)]
    rng = random.Random(stable_seed(config.base_seed, "n8_world", schedule_id))
    rng.shuffle(labels)
    return tuple(labels)


def _canonical_partitions() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    domain = set(range(8))
    left_sides = [side for side in combinations(range(8), 4) if 0 in side]
    return tuple((tuple(side), tuple(sorted(domain - set(side)))) for side in left_sides)


def build_n8_examples(
    config: N8Config | None = None, *, include_endpoint_diagnostics: bool = False
) -> list[ControlledExample]:
    """Build the controlled N=8 bank with paired orientations and fixed replay histories."""

    config = config or N8Config()
    if config.num_turns < 1:
        raise ValueError("num_turns must be positive.")
    if not 1 <= config.subset_size <= 8:
        raise ValueError("subset_size must be between 1 and 8.")
    reliabilities = config.reliabilities + (
        config.endpoint_reliabilities if include_endpoint_diagnostics else ()
    )
    candidate_pairs = tuple(combinations(range(8), 2))
    partitions = _canonical_partitions()
    examples: list[ControlledExample] = []
    for schedule_id in range(config.num_schedules):
        canonical_schedule = _n8_schedule(config, schedule_id)
        displayed_candidates = _n8_world(config, schedule_id)
        candidates = tuple(sorted(displayed_candidates, key=int))
        for pattern_index, answer_pattern in enumerate(answer_patterns(config.num_turns)):
            observations = tuple(
                Observation(
                    turn=turn,
                    subset=tuple(
                        sorted(
                            (displayed_candidates[index] for index in canonical_subset), key=int
                        )
                    ),
                    answer=answer,
                )
                for turn, (canonical_subset, answer) in enumerate(
                    zip(canonical_schedule, answer_pattern), start=1
                )
            )
            pair_indices = candidate_pairs[
                (schedule_id * len(answer_patterns(config.num_turns)) + pattern_index)
                % len(candidate_pairs)
            ]
            partition_indices = partitions[
                (schedule_id * 11 + pattern_index * 3) % len(partitions)
            ]
            candidate_pair = tuple(displayed_candidates[index] for index in pair_indices)
            partition = tuple(
                tuple(
                    sorted((displayed_candidates[index] for index in side), key=int)
                )
                for side in partition_indices
            )
            pattern_tag = "".join(answer[0] for answer in answer_pattern)
            bank_id = f"n8_schedule{schedule_id}_world{schedule_id}_{pattern_tag}"
            for reliability in reliabilities:
                try:
                    posterior, evidence = exact_posterior(
                        candidates, observations, reliability
                    )
                except ValueError as error:
                    if (
                        reliability in ENDPOINT_RELIABILITIES
                        and "zero probability" in str(error)
                    ):
                        continue
                    raise
                probes = (
                    make_candidate_probe(
                        probe_id=f"{bank_id}_candidate_forward",
                        candidates=candidates,
                        observations=observations,
                        posterior=posterior,
                        reliability=reliability,
                        left=candidate_pair[0],
                        right=candidate_pair[1],
                    ),
                    make_candidate_probe(
                        probe_id=f"{bank_id}_candidate_reverse",
                        candidates=candidates,
                        observations=observations,
                        posterior=posterior,
                        reliability=reliability,
                        left=candidate_pair[1],
                        right=candidate_pair[0],
                    ),
                    make_partition_probe(
                        probe_id=f"{bank_id}_partition_forward",
                        candidates=candidates,
                        posterior=posterior,
                        reliability=reliability,
                        left=partition[0],
                        right=partition[1],
                    ),
                    make_partition_probe(
                        probe_id=f"{bank_id}_partition_reverse",
                        candidates=candidates,
                        posterior=posterior,
                        reliability=reliability,
                        left=partition[1],
                        right=partition[0],
                    ),
                )
                examples.append(
                    ControlledExample(
                        example_id=f"{bank_id}_r{reliability.numerator}-{reliability.denominator}",
                        stage="n8",
                        bank_id=bank_id,
                        schedule_id=schedule_id,
                        world_id=schedule_id,
                        candidates=candidates,
                        reliability=reliability,
                        observations=observations,
                        posterior=posterior,
                        prior_predictive_probability=evidence,
                        probes=probes,
                    )
                )
    return examples


def build_elicitation_controls() -> list[ElicitationControl]:
    weight_pairs = (
        (Fraction(9, 10), Fraction(1, 10)),
        (Fraction(7, 10), Fraction(3, 10)),
        (Fraction(1, 2), Fraction(1, 2)),
    )
    controls: list[ElicitationControl] = []
    for index, pair in enumerate(weight_pairs):
        for orientation, (left, right) in enumerate((pair, pair[::-1])):
            controls.append(
                ElicitationControl(
                    control_id=f"elicitation_{index}_{orientation}",
                    left_weight=left,
                    right_weight=right,
                    normative_choice=compare(left, right),
                )
            )
    return controls


def _format_set(values: Sequence[str]) -> str:
    return "{" + ", ".join(values) + "}"


def _format_reliability(reliability: Fraction, format_name: ReliabilityFormat) -> str:
    if format_name == "decimal":
        return f"{float(reliability):g}"
    if format_name == "percent":
        return f"{100 * float(reliability):g}%"
    return fraction_string(reliability)


def _answer_legend(vocabulary: AnswerVocabulary) -> tuple[str, Mapping[Answer, str]]:
    if vocabulary == "yes_no":
        return "Answers use YES for true and NO for false.", {"YES": "YES", "NO": "NO"}
    if vocabulary == "true_false":
        return "Answers use TRUE for true and FALSE for false.", {"YES": "TRUE", "NO": "FALSE"}
    return "Answers use KET for true and ZOG for false.", {"YES": "KET", "NO": "ZOG"}


def _system_prompt(
    example: ControlledExample,
    answer_vocabulary: AnswerVocabulary,
    reliability_format: ReliabilityFormat,
) -> str:
    reliability_text = _format_reliability(example.reliability, reliability_format)
    error_text = _format_reliability(1 - example.reliability, reliability_format)
    legend, _ = _answer_legend(answer_vocabulary)
    return (
        "You are in a hidden-secret inference game. "
        f"The secret was drawn uniformly from {_format_set(example.candidates)}. "
        "For each externally selected membership question, SOURCE reports the proposition's "
        f"true answer with probability r={reliability_text} and the opposite answer with "
        f"probability {error_text}. Outcomes are conditionally independent. "
        "Question selection carries no information about the secret. The reliability may be below "
        f"one half. {legend} At TEST, compare the requested posterior probabilities."
    )


def _probe_text(probe: Probe, semantics_by_label: Sequence[SemanticChoice]) -> str:
    if set(semantics_by_label) != set(SEMANTIC_CHOICES):
        raise ValueError("A label assignment must contain left, right, and tie exactly once.")
    if probe.kind == "candidate":
        question = "TEST: Which candidate has greater posterior probability?"
        option_text = {
            "left": probe.left[0],
            "right": probe.right[0],
            "tie": "They have equal posterior probability.",
        }
    else:
        question = "TEST: Which set has greater total posterior probability?"
        option_text = {
            "left": _format_set(probe.left),
            "right": _format_set(probe.right),
            "tie": "They have equal posterior probability.",
        }
    lines = [question]
    lines.extend(
        f"{label}: {option_text[semantic]}"
        for label, semantic in zip(LABELS, semantics_by_label)
    )
    lines.append("Reply with exactly A, B, or C.")
    return "\n".join(lines)


def messages_for_probe(
    example: ControlledExample,
    probe: Probe,
    semantics_by_label: Sequence[SemanticChoice],
    *,
    transcript_format: TranscriptFormat = "single_user",
    answer_vocabulary: AnswerVocabulary = "yes_no",
    reliability_format: ReliabilityFormat = "decimal",
) -> list[dict[str, str]]:
    system = _system_prompt(example, answer_vocabulary, reliability_format)
    _, answer_mapping = _answer_legend(answer_vocabulary)
    probe_text = _probe_text(probe, semantics_by_label)
    if transcript_format == "single_user":
        evidence_lines = ["OBSERVATIONS:"]
        for observation in example.observations:
            evidence_lines.append(
                f"Question {observation.turn}: Is the secret in {_format_set(observation.subset)}?"
            )
            evidence_lines.append(f"SOURCE answer: {answer_mapping[observation.answer]}.")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join((*evidence_lines, "", probe_text))},
        ]
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for observation in example.observations:
        messages.extend(
            (
                {
                    "role": "assistant",
                    "content": (
                        f"Question {observation.turn}: Is the secret in "
                        f"{_format_set(observation.subset)}?"
                    ),
                },
                {
                    "role": "user",
                    "content": f"SOURCE answer: {answer_mapping[observation.answer]}.",
                },
            )
        )
    messages.append({"role": "user", "content": probe_text})
    return messages


def messages_for_elicitation(
    control: ElicitationControl, semantics_by_label: Sequence[SemanticChoice]
) -> list[dict[str, str]]:
    if set(semantics_by_label) != set(SEMANTIC_CHOICES):
        raise ValueError("A label assignment must contain left, right, and tie exactly once.")
    texts = {
        "left": "LEFT",
        "right": "RIGHT",
        "tie": "They have equal weight.",
    }
    lines = [
        "Compare these two normalized weights:",
        f"LEFT = {fraction_string(control.left_weight)}",
        f"RIGHT = {fraction_string(control.right_weight)}",
    ]
    lines.extend(
        f"{label}: {texts[semantic]}" for label, semantic in zip(LABELS, semantics_by_label)
    )
    lines.append("Reply with exactly A, B, or C.")
    return [
        {
            "role": "system",
            "content": "Choose the option corresponding to the greater weight, or equality.",
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def serialize_probe(probe: Probe) -> dict[str, object]:
    record = asdict(probe)
    record["left_probability_exact"] = fraction_string(probe.left_probability)
    record["right_probability_exact"] = fraction_string(probe.right_probability)
    del record["left_probability"]
    del record["right_probability"]
    return record


def serialize_example(example: ControlledExample) -> dict[str, object]:
    return {
        "example_id": example.example_id,
        "stage": example.stage,
        "bank_id": example.bank_id,
        "schedule_id": example.schedule_id,
        "world_id": example.world_id,
        "candidates": list(example.candidates),
        "reliability_exact": fraction_string(example.reliability),
        "reliability": float(example.reliability),
        "observations": [asdict(observation) for observation in example.observations],
        "posterior_exact": [fraction_string(value) for value in example.posterior],
        "posterior": [float(value) for value in example.posterior],
        "prior_predictive_probability_exact": fraction_string(
            example.prior_predictive_probability
        ),
        "probes": [serialize_probe(probe) for probe in example.probes],
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
