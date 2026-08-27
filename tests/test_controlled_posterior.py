from __future__ import annotations

import math
from fractions import Fraction

import pytest

from mats_experiments.controlled_posterior import (
    LABEL_ASSIGNMENTS,
    ExperimentConfig,
    N8Config,
    Observation,
    build_ladder_examples,
    build_n8_examples,
    candidate_match_count,
    exact_posterior,
    messages_for_probe,
)


def test_exact_candidate_log_odds_uses_all_matching_answers() -> None:
    observations = (
        Observation(1, ("left",), "YES"),
        Observation(2, ("right",), "NO"),
        Observation(3, ("left",), "NO"),
    )
    posterior, _ = exact_posterior(("left", "right"), observations, Fraction(9, 10))
    left_matches = candidate_match_count("left", observations)
    right_matches = candidate_match_count("right", observations)

    assert (left_matches, right_matches) == (2, 1)
    assert math.log(float(posterior[0] / posterior[1])) == pytest.approx(
        (left_matches - right_matches) * math.log(9)
    )


def test_r_half_always_produces_uniform_positive_ties() -> None:
    examples = build_ladder_examples(ExperimentConfig(world_count=1))
    half_examples = [example for example in examples if example.reliability == Fraction(1, 2)]

    assert half_examples
    assert all(len(set(example.posterior)) == 1 for example in half_examples)
    assert all(
        probe.normative_choice == "tie" and probe.tie_type == "uniform_positive"
        for example in half_examples
        for probe in example.probes
    )


def test_interior_equal_evidence_ties_are_not_r_half_nulls() -> None:
    examples = build_ladder_examples(ExperimentConfig(world_count=1))
    tied = [
        probe
        for example in examples
        if example.reliability != Fraction(1, 2)
        for probe in example.probes
        if probe.normative_choice == "tie"
    ]
    assert tied
    assert all(probe.tie_type == "equal_positive" for probe in tied)


def test_reliability_inversion_reverses_candidate_ranking() -> None:
    examples = build_ladder_examples(ExperimentConfig(world_count=1))
    by_key = {
        (example.bank_id, example.reliability): example
        for example in examples
        if example.stage == "one_observation"
    }
    for bank_id in {key[0] for key in by_key}:
        low = by_key[(bank_id, Fraction(1, 10))].probes[0]
        high = by_key[(bank_id, Fraction(9, 10))].probes[0]
        assert {low.normative_choice, high.normative_choice} == {"left", "right"}
        assert low.target_log_odds == pytest.approx(-high.target_log_odds)


def test_n8_replays_hold_observables_and_probe_meanings_fixed() -> None:
    examples = build_n8_examples(N8Config(num_schedules=2))
    bank_ids = {example.bank_id for example in examples}
    assert len(bank_ids) == 2 * 8
    for bank_id in bank_ids:
        variants = [example for example in examples if example.bank_id == bank_id]
        assert len(variants) == 5
        assert all(example.observations == variants[0].observations for example in variants)
        assert all(
            tuple((probe.kind, probe.left, probe.right) for probe in example.probes)
            == tuple((probe.kind, probe.left, probe.right) for probe in variants[0].probes)
            for example in variants
        )


def test_n8_questions_have_distinct_members_and_fixed_size() -> None:
    examples = build_n8_examples(N8Config(num_schedules=4))
    for observation in (observation for example in examples for observation in example.observations):
        assert len(observation.subset) == 4
        assert len(set(observation.subset)) == 4


def test_probe_orientation_is_an_exact_counterfactual() -> None:
    example = build_n8_examples(N8Config(num_schedules=1))[0]
    forward, reverse = example.probes[:2]
    assert forward.left == reverse.right
    assert forward.right == reverse.left
    assert forward.left_probability == reverse.right_probability
    assert forward.right_probability == reverse.left_probability
    assert forward.target_log_odds == pytest.approx(-reverse.target_log_odds)


def test_single_user_and_alternating_formats_preserve_evidence() -> None:
    example = build_ladder_examples(ExperimentConfig(world_count=1))[0]
    probe = example.probes[0]
    single = messages_for_probe(example, probe, LABEL_ASSIGNMENTS[0])
    alternating = messages_for_probe(
        example, probe, LABEL_ASSIGNMENTS[0], transcript_format="alternating"
    )
    observed_answer = example.observations[0].answer

    assert observed_answer in single[-1]["content"]
    assert any(observed_answer in message["content"] for message in alternating)
    assert single[-1]["content"].endswith("Reply with exactly A, B, or C.")
    assert alternating[-1]["content"].endswith("Reply with exactly A, B, or C.")


def test_endpoint_diagnostics_omit_only_zero_evidence_histories() -> None:
    ladder = build_ladder_examples(
        ExperimentConfig(world_count=1, include_endpoint_diagnostics=True)
    )
    n8 = build_n8_examples(
        N8Config(num_schedules=1), include_endpoint_diagnostics=True
    )
    endpoint_rows = [
        example
        for example in (*ladder, *n8)
        if example.reliability in {Fraction(0), Fraction(1)}
    ]
    assert endpoint_rows
    assert all(example.prior_predictive_probability > 0 for example in endpoint_rows)
