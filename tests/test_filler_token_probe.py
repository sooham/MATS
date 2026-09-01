from fractions import Fraction

from mats_experiments.filler_token_probe import (
    FILLER_SURFACES,
    exact_candidate_comparison,
    generate_scaled_decisions,
    n8_target_surface,
    render_n8_messages,
)
from mats_experiments.raw_reasoning_probe import CandidateDecision, RawObservation


def decision(target: str = "right") -> CandidateDecision:
    return CandidateDecision(
        example_id="public-example",
        repeat=0,
        reliability=Fraction(3, 10),
        n=8,
        policy="random_memoryless",
        observations=(
            RawObservation(1, (1, 3, 5, 7), "YES"),
            RawObservation(2, (2, 4, 6, 8), "NO"),
            RawObservation(3, (1, 2, 7, 8), "YES"),
        ),
        left_candidate=2,
        right_candidate=7,
        normative_choice=target,  # type: ignore[arg-type]
    )


def test_prompt_is_target_invariant() -> None:
    assert render_n8_messages(decision("left")) == render_n8_messages(decision("tie"))


def test_prompt_has_no_reasoning_allowance_or_private_fields() -> None:
    prompt = str(render_n8_messages(decision()))
    for forbidden in [
        "180 words",
        "truthful_answer",
        "channel_was_correct",
        "normative",
        "match count",
        "likelihood",
    ]:
        assert forbidden not in prompt
    assert "Do not provide a verbal explanation" in prompt


def test_n8_target_surface() -> None:
    assert n8_target_surface(decision("left")) == "2"
    assert n8_target_surface(decision("right")) == "7"
    assert n8_target_surface(decision("tie")) == "="


def test_filler_surfaces_are_distinct() -> None:
    assert len(FILLER_SURFACES) == len(set(FILLER_SURFACES.values()))


def test_exact_candidate_comparison() -> None:
    observations = (
        RawObservation(1, (2, 3), "YES"),
        RawObservation(2, (2, 7), "NO"),
        RawObservation(3, (1, 7), "NO"),
    )
    assert (
        exact_candidate_comparison(
            first_candidate=2,
            second_candidate=7,
            observations=observations,
            reliability=Fraction(9, 10),
        )
        == 2
    )
    assert (
        exact_candidate_comparison(
            first_candidate=2,
            second_candidate=7,
            observations=observations,
            reliability=Fraction(1, 2),
        )
        == "tie"
    )


def test_scaled_grid_has_exact_target_independent_position_balance() -> None:
    decisions = generate_scaled_decisions(
        n_values=[12],
        reliabilities=[Fraction(3, 10)],
        examples_per_cell=32,
    )
    assert len(decisions) == 32
    assert sum(d.first_candidate < d.second_candidate for d in decisions) == 16
    assert sum(d.first_alias == "X" for d in decisions) == 16
    assert all(len(d.observations) == 3 for d in decisions)
    assert all(len(obs.subset) == 6 for d in decisions for obs in d.observations)
