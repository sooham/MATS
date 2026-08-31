from fractions import Fraction

import pytest

from mats_experiments.raw_reasoning_probe import CandidateDecision, RawObservation
from mats_experiments.single_call_candidate import (
    VARIANTS,
    candidate_order,
    parse_candidate_number,
    render_prompt,
)


@pytest.fixture
def decision() -> CandidateDecision:
    return CandidateDecision(
        example_id="public-id",
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
        normative_choice="right",
    )


def test_renderer_contains_only_public_raw_evidence(decision: CandidateDecision) -> None:
    prompt = render_prompt(decision, variant=VARIANTS["number_reason_filler32"])
    for expected in ["r=0.3", "1-r=0.7", "{1, 3, 5, 7}", "SOURCE reported YES"]:
        assert expected in prompt
    for forbidden in [
        "normative",
        "truthful_answer",
        "channel_was_correct",
        "match count",
        "likelihood",
        "posterior probability is",
    ]:
        assert forbidden not in prompt


def test_all_variants_forbid_thinking_mode() -> None:
    assert all("thinking" not in name for name in VARIANTS)


def test_order_does_not_depend_on_target(decision: CandidateDecision) -> None:
    altered = CandidateDecision(**{**decision.__dict__, "normative_choice": "left"})
    assert candidate_order(decision) == candidate_order(altered)


@pytest.mark.parametrize("variant_name", sorted(VARIANTS))
def test_prompt_does_not_depend_on_target(
    decision: CandidateDecision, variant_name: str
) -> None:
    altered = CandidateDecision(**{**decision.__dict__, "normative_choice": "tie"})
    variant = VARIANTS[variant_name]
    assert render_prompt(decision, variant=variant) == render_prompt(
        altered, variant=variant
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reasoning 2 7\nFINAL: 2", 2),
        ("FINAL: 7", 7),
        ("FINAL: tie", "tie"),
        ("FINAL: 6", None),
        ("candidate 2", None),
    ],
)
def test_parse_candidate_number(text: str, expected: object) -> None:
    assert parse_candidate_number(text, candidates=(2, 7)) == expected
