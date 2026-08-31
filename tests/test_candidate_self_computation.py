from mats_experiments.candidate_self_computation import (
    atomic_membership_prompt,
    choose_qwen_count,
    count_adjudication_prompt,
    endpoint_candidate_final_prompt,
    endpoint_count_final_prompt,
    endpoint_final_prompt,
    interior_candidate_final_prompt,
    interior_final_prompt,
    parse_candidate_final,
    parse_count,
    parse_final_label,
    parse_yes_no,
    semantic_from_label,
    truth_report_pairs,
    visible_count_prompt,
)


def test_atomic_membership_prompt_contains_only_literal_question() -> None:
    prompt = atomic_membership_prompt(7, [1, 4, 6, 7])
    assert prompt == (
        "Inspect the literal set in this single membership question. "
        "Is the integer 7 an element of {1, 4, 6, 7}? Reply with exactly YES or NO."
    )
    assert "posterior" not in prompt.lower()
    assert "count" not in prompt.lower()


def test_count_prompts_route_only_qwen_truths_and_raw_reports() -> None:
    pairs = truth_report_pairs(["YES", "NO", "YES"], ["NO", "YES", "NO"])
    forward = visible_count_prompt(pairs, reverse=False)
    reverse = visible_count_prompt(pairs, reverse=True)
    assert "Q1 [truth=YES, report=NO]" in forward
    assert reverse.index("Q3") < reverse.index("Q1")
    adjudication = count_adjudication_prompt(
        forward_draft="COUNT: 0", reverse_draft="COUNT: 1", pairs=pairs
    )
    assert "unverified truth/report pairs" in adjudication


def test_parsers_and_qwen_count_selection() -> None:
    assert parse_yes_no("Reasoning. YES") == "YES"
    assert parse_count("Q1 SAME; COUNT: 1") == 1
    assert parse_count("COUNT: 9") is None
    assert choose_qwen_count(forward=2, reverse=2, adjudicated=None) == 2
    assert choose_qwen_count(forward=2, reverse=1, adjudicated=1) == 1
    assert parse_final_label("FINAL: B") == "B"
    assert semantic_from_label("B") == "right"


def test_final_prompts_use_only_qwen_generated_intermediates() -> None:
    interior = interior_final_prompt(first_count=0, second_count=1, r=0.9)
    endpoint = endpoint_final_prompt(first_check="NO", second_check="YES")
    assert "unverified match counts" in interior
    assert "FIRST candidate ALPHA has 0" in interior
    assert "FIRST candidate ALPHA check=NO" in endpoint
    endpoint_counts = endpoint_count_final_prompt(
        first_count=0, second_count=3, special_count=3
    )
    assert "SECOND candidate BETA has 3 matches" in endpoint_counts
    assert "count is 3" in endpoint_counts


def test_direct_candidate_final_readout_avoids_abc_mapping() -> None:
    interior = interior_candidate_final_prompt(first_count=0, second_count=1, r=0.9)
    endpoint = endpoint_candidate_final_prompt(first_check="NO", second_check="YES")
    assert "FINAL: ALPHA" in interior
    assert "candidate BETA returned YES" in endpoint
    assert parse_candidate_final("FINAL: ALPHA") == "left"
    assert parse_candidate_final("FINAL: BETA") == "right"
    assert parse_candidate_final("FINAL: TIE") == "tie"
