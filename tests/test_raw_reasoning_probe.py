import json
from pathlib import Path

from mats_experiments.raw_reasoning_probe import (
    aggregate_orientation_pair,
    load_candidate_decisions,
    parse_final_choice,
    render_raw_candidate_prompt,
    score_generated_response,
)


def transcript_record() -> dict[str, object]:
    return {
        "example_id": "example_newer",
        "prompt_variant": "newer",
        "n": 8,
        "repeat": 0,
        "policy": "random_memoryless",
        "reliability": "7/10",
        "observations": [
            {"turn": 1, "subset": [1, 2, 3, 4], "answer": "YES"},
            {"turn": 2, "subset": [2, 4, 6, 8], "answer": "NO"},
            {"turn": 3, "subset": [1, 3, 5, 7], "answer": "YES"},
        ],
        "probes": [
            {
                "kind": "candidate",
                "semantic_options": {"left": "1", "right": "6", "tie": "equal"},
                "normative_semantic_choice": "left",
                "option_probabilities_exact": ["343/370", "27/370", "0/1"],
            }
        ],
    }


def test_loader_and_prompt_exclude_private_target_fields(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    path.write_text(json.dumps(transcript_record()) + "\n", encoding="utf-8")
    decision = load_candidate_decisions(path, repeats=[0])[0]
    prompt = render_raw_candidate_prompt(decision, style="audit", orientation="forward")

    assert "Q1: Is s in {1, 2, 3, 4}?" in prompt
    assert "Observed SOURCE report: YES." in prompt
    assert "P(s=1 | all raw observations)" in prompt
    assert "343/370" not in prompt
    assert "match count" not in prompt.lower()
    assert "posterior_exact" not in prompt


def test_parser_uses_last_explicit_final_field() -> None:
    assert parse_final_choice("Maybe FIRST. FINAL: SECOND") == "SECOND"
    assert parse_final_choice("FINAL: tie") == "TIE"
    assert parse_final_choice("FINAL: ALPHA") == "FIRST"
    assert parse_final_choice("FINAL: BETA") == "SECOND"
    assert parse_final_choice("The first candidate wins.") is None


def test_compact_and_silent_contracts_are_bounded_instructions(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    path.write_text(json.dumps(transcript_record()) + "\n", encoding="utf-8")
    decision = load_candidate_decisions(path)[0]

    silent = render_raw_candidate_prompt(decision, style="silent", orientation="forward")
    compact = render_raw_candidate_prompt(decision, style="compact", orientation="forward")
    structured = render_raw_candidate_prompt(
        decision, style="structured", orientation="forward"
    )
    structured_likelihood = render_raw_candidate_prompt(
        decision, style="structured_likelihood", orientation="forward"
    )
    atomic = render_raw_candidate_prompt(
        decision, style="atomic_likelihood", orientation="reverse"
    )
    assert "single required FINAL line" in silent
    assert "at most six short audit lines" in compact
    assert "Q1 FIRST_TRUE=<YES/NO>" in structured
    assert "each audit row contributes factor r" in structured_likelihood
    assert "Q1 ALPHA(candidate 6)_TRUE=<YES/NO>" in atomic
    assert "Q1 BETA(candidate 1)_TRUE=<YES/NO>" in atomic
    all_prompts = silent + compact + structured + structured_likelihood + atomic
    assert "343/370" not in all_prompts
    assert "FIRST_TRUE=YES" not in all_prompts


def test_orientation_pair_requires_absolute_agreement(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    path.write_text(json.dumps(transcript_record()) + "\n", encoding="utf-8")
    decision = load_candidate_decisions(path)[0]
    forward_score = score_generated_response(
        decision, orientation="forward", generated_text="FINAL: FIRST"
    )
    reverse_score = score_generated_response(
        decision, orientation="reverse", generated_text="FINAL: SECOND"
    )
    records = [
        {
            "example_id": decision.example_id,
            "variant": "audit_thinking",
            "repeat": decision.repeat,
            "reliability": float(decision.reliability),
            "orientation": "forward",
            **forward_score,
        },
        {
            "example_id": decision.example_id,
            "variant": "audit_thinking",
            "repeat": decision.repeat,
            "reliability": float(decision.reliability),
            "orientation": "reverse",
            **reverse_score,
        },
    ]

    aggregate = aggregate_orientation_pair(records)
    assert aggregate["orientation_consistent"] is True
    assert aggregate["correct"] is True
