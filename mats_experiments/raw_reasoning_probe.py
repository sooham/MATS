"""Raw-evidence candidate probes for generated Qwen deliberation experiments.

The renderer in this module intentionally accepts only the public game fields: the
domain size, reliability, raw membership questions/reports, and candidate names.
Exact posteriors and evaluator-derived sufficient statistics are retained only in
the scoring record and can never enter a model prompt through this API.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

PromptStyle = Literal[
    "direct",
    "audit",
    "silent",
    "compact",
    "structured",
    "structured_likelihood",
    "atomic_likelihood",
]
Orientation = Literal["forward", "reverse"]
SemanticChoice = Literal["left", "right", "tie"]
FinalChoice = Literal["FIRST", "SECOND", "TIE"]

FINAL_PATTERN = re.compile(
    r"\bFINAL\s*:\s*(FIRST|SECOND|ALPHA|BETA|TIE)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class RawObservation:
    question: int
    subset: tuple[int, ...]
    report: Literal["YES", "NO"]


@dataclass(frozen=True)
class CandidateDecision:
    example_id: str
    repeat: int
    reliability: Fraction
    n: int
    policy: str
    observations: tuple[RawObservation, ...]
    left_candidate: int
    right_candidate: int
    normative_choice: SemanticChoice


def _fraction_from_record(record: Mapping[str, object]) -> Fraction:
    value = record["reliability"]
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, (int, float)):
        return Fraction(str(value))
    raise TypeError(f"Unsupported reliability value: {value!r}")


def _candidate_probe(record: Mapping[str, object]) -> Mapping[str, object]:
    probes = record.get("probes")
    if not isinstance(probes, list):
        raise TypeError("Transcript record must contain a probes list.")
    candidates = [probe for probe in probes if probe.get("kind") == "candidate"]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one candidate probe, found {len(candidates)}.")
    return candidates[0]


def decision_from_transcript(record: Mapping[str, object]) -> CandidateDecision:
    """Extract the public input and private scoring target from one transcript row."""

    observations_value = record.get("observations")
    if not isinstance(observations_value, list):
        raise TypeError("Transcript record must contain an observations list.")
    observations = tuple(
        RawObservation(
            question=int(observation["turn"]),
            subset=tuple(int(value) for value in observation["subset"]),
            report=str(observation["answer"]),
        )
        for observation in observations_value
    )
    if any(observation.report not in {"YES", "NO"} for observation in observations):
        raise ValueError("Every raw report must be YES or NO.")

    probe = _candidate_probe(record)
    semantic_options = probe.get("semantic_options")
    if not isinstance(semantic_options, Mapping):
        raise TypeError("Candidate probe must contain semantic_options.")
    normative = str(probe["normative_semantic_choice"])
    if normative not in {"left", "right", "tie"}:
        raise ValueError(f"Unsupported normative choice: {normative!r}")

    return CandidateDecision(
        example_id=str(record["example_id"]),
        repeat=int(record["repeat"]),
        reliability=_fraction_from_record(record),
        n=int(record["n"]),
        policy=str(record["policy"]),
        observations=observations,
        left_candidate=int(semantic_options["left"]),
        right_candidate=int(semantic_options["right"]),
        normative_choice=normative,  # type: ignore[arg-type]
    )


def load_candidate_decisions(
    path: Path,
    *,
    repeats: Sequence[int] | None = None,
    prompt_variant: str = "newer",
    policy: str = "random_memoryless",
) -> list[CandidateDecision]:
    """Load one copy of each candidate decision from notebook 01 transcripts."""

    repeat_filter = set(repeats) if repeats is not None else None
    decisions: list[CandidateDecision] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["prompt_variant"] != prompt_variant or record["policy"] != policy:
                continue
            if repeat_filter is not None and int(record["repeat"]) not in repeat_filter:
                continue
            decision = decision_from_transcript(record)
            if decision.n != 8 or len(decision.observations) != 3:
                raise ValueError("This experiment is preregistered for N=8 and K=3.")
            decisions.append(decision)

    ids = [decision.example_id for decision in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate decision IDs are not unique.")
    return decisions


def _set_text(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def oriented_candidates(
    decision: CandidateDecision, orientation: Orientation
) -> tuple[int, int]:
    if orientation == "forward":
        return decision.left_candidate, decision.right_candidate
    if orientation == "reverse":
        return decision.right_candidate, decision.left_candidate
    raise ValueError(f"Unknown orientation: {orientation!r}")


def render_raw_candidate_prompt(
    decision: CandidateDecision,
    *,
    style: PromptStyle,
    orientation: Orientation,
) -> str:
    """Render a prompt containing rules, raw observations, and no derived evidence."""

    first, second = oriented_candidates(decision, orientation)
    reliability = float(decision.reliability)
    error_rate = float(1 - decision.reliability)
    lines = [
        f"A secret integer s was drawn uniformly from {{1, ..., {decision.n}}}.",
        "Each externally chosen question asks whether s belongs to the displayed set.",
        (
            "For each question, the observed SOURCE report equals the truthful YES/NO "
            f"answer with probability r={reliability:g}, and is flipped with probability "
            f"1-r={error_rate:g}."
        ),
        "Channel outcomes are independent conditional on s, and question selection is not evidence.",
        "The reliability can be below, equal to, or above one half.",
        "",
        "RAW OBSERVATIONS:",
    ]
    for observation in decision.observations:
        lines.extend(
            [
                f"Q{observation.question}: Is s in {_set_text(observation.subset)}?",
                f"Observed SOURCE report: {observation.report}.",
            ]
        )
    lines.extend(
        [
            "",
            "TASK:",
            (
                f"Compare P(s={first} | all raw observations) with "
                f"P(s={second} | all raw observations)."
            ),
            f"FIRST means candidate {first}; SECOND means candidate {second}.",
        ]
    )
    if style == "audit":
        lines.extend(
            [
                (
                    "Reason from the raw observations yourself. Audit every displayed question for "
                    "both candidates, and account for the stated reliability without assuming that "
                    "SOURCE is usually truthful."
                ),
                "Do not state normalized posterior probabilities.",
            ]
        )
    elif style == "silent":
        lines.extend(
            [
                "Do the comparison internally; do not show calculations or restate the problem.",
                "Your entire visible response must be the single required FINAL line.",
            ]
        )
    elif style == "compact":
        lines.extend(
            [
                (
                    "Do not restate the setup. In a compact self-audit, check Q1, Q2, and Q3 "
                    "for FIRST and SECOND directly against the raw sets and reports."
                ),
                (
                    "Use at most six short audit lines plus the FINAL line, stay under 120 words, "
                    "and do not state normalized posterior probabilities."
                ),
            ]
        )
    elif style in {"structured", "structured_likelihood", "atomic_likelihood"}:
        if style in {"structured_likelihood", "atomic_likelihood"}:
            lines.extend(
                [
                    (
                        "For each candidate, each audit row contributes factor r when that "
                        "candidate's truthful answer equals the observed report, and factor "
                        "1-r when they differ. The uniform prior is common to both candidates."
                    ),
                    (
                        "Use the product of the three factors to compare the candidates, but do "
                        "not print match counts, products, likelihoods, or posterior values."
                    ),
                ]
            )
        if style == "atomic_likelihood":
            lines.append("Fill these exact six audit lines and add no other prose:")
            for observation in decision.observations:
                lines.extend(
                    [
                        (
                            f"Q{observation.question} ALPHA(candidate {first})_TRUE=<YES/NO>; "
                            f"OBSERVED_REPORT={observation.report}"
                        ),
                        (
                            f"Q{observation.question} BETA(candidate {second})_TRUE=<YES/NO>; "
                            f"OBSERVED_REPORT={observation.report}"
                        ),
                    ]
                )
        else:
            lines.extend(
                [
                    "Fill this exact four-line audit and add no other prose:",
                    "Q1 FIRST_TRUE=<YES/NO>; SECOND_TRUE=<YES/NO>; REPORT=<YES/NO>",
                    "Q2 FIRST_TRUE=<YES/NO>; SECOND_TRUE=<YES/NO>; REPORT=<YES/NO>",
                    "Q3 FIRST_TRUE=<YES/NO>; SECOND_TRUE=<YES/NO>; REPORT=<YES/NO>",
                ]
            )
    elif style != "direct":
        raise ValueError(f"Unknown prompt style: {style!r}")
    if style == "atomic_likelihood":
        lines.append("End with exactly FINAL: ALPHA, FINAL: BETA, or FINAL: TIE.")
    else:
        lines.append("End with exactly FINAL: FIRST, FINAL: SECOND, or FINAL: TIE.")
    return "\n".join(lines)


def parse_final_choice(text: str) -> FinalChoice | None:
    matches = FINAL_PATTERN.findall(text)
    if not matches:
        return None
    raw = matches[-1].upper()
    normalized = {"ALPHA": "FIRST", "BETA": "SECOND"}.get(raw, raw)
    return normalized  # type: ignore[return-value]


def absolute_prediction(
    decision: CandidateDecision,
    *,
    orientation: Orientation,
    parsed_choice: FinalChoice | None,
) -> int | Literal["tie"] | None:
    if parsed_choice is None:
        return None
    if parsed_choice == "TIE":
        return "tie"
    first, second = oriented_candidates(decision, orientation)
    return first if parsed_choice == "FIRST" else second


def normative_absolute(decision: CandidateDecision) -> int | Literal["tie"]:
    if decision.normative_choice == "left":
        return decision.left_candidate
    if decision.normative_choice == "right":
        return decision.right_candidate
    return "tie"


def score_generated_response(
    decision: CandidateDecision,
    *,
    orientation: Orientation,
    generated_text: str,
) -> dict[str, object]:
    parsed = parse_final_choice(generated_text)
    predicted = absolute_prediction(
        decision, orientation=orientation, parsed_choice=parsed
    )
    target = normative_absolute(decision)
    return {
        "parsed_choice": parsed,
        "parse_success": parsed is not None,
        "predicted_absolute": predicted,
        "normative_absolute": target,
        "correct": predicted == target,
    }


def aggregate_orientation_pair(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require forward/reverse responses to agree on an absolute semantic answer."""

    by_orientation = {str(record["orientation"]): record for record in records}
    if set(by_orientation) != {"forward", "reverse"}:
        raise ValueError("An orientation pair must contain one forward and one reverse record.")
    predictions = [by_orientation[key]["predicted_absolute"] for key in ("forward", "reverse")]
    consistent = predictions[0] is not None and predictions[0] == predictions[1]
    target = by_orientation["forward"]["normative_absolute"]
    return {
        "example_id": by_orientation["forward"]["example_id"],
        "variant": by_orientation["forward"]["variant"],
        "repeat": by_orientation["forward"]["repeat"],
        "reliability": by_orientation["forward"]["reliability"],
        "normative_absolute": target,
        "predicted_absolute": predictions[0] if consistent else None,
        "orientation_consistent": consistent,
        "orientation_accuracy": sum(bool(record["correct"]) for record in records) / 2,
        "correct": consistent and predictions[0] == target,
    }
