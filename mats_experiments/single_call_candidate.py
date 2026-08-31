"""Honest single-call candidate-posterior prompts for the noisy-source game.

The prompt renderer accepts only public game information plus the two candidate
identities.  It does not accept posterior values, evaluator-computed memberships,
agreement counts, likelihoods, or a target label.  Presentation order is derived
from the example identifier alone so it cannot leak the scoring target.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .raw_reasoning_probe import CandidateDecision

ReasoningMode = Literal["answer_only", "concise"]
PredictedAnswer = int | Literal["tie"]

FINAL_NUMBER_PATTERN = re.compile(r"\bFINAL\s*:\s*(TIE|[1-9][0-9]*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SingleCallVariant:
    """Generation and prompt settings for one independently evaluated call."""

    reasoning_mode: ReasoningMode
    max_new_tokens: int
    filler_tokens: int = 0
    use_system_message: bool = False


VARIANTS: dict[str, SingleCallVariant] = {
    "number_direct": SingleCallVariant("answer_only", 96),
    "number_reason": SingleCallVariant("concise", 1024),
    "number_reason_filler32": SingleCallVariant("concise", 1024, 32),
    "number_reason_filler128": SingleCallVariant("concise", 1024, 128),
    "number_system_reason": SingleCallVariant("concise", 1024, 0, True),
}


def _format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def candidate_order(decision: CandidateDecision) -> tuple[int, int]:
    """Counterbalance presentation using public identity only, never the target."""

    digest = hashlib.sha256(decision.example_id.encode("utf-8")).digest()
    candidates = (decision.left_candidate, decision.right_candidate)
    return candidates if digest[0] % 2 == 0 else tuple(reversed(candidates))


def render_prompt(
    decision: CandidateDecision,
    *,
    variant: SingleCallVariant,
) -> str:
    """Render one raw-evidence prompt with no task-derived intermediate values."""

    first, second = candidate_order(decision)
    reliability = float(decision.reliability)
    error_rate = float(1 - decision.reliability)
    lines = [
        "A secret integer s was sampled uniformly from the displayed domain.",
        f"DOMAIN: s is one of {{1, 2, ..., {decision.n}}}.",
        (
            "Each question asks whether s is in a displayed set. Its truthful answer is "
            "YES exactly when s is in that set, otherwise NO."
        ),
        (
            f"The observed SOURCE report equals the truthful answer with probability r={reliability:g} "
            f"and is flipped with probability 1-r={error_rate:g}."
        ),
        (
            f"Equivalently: if the truthful answer is YES, SOURCE reports YES with probability "
            f"{reliability:g} and NO with probability {error_rate:g}; if the truthful answer is "
            f"NO, SOURCE reports NO with probability {reliability:g} and YES with probability "
            f"{error_rate:g}."
        ),
        (
            "Channel outcomes are independent conditional on s. The questions were chosen "
            "externally, so their set contents are not evidence about s."
        ),
        "",
        "OBSERVATIONS:",
    ]
    for observation in decision.observations:
        lines.append(
            f"Q{observation.question}: Is s in {_format_set(observation.subset)}? "
            f"SOURCE reported {observation.report}."
        )
    if variant.filler_tokens:
        lines.extend(
            [
                "",
                "NEUTRAL WORKSPACE TOKENS (carry no evidence):",
                " ".join("." for _ in range(variant.filler_tokens)),
            ]
        )
    lines.extend(
        [
            "",
            "QUESTION:",
            (
                f"Given all observations, which has larger posterior probability: s={first} "
                f"or s={second}? If their posterior probabilities are equal, answer TIE."
            ),
            (
                f"End with exactly one of: FINAL: {first} | FINAL: {second} | FINAL: TIE"
            ),
        ]
    )
    if variant.reasoning_mode == "concise":
        lines.insert(
            -1,
            (
                "Reason carefully from the raw observations and the stated channel rule. "
                "Keep the explanation under 180 words."
            ),
        )
    else:
        lines.insert(
            -1,
            (
                "Do the reasoning internally. Do not show calculations or restate the problem; "
                "your entire visible response must be the single required FINAL line."
            ),
        )
    return "\n".join(lines)


def render_messages(
    decision: CandidateDecision,
    *,
    variant: SingleCallVariant,
) -> list[dict[str, str]]:
    prompt = render_prompt(decision, variant=variant)
    if not variant.use_system_message:
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "system",
            "content": (
                "You are a careful Bayesian reasoner. Follow the user's game rules exactly, "
                "including when a stated reliability is below one half."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_candidate_number(text: str, *, candidates: Sequence[int]) -> PredictedAnswer | None:
    """Parse only an explicit FINAL line and reject out-of-option numbers."""

    matches = FINAL_NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    raw = matches[-1].upper()
    if raw == "TIE":
        return "tie"
    value = int(raw)
    return value if value in set(candidates) else None
