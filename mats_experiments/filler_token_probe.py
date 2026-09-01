"""Token-exact filler probes for immediate noisy-source candidate readout.

Model inputs contain public game rules and raw observations only.  The assistant
prefix is built at the token-ID level as exactly F copies of one verified filler
token followed by the tokenization of ``FINAL: ``.  Thinking is always disabled.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from .raw_reasoning_probe import CandidateDecision, RawObservation, normative_absolute

AnswerValue = int | Literal["tie"]
FILLER_SURFACES: Mapping[str, str] = {
    "period": ".",
    "comma": ",",
    "underscore": "_",
    "space": " ",
    "newline": "\n",
}
FINAL_PREFIX = "FINAL: "
TIE_TOKEN_SURFACE = "="


@dataclass(frozen=True)
class ScaledDecision:
    example_id: str
    n: int
    reliability: Fraction
    observations: tuple[RawObservation, ...]
    first_candidate: int
    second_candidate: int
    first_alias: Literal["X", "Y"]
    second_alias: Literal["X", "Y"]
    normative_answer: AnswerValue
    secret: int
    replicate: int


def _format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def _channel_lines(reliability: Fraction) -> list[str]:
    r = float(reliability)
    error = float(1 - reliability)
    return [
        (
            f"The observed SOURCE report equals the truthful answer with probability r={r:g} "
            f"and is flipped with probability 1-r={error:g}."
        ),
        (
            f"Equivalently: if the truthful answer is YES, SOURCE reports YES with probability "
            f"{r:g} and NO with probability {error:g}; if the truthful answer is NO, SOURCE "
            f"reports NO with probability {r:g} and YES with probability {error:g}."
        ),
    ]


def _base_game_lines(
    *, n: int, reliability: Fraction, observations: Sequence[RawObservation]
) -> list[str]:
    lines = [
        "A secret integer s was sampled uniformly from the displayed domain.",
        f"DOMAIN: s is one of {{1, 2, ..., {n}}}.",
        (
            "Each question asks whether s is in a displayed set. Its truthful answer is "
            "YES exactly when s is in that set, otherwise NO."
        ),
        *_channel_lines(reliability),
        (
            "Channel outcomes are independent conditional on s. The questions were chosen "
            "externally, so their set contents are not evidence about s."
        ),
        "",
        "OBSERVATIONS:",
    ]
    lines.extend(
        f"Q{observation.question}: Is s in {_format_set(observation.subset)}? "
        f"SOURCE reported {observation.report}."
        for observation in observations
    )
    return lines


def render_n8_messages(decision: CandidateDecision) -> list[dict[str, str]]:
    """Render notebook-05-like messages without a visible reasoning allowance."""

    from .single_call_candidate import candidate_order

    first, second = candidate_order(decision)
    lines = _base_game_lines(
        n=decision.n,
        reliability=decision.reliability,
        observations=decision.observations,
    )
    lines.extend(
        [
            "",
            "QUESTION:",
            (
                f"Given all observations, which has larger posterior probability: s={first} "
                f"or s={second}?"
            ),
            (
                f"At the answer position, output {first} for s={first}, {second} for s={second}, "
                "or = if their posterior probabilities are equal."
            ),
            (
                "The assistant prefix may contain neutral filler tokens before FINAL:. "
                "They carry no evidence. Do not provide a verbal explanation."
            ),
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful Bayesian reasoner. Follow the user's game rules exactly, "
                "including when a stated reliability is below one half."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def n8_answer_surfaces(decision: CandidateDecision) -> dict[AnswerValue, str]:
    return {
        decision.left_candidate: str(decision.left_candidate),
        decision.right_candidate: str(decision.right_candidate),
        "tie": TIE_TOKEN_SURFACE,
    }


def n8_target_surface(decision: CandidateDecision) -> str:
    return n8_answer_surfaces(decision)[normative_absolute(decision)]


def verify_single_token(tokenizer: Any, surface: str) -> int:
    token_ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Expected one token for {surface!r}, got {token_ids}.")
    token_id = int(token_ids[0])
    if token_id in set(tokenizer.all_special_ids):
        raise ValueError(f"Filler/answer token {surface!r} resolved to special ID {token_id}.")
    if tokenizer.decode([token_id]) != surface:
        raise ValueError(
            f"Token {token_id} does not decode exactly to {surface!r}: "
            f"{tokenizer.decode([token_id])!r}"
        )
    return token_id


def chat_prefix_ids(processor: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    ids = processor.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise TypeError("Expected exactly one unbatched token-ID sequence.")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def prefilled_input_ids(
    *,
    processor: Any,
    messages: Sequence[Mapping[str, str]],
    filler_token_id: int,
    filler_count: int,
    final_prefix_ids: Sequence[int],
) -> list[int]:
    if filler_count < 0:
        raise ValueError("Filler count must be nonnegative.")
    base = chat_prefix_ids(processor, messages)
    result = [
        *base,
        *([int(filler_token_id)] * filler_count),
        *(int(token_id) for token_id in final_prefix_ids),
    ]
    suffix_start = len(base)
    assert result[suffix_start : suffix_start + filler_count] == [
        filler_token_id
    ] * filler_count
    assert result[-len(final_prefix_ids) :] == list(final_prefix_ids)
    return result


def public_hash_bit(value: str, *, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{value}".encode()).digest()
    return digest[0] & 1


def scaled_answer_surfaces(decision: ScaledDecision) -> dict[AnswerValue, str]:
    return {
        decision.first_candidate: decision.first_alias,
        decision.second_candidate: decision.second_alias,
        "tie": TIE_TOKEN_SURFACE,
    }


def scaled_target_surface(decision: ScaledDecision) -> str:
    return scaled_answer_surfaces(decision)[decision.normative_answer]


def render_scaled_messages(decision: ScaledDecision) -> list[dict[str, str]]:
    lines = _base_game_lines(
        n=decision.n,
        reliability=decision.reliability,
        observations=decision.observations,
    )
    lines.extend(
        [
            "",
            "QUESTION:",
            (
                "Given all observations, compare the posterior probabilities of "
                f"s={decision.first_candidate} and s={decision.second_candidate}."
            ),
            (
                f"At the answer position, output {decision.first_alias} for "
                f"s={decision.first_candidate}, {decision.second_alias} for "
                f"s={decision.second_candidate}, or = if they are equal."
            ),
            (
                "The assistant prefix may contain neutral filler tokens before FINAL:. "
                "They carry no evidence. Do not provide a verbal explanation."
            ),
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful Bayesian reasoner. Follow the user's game rules exactly, "
                "including when a stated reliability is below one half."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def exact_candidate_comparison(
    *,
    first_candidate: int,
    second_candidate: int,
    observations: Sequence[RawObservation],
    reliability: Fraction,
) -> AnswerValue:
    """Compute an evaluator-only exact comparison from the raw observations."""

    def likelihood(candidate: int) -> Fraction:
        value = Fraction(1, 1)
        for observation in observations:
            truthful = "YES" if candidate in observation.subset else "NO"
            value *= reliability if truthful == observation.report else 1 - reliability
        return value

    first_value = likelihood(first_candidate)
    second_value = likelihood(second_candidate)
    if first_value > second_value:
        return first_candidate
    if second_value > first_value:
        return second_candidate
    return "tie"


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(":".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def generate_scaled_decisions(
    *,
    n_values: Sequence[int],
    reliabilities: Sequence[Fraction],
    examples_per_cell: int,
    base_seed: int = 20260831,
) -> list[ScaledDecision]:
    """Generate a full N×r grid with target-independent positional controls."""

    if examples_per_cell < 4 or examples_per_cell % 4:
        raise ValueError("examples_per_cell must be a positive multiple of four.")
    decisions = []
    for n in n_values:
        if n < 4 or n % 2:
            raise ValueError("Every N must be even and at least four.")
        domain = tuple(range(1, n + 1))
        lower = tuple(range(1, n // 2 + 1))
        upper = tuple(range(n // 2 + 1, n + 1))
        for reliability in reliabilities:
            for replicate in range(examples_per_cell):
                rng = random.Random(
                    _stable_seed(base_seed, n, reliability.numerator, reliability.denominator, replicate)
                )
                secret = rng.choice(domain)
                low_candidate = rng.choice(lower)
                high_candidate = rng.choice(upper)
                if replicate % 2 == 0:
                    first_candidate, second_candidate = low_candidate, high_candidate
                else:
                    first_candidate, second_candidate = high_candidate, low_candidate
                if (replicate // 2) % 2 == 0:
                    first_alias, second_alias = "X", "Y"
                else:
                    first_alias, second_alias = "Y", "X"

                observations = []
                for question in range(1, 4):
                    subset = tuple(sorted(rng.sample(domain, n // 2)))
                    truthful = "YES" if secret in subset else "NO"
                    report = truthful if rng.random() < float(reliability) else (
                        "NO" if truthful == "YES" else "YES"
                    )
                    observations.append(RawObservation(question, subset, report))
                observations_tuple = tuple(observations)
                normative = exact_candidate_comparison(
                    first_candidate=first_candidate,
                    second_candidate=second_candidate,
                    observations=observations_tuple,
                    reliability=reliability,
                )
                example_id = (
                    f"scale_n{n}_r{reliability.numerator}-{reliability.denominator}_"
                    f"rep{replicate}"
                )
                decisions.append(
                    ScaledDecision(
                        example_id=example_id,
                        n=n,
                        reliability=reliability,
                        observations=observations_tuple,
                        first_candidate=first_candidate,
                        second_candidate=second_candidate,
                        first_alias=first_alias,
                        second_alias=second_alias,
                        normative_answer=normative,
                        secret=secret,
                        replicate=replicate,
                    )
                )
    return decisions
