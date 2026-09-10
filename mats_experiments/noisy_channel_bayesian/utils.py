import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from .constants import NO, YES


@dataclass(frozen=True)
class SystemPrompt:
    content: str | None = None


def as_fraction(value: float | str | Fraction) -> Fraction:
    """
    Convert a public reliability value without introducing binary-float noise.
    VERIFIED
    """

    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, float):
        result = Fraction(str(value))
    else:
        result = Fraction(value)
    if not 0 <= result <= 1:
        raise ValueError(f"Reliability must lie in [0, 1], got {value!r}.")
    return result

def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def natural_log_ratio(numerator: Fraction, denominator: Fraction) -> float:
    """Return ln(numerator / denominator) on PyTorch softmax's logit scale.

    Computing the two integer logarithms separately preserves the exact rational
    ratio until the final floating-point operation and avoids an intermediate
    ``float(Fraction)`` overflow for unusually large exact values.

    VERIFIED
    """

    if numerator <= 0 or denominator <= 0:
        raise ValueError("A finite natural-log ratio requires positive values.")
    ratio = numerator / denominator
    return math.log(ratio.numerator) - math.log(ratio.denominator)


def exact_bayesian_target(
    *,
    domain: Sequence[int],
    membership_sets: Sequence[Sequence[int]],
    reports: Sequence[str],
    reliabilities: Sequence[Fraction],
) -> tuple[Fraction, dict[int, Fraction] | None]:
    """
    Return prior-predictive evidence and the exact posterior, if defined.
    VERIFIED
    """
    if not (len(membership_sets) == len(reports) == len(reliabilities)):
        raise ValueError("Questions, reports, and reliabilities must have equal lengths.")
    if not domain:
        raise ValueError("domain must not be empty.")
    likelihoods: dict[int, Fraction] = {}
    set_views = [set(values) for values in membership_sets]
    for candidate in domain:
        likelihood = Fraction(1)
        for membership, report, reliability in zip(set_views, reports, reliabilities):
            if report not in (YES, NO):
                raise ValueError(f"Unknown report {report!r}.")
            expected = YES if candidate in membership else NO
            likelihood *= reliability if report == expected else 1 - reliability
        likelihoods[candidate] = likelihood
    total = sum(likelihoods.values(), Fraction(0))
    evidence = total / len(domain)
    if total == 0:
        return evidence, None
    return evidence, {candidate: value / total for candidate, value in likelihoods.items()}

def candidate_agreements(
    *,
    membership_sets: Sequence[Sequence[int]],
    reports: Sequence[str],
    candidate: int,
) -> list[int]:
    """
    Return per-question agreement between a candidate and the observed reports.
    VERIFIED
    """

    if len(membership_sets) != len(reports):
        raise ValueError("membership_sets and reports must have equal lengths.")
    flags: list[int] = []
    for membership, report in zip(membership_sets, reports):
        if report not in (YES, NO):
            raise ValueError(f"Unknown report {report!r}.")
        predicted_report = YES if candidate in set(membership) else NO
        flags.append(int(report == predicted_report))
    return flags

def reliability_surface(value: Fraction) -> str:
    """
    Render a reliability exactly, preferring a terminating decimal when possible.
    VERIFIED
    """

    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return fraction_text(value)
    digits = max(twos, fives)
    if digits == 0:
        return str(value.numerator)
    scaled = value.numerator * 10**digits // value.denominator
    text = str(abs(scaled)).rjust(digits + 1, "0")
    sign = "-" if scaled < 0 else ""
    return f"{sign}{text[:-digits]}.{text[-digits:]}"


def answer_patterns(k: int) -> list[tuple[str, ...]]:
    """
    Stable exhaustive order: YES precedes NO at every position.
    VERIFIED
    """

    return list(itertools.product((YES, NO), repeat=k))

def _format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"

def initial_messages(
    *, observable_prompt: str, system_prompt: SystemPrompt
) -> list[dict[str, str]]:
    """
    VERIFIED
    """
    messages: list[dict[str, str]] = []
    if system_prompt.content:
        messages.append({"role": "system", "content": system_prompt.content})
    messages.append({"role": "user", "content": observable_prompt})
    return messages
