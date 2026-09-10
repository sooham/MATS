import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .constants import NO, YES

AgreementPattern = tuple[int, ...]


def agreement_patterns(k: int) -> list[AgreementPattern]:
    """Return every length-``k`` agreement vector in stable Y-before-N order."""

    if k < 1:
        raise ValueError("k must be positive.")
    return list(itertools.product((1, 0), repeat=k))


def agreement_pattern_text(pattern: Sequence[int | bool]) -> str:
    """Render an agreement vector using the notebook's Y/N notation."""

    return "".join("Y" if bool(value) else "N" for value in pattern)


def _validated_agreement_pattern(
    pattern: Sequence[int | bool], *, k: int, name: str
) -> AgreementPattern:
    values = tuple(pattern)
    if len(values) != k:
        raise ValueError(f"{name} must contain exactly {k} values.")
    if any(type(value) not in (bool, int) or int(value) not in (0, 1) for value in values):
        raise ValueError(f"{name} values must all be booleans or the integers 0 and 1.")
    return tuple(int(value) for value in values)

@dataclass(frozen=True)
class CandidateEvidenceQuestion:
    """
    Presentation contract for the reduced, set-membership-free control.
    VERIFIED
    """

    agreement_surface: str = "AGREES"
    disagreement_surface: str = "DISAGREES"
    reliability_format: Literal["decimal_or_exact_fraction"] = "decimal_or_exact_fraction"
    layout: Literal["grouped_by_candidate"] = "grouped_by_candidate"

    def __post_init__(self) -> None:
        if not self.agreement_surface.strip() or not self.disagreement_surface.strip():
            raise ValueError("Candidate-evidence relation surfaces must not be empty.")
        if self.agreement_surface.strip() == self.disagreement_surface.strip():
            raise ValueError("Agreement and disagreement surfaces must be distinct.")
        if self.reliability_format != "decimal_or_exact_fraction":
            raise ValueError("Unsupported candidate-evidence reliability format.")
        if self.layout != "grouped_by_candidate":
            raise ValueError("Unsupported candidate-evidence layout.")

@dataclass(frozen=True)
class FixedSubsetQuestion:
    """
    A single, explicitly supplied question schedule.
    VERIFIED
    """

    subsets: Sequence[Sequence[int]]

    def sample(self, *, rng: random.Random, n: int, k: int) -> list[dict[str, object]]:
        del rng
        if len(self.subsets) != k:
            raise ValueError(f"Expected {k} fixed subsets, got {len(self.subsets)}.")
        result: list[dict[str, object]] = []
        for subset in self.subsets:
            raw = list(subset)
            if not raw:
                raise ValueError("Fixed subsets must not be empty.")
            if any(not isinstance(value, int) or not 1 <= value <= n for value in raw):
                raise ValueError(f"Fixed subset values must be integers in 1..{n}.")
            if len(set(raw)) != len(raw):
                raise ValueError("Fixed subsets may not contain duplicate values.")
            result.append({"membership_set": raw})
        return result

@dataclass(frozen=True)
class RandomSubsetQuestion:
    """
    Draw one subset independently for every question in a schedule.
    VERIFIED
    """

    subset_size: int = 4
    replacement: bool = False
    sort: bool = True

    def validate(self, n: int) -> None:
        if self.subset_size < 1:
            raise ValueError("subset_size must be positive.")
        if not self.replacement and self.subset_size > n:
            raise ValueError("subset_size cannot exceed n without replacement.")

    def sample(self, *, rng: random.Random, n: int, k: int) -> list[dict[str, object]]:
        self.validate(n)
        domain = list(range(1, n + 1))
        result: list[dict[str, object]] = []
        for _ in range(k):
            raw = (
                [rng.choice(domain) for _ in range(self.subset_size)]
                if self.replacement
                else rng.sample(domain, self.subset_size)
            )
            unique = list(dict.fromkeys(raw))
            membership = sorted(unique) if self.sort else unique
            result.append({"membership_set": membership})
        return result


@dataclass(frozen=True)
class AgreementSubsetQuestion:
    """Sample subset questions conditioned on two candidates' agreement vectors.

    A target agreement bit says whether the membership-implied answer for that
    candidate must equal the sampled observed report. For each question, the
    report and subset are sampled uniformly from all compatible
    ``(observed_report, membership_set)`` pairs of the configured size.
    """

    subset_size: int = 4
    sort: bool = True

    def validate(self, *, n: int, x: int, y: int) -> None:
        if n < 1:
            raise ValueError("n must be positive.")
        if self.subset_size < 1 or self.subset_size > n:
            raise ValueError("subset_size must lie in 1..n.")
        if x == y or x not in range(1, n + 1) or y not in range(1, n + 1):
            raise ValueError("x and y must be distinct members of the domain.")

        other_count = n - 2
        for x_agrees, y_agrees in itertools.product((0, 1), repeat=2):
            if not any(
                0
                <= self.subset_size
                - sum(
                    (report == YES) == bool(agrees)
                    for agrees in (x_agrees, y_agrees)
                )
                <= other_count
                for report in (YES, NO)
            ):
                raise ValueError(
                    "subset_size cannot realize every X/Y agreement combination "
                    f"for n={n}."
                )

    def sample(
        self,
        *,
        rng: random.Random,
        n: int,
        k: int,
        x: int,
        y: int,
        x_agreements: Sequence[int | bool],
        y_agreements: Sequence[int | bool],
    ) -> tuple[list[dict[str, object]], tuple[str, ...]]:
        """Draw one compatible question schedule and its observed reports."""

        self.validate(n=n, x=x, y=y)
        x_pattern = _validated_agreement_pattern(x_agreements, k=k, name="x_agreements")
        y_pattern = _validated_agreement_pattern(y_agreements, k=k, name="y_agreements")
        other_values = [value for value in range(1, n + 1) if value not in (x, y)]

        questions: list[dict[str, object]] = []
        reports: list[str] = []
        for x_agrees, y_agrees in zip(x_pattern, y_pattern, strict=True):
            compatible: list[tuple[str, list[int], int, int]] = []
            for report in (YES, NO):
                included = [
                    candidate
                    for candidate, agrees in ((x, x_agrees), (y, y_agrees))
                    if (report == YES) == bool(agrees)
                ]
                filler_count = self.subset_size - len(included)
                if 0 <= filler_count <= len(other_values):
                    compatible.append(
                        (report, included, filler_count, math.comb(len(other_values), filler_count))
                    )

            total_outcomes = sum(weight for _, _, _, weight in compatible)
            draw = rng.randrange(total_outcomes)
            selected_report = ""
            selected_included: list[int] = []
            selected_filler_count = 0
            for report, included, filler_count, weight in compatible:
                if draw < weight:
                    selected_report = report
                    selected_included = included
                    selected_filler_count = filler_count
                    break
                draw -= weight

            membership = selected_included + rng.sample(other_values, selected_filler_count)
            if self.sort:
                membership.sort()
            else:
                rng.shuffle(membership)
            questions.append({"membership_set": membership})
            reports.append(selected_report)

        return questions, tuple(reports)


class ExclusiveCandidateSubsetQuestion(RandomSubsetQuestion):
    """
    Sample sets conditioned on containing exactly one focal candidate.
    VERIFIED
    """

    x: int = 2
    y: int = 7

    def sample(self, *, rng: random.Random, n: int, k: int) -> list[dict[str, object]]:
        self.validate(n)
        if self.replacement:
            raise ValueError("The clean sign-gate sampler requires replacement=False.")
        if self.x == self.y or self.x not in range(1, n + 1) or self.y not in range(1, n + 1):
            raise ValueError("x and y must be distinct members of the domain.")
        if self.subset_size >= n:
            raise ValueError("subset_size must leave room to exclude one focal candidate.")
        domain = list(range(1, n + 1))
        sampled: list[dict[str, object]] = []
        for _ in range(k):
            while True:
                raw = rng.sample(domain, self.subset_size)
                if (self.x in raw) != (self.y in raw):
                    membership = sorted(raw) if self.sort else list(raw)
                    sampled.append({"raw_draws": raw, "membership_set": membership})
                    break
        return sampled
