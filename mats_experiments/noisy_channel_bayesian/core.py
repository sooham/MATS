"""Exact data generation for exhaustive noisy-channel transcripts.

This module intentionally has no dependency on the older experiment modules.  It
contains only deterministic prompt construction and exact rational arithmetic;
model execution lives in :mod:`runner`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

SCHEMA_VERSION = "1.0"
YES = "YES"
NO = "NO"

# Qwen returns unnormalized pre-softmax scores, so individual raw logits do not
# have a logarithm base.  PyTorch's softmax uses exp(), however, which makes
# log-probabilities and logit differences natural-log quantities (nats).
SOFTMAX_LOG_BASE = "e"
SOFTMAX_LOG_UNIT = "nats"


def as_fraction(value: float | str | Fraction) -> Fraction:
    """Convert a public reliability value without introducing binary-float noise."""

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
    """

    if numerator <= 0 or denominator <= 0:
        raise ValueError("A finite natural-log ratio requires positive values.")
    ratio = numerator / denominator
    return math.log(ratio.numerator) - math.log(ratio.denominator)


@dataclass(frozen=True)
class NoisyChannelBayesianEnvironment:
    """Finite uniform domain and the reliability of each observed report."""

    n: int = 8
    k: int = 3
    r_values: int | float | str | Fraction | Sequence[int | float | str | Fraction] = Fraction(3, 4)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive.")
        if self.k < 1:
            raise ValueError("k must be positive.")
        values = self.r_values
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            resolved = tuple(as_fraction(value) for value in values)
            if len(resolved) != self.k:
                raise ValueError(f"Expected {self.k} reliability values, got {len(resolved)}.")
            shared = False
        else:
            resolved = (as_fraction(values),) * self.k
            shared = True
        object.__setattr__(self, "_reliabilities", resolved)
        object.__setattr__(self, "_shared_reliability", shared)

    @property
    def domain(self) -> tuple[int, ...]:
        return tuple(range(1, self.n + 1))

    @property
    def reliabilities(self) -> tuple[Fraction, ...]:
        return self._reliabilities  # type: ignore[attr-defined]

    @property
    def shared_reliability(self) -> bool:
        return self._shared_reliability  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CandidateEvidenceBayesianEnvironment(NoisyChannelBayesianEnvironment):
    """The same channel model presented through candidate-specific evidence relations."""


@dataclass(frozen=True)
class CandidateEvidenceQuestion:
    """Presentation contract for the reduced, set-membership-free control."""

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
class RandomSubsetQuestion:
    """Draw one subset independently for every question in a schedule."""

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
            result.append({"raw_draws": raw, "membership_set": membership})
        return result


@dataclass(frozen=True)
class FixedSubsetQuestion:
    """A single, explicitly supplied question schedule."""

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
            result.append({"raw_draws": raw, "membership_set": raw})
        return result


CallLayout = Literal["conversation", "replay_user"]


@dataclass(frozen=True)
class XVsYPosteriorProbe:
    x: int = 2
    y: int = 7
    reasoning_budget: int = 0
    allow_same: bool = False
    call_layout: CallLayout = "conversation"
    answer_prefix: str = "ANSWER:"

    def validate(self, n: int) -> None:
        if self.x == self.y:
            raise ValueError("x and y must be distinct; allow_same only controls tie answers.")
        if self.x not in range(1, n + 1) or self.y not in range(1, n + 1):
            raise ValueError(f"x and y must both lie in 1..{n}.")
        if self.reasoning_budget < 0:
            raise ValueError("reasoning_budget must be non-negative.")
        if self.call_layout not in ("conversation", "replay_user"):
            raise ValueError("call_layout must be 'conversation' or 'replay_user'.")
        if (
            not self.answer_prefix
            or self.answer_prefix != self.answer_prefix.strip()
            or "\n" in self.answer_prefix
        ):
            raise ValueError(
                "answer_prefix must be non-empty, single-line, and have no surrounding whitespace."
            )


@dataclass(frozen=True)
class SystemPrompt:
    content: str | None = None


@dataclass(frozen=True)
class TokenizerBinding:
    """Bind generation to one tokenizer and its exact chat-template serialization."""

    tokenizer: Any
    enable_thinking: bool = False
    template_label: str | None = None

    def _template_kwargs(self, *, tokenize: bool) -> dict[str, object]:
        return {
            "tokenize": tokenize,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }

    def apply(self, messages: Sequence[Mapping[str, str]], *, tokenize: bool) -> Any:
        kwargs = self._template_kwargs(tokenize=tokenize)
        try:
            return self.tokenizer.apply_chat_template(list(messages), **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            return self.tokenizer.apply_chat_template(list(messages), **kwargs)

    def serialize(self, messages: Sequence[Mapping[str, str]]) -> str:
        rendered = self.apply(messages, tokenize=False)
        if not isinstance(rendered, str):
            raise TypeError("apply_chat_template(..., tokenize=False) must return a string.")
        return rendered

    def input_ids(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        encoded = self.apply(messages, tokenize=True)
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    @property
    def fingerprint(self) -> str:
        template = getattr(self.tokenizer, "chat_template", None)
        identity = {
            "class": type(self.tokenizer).__qualname__,
            "name_or_path": getattr(self.tokenizer, "name_or_path", None),
            "chat_template": template,
            "template_label": self.template_label,
            "enable_thinking": self.enable_thinking,
        }
        payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=repr)
        return hashlib.sha256(payload.encode()).hexdigest()


def answer_patterns(k: int) -> list[tuple[str, ...]]:
    """Stable exhaustive order: YES precedes NO at every position."""

    return list(itertools.product((YES, NO), repeat=k))


def exact_bayesian_target(
    *,
    domain: Sequence[int],
    membership_sets: Sequence[Sequence[int]],
    reports: Sequence[str],
    reliabilities: Sequence[Fraction],
) -> tuple[Fraction, dict[int, Fraction] | None]:
    """Return prior-predictive evidence and the exact posterior, if defined."""

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


def reliability_surface(value: Fraction) -> str:
    """Render a reliability exactly, preferring a terminating decimal when possible."""

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


def derive_candidate_evidence(
    *,
    membership_sets: Sequence[Sequence[int]],
    reports: Sequence[str],
    reliabilities: Sequence[Fraction],
    probe: XVsYPosteriorProbe,
    question: CandidateEvidenceQuestion,
) -> dict[str, dict[str, object]]:
    """Project raw membership observations into pairwise sufficient evidence."""

    if not (len(membership_sets) == len(reports) == len(reliabilities)):
        raise ValueError("Questions, reports, and reliabilities must have equal lengths.")
    evidence: dict[str, dict[str, object]] = {}
    for candidate in (probe.x, probe.y):
        observations: list[dict[str, object]] = []
        for index, (membership, report, reliability) in enumerate(
            zip(membership_sets, reports, reliabilities), start=1
        ):
            if report not in (YES, NO):
                raise ValueError(f"Unknown report {report!r}.")
            predicted = YES if candidate in set(membership) else NO
            agrees = predicted == report
            observations.append(
                {
                    "observation_index": index,
                    "relation": (
                        question.agreement_surface if agrees else question.disagreement_surface
                    ),
                    "agrees": agrees,
                    "reliability_exact": fraction_text(reliability),
                    "reliability": float(reliability),
                    "reliability_surface": reliability_surface(reliability),
                }
            )
        evidence[str(candidate)] = {
            "candidate_value": candidate,
            "observations": observations,
        }
    return evidence


def render_candidate_evidence_prompt(
    *,
    n: int,
    candidate_evidence: Mapping[str, Mapping[str, object]],
    probe: XVsYPosteriorProbe,
    question: CandidateEvidenceQuestion,
) -> str:
    """Render the reduced control without accepting raw questions or reports."""

    lines = [
        f"A value s is uniformly distributed over the integers 1 through {n}.",
        (
            f"The candidates s={probe.x} and s={probe.y} have equal prior probability."
        ),
        "",
        (
            f"For a candidate, {question.agreement_surface} means that the observed report "
            "matches the answer predicted by that candidate. "
            f"A {question.agreement_surface} observation contributes its stated reliability "
            f"to that candidate's likelihood. {question.disagreement_surface} contributes "
            "one minus that reliability. Observations are conditionally independent."
        ),
        "",
    ]
    for candidate in (probe.x, probe.y):
        evidence = candidate_evidence[str(candidate)]
        if int(evidence["candidate_value"]) != candidate:
            raise ValueError(
                f"Candidate evidence for s={candidate} has the wrong candidate value."
            )
        lines.append(f"Candidate s={evidence['candidate_value']}:")
        observations = evidence["observations"]
        if not isinstance(observations, Sequence):
            raise TypeError("candidate observations must be a sequence.")
        for raw_observation in observations:
            if not isinstance(raw_observation, Mapping):
                raise TypeError("candidate observations must be mappings.")
            lines.append(
                f"Observation {raw_observation['observation_index']}: "
                f"{raw_observation['relation']}; reliability "
                f"{raw_observation['reliability_surface']}."
            )
        lines.append("")
    choices = f"{probe.x} or {probe.y}" + (" or SAME" if probe.allow_same else "")
    allowed_values = [str(probe.x), str(probe.y)] + (["SAME"] if probe.allow_same else [])
    allowed_responses = " | ".join(
        f"{probe.answer_prefix}{value}" for value in allowed_values
    )
    lines.extend(
        [
            "Which candidate has greater posterior probability after all observations?",
            f"The decision value must be exactly one of: {choices}.",
            *(
                ["If the two posterior probabilities are equal, the required value is SAME."]
                if probe.allow_same
                else []
            ),
            "Do not provide reasoning, explanation, calculations, or intermediate work.",
            (
                f"Output exactly one of: {allowed_responses}. Do not include any other text "
                "or punctuation, and do not insert whitespace after the answer prefix."
            ),
        ]
    )
    return "\n".join(lines)


def _format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def render_observable_prompt(
    *,
    n: int,
    questions: Sequence[Mapping[str, object]],
    reports: Sequence[str],
    reliabilities: Sequence[Fraction],
    shared_reliability: bool,
    probe: XVsYPosteriorProbe,
    stage: Literal["answer", "reasoning"],
) -> str:
    """Render exclusively from fields available to an observer of the transcript."""

    lines = [
        "A secret integer s was sampled uniformly from the displayed domain.",
        f"DOMAIN: s is one of {_format_set(range(1, n + 1))}.",
        (
            "Each question asks whether s is in a displayed set. Its truthful answer is "
            "YES exactly when s is in that set, otherwise NO."
        ),
    ]
    if shared_reliability:
        exact_r = fraction_text(reliabilities[0])
        displayed_r = reliability_surface(reliabilities[0])
        lines.extend(
            [
                (
                    "The observed SOURCE report equals the truthful answer with probability "
                    f"r={displayed_r} (exactly {exact_r}) and is flipped with probability "
                    f"1-r={reliability_surface(1 - reliabilities[0])}."
                ),
                (
                    "Equivalently: if the truthful answer is YES, SOURCE reports YES with "
                    f"probability {displayed_r} and NO with probability "
                    f"{reliability_surface(1 - reliabilities[0])}; if the truthful answer is "
                    f"NO, SOURCE reports NO with probability {displayed_r} and YES with "
                    f"probability {reliability_surface(1 - reliabilities[0])}."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "Each question states its own reliability r_i. For that question, the "
                    "observed SOURCE report equals the truthful answer with probability r_i "
                    "and is flipped with probability 1-r_i."
                ),
                (
                    "Equivalently: SOURCE reports the truthful YES/NO value with probability "
                    "r_i and the opposite YES/NO value with probability 1-r_i."
                ),
            ]
        )
    lines.extend(
        [
            "Channel outcomes are independent conditional on s.",
            (
                "The questions were chosen externally, so their displayed set contents are "
                "not evidence about s."
            ),
            "",
            "OBSERVATIONS:",
        ]
    )
    for index, (question, report, reliability) in enumerate(
        zip(questions, reports, reliabilities), start=1
    ):
        suffix = (
            ""
            if shared_reliability
            else (
                f" [SOURCE reliability r={reliability_surface(reliability)} "
                f"(exactly {fraction_text(reliability)})]"
            )
        )
        membership = question["membership_set"]
        lines.append(f"Q{index}: Is s in {_format_set(membership)}?{suffix}")  # type: ignore[arg-type]
        lines.append(f"SOURCE reported {report}.")
    lines.extend(
        [
            "",
            "QUESTION:",
            (
                f"Given all observations, which has larger posterior probability: s={probe.x} "
                f"or s={probe.y}?"
            ),
        ]
    )
    choices = f"{probe.x} or {probe.y}" + (" or SAME" if probe.allow_same else "")
    lines.append(f"The decision value must be exactly one of: {choices}.")
    if probe.allow_same:
        lines.append(
            "If the two posterior probabilities are equal, the required decision value is SAME."
        )
    allowed_values = [str(probe.x), str(probe.y)] + (["SAME"] if probe.allow_same else [])
    allowed_responses = " | ".join(
        f"{probe.answer_prefix}{value}" for value in allowed_values
    )
    if stage == "reasoning":
        lines.extend(
            [
                (
                    "Reason carefully from the raw observations and the stated channel rules. "
                    f"Keep the explanation at most {probe.reasoning_budget} words."
                ),
                (
                    f"When asked for the final answer, output exactly one of: "
                    f"{allowed_responses}. Do not include any other text in the answer, and do "
                    "not insert whitespace after the answer prefix."
                ),
                "REASONING:",
            ]
        )
    else:
        lines.extend(
            [
                "Do not provide reasoning, explanation, calculations, or intermediate work.",
                (
                    f"Output exactly one of: {allowed_responses}. Your entire assistant "
                    "response must be that allowed response with no other words or punctuation. "
                    "Do not insert whitespace after the answer prefix."
                ),
            ]
        )
    return "\n".join(lines)


def initial_messages(
    *, observable_prompt: str, system_prompt: SystemPrompt
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.content:
        messages.append({"role": "system", "content": system_prompt.content})
    messages.append({"role": "user", "content": observable_prompt})
    return messages


def strip_thinking_markers(text: str) -> str:
    """Remove Qwen-native think delimiters from a visible reasoning completion."""

    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def enforce_reasoning(text: str, budget: int) -> str:
    """Remove a generated answer tail and enforce a whitespace-delimited word cap."""

    if budget <= 0:
        return ""
    text = strip_thinking_markers(text)
    # A model may anticipate the next stage.  Anything from that marker onward is discarded.
    upper = text.upper()
    marker = upper.find("ANSWER:")
    if marker >= 0:
        text = text[:marker]
    return " ".join(text.strip().split()[:budget])


def stage_two_messages(
    *,
    first_messages: Sequence[Mapping[str, str]],
    enforced_reasoning: str,
    layout: CallLayout,
) -> list[dict[str, str]]:
    if not first_messages:
        raise ValueError("first_messages must not be empty.")
    if layout == "conversation":
        return [
            *(dict(message) for message in first_messages),
            {"role": "assistant", "content": enforced_reasoning},
            {
                "role": "user",
                "content": "Now provide only the final answer in the required format.",
            },
        ]
    if layout == "replay_user":
        system = [dict(message) for message in first_messages if message["role"] == "system"]
        user = next(message["content"] for message in first_messages if message["role"] == "user")
        replay = (
            f"{user}\n{enforced_reasoning}\n"
            "Now provide only the final answer in the required format."
        )
        return [*system, {"role": "user", "content": replay}]
    raise ValueError(f"Unknown call layout {layout!r}.")


@dataclass(frozen=True)
class CaptureSpec:
    """Tensor capture policy. Persistence is disabled unless at least one field is set."""

    logits_boundaries: tuple[str, ...] = ()
    streams: tuple[str, ...] = ()
    layers: object = "all"
    tokens: object = "last"
    every_decode_position: bool = False

    def __post_init__(self) -> None:
        valid_boundaries = {"reasoning", "answer"}
        if not set(self.logits_boundaries) <= valid_boundaries:
            raise ValueError("logits_boundaries may contain only 'reasoning' and 'answer'.")
        valid_streams = {"resid_pre", "token_mixer_out", "mlp_out", "resid_post"}
        if not set(self.streams) <= valid_streams:
            raise ValueError(f"Unknown activation stream: {set(self.streams) - valid_streams}")

    @property
    def enabled(self) -> bool:
        return bool(self.logits_boundaries or self.streams or self.every_decode_position)


@dataclass(frozen=True)
class MetricSpec:
    """Answer scoring surfaces; omitted X/Y surfaces resolve from each row's probe."""

    x_surface: str | None = None
    y_surface: str | None = None
    same_surface: str = "SAME"
    sequence_scores: bool = True

    def resolve(self, *, x: int, y: int) -> MetricSpec:
        resolved = MetricSpec(
            x_surface=str(x) if self.x_surface is None else self.x_surface,
            y_surface=str(y) if self.y_surface is None else self.y_surface,
            same_surface=self.same_surface,
            sequence_scores=self.sequence_scores,
        )
        surfaces = list(resolved.surfaces.values())
        if any(not surface for surface in surfaces):
            raise ValueError("Metric answer surfaces must not be empty.")
        if len({surface.casefold() for surface in surfaces}) != len(surfaces):
            raise ValueError("Metric answer surfaces must be distinct (ignoring case).")
        return resolved

    @property
    def surfaces(self) -> dict[str, str]:
        if self.x_surface is None or self.y_surface is None:
            raise ValueError("Resolve MetricSpec with probe x and y before reading surfaces.")
        return {"X": self.x_surface, "Y": self.y_surface, "SAME": self.same_surface}


def stable_row_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


def _posterior_target_fields(
    *,
    posterior: Mapping[int, Fraction] | None,
    probe: XVsYPosteriorProbe,
) -> dict[str, object]:
    if posterior is None:
        return {
            "posterior_exact": None,
            "posterior": None,
            "x_posterior_exact": None,
            "y_posterior_exact": None,
            "x_posterior": None,
            "y_posterior": None,
            "posterior_difference": None,
            "posterior_log_odds": None,
            "posterior_log_odds_base": SOFTMAX_LOG_BASE,
            "posterior_log_odds_unit": SOFTMAX_LOG_UNIT,
            "ground_truth_choice": None,
            "normative_comparison": None,
        }
    x_probability = posterior[probe.x]
    y_probability = posterior[probe.y]
    if x_probability > y_probability:
        comparison = "X"
    elif y_probability > x_probability:
        comparison = "Y"
    else:
        comparison = "SAME"
    ground_truth = comparison if comparison != "SAME" or probe.allow_same else None
    if x_probability > 0 and y_probability > 0:
        log_odds: float | None = natural_log_ratio(x_probability, y_probability)
    elif x_probability == y_probability:
        log_odds = 0.0
    else:
        log_odds = None
    return {
        "posterior_exact": {
            str(candidate): fraction_text(probability)
            for candidate, probability in posterior.items()
        },
        "posterior": {
            str(candidate): float(probability) for candidate, probability in posterior.items()
        },
        "x_posterior_exact": fraction_text(x_probability),
        "y_posterior_exact": fraction_text(y_probability),
        "x_posterior": float(x_probability),
        "y_posterior": float(y_probability),
        "posterior_difference": float(x_probability - y_probability),
        "posterior_log_odds": log_odds,
        "posterior_log_odds_base": SOFTMAX_LOG_BASE,
        "posterior_log_odds_unit": SOFTMAX_LOG_UNIT,
        "ground_truth_choice": ground_truth,
        "normative_comparison": comparison,
    }


def build_candidate_evidence_row(
    *,
    source_row: Mapping[str, object],
    environment: CandidateEvidenceBayesianEnvironment,
    question: CandidateEvidenceQuestion,
    probe: XVsYPosteriorProbe,
    system_prompt: SystemPrompt,
    tokenizer_binding: TokenizerBinding,
) -> dict[str, object]:
    """Build one reduced row while retaining the raw transcript for auditing only."""

    if source_row.get("representation") == "candidate_evidence":
        raise ValueError("Candidate-evidence rows cannot be projected a second time.")
    if "answer_completion" in source_row:
        raise ValueError("Project a pre-run raw dataset, not an inference result dataset.")
    if int(source_row["n"]) != environment.n or int(source_row["k"]) != environment.k:
        raise ValueError("Source row does not match the candidate-evidence environment.")
    if list(source_row["domain"]) != list(environment.domain):  # type: ignore[arg-type]
        raise ValueError("Source row domain does not match the environment.")
    probe.validate(environment.n)
    if probe.reasoning_budget != 0:
        raise ValueError("The candidate-evidence control requires reasoning_budget=0.")
    for key, expected in (("x", probe.x), ("y", probe.y)):
        if int(source_row[key]) != expected:
            raise ValueError(f"Source row {key} does not match the reduced probe.")
    if bool(source_row["allow_same"]) != probe.allow_same:
        raise ValueError("Reduced and raw probes must use the same allow_same policy.")
    if str(source_row.get("answer_prefix", "ANSWER:")) != probe.answer_prefix:
        raise ValueError("Reduced and raw probes must use the same answer prefix.")

    reliabilities = tuple(
        as_fraction(str(value))
        for value in source_row["reliabilities_exact"]  # type: ignore[union-attr]
    )
    if reliabilities != environment.reliabilities:
        raise ValueError("Source row reliabilities do not match the environment.")
    membership_sets = [
        list(values)
        for values in source_row["membership_sets"]  # type: ignore[union-attr]
    ]
    reports = [str(value) for value in source_row["observed_reports"]]  # type: ignore[union-attr]
    evidence_mass, posterior = exact_bayesian_target(
        domain=environment.domain,
        membership_sets=membership_sets,
        reports=reports,
        reliabilities=reliabilities,
    )
    target_fields = _posterior_target_fields(posterior=posterior, probe=probe)
    if source_row.get("prior_predictive_exact") != fraction_text(evidence_mass):
        raise ValueError("Source prior-predictive mass failed exact recomputation.")
    if source_row.get("posterior_exact") != target_fields["posterior_exact"]:
        raise ValueError("Source posterior failed exact recomputation.")
    for key in (
        "x_posterior_exact",
        "y_posterior_exact",
        "ground_truth_choice",
        "normative_comparison",
    ):
        if source_row.get(key) != target_fields[key]:
            raise ValueError(f"Source target field {key!r} failed exact recomputation.")

    candidate_evidence = derive_candidate_evidence(
        membership_sets=membership_sets,
        reports=reports,
        reliabilities=reliabilities,
        probe=probe,
        question=question,
    )
    observable = render_candidate_evidence_prompt(
        n=environment.n,
        candidate_evidence=candidate_evidence,
        probe=probe,
        question=question,
    )
    messages = initial_messages(observable_prompt=observable, system_prompt=system_prompt)
    serialized_prompt = tokenizer_binding.serialize(messages)
    input_ids = tokenizer_binding.input_ids(messages)
    source_row_id = str(source_row["row_id"])
    identity = {
        "schema_version": SCHEMA_VERSION,
        "representation": "candidate_evidence",
        "source_row_id": source_row_id,
        "question": {
            "agreement_surface": question.agreement_surface,
            "disagreement_surface": question.disagreement_surface,
            "reliability_format": question.reliability_format,
            "layout": question.layout,
        },
        "probe": {
            "x": probe.x,
            "y": probe.y,
            "allow_same": probe.allow_same,
            "reasoning_budget": probe.reasoning_budget,
            "call_layout": probe.call_layout,
            "answer_prefix": probe.answer_prefix,
        },
        "system_prompt": system_prompt.content,
        "messages": messages,
        "serialized_prompt": serialized_prompt,
        "input_ids": input_ids,
        "tokenizer_template_fingerprint": tokenizer_binding.fingerprint,
    }
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "row_id": stable_row_id(identity),
        "source_row_id": source_row_id,
        "representation": "candidate_evidence",
        "environment_type": "candidate_evidence_bayesian",
        "question_set_index": int(source_row["question_set_index"]),
        "answer_pattern_index": int(source_row["answer_pattern_index"]),
        "domain": list(environment.domain),
        "n": environment.n,
        "k": environment.k,
        "reliabilities_exact": [fraction_text(value) for value in reliabilities],
        "reliabilities": [float(value) for value in reliabilities],
        "shared_reliability": environment.shared_reliability,
        "x": probe.x,
        "y": probe.y,
        "allow_same": probe.allow_same,
        "reasoning_budget": probe.reasoning_budget,
        "call_layout": probe.call_layout,
        "answer_prefix": probe.answer_prefix,
        "candidate_evidence": candidate_evidence,
        "audit_metadata": {
            "source_row_id": source_row_id,
            "answer_pattern": source_row.get("answer_pattern"),
            "questions": source_row.get("questions"),
            "membership_sets": membership_sets,
            "observed_reports": reports,
        },
        "messages": messages,
        "serialized_prompt": serialized_prompt,
        "input_ids": input_ids,
        "tokenizer_template_fingerprint": tokenizer_binding.fingerprint,
        "prior_predictive_exact": fraction_text(evidence_mass),
        "prior_predictive": float(evidence_mass),
        "posterior_state": ("defined" if posterior is not None else "undefined_zero_evidence"),
        **target_fields,
    }
    return row


def build_row(
    *,
    environment: NoisyChannelBayesianEnvironment,
    questions: Sequence[Mapping[str, object]],
    question_set_index: int,
    reports: Sequence[str],
    answer_pattern_index: int,
    probe: XVsYPosteriorProbe,
    system_prompt: SystemPrompt,
    tokenizer_binding: TokenizerBinding,
) -> dict[str, object]:
    evidence, posterior = exact_bayesian_target(
        domain=environment.domain,
        membership_sets=[question["membership_set"] for question in questions],  # type: ignore[list-item]
        reports=reports,
        reliabilities=environment.reliabilities,
    )
    stage = "reasoning" if probe.reasoning_budget > 0 else "answer"
    observable = render_observable_prompt(
        n=environment.n,
        questions=questions,
        reports=reports,
        reliabilities=environment.reliabilities,
        shared_reliability=environment.shared_reliability,
        probe=probe,
        stage=stage,
    )
    messages = initial_messages(observable_prompt=observable, system_prompt=system_prompt)
    serialized_prompt = tokenizer_binding.serialize(messages)
    input_ids = tokenizer_binding.input_ids(messages)
    pattern_surface = "".join("Y" if report == YES else "N" for report in reports)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "question_set_index": question_set_index,
        "answer_pattern_index": answer_pattern_index,
        "answer_pattern": pattern_surface,
        "membership_sets": [question["membership_set"] for question in questions],
        "reports": list(reports),
        "reliabilities": [fraction_text(value) for value in environment.reliabilities],
        "x": probe.x,
        "y": probe.y,
        "allow_same": probe.allow_same,
        "reasoning_budget": probe.reasoning_budget,
        "call_layout": probe.call_layout,
        "answer_prefix": probe.answer_prefix,
        "system_prompt": system_prompt.content,
        "messages": messages,
        "serialized_prompt": serialized_prompt,
        "input_ids": input_ids,
        "tokenizer_template_fingerprint": tokenizer_binding.fingerprint,
    }
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "row_id": stable_row_id(identity),
        "question_set_index": question_set_index,
        "answer_pattern_index": answer_pattern_index,
        "answer_pattern": pattern_surface,
        "domain": list(environment.domain),
        "n": environment.n,
        "k": environment.k,
        "reliabilities_exact": [fraction_text(value) for value in environment.reliabilities],
        "reliabilities": [float(value) for value in environment.reliabilities],
        "shared_reliability": environment.shared_reliability,
        "questions": [dict(question) for question in questions],
        "membership_sets": [list(question["membership_set"]) for question in questions],  # type: ignore[arg-type]
        "observed_reports": list(reports),
        "x": probe.x,
        "y": probe.y,
        "allow_same": probe.allow_same,
        "reasoning_budget": probe.reasoning_budget,
        "call_layout": probe.call_layout,
        "answer_prefix": probe.answer_prefix,
        "messages": messages,
        "serialized_prompt": serialized_prompt,
        "input_ids": input_ids,
        "tokenizer_template_fingerprint": tokenizer_binding.fingerprint,
        "prior_predictive_exact": fraction_text(evidence),
        "prior_predictive": float(evidence),
        "posterior_state": "defined" if posterior is not None else "undefined_zero_evidence",
    }
    if posterior is None:
        row.update(
            {
                "posterior_exact": None,
                "posterior": None,
                "x_posterior_exact": None,
                "y_posterior_exact": None,
                "x_posterior": None,
                "y_posterior": None,
                "posterior_difference": None,
                "posterior_log_odds": None,
                "posterior_log_odds_base": SOFTMAX_LOG_BASE,
                "posterior_log_odds_unit": SOFTMAX_LOG_UNIT,
                "ground_truth_choice": None,
                "normative_comparison": None,
            }
        )
        return row
    x_probability = posterior[probe.x]
    y_probability = posterior[probe.y]
    if x_probability > y_probability:
        comparison = "X"
    elif y_probability > x_probability:
        comparison = "Y"
    else:
        comparison = "SAME"
    ground_truth = comparison if comparison != "SAME" or probe.allow_same else None
    if x_probability > 0 and y_probability > 0:
        log_odds: float | None = natural_log_ratio(x_probability, y_probability)
    elif x_probability == y_probability:
        log_odds = 0.0
    else:
        # The finite real-valued log-odds is undefined at probability endpoints.
        log_odds = None
    row.update(
        {
            # These candidate posteriors are deterministic functions of the displayed
            # transcript and are deliberately part of the persisted target schema.
            "posterior_exact": {
                str(candidate): fraction_text(probability)
                for candidate, probability in posterior.items()
            },
            "posterior": {
                str(candidate): float(probability) for candidate, probability in posterior.items()
            },
            "x_posterior_exact": fraction_text(x_probability),
            "y_posterior_exact": fraction_text(y_probability),
            "x_posterior": float(x_probability),
            "y_posterior": float(y_probability),
            "posterior_difference": float(x_probability - y_probability),
            "posterior_log_odds": log_odds,
            "posterior_log_odds_base": SOFTMAX_LOG_BASE,
            "posterior_log_odds_unit": SOFTMAX_LOG_UNIT,
            "ground_truth_choice": ground_truth,
            "normative_comparison": comparison,
        }
    )
    return row
