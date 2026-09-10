"""Exact data generation for exhaustive noisy-channel transcripts.

This module intentionally has no dependency on the older experiment modules.  It
contains only deterministic prompt construction and exact rational arithmetic;
model execution lives in :mod:`runner`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from .constants import NO, SOFTMAX_LOG_BASE, SOFTMAX_LOG_UNIT, YES
from .env import CandidateEvidenceBayesianEnvironment, NoisyChannelBayesianEnvironment
from .probes import XVsYPosteriorProbe, _posterior_target_fields
from .questions import CandidateEvidenceQuestion
from .utils import (
    SystemPrompt,
    _format_set,
    as_fraction,
    candidate_agreements,
    exact_bayesian_target,
    fraction_text,
    initial_messages,
    natural_log_ratio,
    reliability_surface,
)

SCHEMA_VERSION = "1.0"

@dataclass(frozen=True)
class TokenizerBinding:
    """
    Bind generation to one tokenizer and its exact chat-template serialization.
    VERIFIED
    """

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


def derive_candidate_evidence(
    *,
    membership_sets: Sequence[Sequence[int]],
    reports: Sequence[str],
    reliabilities: Sequence[Fraction],
    probe: XVsYPosteriorProbe,
    question: CandidateEvidenceQuestion,
) -> dict[str, dict[str, object]]:
    """
    Project raw membership observations into pairwise sufficient evidence.
    VERIFIED
    """

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


def _final_answer_format_lines(
    *, probe: XVsYPosteriorProbe, allowed_values: Sequence[str]
) -> list[str]:
    """VERIFIED"""
    return [
        "FINAL-ANSWER FORMAT:",
        "End your response with exactly one of these lines:",
        *(f"{probe.answer_prefix} {value}" for value in allowed_values),
        "Use exactly one ASCII space between the answer prefix and the decision value.",
        f'Do not use "{probe.answer_prefix}" anywhere else in your response.',
        "Do not output anything after the final answer line.",
    ]


def render_candidate_evidence_prompt(
    *,
    n: int,
    candidate_evidence: Mapping[str, Mapping[str, object]],
    probe: XVsYPosteriorProbe,
    question: CandidateEvidenceQuestion,
) -> str:
    """
    Render the reduced control without accepting raw questions or reports.
    VERIFIED
    """

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
            "Output only the final answer line.",
            "",
            *_final_answer_format_lines(probe=probe, allowed_values=allowed_values),
        ]
    )
    return "\n".join(lines)


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
    """
    Render exclusively from fields available to an observer of the transcript.
    VERIFIED
    """

    lines = [
        "A secret integer s was sampled uniformly from the displayed domain.",
        f"DOMAIN: s is one of {_format_set(range(1, n + 1))}.",
        (
            "Each question asks whether s is in a displayed set. Its truthful answer is "
            "YES exactly when s is in that set, otherwise NO."
        ),
    ]
    if shared_reliability:
        displayed_r = reliability_surface(reliabilities[0])
        lines.extend(
            [
                (
                    "The observed SOURCE report equals the truthful answer with probability "
                    f"r={displayed_r} and is flipped with probability "
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
            else f" [SOURCE reliability r={reliability_surface(reliability)}]"
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
    else:
        lines.append(
            f"If the two posterior probabilities are equal, output either {probe.x} or "
            f"{probe.y}; either value is valid."
        )
    allowed_values = [str(probe.x), str(probe.y)] + (["SAME"] if probe.allow_same else [])
    permitted_values = ", ".join(allowed_values)
    reliability_name = "r" if shared_reliability else "r_i"
    lines.extend(
        [
            (
                f"Compare only candidates {probe.x} and {probe.y}. For each report, use "
                f"probability {reliability_name} when the candidate's truthful answer matches "
                f"the report and 1-{reliability_name} otherwise."
            ),
            f"The permitted values are {permitted_values}.",
        ]
    )
    if stage == "reasoning":
        lines.extend(
            [
                "Reason carefully from the raw observations and the stated channel rules.",
                "Explain your reasoning before the final answer line.",
            ]
        )
    else:
        lines.extend(
            [
                "Do not provide reasoning, explanation, calculations, or intermediate work.",
                "Output only the final answer line. Thanks.",
            ]
        )
    lines.extend(
        [
            "",
            *_final_answer_format_lines(probe=probe, allowed_values=allowed_values),
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class MetricSpec:
    """
    Answer scoring surfaces; omitted X/Y surfaces resolve from each row's probe.
    VERIFIED
    """

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


def _canonical_candidate_target_fields(
    *,
    posterior: Mapping[int, Fraction] | None,
    candidate_1: int,
    candidate_2: int,
    allow_same: bool,
) -> dict[str, object]:
    """
    Posterior fields whose C1/C2 orientation is invariant to presentation order.
    VERIFIED
    """

    if posterior is None:
        return {
            "candidate_1_posterior_exact": None,
            "candidate_2_posterior_exact": None,
            "candidate_1_posterior": None,
            "candidate_2_posterior": None,
            "candidate_1_minus_candidate_2_posterior": None,
            "candidate_1_minus_candidate_2_log_odds": None,
            "candidate_1_minus_candidate_2_log_odds_base": SOFTMAX_LOG_BASE,
            "candidate_1_minus_candidate_2_log_odds_unit": SOFTMAX_LOG_UNIT,
            "canonical_ground_truth_choice": None,
            "canonical_normative_comparison": None,
        }
    candidate_1_probability = posterior[candidate_1]
    candidate_2_probability = posterior[candidate_2]
    if candidate_1_probability > candidate_2_probability:
        comparison = "C1"
    elif candidate_2_probability > candidate_1_probability:
        comparison = "C2"
    else:
        comparison = "SAME"
    ground_truth = comparison if comparison != "SAME" or allow_same else None
    if candidate_1_probability > 0 and candidate_2_probability > 0:
        log_odds: float | None = natural_log_ratio(
            candidate_1_probability, candidate_2_probability
        )
    elif candidate_1_probability == candidate_2_probability:
        log_odds = 0.0
    else:
        log_odds = None
    return {
        "candidate_1_posterior_exact": fraction_text(candidate_1_probability),
        "candidate_2_posterior_exact": fraction_text(candidate_2_probability),
        "candidate_1_posterior": float(candidate_1_probability),
        "candidate_2_posterior": float(candidate_2_probability),
        "candidate_1_minus_candidate_2_posterior": float(
            candidate_1_probability - candidate_2_probability
        ),
        "candidate_1_minus_candidate_2_log_odds": log_odds,
        "candidate_1_minus_candidate_2_log_odds_base": SOFTMAX_LOG_BASE,
        "candidate_1_minus_candidate_2_log_odds_unit": SOFTMAX_LOG_UNIT,
        "canonical_ground_truth_choice": ground_truth,
        "canonical_normative_comparison": comparison,
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
    """
    Build one reduced row while retaining the raw transcript for auditing only.
    VERIFIED
    """

    if source_row.get("representation") == "candidate_evidence":
        raise ValueError("Candidate-evidence rows cannot be projected a second time.")
    if "answer_completion" in source_row:
        raise ValueError("Project a pre-run raw dataset, not an inference result dataset.")
    if int(source_row["n"]) != environment.n or int(source_row["k"]) != environment.k:
        raise ValueError("Source row does not match the candidate-evidence environment.")
    if list(source_row["domain"]) != list(environment.domain):  # type: ignore[arg-type]
        raise ValueError("Source row domain does not match the environment.")
    probe.validate(environment.n)
    if probe.reasoning:
        raise ValueError("The candidate-evidence control requires reasoning=False.")
    for key, expected in (("x", probe.x), ("y", probe.y)):
        if int(source_row[key]) != expected:
            raise ValueError(f"Source row {key} does not match the reduced probe.")
    if bool(source_row["allow_same"]) != probe.allow_same:
        raise ValueError("Reduced and raw probes must use the same allow_same policy.")
    if str(source_row.get("answer_prefix", "ANSWER:")) != probe.answer_prefix:
        raise ValueError("Reduced and raw probes must use the same answer prefix.")
    if bool(source_row.get("control_positional_bias", False)) != environment.control_positional_bias:
        raise ValueError("Reduced and raw environments disagree on positional-bias control.")
    candidate_1 = int(source_row.get("candidate_1", probe.x))
    candidate_2 = int(source_row.get("candidate_2", probe.y))
    if {probe.x, probe.y} != {candidate_1, candidate_2}:
        raise ValueError("Source presented candidates do not match canonical C1/C2 candidates.")

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
    canonical_target_fields = _canonical_candidate_target_fields(
        posterior=posterior,
        candidate_1=candidate_1,
        candidate_2=candidate_2,
        allow_same=probe.allow_same,
    )
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
            "reasoning": probe.reasoning,
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
        "control_positional_bias": environment.control_positional_bias,
        "presentations_per_scenario": int(
            source_row.get("presentations_per_scenario", 1)
        ),
        "positional_control_pair_id": source_row.get("positional_control_pair_id"),
        "presentation_index": int(source_row.get("presentation_index", 0)),
        "presentation_order": source_row.get("presentation_order", "C1_C2"),
        "candidate_1": candidate_1,
        "candidate_2": candidate_2,
        "candidate_1_position": int(source_row.get("candidate_1_position", 1)),
        "candidate_2_position": int(source_row.get("candidate_2_position", 2)),
        "candidate_value_order": [probe.x, probe.y],
        "x": probe.x,
        "y": probe.y,
        "allow_same": probe.allow_same,
        "reasoning": probe.reasoning,
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
        **canonical_target_fields,
    }
    candidate_1_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=candidate_1
    )
    candidate_2_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=candidate_2
    )
    x_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=probe.x
    )
    y_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=probe.y
    )
    row.update(
        {
            "agreement_x_by_question": x_flags,
            "agreement_y_by_question": y_flags,
            "total_agreement_x": sum(x_flags),
            "total_agreement_y": sum(y_flags),
            "agreement_candidate_1_by_question": candidate_1_flags,
            "agreement_candidate_2_by_question": candidate_2_flags,
            "total_agreement_candidate_1": sum(candidate_1_flags),
            "total_agreement_candidate_2": sum(candidate_2_flags),
        }
    )
    return row


def build_row(
    *,
    environment: NoisyChannelBayesianEnvironment,
    questions: Sequence[Mapping[str, object]],
    question_set_index: int,
    reports: Sequence[str],
    answer_pattern_index: int,
    probe: XVsYPosteriorProbe,
    canonical_probe: XVsYPosteriorProbe | None = None,
    presentation_index: int = 0,
    parameterization_index: int = 0,
    environment_parameter_index: int = 0,
    question_parameter_index: int = 0,
    probe_parameter_index: int = 0,
    system_prompt: SystemPrompt,
    tokenizer_binding: TokenizerBinding,
) -> dict[str, object]:
    canonical_probe = probe if canonical_probe is None else canonical_probe
    candidate_1 = canonical_probe.x
    candidate_2 = canonical_probe.y
    if {probe.x, probe.y} != {candidate_1, candidate_2}:
        raise ValueError("Presented probe candidates must match canonical C1/C2 candidates.")
    for field_name in ("allow_same", "reasoning", "call_layout", "answer_prefix"):
        if getattr(probe, field_name) != getattr(canonical_probe, field_name):
            raise ValueError(f"Presented and canonical probes disagree on {field_name}.")
    presentation_count = 2 if environment.control_positional_bias else 1
    if presentation_index not in range(presentation_count):
        raise ValueError(
            f"presentation_index must lie in 0..{presentation_count - 1} for this environment."
        )
    expected_order = (
        (candidate_1, candidate_2) if presentation_index == 0 else (candidate_2, candidate_1)
    )
    if (probe.x, probe.y) != expected_order:
        raise ValueError("Presented probe order does not match presentation_index.")

    membership_sets = [
        list(question["membership_set"]) for question in questions  # type: ignore[arg-type]
    ]
    evidence, posterior = exact_bayesian_target(
        domain=environment.domain,
        membership_sets=membership_sets,
        reports=reports,
        reliabilities=environment.reliabilities,
    )
    stage = "reasoning" if probe.reasoning else "answer"
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
    presentation_order = "C1_C2" if presentation_index == 0 else "C2_C1"
    pair_identity = {
        "schema_version": SCHEMA_VERSION,
        "question_set_index": question_set_index,
        "answer_pattern_index": answer_pattern_index,
        "membership_sets": membership_sets,
        "reports": list(reports),
        "reliabilities": [fraction_text(value) for value in environment.reliabilities],
        "candidate_1": candidate_1,
        "candidate_2": candidate_2,
        "allow_same": canonical_probe.allow_same,
        "reasoning": canonical_probe.reasoning,
        "call_layout": canonical_probe.call_layout,
        "answer_prefix": canonical_probe.answer_prefix,
        "system_prompt": system_prompt.content,
        "tokenizer_template_fingerprint": tokenizer_binding.fingerprint,
    }
    positional_control_pair_id = stable_row_id(pair_identity)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "control_positional_bias": environment.control_positional_bias,
        "positional_control_pair_id": positional_control_pair_id,
        "presentation_index": presentation_index,
        "presentation_order": presentation_order,
        "question_set_index": question_set_index,
        "answer_pattern_index": answer_pattern_index,
        "answer_pattern": pattern_surface,
        "membership_sets": [question["membership_set"] for question in questions],
        "reports": list(reports),
        "reliabilities": [fraction_text(value) for value in environment.reliabilities],
        "x": probe.x,
        "y": probe.y,
        "allow_same": probe.allow_same,
        "reasoning": probe.reasoning,
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
        "parameterization_index": parameterization_index,
        "environment_parameter_index": environment_parameter_index,
        "question_parameter_index": question_parameter_index,
        "probe_parameter_index": probe_parameter_index,
        "question_set_index": question_set_index,
        "answer_pattern_index": answer_pattern_index,
        "answer_pattern": pattern_surface,
        "domain": list(environment.domain),
        "n": environment.n,
        "k": environment.k,
        "reliabilities_exact": [fraction_text(value) for value in environment.reliabilities],
        "reliabilities": [float(value) for value in environment.reliabilities],
        "shared_reliability": environment.shared_reliability,
        "control_positional_bias": environment.control_positional_bias,
        "presentations_per_scenario": presentation_count,
        "positional_control_pair_id": positional_control_pair_id,
        "presentation_index": presentation_index,
        "presentation_order": presentation_order,
        "candidate_1": candidate_1,
        "candidate_2": candidate_2,
        "candidate_1_position": 1 if probe.x == candidate_1 else 2,
        "candidate_2_position": 1 if probe.x == candidate_2 else 2,
        "candidate_value_order": [probe.x, probe.y],
        "questions": [dict(question) for question in questions],
        "membership_sets": membership_sets,
        "observed_reports": list(reports),
        "x": probe.x,
        "y": probe.y,
        "allow_same": probe.allow_same,
        "reasoning": probe.reasoning,
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
    x_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=probe.x
    )
    y_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=probe.y
    )
    candidate_1_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=candidate_1
    )
    candidate_2_flags = candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=candidate_2
    )
    row.update(
        {
            "agreement_x_by_question": x_flags,
            "agreement_y_by_question": y_flags,
            "total_agreement_x": sum(x_flags),
            "total_agreement_y": sum(y_flags),
            "agreement_candidate_1_by_question": candidate_1_flags,
            "agreement_candidate_2_by_question": candidate_2_flags,
            "total_agreement_candidate_1": sum(candidate_1_flags),
            "total_agreement_candidate_2": sum(candidate_2_flags),
        }
    )
    row.update(_posterior_target_fields(posterior=posterior, probe=probe))
    row.update(
        _canonical_candidate_target_fields(
            posterior=posterior,
            candidate_1=candidate_1,
            candidate_2=candidate_2,
            allow_same=canonical_probe.allow_same,
        )
    )
    return row
