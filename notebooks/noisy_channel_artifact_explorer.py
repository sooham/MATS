"""Read-only exploration helpers for noisy-channel Bayesian artifacts.

This module deliberately has no dependency on ``noisy_channel_bayesian``.  It
reads the completed JSONL artifact, derives analysis-friendly scalar columns,
and registers three in-memory SQLite tables.  It never writes to the artifact
directory.
"""

from __future__ import annotations

import html
import json
import math
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

DEFAULT_MODEL_KEY = "Qwen_Qwen3.5-9B_4c87a623"
DEFAULT_RUN_ID = "agreement_logits_reliability_sweep_generated_answer_line_v1_transformers"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository containing ``pyproject.toml``."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root (pyproject.toml).")


def default_results_path(repo_root: Path | None = None) -> Path:
    """Return notebook 13's checked-in Qwen3.5-9B results path."""
    root = repo_root or find_repo_root()
    return (
        root
        / "artifacts"
        / "noisy_channel_bayesian_experiment_2"
        / DEFAULT_MODEL_KEY
        / "runs"
        / DEFAULT_RUN_ID
        / "results.jsonl"
    )


def load_jsonl_readonly(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records using an explicitly read-only file handle."""
    resolved = path.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{resolved}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _mean_optional(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return fmean(finite) if finite else None


def _sign(value: float | None, *, tolerance: float = 1e-12) -> int | None:
    if value is None:
        return None
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _strict_log_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """A log ratio that leaves 0/0 undefined instead of silently returning zero."""
    if numerator is None or denominator is None:
        return None
    if numerator == 0.0 and denominator == 0.0:
        return None
    if numerator == 0.0:
        return -math.inf
    if denominator == 0.0:
        return math.inf
    return math.log(numerator) - math.log(denominator)


def _choice_from_z(value: float | None) -> str:
    sign = _sign(value)
    return {1: "C1", -1: "C2", 0: "TIE", None: "UNAVAILABLE"}[sign]


def _q_true(
    candidate_1_posterior: float | None, candidate_2_posterior: float | None
) -> float | None:
    if candidate_1_posterior is None or candidate_2_posterior is None:
        return None
    denominator = candidate_1_posterior + candidate_2_posterior
    return candidate_1_posterior / denominator if denominator > 0.0 else None


def _correct_from_z(z: float | None, q_true: float | None) -> bool | None:
    """Score only non-tied, candidate-defined Bayesian comparisons."""
    if z is None or q_true is None or math.isclose(q_true, 0.5, abs_tol=1e-12):
        return None
    return bool((z > 0.0) == (q_true > 0.5))


def _transition(before: bool | None, after: bool | None) -> str:
    if before is None or after is None:
        return "not scored"
    if not before and after:
        return "wrong→right"
    if before and not after:
        return "right→wrong"
    return "unchanged right" if before else "unchanged wrong"


def _endpoint_category(
    reliability: float,
    evidence: float,
    candidate_1_posterior: float | None,
    candidate_2_posterior: float | None,
) -> str | None:
    if reliability not in (0.0, 1.0):
        return None
    if evidence == 0.0:
        return "entire observation pattern has zero evidence"
    if candidate_1_posterior is None or candidate_2_posterior is None:
        raise ValueError("Defined endpoint evidence unexpectedly has undefined posteriors.")
    c1_positive = candidate_1_posterior > 0.0
    c2_positive = candidate_2_posterior > 0.0
    if c1_positive and not c2_positive:
        return "C1 possible and C2 impossible"
    if c2_positive and not c1_positive:
        return "C2 possible and C1 impossible"
    if not c1_positive and not c2_positive:
        return "both candidates have zero posterior"
    if not math.isclose(candidate_1_posterior, candidate_2_posterior, abs_tol=1e-12):
        raise ValueError("At a deterministic endpoint, two possible candidates should tie.")
    return "both candidates positive/tied"


def _candidate_probability_assigned_to_truth(
    q_llm: float | None, q_true: float | None
) -> float | None:
    if q_llm is None or q_true is None or math.isclose(q_true, 0.5, abs_tol=1e-12):
        return None
    return q_llm if q_true > 0.5 else 1.0 - q_llm


def _extract_surface_logit(row: Mapping[str, Any], candidate: int) -> float | None:
    values = row.get("answer_surface_raw_logits")
    if not isinstance(values, Mapping):
        return None
    value = values.get(str(candidate))
    return float(value) if value is not None else None


def flatten_transcript_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one source record into scalar, SQL-friendly columns."""
    reliabilities = [float(value) for value in row["reliabilities"]]
    exact_reliabilities = [str(value) for value in row["reliabilities_exact"]]
    if not reliabilities or len(set(reliabilities)) != 1 or len(set(exact_reliabilities)) != 1:
        raise ValueError(f"Row {row.get('row_id')} does not have one shared reliability.")
    reliability = reliabilities[0]
    reliability_exact = exact_reliabilities[0]
    candidate_1 = int(row["candidate_1"])
    candidate_2 = int(row["candidate_2"])
    agreements_1 = [int(value) for value in row["agreement_candidate_1_by_question"]]
    agreements_2 = [int(value) for value in row["agreement_candidate_2_by_question"]]
    factors_1 = [reliability if value else 1.0 - reliability for value in agreements_1]
    factors_2 = [reliability if value else 1.0 - reliability for value in agreements_2]
    likelihood_1 = math.prod(factors_1)
    likelihood_2 = math.prod(factors_2)
    posterior_1 = (
        float(row["candidate_1_posterior"])
        if row.get("candidate_1_posterior") is not None
        else None
    )
    posterior_2 = (
        float(row["candidate_2_posterior"])
        if row.get("candidate_2_posterior") is not None
        else None
    )
    evidence = float(row["prior_predictive"])
    z1 = _extract_surface_logit(row, candidate_1)
    z2 = _extract_surface_logit(row, candidate_2)
    z = z1 - z2 if z1 is not None and z2 is not None else None
    q_true = _q_true(posterior_1, posterior_2)
    q_llm = _sigmoid(z) if z is not None else None
    subsets = row["membership_sets"]
    reports = row["observed_reports"]

    flattened: dict[str, Any] = {
        "row_id": str(row["row_id"]),
        "positional_control_pair_id": str(row["positional_control_pair_id"]),
        "model": "Qwen/Qwen3.5-9B",
        "question_set_index": int(row["question_set_index"]),
        "answer_pattern_index": int(row["answer_pattern_index"]),
        "answer_pattern": str(row["answer_pattern"]),
        "reasoning": bool(row["reasoning"]),
        "reliability": reliability,
        "reliability_exact": reliability_exact,
        "presentation_order": str(row["presentation_order"]),
        "candidate_1_first": str(row["presentation_order"]) == "C1_C2",
        "candidate_2_first": str(row["presentation_order"]) == "C2_C1",
        "candidate_1": candidate_1,
        "candidate_2": candidate_2,
        "candidate_value_order_json": _json(row["candidate_value_order"]),
        "question_subsets_json": _json(subsets),
        "observed_reports_json": _json(reports),
        "candidate_1_agreements_json": _json(agreements_1),
        "candidate_2_agreements_json": _json(agreements_2),
        "candidate_1_likelihood_factors_json": _json(factors_1),
        "candidate_2_likelihood_factors_json": _json(factors_2),
        "total_agreement_candidate_1": int(row["total_agreement_candidate_1"]),
        "total_agreement_candidate_2": int(row["total_agreement_candidate_2"]),
        "agreement_difference": int(row["total_agreement_candidate_1"])
        - int(row["total_agreement_candidate_2"]),
        "candidate_1_likelihood_total": likelihood_1,
        "candidate_2_likelihood_total": likelihood_2,
        "candidate_1_minus_candidate_2_log_likelihood": _strict_log_ratio(
            likelihood_1, likelihood_2
        ),
        "candidate_1_posterior": posterior_1,
        "candidate_2_posterior": posterior_2,
        "candidate_1_posterior_exact": row.get("candidate_1_posterior_exact"),
        "candidate_2_posterior_exact": row.get("candidate_2_posterior_exact"),
        "candidate_1_minus_candidate_2_log_posterior": _strict_log_ratio(posterior_1, posterior_2),
        "total_marginalized_probability": evidence,
        "total_marginalized_probability_exact": str(row["prior_predictive_exact"]),
        "posterior_state": str(row["posterior_state"]),
        "q_true": q_true,
        "candidate_1_raw_logit": z1,
        "candidate_2_raw_logit": z2,
        "candidate_1_minus_candidate_2_raw_logit": z,
        "z": z,
        "q_llm": q_llm,
        "logit_choice": _choice_from_z(z),
        "logit_correct": _correct_from_z(z, q_true),
        "model_choice_canonical": row.get("model_choice_canonical"),
        "model_choice_surface": row.get("model_choice_surface"),
        "canonical_posterior_correct": row.get("canonical_posterior_correct"),
        "reasoning_length_tokens": row.get("reasoning_length_tokens"),
        "strict_answer_compliance": row.get("strict_answer_compliance"),
        "semantic_answer_compliance": row.get("semantic_answer_compliance"),
        "answer_boundary_available": z is not None,
        "full_completion": row.get("full_completion"),
        "endpoint_category": _endpoint_category(reliability, evidence, posterior_1, posterior_2),
        "answer_surface_raw_logits_json": _json(row.get("answer_surface_raw_logits")),
    }
    for index, (subset, report, agreement_1, agreement_2, factor_1, factor_2) in enumerate(
        zip(subsets, reports, agreements_1, agreements_2, factors_1, factors_2, strict=True),
        start=1,
    ):
        flattened[f"question_subset_{index}_json"] = _json(subset)
        flattened[f"report_{index}"] = str(report)
        flattened[f"candidate_1_agrees_q{index}"] = agreement_1
        flattened[f"candidate_2_agrees_q{index}"] = agreement_2
        flattened[f"candidate_1_likelihood_factor_q{index}"] = factor_1
        flattened[f"candidate_2_likelihood_factor_q{index}"] = factor_2
    return flattened


def make_balanced_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair C1-first/C2-first presentations and compute semantic contrasts."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["positional_control_pair_id"])].append(row)

    balanced_rows: list[dict[str, Any]] = []
    for pair_id, pair in groups.items():
        by_order = {str(row["presentation_order"]): row for row in pair}
        if set(by_order) != {"C1_C2", "C2_C1"} or len(pair) != 2:
            raise ValueError(f"Position pair {pair_id} is incomplete or duplicated.")
        c1_first = by_order["C1_C2"]
        c2_first = by_order["C2_C1"]
        invariant_fields = (
            "reasoning",
            "reliability",
            "reliability_exact",
            "question_set_index",
            "answer_pattern_index",
            "answer_pattern",
            "question_subsets_json",
            "observed_reports_json",
            "candidate_1_posterior",
            "candidate_2_posterior",
            "total_marginalized_probability",
        )
        for field in invariant_fields:
            if c1_first[field] != c2_first[field]:
                raise ValueError(f"Position pair {pair_id} differs at {field}.")
        z_c1_first = c1_first["z"]
        z_c2_first = c2_first["z"]
        complete = z_c1_first is not None and z_c2_first is not None
        z_balanced = 0.5 * (z_c1_first + z_c2_first) if complete else None
        delta_position = 0.5 * (z_c1_first - z_c2_first) if complete else None
        q_true = c1_first["q_true"]
        q_llm = _sigmoid(z_balanced) if z_balanced is not None else None
        choice_1 = c1_first["model_choice_canonical"]
        choice_2 = c2_first["model_choice_canonical"]
        calibration_error = q_llm - q_true if q_llm is not None and q_true is not None else None
        if q_llm is not None and q_true is not None:
            clipped = min(max(q_llm, 1e-12), 1.0 - 1e-12)
            brier = (q_llm - q_true) ** 2
            log_loss = -(q_true * math.log(clipped) + (1.0 - q_true) * math.log(1.0 - clipped))
        else:
            brier = None
            log_loss = None

        # Start from fields that are invariant and useful in downstream queries.
        kept_fields = (
            "model",
            "question_set_index",
            "answer_pattern_index",
            "answer_pattern",
            "reasoning",
            "reliability",
            "reliability_exact",
            "candidate_1",
            "candidate_2",
            "question_subsets_json",
            "observed_reports_json",
            "candidate_1_agreements_json",
            "candidate_2_agreements_json",
            "candidate_1_likelihood_factors_json",
            "candidate_2_likelihood_factors_json",
            "total_agreement_candidate_1",
            "total_agreement_candidate_2",
            "agreement_difference",
            "candidate_1_likelihood_total",
            "candidate_2_likelihood_total",
            "candidate_1_minus_candidate_2_log_likelihood",
            "candidate_1_posterior",
            "candidate_2_posterior",
            "candidate_1_posterior_exact",
            "candidate_2_posterior_exact",
            "candidate_1_minus_candidate_2_log_posterior",
            "total_marginalized_probability",
            "total_marginalized_probability_exact",
            "posterior_state",
            "q_true",
            "endpoint_category",
        )
        balanced = {field: c1_first[field] for field in kept_fields}
        balanced.update(
            {
                "positional_control_pair_id": pair_id,
                "source_c1_first_row_id": c1_first["row_id"],
                "source_c2_first_row_id": c2_first["row_id"],
                "z_c1_first": z_c1_first,
                "z_c2_first": z_c2_first,
                "z_balanced": z_balanced,
                "delta_position": delta_position,
                "absolute_order_gap": abs(z_c1_first - z_c2_first) if complete else None,
                "logit_sign_flip_after_swap": (
                    _sign(z_c1_first) != _sign(z_c2_first) if complete else None
                ),
                "semantic_choice_c1_first": choice_1,
                "semantic_choice_c2_first": choice_2,
                "semantic_choice_flip_after_swap": (
                    choice_1 != choice_2 if choice_1 is not None and choice_2 is not None else None
                ),
                "balanced_logit_choice": _choice_from_z(z_balanced),
                "balanced_logit_correct": _correct_from_z(z_balanced, q_true),
                "q_llm": q_llm,
                "calibration_error": calibration_error,
                "brier_score": brier,
                "log_loss": log_loss,
                "reasoning_length_tokens": _mean_optional(
                    [c1_first["reasoning_length_tokens"], c2_first["reasoning_length_tokens"]]
                ),
                "reasoning_length_c1_first": c1_first["reasoning_length_tokens"],
                "reasoning_length_c2_first": c2_first["reasoning_length_tokens"],
                "strict_answer_compliance_c1_first": c1_first["strict_answer_compliance"],
                "strict_answer_compliance_c2_first": c2_first["strict_answer_compliance"],
                "actual_correct_c1_first": c1_first["canonical_posterior_correct"],
                "actual_correct_c2_first": c2_first["canonical_posterior_correct"],
                "pair_has_answer_boundary_logits": complete,
            }
        )
        balanced_rows.append(balanced)
    balanced_rows.sort(
        key=lambda row: (
            row["reliability"],
            bool(row["reasoning"]),
            row["question_set_index"],
            row["answer_pattern_index"],
        )
    )
    return balanced_rows


def make_reasoning_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair position-balanced reasoning-off/on rows."""
    groups: dict[tuple[Any, ...], dict[bool, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["reliability_exact"],
            row["question_set_index"],
            row["answer_pattern_index"],
        )
        reasoning = bool(row["reasoning"])
        if reasoning in groups[key]:
            raise ValueError(f"Duplicate reasoning={reasoning} row for {key}.")
        groups[key][reasoning] = row

    paired_rows: list[dict[str, Any]] = []
    for key, pair in groups.items():
        if set(pair) != {False, True}:
            raise ValueError(f"Reasoning pair {key} is incomplete.")
        off = pair[False]
        on = pair[True]
        z_off = off["z_balanced"]
        z_on = on["z_balanced"]
        q_true = off["q_true"]
        q_llm_off = off["q_llm"]
        q_llm_on = on["q_llm"]
        correct_off = off["balanced_logit_correct"]
        correct_on = on["balanced_logit_correct"]
        truth_probability_off = _candidate_probability_assigned_to_truth(q_llm_off, q_true)
        truth_probability_on = _candidate_probability_assigned_to_truth(q_llm_on, q_true)
        absolute_calibration_error_off = (
            abs(q_llm_off - q_true) if q_llm_off is not None and q_true is not None else None
        )
        absolute_calibration_error_on = (
            abs(q_llm_on - q_true) if q_llm_on is not None and q_true is not None else None
        )
        if correct_off is False and q_llm_off is not None:
            off_wrong_confidence = max(q_llm_off, 1.0 - q_llm_off)
        else:
            off_wrong_confidence = None
        paired_rows.append(
            {
                "model": off["model"],
                "reliability": off["reliability"],
                "reliability_exact": off["reliability_exact"],
                "question_set_index": off["question_set_index"],
                "answer_pattern_index": off["answer_pattern_index"],
                "answer_pattern": off["answer_pattern"],
                "candidate_1": off["candidate_1"],
                "candidate_2": off["candidate_2"],
                "question_subsets_json": off["question_subsets_json"],
                "observed_reports_json": off["observed_reports_json"],
                "total_agreement_candidate_1": off["total_agreement_candidate_1"],
                "total_agreement_candidate_2": off["total_agreement_candidate_2"],
                "candidate_1_likelihood_total": off["candidate_1_likelihood_total"],
                "candidate_2_likelihood_total": off["candidate_2_likelihood_total"],
                "candidate_1_posterior": off["candidate_1_posterior"],
                "candidate_2_posterior": off["candidate_2_posterior"],
                "candidate_1_minus_candidate_2_log_posterior": off[
                    "candidate_1_minus_candidate_2_log_posterior"
                ],
                "total_marginalized_probability": off["total_marginalized_probability"],
                "q_true": q_true,
                "endpoint_category": off["endpoint_category"],
                "z_off": z_off,
                "z_on": z_on,
                "z_on_minus_off": (
                    z_on - z_off if z_on is not None and z_off is not None else None
                ),
                "q_llm_off": q_llm_off,
                "q_llm_on": q_llm_on,
                "probability_true_choice_off": truth_probability_off,
                "probability_true_choice_on": truth_probability_on,
                "delta_probability_true_choice": (
                    truth_probability_on - truth_probability_off
                    if truth_probability_on is not None and truth_probability_off is not None
                    else None
                ),
                "absolute_calibration_error_off": absolute_calibration_error_off,
                "absolute_calibration_error_on": absolute_calibration_error_on,
                "change_absolute_calibration_error": (
                    absolute_calibration_error_on - absolute_calibration_error_off
                    if absolute_calibration_error_on is not None
                    and absolute_calibration_error_off is not None
                    else None
                ),
                "off_wrong_confidence": off_wrong_confidence,
                "balanced_correct_off": correct_off,
                "balanced_correct_on": correct_on,
                "correctness_transition": _transition(correct_off, correct_on),
                "choice_sign_flip_reasoning": (
                    _sign(z_off) != _sign(z_on) if z_off is not None and z_on is not None else None
                ),
                "confidence_amplification": (
                    abs(z_on) - abs(z_off) if z_off is not None and z_on is not None else None
                ),
                "reasoning_length_tokens": on["reasoning_length_tokens"],
                "semantic_choice_c1_first_off": off["semantic_choice_c1_first"],
                "semantic_choice_c1_first_on": on["semantic_choice_c1_first"],
                "semantic_choice_c2_first_off": off["semantic_choice_c2_first"],
                "semantic_choice_c2_first_on": on["semantic_choice_c2_first"],
                "off_c1_first_row_id": off["source_c1_first_row_id"],
                "off_c2_first_row_id": off["source_c2_first_row_id"],
                "on_c1_first_row_id": on["source_c1_first_row_id"],
                "on_c2_first_row_id": on["source_c2_first_row_id"],
            }
        )
    paired_rows.sort(
        key=lambda row: (
            row["reliability"],
            row["question_set_index"],
            row["answer_pattern_index"],
        )
    )
    return paired_rows


class SQLResult(Sequence[dict[str, Any]]):
    """Small notebook-friendly wrapper around SQL result dictionaries."""

    def __init__(self, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        self.columns = tuple(columns)
        self._rows = [dict(zip(self.columns, row, strict=True)) for row in rows]

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    def __repr__(self) -> str:
        preview = self._rows[:8]
        suffix = "" if len(self) <= 8 else f"\n... {len(self) - 8} more rows"
        return f"SQLResult({preview!r}){suffix}"

    def _repr_html_(self) -> str:
        max_rows = 50
        headers = "".join(f"<th>{html.escape(column)}</th>" for column in self.columns)
        body = []
        for row in self._rows[:max_rows]:
            cells = "".join(
                f"<td style='max-width:420px;white-space:pre-wrap'>{html.escape(str(row[column]))}</td>"
                for column in self.columns
            )
            body.append(f"<tr>{cells}</tr>")
        note = ""
        if len(self) > max_rows:
            note = f"<p>Showing {max_rows} of {len(self)} rows.</p>"
        return (
            "<div style='overflow-x:auto'>"
            "<table><thead><tr>"
            + headers
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div>"
            + note
        )

    def head(self, count: int = 5) -> SQLResult:
        rows = [[row[column] for column in self.columns] for row in self._rows[:count]]
        return SQLResult(self.columns, rows)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


@dataclass(frozen=True)
class Query:
    """Immutable Pythonic builder for SELECT/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT."""

    explorer: ExperimentExplorer
    table: str
    selections: tuple[str, ...] = ("*",)
    predicates: tuple[str, ...] = ()
    parameters: tuple[Any, ...] = ()
    grouping: tuple[str, ...] = ()
    having_predicates: tuple[str, ...] = ()
    ordering: tuple[str, ...] = ()
    row_limit: int | None = None

    def select(self, *expressions: str) -> Query:
        return replace(self, selections=tuple(expressions) or ("*",))

    def where(self, clause: str, *parameters: Any) -> Query:
        return replace(
            self,
            predicates=(*self.predicates, clause),
            parameters=(*self.parameters, *parameters),
        )

    def where_eq(self, **values: Any) -> Query:
        query = self
        for column, value in values.items():
            if not _IDENTIFIER.fullmatch(column):
                raise ValueError(f"Unsafe column name: {column!r}")
            query = query.where(f'"{column}" = ?', value)
        return query

    def group_by(self, *expressions: str) -> Query:
        return replace(self, grouping=tuple(expressions))

    def having(self, clause: str, *parameters: Any) -> Query:
        return replace(
            self,
            having_predicates=(*self.having_predicates, clause),
            parameters=(*self.parameters, *parameters),
        )

    def order_by(self, *expressions: str) -> Query:
        return replace(self, ordering=tuple(expressions))

    def sort(self, *expressions: str) -> Query:
        return self.order_by(*expressions)

    def limit(self, count: int) -> Query:
        if count < 0:
            raise ValueError("LIMIT must be non-negative.")
        return replace(self, row_limit=count)

    def sql_text(self) -> str:
        parts = [f'SELECT {", ".join(self.selections)} FROM "{self.table}"']
        if self.predicates:
            parts.append("WHERE " + " AND ".join(f"({value})" for value in self.predicates))
        if self.grouping:
            parts.append("GROUP BY " + ", ".join(self.grouping))
        if self.having_predicates:
            parts.append("HAVING " + " AND ".join(f"({value})" for value in self.having_predicates))
        if self.ordering:
            parts.append("ORDER BY " + ", ".join(self.ordering))
        if self.row_limit is not None:
            parts.append(f"LIMIT {self.row_limit}")
        return "\n".join(parts)

    def run(self) -> SQLResult:
        return self.explorer.sql(self.sql_text(), self.parameters)

    def count(self) -> int:
        count_query = replace(
            self,
            selections=("COUNT(*) AS n",),
            ordering=(),
            row_limit=None,
        )
        return int(count_query.run()[0]["n"])


class ExperimentExplorer:
    """Raw record access plus optional paired SQLite views of experiment artifacts."""

    def __init__(
        self,
        results_path: Path,
        *,
        build_position_pairs: bool = True,
        build_reasoning_pairs: bool = True,
    ) -> None:
        if build_reasoning_pairs and not build_position_pairs:
            raise ValueError("Reasoning pairs require position-balanced rows.")
        self.results_path = results_path.resolve(strict=True)
        self.source_rows = load_jsonl_readonly(self.results_path)
        if not self.source_rows:
            raise ValueError(f"No rows found in {self.results_path}.")
        row_ids = [str(row["row_id"]) for row in self.source_rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("Source JSONL contains duplicate row IDs.")
        self._raw_by_id = dict(zip(row_ids, self.source_rows, strict=True))
        self.transcript_rows = [flatten_transcript_row(row) for row in self.source_rows]
        self.balanced_rows = (
            make_balanced_rows(self.transcript_rows) if build_position_pairs else []
        )
        self.reasoning_pairs = (
            make_reasoning_pairs(self.balanced_rows) if build_reasoning_pairs else []
        )
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self._register_table("transcript_rows", self.transcript_rows)
        table_names = ["transcript_rows"]
        if self.balanced_rows:
            self._register_table("balanced_rows", self.balanced_rows)
            table_names.append("balanced_rows")
        if self.reasoning_pairs:
            self._register_table("reasoning_pairs", self.reasoning_pairs)
            table_names.append("reasoning_pairs")
        self._table_names = tuple(table_names)
        # Defense in depth: even the derived in-memory database rejects writes after setup.
        self.connection.execute("PRAGMA query_only = ON")

    def _register_table(self, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not _IDENTIFIER.fullmatch(name) or not rows:
            raise ValueError(f"Invalid or empty table: {name!r}")
        columns = list(rows[0])
        if any(set(row) != set(columns) for row in rows):
            raise ValueError(f"Rows for {name} do not have a consistent schema.")
        declarations = []
        for column in columns:
            values = [row[column] for row in rows if row[column] is not None]
            if values and all(isinstance(value, (bool, int)) for value in values):
                kind = "INTEGER"
            elif values and all(isinstance(value, (bool, int, float)) for value in values):
                kind = "REAL"
            else:
                kind = "TEXT"
            declarations.append(f'"{column}" {kind}')
        self.connection.execute(f'CREATE TABLE "{name}" ({", ".join(declarations)})')
        placeholders = ", ".join("?" for _ in columns)
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        values = [
            tuple(
                int(value) if isinstance(value, bool) else value
                for value in (row[c] for c in columns)
            )
            for row in rows
        ]
        self.connection.executemany(
            f'INSERT INTO "{name}" ({quoted_columns}) VALUES ({placeholders})', values
        )
        self.connection.commit()

    def table_names(self) -> tuple[str, ...]:
        return self._table_names

    def columns(self, table: str) -> SQLResult:
        if table not in self.table_names():
            raise KeyError(table)
        return self.sql(f'PRAGMA table_info("{table}")')

    def query(self, table: str = "transcript_rows") -> Query:
        if table not in self.table_names():
            raise KeyError(table)
        return Query(self, table)

    def sql(self, statement: str, parameters: Sequence[Any] = ()) -> SQLResult:
        """Run one read-only SQL statement against the in-memory derived tables."""
        first_word = statement.lstrip().split(maxsplit=1)[0].upper() if statement.strip() else ""
        if first_word not in {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}:
            raise ValueError("Only read-only SELECT/WITH/PRAGMA/EXPLAIN statements are allowed.")
        cursor = self.connection.execute(statement, tuple(parameters))
        columns = [item[0] for item in cursor.description or ()]
        return SQLResult(columns, cursor.fetchall())

    def raw_row(self, row_id: str) -> dict[str, Any]:
        """Return the untouched source JSON object for prompt/completion inspection."""
        return self._raw_by_id[row_id]

    def summary(self) -> dict[str, Any]:
        return {
            "source": str(self.results_path),
            "artifact_mode": "read-only source; in-memory SQLite derivatives",
            "transcript_rows": len(self.transcript_rows),
            "balanced_rows": len(self.balanced_rows),
            "reasoning_pairs": len(self.reasoning_pairs),
            "models": sorted({row["model"] for row in self.transcript_rows}),
            "reliabilities": sorted({row["reliability_exact"] for row in self.transcript_rows}),
            "missing_answer_boundary_logits": sum(
                not row["answer_boundary_available"] for row in self.transcript_rows
            ),
        }


def load_explorer(results_path: Path | None = None) -> ExperimentExplorer:
    """Convenience constructor used by the exploration notebook."""
    return ExperimentExplorer(results_path or default_results_path())


def load_transcript_explorer(results_path: Path) -> ExperimentExplorer:
    """Load a run that has no position or reasoning pairs."""
    return ExperimentExplorer(
        results_path,
        build_position_pairs=False,
        build_reasoning_pairs=False,
    )
