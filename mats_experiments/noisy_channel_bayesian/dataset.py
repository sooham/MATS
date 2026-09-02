"""Dataset container, persistence, and aggregate reporting."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, overload

from .core import (
    SCHEMA_VERSION,
    CandidateEvidenceBayesianEnvironment,
    CandidateEvidenceQuestion,
    FixedSubsetQuestion,
    NoisyChannelBayesianEnvironment,
    RandomSubsetQuestion,
    SystemPrompt,
    TokenizerBinding,
    XVsYPosteriorProbe,
    answer_patterns,
    build_candidate_evidence_row,
    build_row,
    stable_row_id,
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class TranscriptDataset:
    """A small table-like wrapper whose individual rows are plain dictionaries."""

    def __init__(
        self,
        rows: Iterable[dict[str, object]],
        *,
        manifest: dict[str, object] | None = None,
        experiment_dir: Path | None = None,
    ) -> None:
        self._rows = list(rows)
        self.manifest = dict(manifest or {})
        self.experiment_dir = experiment_dir
        row_ids = [row.get("row_id") for row in self._rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("row_id values must be unique.")

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self._rows)

    @overload
    def __getitem__(self, key: int | slice) -> dict[str, object] | list[dict[str, object]]: ...

    @overload
    def __getitem__(self, key: str) -> list[object]: ...

    @overload
    def __getitem__(self, key: Sequence[str]) -> list[dict[str, object]]: ...

    def __getitem__(self, key: int | slice | str | Sequence[str]) -> object:
        if isinstance(key, (int, slice)):
            return self._rows[key]
        if isinstance(key, str):
            return [row.get(key) for row in self._rows]
        return [{column: row.get(column) for column in key} for row in self._rows]

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys(key for row in self._rows for key in row))

    def head(self, n: int = 5) -> list[dict[str, object]]:
        if n < 0:
            raise ValueError("n must be non-negative.")
        return self._rows[:n]

    def save(self, experiment_dir: str | Path) -> Path:
        directory = Path(experiment_dir)
        directory.mkdir(parents=True, exist_ok=True)
        dataset_path = directory / "dataset.jsonl"
        contents = "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in self._rows
        )
        _atomic_write_text(dataset_path, contents)
        manifest = {
            **self.manifest,
            "schema_version": SCHEMA_VERSION,
            "row_count": len(self),
            "columns": self.columns,
            "dataset_file": "dataset.jsonl",
        }
        _atomic_write_text(
            directory / "dataset_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
        self.manifest = manifest
        self.experiment_dir = directory
        return dataset_path

    @classmethod
    def load(cls, experiment_dir: str | Path) -> TranscriptDataset:
        directory = Path(experiment_dir)
        manifest_path = directory / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version {manifest.get('schema_version')!r}.")
        rows = [
            json.loads(line)
            for line in (directory / manifest.get("dataset_file", "dataset.jsonl"))
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if manifest.get("row_count") != len(rows):
            raise ValueError("Dataset manifest row count does not match dataset.jsonl.")
        return cls(rows, manifest=manifest, experiment_dir=directory)

    def execute(self, runner: Any, execution_config: Any) -> TranscriptDataset:
        return runner.execute(self, execution_config)

    def summarize(self) -> dict[str, object]:
        """Compute uniform-history and natural-distribution accuracy aggregates."""

        defined = [row for row in self if row.get("posterior_state") == "defined"]
        eligible = [row for row in defined if row.get("ground_truth_choice") is not None]
        scored = [row for row in eligible if row.get("posterior_correct") is not None]
        uniform = (
            sum(bool(row["posterior_correct"]) for row in scored) / len(scored) if scored else None
        )
        set_indices = sorted({int(row["question_set_index"]) for row in self})
        per_set: list[float] = []
        for index in set_indices:
            set_rows = [row for row in scored if row["question_set_index"] == index]
            denominator = sum(float(row["prior_predictive"]) for row in set_rows)
            if denominator:
                numerator = sum(
                    float(row["prior_predictive"]) * bool(row["posterior_correct"])
                    for row in set_rows
                )
                per_set.append(numerator / denominator)
        natural = sum(per_set) / len(per_set) if per_set else None
        undefined = [row for row in self if row.get("posterior_state") != "defined"]
        forced_ties = [
            row
            for row in defined
            if row.get("normative_comparison") == "SAME" and row.get("ground_truth_choice") is None
        ]

        def average_set_mass(rows: Sequence[dict[str, object]]) -> float:
            if not set_indices:
                return 0.0
            return sum(
                sum(
                    float(row["prior_predictive"])
                    for row in rows
                    if row["question_set_index"] == index
                )
                for index in set_indices
            ) / len(set_indices)

        compliance_rows = [row for row in self if row.get("parse_compliance") is not None]
        uniform_compliance = (
            sum(bool(row["parse_compliance"]) for row in compliance_rows) / len(compliance_rows)
            if compliance_rows
            else None
        )
        compliance_per_set: list[float] = []
        for index in set_indices:
            set_rows = [row for row in compliance_rows if row["question_set_index"] == index]
            denominator = sum(float(row["prior_predictive"]) for row in set_rows)
            if denominator:
                compliance_per_set.append(
                    sum(
                        float(row["prior_predictive"]) * bool(row["parse_compliance"])
                        for row in set_rows
                    )
                    / denominator
                )

        return {
            "row_count": len(self),
            "defined_row_count": len(defined),
            "undefined_row_count": len(undefined),
            "undefined_uniform_fraction": len(undefined) / len(self) if self else 0.0,
            "undefined_natural_mass": average_set_mass(undefined),
            "forced_tie_count": len(forced_ties),
            "forced_tie_uniform_fraction": len(forced_ties) / len(defined) if defined else 0.0,
            "forced_tie_natural_mass": average_set_mass(forced_ties),
            "accuracy_eligible_count": len(eligible),
            "accuracy_scored_count": len(scored),
            "uniform_defined_history_accuracy": uniform,
            "natural_distribution_accuracy": natural,
            "natural_distribution_question_set_count": len(per_set),
            "uniform_history_parse_compliance": uniform_compliance,
            "natural_distribution_parse_compliance": (
                sum(compliance_per_set) / len(compliance_per_set) if compliance_per_set else None
            ),
        }


@dataclass(frozen=True)
class TranscriptDatasetGenerator:
    environment: NoisyChannelBayesianEnvironment
    question: RandomSubsetQuestion | FixedSubsetQuestion
    probe: XVsYPosteriorProbe
    tokenizer_binding: TokenizerBinding
    system_prompt: SystemPrompt = field(default_factory=SystemPrompt)
    seed: int = 0

    def generate(self, *, num_question_sets: int) -> TranscriptDataset:
        if num_question_sets < 1:
            raise ValueError("num_question_sets must be positive.")
        if isinstance(self.question, FixedSubsetQuestion) and num_question_sets != 1:
            raise ValueError("FixedSubsetQuestion requires num_question_sets=1.")
        self.probe.validate(self.environment.n)
        rng = random.Random(self.seed)
        rows: list[dict[str, object]] = []
        schedule_fingerprints: list[list[list[int]]] = []
        for question_set_index in range(num_question_sets):
            questions = self.question.sample(rng=rng, n=self.environment.n, k=self.environment.k)
            schedule_fingerprints.append(
                [list(question["membership_set"]) for question in questions]  # type: ignore[arg-type]
            )
            for pattern_index, reports in enumerate(answer_patterns(self.environment.k)):
                rows.append(
                    build_row(
                        environment=self.environment,
                        questions=questions,
                        question_set_index=question_set_index,
                        reports=reports,
                        answer_pattern_index=pattern_index,
                        probe=self.probe,
                        system_prompt=self.system_prompt,
                        tokenizer_binding=self.tokenizer_binding,
                    )
                )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generator_seed": self.seed,
            "num_question_sets": num_question_sets,
            "patterns_per_question_set": 2**self.environment.k,
            "n": self.environment.n,
            "k": self.environment.k,
            "reliabilities_exact": [str(value) for value in self.environment.reliabilities],
            "question_type": type(self.question).__name__,
            "probe": {
                "x": self.probe.x,
                "y": self.probe.y,
                "allow_same": self.probe.allow_same,
                "reasoning_budget": self.probe.reasoning_budget,
                "call_layout": self.probe.call_layout,
            },
            "system_prompt": self.system_prompt.content,
            "tokenizer_template_fingerprint": self.tokenizer_binding.fingerprint,
            "question_schedules": schedule_fingerprints,
        }
        return TranscriptDataset(rows, manifest=manifest)


@dataclass(frozen=True)
class CandidateEvidenceDatasetGenerator:
    """Project a raw transcript dataset into a strictly paired reduced control."""

    environment: CandidateEvidenceBayesianEnvironment
    question: CandidateEvidenceQuestion
    probe: XVsYPosteriorProbe
    tokenizer_binding: TokenizerBinding
    system_prompt: SystemPrompt = field(default_factory=SystemPrompt)

    def generate(self, *, source_dataset: TranscriptDataset) -> TranscriptDataset:
        if not source_dataset:
            raise ValueError("source_dataset must not be empty.")
        if self.probe.reasoning_budget != 0:
            raise ValueError("The candidate-evidence control requires reasoning_budget=0.")
        source_system_prompt = source_dataset.manifest.get("system_prompt")
        if source_system_prompt != self.system_prompt.content:
            raise ValueError("Reduced and raw datasets must use the same system prompt.")
        source_tokenizer = source_dataset.manifest.get("tokenizer_template_fingerprint")
        if source_tokenizer != self.tokenizer_binding.fingerprint:
            raise ValueError("Reduced and raw datasets must use the same tokenizer binding.")
        rows = [
            build_candidate_evidence_row(
                source_row=source_row,
                environment=self.environment,
                question=self.question,
                probe=self.probe,
                system_prompt=self.system_prompt,
                tokenizer_binding=self.tokenizer_binding,
            )
            for source_row in source_dataset
        ]
        source_ids = [str(row["row_id"]) for row in source_dataset]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "representation": "candidate_evidence",
            "environment_type": type(self.environment).__name__,
            "question_type": type(self.question).__name__,
            "source_schema_version": source_dataset.manifest.get("schema_version"),
            "source_row_count": len(source_dataset),
            "source_dataset_fingerprint": stable_row_id({"row_ids": source_ids}),
            "num_question_sets": source_dataset.manifest.get("num_question_sets"),
            "patterns_per_question_set": source_dataset.manifest.get("patterns_per_question_set"),
            "n": self.environment.n,
            "k": self.environment.k,
            "reliabilities_exact": [str(value) for value in self.environment.reliabilities],
            "probe": {
                "x": self.probe.x,
                "y": self.probe.y,
                "allow_same": self.probe.allow_same,
                "reasoning_budget": self.probe.reasoning_budget,
                "call_layout": self.probe.call_layout,
            },
            "question": {
                "agreement_surface": self.question.agreement_surface,
                "disagreement_surface": self.question.disagreement_surface,
                "reliability_format": self.question.reliability_format,
                "layout": self.question.layout,
            },
            "system_prompt": self.system_prompt.content,
            "tokenizer_template_fingerprint": self.tokenizer_binding.fingerprint,
        }
        return TranscriptDataset(rows, manifest=manifest)


def summarize_representation_control(
    raw_results: TranscriptDataset, reduced_results: TranscriptDataset
) -> dict[str, object]:
    """Compute strictly paired raw-versus-reduced behavioral summaries."""

    raw_by_id = {str(row["row_id"]): row for row in raw_results}
    reduced_by_source: dict[str, dict[str, object]] = {}
    for row in reduced_results:
        source_id = str(row.get("source_row_id"))
        if source_id in reduced_by_source:
            raise ValueError(f"Duplicate reduced source_row_id {source_id!r}.")
        reduced_by_source[source_id] = row
    if set(raw_by_id) != set(reduced_by_source):
        missing_reduced = sorted(set(raw_by_id) - set(reduced_by_source))
        missing_raw = sorted(set(reduced_by_source) - set(raw_by_id))
        raise ValueError(
            "Raw and reduced result IDs do not form complete pairs: "
            f"missing_reduced={missing_reduced}, missing_raw={missing_raw}."
        )
    pairs = [(raw_by_id[row_id], reduced_by_source[row_id]) for row_id in raw_by_id]
    exact_keys = (
        "prior_predictive_exact",
        "posterior_state",
        "posterior_exact",
        "x_posterior_exact",
        "y_posterior_exact",
        "ground_truth_choice",
        "normative_comparison",
        "x",
        "y",
        "allow_same",
        "question_set_index",
    )
    for raw, reduced in pairs:
        for key in exact_keys:
            if raw.get(key) != reduced.get(key):
                raise ValueError(f"Paired field {key!r} does not match exactly.")
        if raw.get("posterior_correct") is None and raw.get("ground_truth_choice") is not None:
            raise ValueError("Raw result is missing posterior_correct for an eligible row.")
        if (
            reduced.get("posterior_correct") is None
            and reduced.get("ground_truth_choice") is not None
        ):
            raise ValueError("Reduced result is missing posterior_correct for an eligible row.")

    eligible = [pair for pair in pairs if pair[0].get("ground_truth_choice") is not None]

    def uniform_accuracy(index: int) -> float | None:
        if not eligible:
            return None
        return sum(bool(pair[index]["posterior_correct"]) for pair in eligible) / len(eligible)

    question_sets = sorted({int(raw["question_set_index"]) for raw, _ in pairs})

    def natural_average(key: str, index: int, *, eligible_only: bool) -> float | None:
        values: list[float] = []
        selected_pairs = eligible if eligible_only else pairs
        for question_set in question_sets:
            set_pairs = [
                pair
                for pair in selected_pairs
                if int(pair[0]["question_set_index"]) == question_set
            ]
            denominator = sum(float(pair[0]["prior_predictive"]) for pair in set_pairs)
            if denominator:
                values.append(
                    sum(
                        float(pair[0]["prior_predictive"]) * bool(pair[index].get(key))
                        for pair in set_pairs
                    )
                    / denominator
                )
        return sum(values) / len(values) if values else None

    raw_uniform = uniform_accuracy(0)
    reduced_uniform = uniform_accuracy(1)
    raw_natural = natural_average("posterior_correct", 0, eligible_only=True)
    reduced_natural = natural_average("posterior_correct", 1, eligible_only=True)
    transitions = {
        "both_correct": 0,
        "raw_only_correct": 0,
        "reduced_only_correct": 0,
        "neither_correct": 0,
    }
    for raw, reduced in eligible:
        raw_correct = bool(raw["posterior_correct"])
        reduced_correct = bool(reduced["posterior_correct"])
        if raw_correct and reduced_correct:
            transitions["both_correct"] += 1
        elif raw_correct:
            transitions["raw_only_correct"] += 1
        elif reduced_correct:
            transitions["reduced_only_correct"] += 1
        else:
            transitions["neither_correct"] += 1
    raw_failures = transitions["reduced_only_correct"] + transitions["neither_correct"]
    raw_successes = transitions["both_correct"] + transitions["raw_only_correct"]
    undefined = sum(raw.get("posterior_state") != "defined" for raw, _ in pairs)
    forced_ties = sum(
        raw.get("normative_comparison") == "SAME" and raw.get("ground_truth_choice") is None
        for raw, _ in pairs
    )
    return {
        "pair_count": len(pairs),
        "eligible_pair_count": len(eligible),
        "undefined_pair_count": undefined,
        "forced_tie_pair_count": forced_ties,
        "uniform": {
            "raw_accuracy": raw_uniform,
            "reduced_accuracy": reduced_uniform,
            "accuracy_delta": (
                reduced_uniform - raw_uniform
                if reduced_uniform is not None and raw_uniform is not None
                else None
            ),
            "raw_parse_compliance": (
                sum(bool(raw.get("parse_compliance")) for raw, _ in pairs) / len(pairs)
                if pairs
                else None
            ),
            "reduced_parse_compliance": (
                sum(bool(reduced.get("parse_compliance")) for _, reduced in pairs) / len(pairs)
                if pairs
                else None
            ),
        },
        "natural_distribution": {
            "raw_accuracy": raw_natural,
            "reduced_accuracy": reduced_natural,
            "accuracy_delta": (
                reduced_natural - raw_natural
                if reduced_natural is not None and raw_natural is not None
                else None
            ),
            "raw_parse_compliance": natural_average("parse_compliance", 0, eligible_only=False),
            "reduced_parse_compliance": natural_average("parse_compliance", 1, eligible_only=False),
        },
        "transitions": transitions,
        "rescue_rate": (
            transitions["reduced_only_correct"] / raw_failures if raw_failures else None
        ),
        "regression_rate": (
            transitions["raw_only_correct"] / raw_successes if raw_successes else None
        ),
    }


def exact_pattern_mass(dataset: TranscriptDataset, question_set_index: int) -> Fraction:
    """Convenience used by notebooks/tests to verify exact normalization."""

    return sum(
        (
            Fraction(str(row["prior_predictive_exact"]))
            for row in dataset
            if row["question_set_index"] == question_set_index
        ),
        Fraction(0),
    )
