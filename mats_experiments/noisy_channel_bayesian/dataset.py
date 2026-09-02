"""Dataset container, persistence, and aggregate reporting."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
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

        def natural_weight(row: dict[str, object]) -> float:
            presentations = int(row.get("presentations_per_scenario", 1))
            if presentations < 1:
                raise ValueError("presentations_per_scenario must be positive.")
            return float(row["prior_predictive"]) / presentations

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
            denominator = sum(natural_weight(row) for row in set_rows)
            if denominator:
                numerator = sum(
                    natural_weight(row) * bool(row["posterior_correct"])
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
                    natural_weight(row)
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
            denominator = sum(natural_weight(row) for row in set_rows)
            if denominator:
                compliance_per_set.append(
                    sum(
                        natural_weight(row) * bool(row["parse_compliance"])
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


RawQuestion = RandomSubsetQuestion | FixedSubsetQuestion


def _parameter_axis(
    value: object,
    *,
    expected_type: type | tuple[type, ...],
    name: str,
) -> tuple[Any, ...]:
    """Normalize a fixed component or a non-empty sequence of component values."""

    if isinstance(value, expected_type):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if not values:
            raise ValueError(f"The {name} parameter axis must not be empty.")
        if not all(isinstance(item, expected_type) for item in values):
            raise TypeError(f"Every {name} parameter value must have the expected type.")
        return values
    raise TypeError(f"{name} must be one value or a sequence of values.")


def _environment_manifest(environment: NoisyChannelBayesianEnvironment) -> dict[str, object]:
    return {
        "type": type(environment).__name__,
        "n": environment.n,
        "k": environment.k,
        "reliabilities_exact": [str(value) for value in environment.reliabilities],
        "control_positional_bias": environment.control_positional_bias,
    }


def _question_manifest(question: RawQuestion) -> dict[str, object]:
    if isinstance(question, RandomSubsetQuestion):
        return {
            "type": type(question).__name__,
            "subset_size": question.subset_size,
            "replacement": question.replacement,
            "sort": question.sort,
        }
    return {
        "type": type(question).__name__,
        "subsets": [list(subset) for subset in question.subsets],
    }


def _probe_manifest(probe: XVsYPosteriorProbe) -> dict[str, object]:
    return {
        "x": probe.x,
        "y": probe.y,
        "allow_same": probe.allow_same,
        "reasoning_budget": probe.reasoning_budget,
        "call_layout": probe.call_layout,
        "answer_prefix": probe.answer_prefix,
    }


def _common_or_none(values: Sequence[Any]) -> Any | None:
    return values[0] if values and all(value == values[0] for value in values[1:]) else None


@dataclass(frozen=True)
class TranscriptDatasetGenerator:
    """Generate the Cartesian product of environment, question, and probe axes.

    Each component may remain a single fixed instance (the backward-compatible
    path) or be a sequence. Random question schedules are sampled once per
    environment/question pair and reused for every probe value, making probe
    sweeps exactly paired.
    """

    environment: NoisyChannelBayesianEnvironment | Sequence[NoisyChannelBayesianEnvironment]
    question: RawQuestion | Sequence[RawQuestion]
    probe: XVsYPosteriorProbe | Sequence[XVsYPosteriorProbe]
    tokenizer_binding: TokenizerBinding
    system_prompt: SystemPrompt = field(default_factory=SystemPrompt)
    seed: int = 0

    def generate(self, *, num_question_sets: int) -> TranscriptDataset:
        if num_question_sets < 1:
            raise ValueError("num_question_sets must be positive.")
        environments = _parameter_axis(
            self.environment,
            expected_type=NoisyChannelBayesianEnvironment,
            name="environment",
        )
        questions = _parameter_axis(
            self.question,
            expected_type=(RandomSubsetQuestion, FixedSubsetQuestion),
            name="question",
        )
        probes = _parameter_axis(
            self.probe,
            expected_type=XVsYPosteriorProbe,
            name="probe",
        )

        rows: list[dict[str, object]] = []
        parameterizations: list[dict[str, object]] = []
        schedule_banks: list[dict[str, object]] = []
        parameterization_index = 0
        for environment_index, environment in enumerate(environments):
            for question_index, question in enumerate(questions):
                if isinstance(question, FixedSubsetQuestion) and num_question_sets != 1:
                    raise ValueError("FixedSubsetQuestion requires num_question_sets=1.")
                # Resetting from the same seed makes compatible question variants paired too.
                rng = random.Random(self.seed)
                sampled_question_sets = [
                    question.sample(rng=rng, n=environment.n, k=environment.k)
                    for _ in range(num_question_sets)
                ]
                schedule_banks.append(
                    {
                        "environment_parameter_index": environment_index,
                        "question_parameter_index": question_index,
                        "question_schedules": [
                            [
                                list(sampled_question["membership_set"])  # type: ignore[arg-type]
                                for sampled_question in sampled_questions
                            ]
                            for sampled_questions in sampled_question_sets
                        ],
                    }
                )
                for probe_index, probe in enumerate(probes):
                    probe.validate(environment.n)
                    presented_probes = [(0, probe)]
                    if environment.control_positional_bias:
                        presented_probes.append(
                            (1, replace(probe, x=probe.y, y=probe.x))
                        )
                    parameterizations.append(
                        {
                            "parameterization_index": parameterization_index,
                            "environment_parameter_index": environment_index,
                            "question_parameter_index": question_index,
                            "probe_parameter_index": probe_index,
                            "environment": _environment_manifest(environment),
                            "question": _question_manifest(question),
                            "probe": _probe_manifest(probe),
                            "row_count": (
                                num_question_sets
                                * 2**environment.k
                                * len(presented_probes)
                            ),
                        }
                    )
                    for question_set_index, sampled_questions in enumerate(
                        sampled_question_sets
                    ):
                        for pattern_index, reports in enumerate(
                            answer_patterns(environment.k)
                        ):
                            for presentation_index, presented_probe in presented_probes:
                                rows.append(
                                    build_row(
                                        environment=environment,
                                        questions=sampled_questions,
                                        question_set_index=question_set_index,
                                        reports=reports,
                                        answer_pattern_index=pattern_index,
                                        probe=presented_probe,
                                        canonical_probe=probe,
                                        presentation_index=presentation_index,
                                        parameterization_index=parameterization_index,
                                        environment_parameter_index=environment_index,
                                        question_parameter_index=question_index,
                                        probe_parameter_index=probe_index,
                                        system_prompt=self.system_prompt,
                                        tokenizer_binding=self.tokenizer_binding,
                                    )
                                )
                    parameterization_index += 1

        environment_manifests = [_environment_manifest(value) for value in environments]
        question_manifests = [_question_manifest(value) for value in questions]
        probe_manifests = [_probe_manifest(value) for value in probes]
        pattern_counts = [2**environment.k for environment in environments]
        presentation_counts = [
            2 if environment.control_positional_bias else 1
            for environment in environments
        ]
        rows_per_question_set = sum(
            2**environment.k
            * (2 if environment.control_positional_bias else 1)
            * len(probes)
            * len(questions)
            for environment in environments
        )
        single_schedule_bank = (
            schedule_banks[0]["question_schedules"] if len(schedule_banks) == 1 else None
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generator_seed": self.seed,
            "num_question_sets": num_question_sets,
            "parameterization_count": len(parameterizations),
            "environment_parameter_count": len(environments),
            "question_parameter_count": len(questions),
            "probe_parameter_count": len(probes),
            "patterns_per_question_set": _common_or_none(pattern_counts),
            "presentations_per_scenario": _common_or_none(presentation_counts),
            "rows_per_question_set": rows_per_question_set,
            "control_positional_bias": _common_or_none(
                [value.control_positional_bias for value in environments]
            ),
            "n": _common_or_none([value.n for value in environments]),
            "k": _common_or_none([value.k for value in environments]),
            "reliabilities_exact": _common_or_none(
                [list(item["reliabilities_exact"]) for item in environment_manifests]
            ),
            "question_type": _common_or_none(
                [str(item["type"]) for item in question_manifests]
            ),
            "probe": probe_manifests[0] if len(probe_manifests) == 1 else None,
            "environments": environment_manifests,
            "questions": question_manifests,
            "probes": probe_manifests,
            "reasoning_budgets": [probe.reasoning_budget for probe in probes],
            "parameterizations": parameterizations,
            "system_prompt": self.system_prompt.content,
            "tokenizer_template_fingerprint": self.tokenizer_binding.fingerprint,
            "question_schedules": single_schedule_bank,
            "question_schedule_banks": schedule_banks,
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
        source_control = bool(source_dataset.manifest.get("control_positional_bias", False))
        if source_control != self.environment.control_positional_bias:
            raise ValueError(
                "Reduced and raw environments must use the same positional-bias control."
            )
        rows = []
        for source_row in source_dataset:
            candidate_1 = int(source_row.get("candidate_1", source_row["x"]))
            candidate_2 = int(source_row.get("candidate_2", source_row["y"]))
            if (candidate_1, candidate_2) != (self.probe.x, self.probe.y):
                raise ValueError("Source canonical candidates do not match the reduced probe.")
            presented_probe = replace(
                self.probe, x=int(source_row["x"]), y=int(source_row["y"])
            )
            rows.append(
                build_candidate_evidence_row(
                    source_row=source_row,
                    environment=self.environment,
                    question=self.question,
                    probe=presented_probe,
                    system_prompt=self.system_prompt,
                    tokenizer_binding=self.tokenizer_binding,
                )
            )
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
            "presentations_per_scenario": source_dataset.manifest.get(
                "presentations_per_scenario", 1
            ),
            "rows_per_question_set": source_dataset.manifest.get("rows_per_question_set"),
            "control_positional_bias": self.environment.control_positional_bias,
            "n": self.environment.n,
            "k": self.environment.k,
            "reliabilities_exact": [str(value) for value in self.environment.reliabilities],
            "probe": {
                "x": self.probe.x,
                "y": self.probe.y,
                "allow_same": self.probe.allow_same,
                "reasoning_budget": self.probe.reasoning_budget,
                "call_layout": self.probe.call_layout,
                "answer_prefix": self.probe.answer_prefix,
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

    masses_by_pattern: dict[int, Fraction] = {}
    for row in dataset:
        if row["question_set_index"] != question_set_index:
            continue
        pattern_index = int(row["answer_pattern_index"])
        mass = Fraction(str(row["prior_predictive_exact"]))
        previous = masses_by_pattern.setdefault(pattern_index, mass)
        if previous != mass:
            raise ValueError("Presentation-order pair has inconsistent prior-predictive mass.")
    return sum(masses_by_pattern.values(), Fraction(0))
