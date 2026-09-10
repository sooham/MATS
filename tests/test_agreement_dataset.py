import json
import random
from collections import Counter

import pytest

from mats_experiments.noisy_channel_bayesian import (
    AgreementSubsetQuestion,
    AgreementTranscriptDatasetGenerator,
    NoisyChannelBayesianEnvironment,
    SystemPrompt,
    TokenizerBinding,
    TranscriptDataset,
    XVsYPosteriorProbe,
    candidate_agreements,
)


class FakeTokenizer:
    name_or_path = "fake-tokenizer"
    chat_template = "fake-chat-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        rendered = json.dumps(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            },
            sort_keys=True,
        )
        return list(rendered.encode()) if tokenize else rendered


def test_agreement_subset_question_realizes_requested_vectors() -> None:
    question = AgreementSubsetQuestion(subset_size=4, sort=True)
    x_pattern = (0, 0, 1, 1)
    y_pattern = (0, 1, 0, 1)

    questions, reports = question.sample(
        rng=random.Random(17),
        n=8,
        k=4,
        x=2,
        y=7,
        x_agreements=x_pattern,
        y_agreements=y_pattern,
    )

    membership_sets = [item["membership_set"] for item in questions]
    assert all(len(values) == 4 for values in membership_sets)
    assert all(values == sorted(values) for values in membership_sets)
    assert candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=2
    ) == list(x_pattern)
    assert candidate_agreements(
        membership_sets=membership_sets, reports=reports, candidate=7
    ) == list(y_pattern)


def test_agreement_generator_covers_grid_uniformly_and_round_trips(tmp_path) -> None:
    environments = tuple(
        NoisyChannelBayesianEnvironment(
            n=8,
            k=3,
            r_values=reliability,
            control_positional_bias=True,
        )
        for reliability in (0.3, 0.7)
    )
    probes = tuple(
        XVsYPosteriorProbe(x=2, y=7, reasoning=reasoning)
        for reasoning in (False, True)
    )
    binding = TokenizerBinding(FakeTokenizer(), enable_thinking=False)
    dataset = AgreementTranscriptDatasetGenerator(
        environment=environments,
        question=AgreementSubsetQuestion(subset_size=4),
        probe=probes,
        tokenizer_binding=binding,
        system_prompt=SystemPrompt("Test system prompt."),
        seed=123,
    ).generate(num_question_sets=2)

    assert len(dataset) == 2 * 2 * 8 * 8 * 2 * 2
    assert dataset.manifest["generator_type"] == "AgreementTranscriptDatasetGenerator"
    assert dataset.manifest["sampling_design"] == "agreement_cartesian_product"
    assert dataset.manifest["agreement_patterns_per_candidate"] == 8
    assert dataset.manifest["agreement_cell_count"] == 64
    assert dataset.manifest["num_question_sets_per_agreement_cell"] == 2
    assert dataset.manifest["num_question_sets"] == 128
    assert dataset.manifest["patterns_per_question_set"] == 1
    assert dataset.manifest["rows_per_question_set"] == 8
    assert dataset.manifest["rows_per_agreement_cell"] == 16

    cell_counts = Counter(
        (
            int(row["environment_parameter_index"]),
            int(row["probe_parameter_index"]),
            str(row["presentation_order"]),
            str(row["agreement_x_pattern"]),
            str(row["agreement_y_pattern"]),
        )
        for row in dataset
    )
    assert len(cell_counts) == 2 * 2 * 2 * 64
    assert set(cell_counts.values()) == {2}

    for row in dataset:
        assert row["candidate_1"] == row["agreement_target_x"] == 2
        assert row["candidate_2"] == row["agreement_target_y"] == 7
        assert row["agreement_candidate_1_by_question"] == row[
            "target_agreement_x_by_question"
        ]
        assert row["agreement_candidate_2_by_question"] == row[
            "target_agreement_y_by_question"
        ]
        canonical_vectors = {
            2: row["agreement_candidate_1_by_question"],
            7: row["agreement_candidate_2_by_question"],
        }
        first_value, second_value = row["candidate_value_order"]
        assert row["agreement_x_by_question"] == canonical_vectors[first_value]
        assert row["agreement_y_by_question"] == canonical_vectors[second_value]

    scenario_by_environment: dict[int, dict[int, tuple[object, object]]] = {}
    for row in dataset:
        if row["probe_parameter_index"] != 0 or row["presentation_index"] != 0:
            continue
        scenario_by_environment.setdefault(int(row["environment_parameter_index"]), {})[
            int(row["question_set_index"])
        ] = (row["membership_sets"], row["observed_reports"])
    assert scenario_by_environment[0] == scenario_by_environment[1]

    dataset.save(tmp_path)
    loaded = TranscriptDataset.load(tmp_path)
    assert list(loaded) == list(dataset)
    assert loaded.manifest == dataset.manifest


def test_agreement_generator_requires_one_fixed_candidate_pair() -> None:
    binding = TokenizerBinding(FakeTokenizer())
    generator = AgreementTranscriptDatasetGenerator(
        environment=NoisyChannelBayesianEnvironment(n=8, k=3),
        question=AgreementSubsetQuestion(),
        probe=(
            XVsYPosteriorProbe(x=2, y=7, reasoning=False),
            XVsYPosteriorProbe(x=3, y=7, reasoning=True),
        ),
        tokenizer_binding=binding,
    )

    with pytest.raises(ValueError, match="same ordered X/Y candidate pair"):
        generator.generate(num_question_sets=1)


def test_agreement_question_rejects_an_infeasible_subset_size() -> None:
    with pytest.raises(ValueError, match="cannot realize every"):
        AgreementSubsetQuestion(subset_size=1).validate(n=2, x=1, y=2)
