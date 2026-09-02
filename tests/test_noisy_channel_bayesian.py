import json
import math
import random
from fractions import Fraction
from pathlib import Path

import pytest

from mats_experiments.noisy_channel_bayesian import (
    SOFTMAX_LOG_BASE,
    SOFTMAX_LOG_UNIT,
    CandidateEvidenceBayesianEnvironment,
    CandidateEvidenceDatasetGenerator,
    CandidateEvidenceQuestion,
    FixedSubsetQuestion,
    MetricSpec,
    NoisyChannelBayesianEnvironment,
    RandomSubsetQuestion,
    SystemPrompt,
    TokenizerBinding,
    TranscriptDataset,
    TranscriptDatasetGenerator,
    XVsYPosteriorProbe,
    enforce_reasoning,
    exact_bayesian_target,
    exact_pattern_mass,
    natural_log_ratio,
    resolve_selector,
    stage_two_messages,
    strip_thinking_markers,
    summarize_representation_control,
)


class TinyTokenizer:
    chat_template = "tiny-v1"
    name_or_path = "tiny"
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        del enable_thinking
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return self(rendered, add_special_tokens=False)["input_ids"] if tokenize else rendered

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        if isinstance(text, list):
            rows = [[ord(character) % 251 + 3 for character in item] for item in text]
            return {"input_ids": rows}
        return {"input_ids": [ord(character) % 251 + 3 for character in text]}


def generator(*, environment=None, question=None, probe=None, seed=17):
    return TranscriptDatasetGenerator(
        environment or NoisyChannelBayesianEnvironment(),
        question or RandomSubsetQuestion(),
        probe or XVsYPosteriorProbe(),
        TokenizerBinding(TinyTokenizer()),
        SystemPrompt("Be a careful Bayesian."),
        seed,
    )


def test_exhaustive_patterns_reproducibility_and_question_independence() -> None:
    first = generator().generate(num_question_sets=2)
    second = generator().generate(num_question_sets=2)
    assert len(first) == 16
    assert [row["row_id"] for row in first] == [row["row_id"] for row in second]
    expected = ["YYY", "YYN", "YNY", "YNN", "NYY", "NYN", "NNY", "NNN"]
    for question_set in range(2):
        rows = [row for row in first if row["question_set_index"] == question_set]
        assert [row["answer_pattern"] for row in rows] == expected
        assert len({tuple(map(tuple, row["membership_sets"])) for row in rows}) == 1
        assert exact_pattern_mass(first, question_set) == 1


def test_positional_control_pairs_each_evidence_scenario_in_both_orders() -> None:
    dataset = generator(
        environment=NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="3/4", control_positional_bias=True
        ),
        question=FixedSubsetQuestion([[1]]),
        probe=XVsYPosteriorProbe(x=1, y=2),
    ).generate(num_question_sets=1)

    assert len(dataset) == 4
    assert dataset.manifest["control_positional_bias"] is True
    assert dataset.manifest["presentations_per_scenario"] == 2
    assert [(row["answer_pattern"], row["x"], row["y"]) for row in dataset] == [
        ("Y", 1, 2),
        ("Y", 2, 1),
        ("N", 1, 2),
        ("N", 2, 1),
    ]
    for pattern_index in range(2):
        first, second = dataset[pattern_index * 2 : pattern_index * 2 + 2]
        assert first["positional_control_pair_id"] == second["positional_control_pair_id"]
        assert first["presentation_order"] == "C1_C2"
        assert second["presentation_order"] == "C2_C1"
        assert first["candidate_1"] == second["candidate_1"] == 1
        assert first["candidate_2"] == second["candidate_2"] == 2
        assert first["membership_sets"] == second["membership_sets"]
        assert first["observed_reports"] == second["observed_reports"]
        assert first["prior_predictive_exact"] == second["prior_predictive_exact"]
        assert first["posterior_exact"] == second["posterior_exact"]
        assert first["agreement_candidate_1_by_question"] == second[
            "agreement_candidate_1_by_question"
        ]
        assert first["agreement_candidate_2_by_question"] == second[
            "agreement_candidate_2_by_question"
        ]
        assert first["posterior_log_odds"] == pytest.approx(
            -float(second["posterior_log_odds"])
        )
        assert first["candidate_1_minus_candidate_2_log_odds"] == pytest.approx(
            float(second["candidate_1_minus_candidate_2_log_odds"])
        )
    assert exact_pattern_mass(dataset, 0) == 1

    with pytest.raises(TypeError, match="control_positional_bias"):
        NoisyChannelBayesianEnvironment(control_positional_bias=1)  # type: ignore[arg-type]


def test_generator_parameter_axes_form_a_cartesian_product() -> None:
    environments = (
        NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="3/4", control_positional_bias=True
        ),
        NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="2/3", control_positional_bias=True
        ),
    )
    questions = (FixedSubsetQuestion([[1]]), FixedSubsetQuestion([[2]]))
    probes = (
        XVsYPosteriorProbe(x=1, y=2, reasoning_budget=0),
        XVsYPosteriorProbe(x=1, y=2, reasoning_budget=10),
    )
    dataset = TranscriptDatasetGenerator(
        environment=environments,
        question=questions,
        probe=probes,
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
        system_prompt=SystemPrompt("Be a careful Bayesian."),
        seed=17,
    ).generate(num_question_sets=1)

    assert len(dataset) == 2 * 2 * 2 * 2 * 2
    assert dataset.manifest["parameterization_count"] == 8
    assert dataset.manifest["environment_parameter_count"] == 2
    assert dataset.manifest["question_parameter_count"] == 2
    assert dataset.manifest["probe_parameter_count"] == 2
    assert dataset.manifest["reasoning_budgets"] == [0, 10]
    assert dataset.manifest["rows_per_question_set"] == len(dataset)
    assert {
        (
            row["environment_parameter_index"],
            row["question_parameter_index"],
            row["probe_parameter_index"],
        )
        for row in dataset
    } == {(environment, question, probe) for environment in range(2)
          for question in range(2) for probe in range(2)}


def test_reasoning_probe_sweep_reuses_one_question_bank() -> None:
    budgets = tuple(range(0, 200, 10))
    dataset = TranscriptDatasetGenerator(
        environment=NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="3/4", control_positional_bias=True
        ),
        question=FixedSubsetQuestion([[1]]),
        probe=tuple(
            XVsYPosteriorProbe(x=1, y=2, reasoning_budget=budget)
            for budget in budgets
        ),
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
    ).generate(num_question_sets=1)

    assert len(budgets) == 20
    assert len(dataset) == 20 * 2 * 1 * 2
    assert dataset.manifest["reasoning_budgets"] == list(budgets)
    assert all(
        sum(row["reasoning_budget"] == budget for row in dataset) == 4
        for budget in budgets
    )
    assert len({
        tuple(tuple(values) for values in row["membership_sets"])
        for row in dataset
    }) == 1


def test_exact_scalar_and_per_question_reliabilities() -> None:
    evidence, posterior = exact_bayesian_target(
        domain=[1, 2],
        membership_sets=[[1]],
        reports=["YES"],
        reliabilities=[Fraction(3, 4)],
    )
    assert evidence == Fraction(1, 2)
    assert posterior == {1: Fraction(3, 4), 2: Fraction(1, 4)}

    evidence, posterior = exact_bayesian_target(
        domain=[1, 2],
        membership_sets=[[1], [2]],
        reports=["YES", "NO"],
        reliabilities=[Fraction(1, 4), Fraction(1)],
    )
    assert evidence == Fraction(1, 8)
    assert posterior == {1: Fraction(1), 2: Fraction(0)}


def test_zero_evidence_rows_are_retained_and_null(tmp_path: Path) -> None:
    dataset = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values=1),
        question=FixedSubsetQuestion([[1, 2]]),
        probe=XVsYPosteriorProbe(x=1, y=2),
    ).generate(num_question_sets=1)
    impossible = dataset[1]
    assert impossible["answer_pattern"] == "N"
    assert impossible["prior_predictive_exact"] == "0/1"
    assert impossible["posterior_state"] == "undefined_zero_evidence"
    assert impossible["posterior"] is None
    assert impossible["ground_truth_choice"] is None
    assert exact_pattern_mass(dataset, 0) == 1
    dataset.save(tmp_path)
    assert len(TranscriptDataset.load(tmp_path)) == 2


def test_valid_transcripts_persist_x_and_y_posteriors_exactly(tmp_path: Path) -> None:
    dataset = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values="3/4"),
        question=FixedSubsetQuestion([[1]]),
        probe=XVsYPosteriorProbe(x=1, y=2),
    ).generate(num_question_sets=1)
    observed_yes = dataset[0]
    assert observed_yes["posterior_state"] == "defined"
    assert observed_yes["x_posterior_exact"] == "3/4"
    assert observed_yes["y_posterior_exact"] == "1/4"
    assert observed_yes["x_posterior"] == pytest.approx(0.75)
    assert observed_yes["y_posterior"] == pytest.approx(0.25)
    assert observed_yes["posterior_log_odds"] == pytest.approx(math.log(3))
    assert observed_yes["posterior_log_odds_base"] == SOFTMAX_LOG_BASE == "e"
    assert observed_yes["posterior_log_odds_unit"] == SOFTMAX_LOG_UNIT == "nats"

    dataset.save(tmp_path)
    persisted = TranscriptDataset.load(tmp_path)[0]
    assert persisted["x_posterior_exact"] == "3/4"
    assert persisted["y_posterior_exact"] == "1/4"
    assert persisted["x_posterior"] == pytest.approx(0.75)
    assert persisted["y_posterior"] == pytest.approx(0.25)


def test_natural_log_ratio_matches_softmax_logit_scale() -> None:
    assert natural_log_ratio(Fraction(8), Fraction(1)) == pytest.approx(math.log(8))
    assert natural_log_ratio(Fraction(8), Fraction(1)) / math.log(2) == pytest.approx(3)
    assert natural_log_ratio(Fraction(1), Fraction(8)) == pytest.approx(-math.log(8))
    with pytest.raises(ValueError, match="positive"):
        natural_log_ratio(Fraction(0), Fraction(1))


def test_rows_and_prompts_do_not_contain_prohibited_latent_metadata() -> None:
    serialized = json.dumps(generator().generate(num_question_sets=1)[0]).lower()
    for prohibited in (
        '"secret"',
        '"truthful_answer"',
        '"channel_reliable"',
        '"channel_was_correct"',
        '"repeat"',
    ):
        assert prohibited not in serialized


def test_fixed_and_random_subset_validation_and_replacement() -> None:
    with pytest.raises(ValueError, match="num_question_sets=1"):
        generator(question=FixedSubsetQuestion([[1], [2], [3]])).generate(num_question_sets=2)
    with pytest.raises(ValueError, match="duplicate"):
        generator(question=FixedSubsetQuestion([[1, 1], [2], [3]])).generate(num_question_sets=1)
    with pytest.raises(ValueError, match="without replacement"):
        generator(question=RandomSubsetQuestion(9)).generate(num_question_sets=1)

    sampled = RandomSubsetQuestion(4, replacement=True, sort=False).sample(
        rng=random.Random(0), n=2, k=1
    )[0]
    assert len(sampled["raw_draws"]) == 4
    assert sampled["membership_set"] == list(dict.fromkeys(sampled["raw_draws"]))


def test_multistage_layouts_word_cap_and_ties() -> None:
    assert enforce_reasoning("one two three ANSWER: X", 2) == "one two"
    assert strip_thinking_markers("\n<think>\none two\n</think>\n") == "one two"
    assert enforce_reasoning("<think>one two three</think>", 2) == "one two"
    base = [{"role": "user", "content": "REASONING:"}]
    conversation = stage_two_messages(
        first_messages=base, enforced_reasoning="work", layout="conversation"
    )
    assert [message["role"] for message in conversation] == ["user", "assistant"]
    assert conversation[-1]["content"] == "work\nANSWER:"
    replay = stage_two_messages(
        first_messages=base, enforced_reasoning="work", layout="replay_user"
    )
    assert [message["role"] for message in replay] == ["user", "assistant"]
    assert replay[-1]["content"] == "work\nANSWER:"

    tied = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values=Fraction(1, 2)),
        question=FixedSubsetQuestion([[1]]),
        probe=XVsYPosteriorProbe(x=1, y=2, allow_same=False),
    ).generate(num_question_sets=1)
    assert all(row["normative_comparison"] == "SAME" for row in tied)
    assert all(row["ground_truth_choice"] is None for row in tied)

    same_allowed = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values=Fraction(1, 2)),
        question=FixedSubsetQuestion([[1]]),
        probe=XVsYPosteriorProbe(x=1, y=2, allow_same=True),
    ).generate(num_question_sets=1)
    assert all(row["ground_truth_choice"] == "SAME" for row in same_allowed)
    assert all(
        "If the two posterior probabilities are equal, the required decision value is SAME."
        in row["messages"][-1]["content"]
        for row in same_allowed
    )


@pytest.mark.parametrize("reasoning_budget", [0, 12])
def test_probe_values_are_the_model_facing_answer_surfaces(reasoning_budget: int) -> None:
    dataset = generator(
        environment=NoisyChannelBayesianEnvironment(n=8, k=1, r_values="3/4"),
        question=FixedSubsetQuestion([[2, 4, 7]]),
        probe=XVsYPosteriorProbe(x=2, y=7, reasoning_budget=reasoning_budget),
    ).generate(num_question_sets=1)
    prompt = dataset[0]["messages"][-1]["content"]
    assert "A secret integer s was sampled uniformly from the displayed domain." in prompt
    assert "if the truthful answer is NO, SOURCE reports NO" in prompt
    assert "displayed set contents are not evidence about s" in prompt
    assert "The decision value must be exactly one of: 2 or 7." in prompt
    assert "where X means" not in prompt
    if reasoning_budget:
        assert "Reason carefully from the raw observations" in prompt
        assert "output exactly one of: ANSWER:2 | ANSWER:7" in prompt
        assert "ANSWER:SAME" not in prompt
        assert "Do not provide reasoning" not in prompt
        assert prompt.endswith("REASONING:")
    else:
        assert "Do not provide reasoning, explanation, calculations" in prompt
        assert "Output exactly one of: ANSWER:2 | ANSWER:7." in prompt
        assert "Your entire assistant response must be that allowed response" in prompt
        assert not prompt.endswith("ANSWER:")
    assert dataset[0]["answer_prefix"] == "ANSWER:"

    resolved = MetricSpec().resolve(x=2, y=7)
    assert resolved.surfaces == {"X": "2", "Y": "7", "SAME": "SAME"}


def test_candidate_evidence_projection_is_exact_paired_and_set_free(
    tmp_path: Path,
) -> None:
    raw_environment = NoisyChannelBayesianEnvironment(
        n=8, k=2, r_values=(Fraction(19, 20), Fraction(3, 5))
    )
    probe = XVsYPosteriorProbe(x=2, y=7, reasoning_budget=0)
    raw = generator(
        environment=raw_environment,
        question=FixedSubsetQuestion([[2, 4], [2, 7]]),
        probe=probe,
    ).generate(num_question_sets=1)
    reduced = CandidateEvidenceDatasetGenerator(
        environment=CandidateEvidenceBayesianEnvironment(
            n=8, k=2, r_values=(Fraction(19, 20), Fraction(3, 5))
        ),
        question=CandidateEvidenceQuestion(),
        probe=probe,
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
        system_prompt=SystemPrompt("Be a careful Bayesian."),
    ).generate(source_dataset=raw)

    assert len(reduced) == len(raw) == 4
    assert [row["source_row_id"] for row in reduced] == [row["row_id"] for row in raw]
    row = reduced[1]  # reports YES, NO
    assert set(row["candidate_evidence"]) == {"2", "7"}
    assert [
        observation["relation"] for observation in row["candidate_evidence"]["2"]["observations"]
    ] == ["AGREES", "DISAGREES"]
    assert [
        observation["relation"] for observation in row["candidate_evidence"]["7"]["observations"]
    ] == ["DISAGREES", "DISAGREES"]
    assert [
        observation["reliability_surface"]
        for observation in row["candidate_evidence"]["2"]["observations"]
    ] == ["0.95", "0.6"]
    assert row["posterior_exact"] == raw[1]["posterior_exact"]
    assert row["x_posterior_exact"] == raw[1]["x_posterior_exact"]
    assert row["y_posterior_exact"] == raw[1]["y_posterior_exact"]
    assert row["prior_predictive_exact"] == raw[1]["prior_predictive_exact"]
    assert row["audit_metadata"]["membership_sets"] == [[2, 4], [2, 7]]
    assert row["audit_metadata"]["observed_reports"] == ["YES", "NO"]
    prompt = row["messages"][-1]["content"]
    assert "Candidate s=2:" in prompt
    assert "Candidate s=7:" in prompt
    assert "The decision value must be exactly one of: 2 or 7." in prompt
    assert "Candidate X" not in prompt
    assert "Observation 1: AGREES; reliability 0.95." in prompt
    assert "Is s in" not in prompt
    assert "Observed report:" not in prompt
    assert "{2, 4}" not in prompt
    assert exact_pattern_mass(reduced, 0) == 1

    reduced.save(tmp_path)
    persisted = TranscriptDataset.load(tmp_path)[1]
    assert persisted["candidate_evidence"] == row["candidate_evidence"]
    assert persisted["x_posterior_exact"] == row["x_posterior_exact"]
    assert persisted["y_posterior_exact"] == row["y_posterior_exact"]


def test_candidate_evidence_retains_zero_evidence_and_tie_policies() -> None:
    probe = XVsYPosteriorProbe(x=1, y=2, allow_same=False)
    raw = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values=1),
        question=FixedSubsetQuestion([[1, 2]]),
        probe=probe,
    ).generate(num_question_sets=1)
    reduced = CandidateEvidenceDatasetGenerator(
        environment=CandidateEvidenceBayesianEnvironment(n=2, k=1, r_values=1),
        question=CandidateEvidenceQuestion(),
        probe=probe,
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
        system_prompt=SystemPrompt("Be a careful Bayesian."),
    ).generate(source_dataset=raw)
    impossible = reduced[1]
    assert impossible["posterior_state"] == "undefined_zero_evidence"
    assert impossible["x_posterior"] is None
    assert impossible["y_posterior"] is None

    tie_probe = XVsYPosteriorProbe(x=1, y=2, allow_same=False)
    tied_raw = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values="1/2"),
        question=FixedSubsetQuestion([[1]]),
        probe=tie_probe,
    ).generate(num_question_sets=1)
    tied_reduced = CandidateEvidenceDatasetGenerator(
        environment=CandidateEvidenceBayesianEnvironment(n=2, k=1, r_values="1/2"),
        question=CandidateEvidenceQuestion(),
        probe=tie_probe,
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
        system_prompt=SystemPrompt("Be a careful Bayesian."),
    ).generate(source_dataset=tied_raw)
    assert all(row["normative_comparison"] == "SAME" for row in tied_reduced)
    assert all(row["ground_truth_choice"] is None for row in tied_reduced)


def test_candidate_evidence_rejects_target_or_pairing_mismatches() -> None:
    raw = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values="3/4"),
        question=FixedSubsetQuestion([[1]]),
        probe=XVsYPosteriorProbe(x=1, y=2),
    ).generate(num_question_sets=1)
    projector = CandidateEvidenceDatasetGenerator(
        environment=CandidateEvidenceBayesianEnvironment(n=2, k=1, r_values="3/4"),
        question=CandidateEvidenceQuestion(),
        probe=XVsYPosteriorProbe(x=1, y=2),
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
        system_prompt=SystemPrompt("Different system prompt"),
    )
    with pytest.raises(ValueError, match="same system prompt"):
        projector.generate(source_dataset=raw)

    corrupted_rows = [dict(row) for row in raw]
    corrupted_rows[0]["x_posterior_exact"] = "0/1"
    corrupted = TranscriptDataset(corrupted_rows, manifest=raw.manifest)
    projector = CandidateEvidenceDatasetGenerator(
        environment=CandidateEvidenceBayesianEnvironment(n=2, k=1, r_values="3/4"),
        question=CandidateEvidenceQuestion(),
        probe=XVsYPosteriorProbe(x=1, y=2),
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
        system_prompt=SystemPrompt("Be a careful Bayesian."),
    )
    with pytest.raises(ValueError, match="failed exact recomputation"):
        projector.generate(source_dataset=corrupted)


def test_paired_representation_metrics_report_rescues_and_deltas() -> None:
    raw = generator(
        environment=NoisyChannelBayesianEnvironment(n=2, k=1, r_values="3/4"),
        question=FixedSubsetQuestion([[1]]),
        probe=XVsYPosteriorProbe(x=1, y=2),
    ).generate(num_question_sets=1)
    reduced = CandidateEvidenceDatasetGenerator(
        environment=CandidateEvidenceBayesianEnvironment(n=2, k=1, r_values="3/4"),
        question=CandidateEvidenceQuestion(),
        probe=XVsYPosteriorProbe(x=1, y=2),
        tokenizer_binding=TokenizerBinding(TinyTokenizer()),
        system_prompt=SystemPrompt("Be a careful Bayesian."),
    ).generate(source_dataset=raw)
    raw_results = TranscriptDataset(
        [
            {**raw[0], "posterior_correct": False, "parse_compliance": True},
            {**raw[1], "posterior_correct": True, "parse_compliance": True},
        ]
    )
    reduced_results = TranscriptDataset(
        [
            {**reduced[0], "posterior_correct": True, "parse_compliance": True},
            {**reduced[1], "posterior_correct": True, "parse_compliance": True},
        ]
    )
    summary = summarize_representation_control(raw_results, reduced_results)
    assert summary["uniform"]["raw_accuracy"] == pytest.approx(0.5)
    assert summary["uniform"]["reduced_accuracy"] == pytest.approx(1)
    assert summary["uniform"]["accuracy_delta"] == pytest.approx(0.5)
    assert summary["natural_distribution"]["accuracy_delta"] == pytest.approx(0.5)
    assert summary["transitions"] == {
        "both_correct": 1,
        "raw_only_correct": 0,
        "reduced_only_correct": 1,
        "neither_correct": 0,
    }
    assert summary["rescue_rate"] == pytest.approx(1)
    assert summary["regression_rate"] == pytest.approx(0)


@pytest.mark.parametrize(
    ("selector", "length", "expected"),
    [("all", 3, [0, 1, 2]), ("last", 3, [2]), (-1, 3, [2]), ("1:3", 4, [1, 2])],
)
def test_capture_selectors(selector, length, expected) -> None:
    assert resolve_selector(selector, length) == expected
