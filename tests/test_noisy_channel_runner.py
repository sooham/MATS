from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from mats_experiments.noisy_channel_bayesian import (
    CaptureSpec,
    ExecutionConfig,
    FixedSubsetQuestion,
    NoisyChannelBayesianEnvironment,
    QwenRunner,
    TokenizerBinding,
    TranscriptDatasetGenerator,
    XVsYPosteriorProbe,
    get_activation,
    get_answer_surface_logits,
    parse_model_choice,
    select_unpadded_tokens,
)


class FakeTokenizer:
    chat_template = "fake"
    name_or_path = "fake"
    padding_side = "left"
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text):
        return [ord(character) + 3 for character in text]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        del enable_thinking
        text = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text

    def __call__(self, text, *, padding=False, add_special_tokens=False, return_tensors=None):
        del add_special_tokens
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        rows = [self.encode(value) for value in text]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(masks),
            }
        assert padding
        return {"input_ids": padded, "attention_mask": masks}

    def decode(self, ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token - 3) for token in ids if token > 2)


class FakeLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden):
        return hidden + self.mlp(self.self_attn(hidden))


class FakeModel(nn.Module):
    def __init__(self, *, oom_batches=False):
        super().__init__()
        self.embedding = nn.Embedding(300, 4)
        self.layers = nn.ModuleList([FakeLayer(4), FakeLayer(4)])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.layers))
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.generate_calls = 0
        self.oom_batches = oom_batches

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = torch.zeros(*input_ids.shape, 300)
        logits[..., ord("1") + 3] = 3
        logits[..., ord("2") + 3] = 1
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens, kwargs
        self.generate_calls += 1
        if self.oom_batches and len(input_ids) > 1:
            raise RuntimeError("CUDA out of memory")
        suffix = torch.tensor([[ord("1") + 3, 2]] * len(input_ids))
        return torch.cat([input_ids, suffix], dim=1)


def make_dataset(tmp_path: Path, *, reasoning_budget=3):
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(n=2, k=1, r_values="3/4"),
        FixedSubsetQuestion([[1]]),
        XVsYPosteriorProbe(x=1, y=2, reasoning_budget=reasoning_budget),
        TokenizerBinding(tokenizer),
    ).generate(num_question_sets=1)
    dataset.save(tmp_path)
    return dataset, tokenizer


def test_padding_is_removed_before_token_selection() -> None:
    tensor = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    selected = select_unpadded_tokens(tensor, mask, "all")
    assert selected[0].shape == (2, 2)
    assert torch.equal(selected[0], tensor[0, 2:])
    assert selected[1].shape == (3, 2)


def test_batched_multistage_capture_scoring_and_resume(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path)
    model = FakeModel()
    runner = QwenRunner(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    config = ExecutionConfig(
        experiment_dir=tmp_path,
        run_id="fake",
        batch_size=2,
        capture=CaptureSpec(
            logits_boundaries=("reasoning", "answer"),
            streams=("resid_pre", "token_mixer_out", "mlp_out", "resid_post"),
            layers="all",
            tokens="last",
        ),
    )
    results = runner.execute(dataset, config)
    assert len(results) == 2
    assert model.generate_calls == 2
    assert all(row["model_choice"] == "X" for row in results)
    assert all(row["model_choice_surface"] == "1" for row in results)
    assert all(row["assistant_completion"] == "ANSWER:1" for row in results)
    assert all(row["strict_answer_compliance"] is True for row in results)
    assert all(row["answer_messages"][-1] == {
        "role": "assistant", "content": "1\nANSWER:"
    } for row in results)
    assert all(
        row["answer_serialized_prompt"].endswith("<assistant>1\nANSWER:")
        for row in results
    )
    assert all("Now provide only" not in row["answer_serialized_prompt"] for row in results)
    assert all(row["answer_surfaces"]["X"] == "1" for row in results)
    assert all(row["answer_surfaces"]["Y"] == "2" for row in results)
    assert all(row["answer_surface_token_ids"]["X"] == [ord("1") + 3] for row in results)
    assert all(set(row["sequence_log_probabilities"]) == {"1", "2", "SAME"} for row in results)
    assert all(row["sequence_log_probability_base"] == "e" for row in results)
    assert all(row["sequence_log_probability_unit"] == "nats" for row in results)
    assert all(
        row["sequence_log_probabilities"]["1"]
        == row["x_sequence_log_probability"]
        for row in results
    )
    assert all(row["x_minus_y_logit"] == pytest.approx(2) for row in results)
    assert all(len(row["enforced_reasoning"].split()) <= 3 for row in results)
    assert all(row["reasoning_serialized_prompt"] for row in results)
    assert all(row["reasoning_input_ids"] for row in results)
    assert all(row["reasoning_input_tokens"] for row in results)
    assert all(row["answer_input_ids"] == tokenizer.encode(row["answer_serialized_prompt"]) for row in results)
    tensors = load_file(tmp_path / "runs/fake/activations" / f"{results[0]['row_id']}.safetensors")
    assert "reasoning.resid_pre.layer_0" in tensors
    assert "answer.resid_post.layer_1" in tensors
    logits = load_file(tmp_path / "runs/fake/logits" / f"{results[0]['row_id']}.safetensors")
    assert logits["answer.logits"].shape == (300,)
    surface_logits = get_answer_surface_logits(results[0], tmp_path / "runs/fake")
    assert surface_logits["1"] == pytest.approx(3)
    assert surface_logits["2"] == pytest.approx(1)
    assert get_activation(
        results[0],
        tmp_path / "runs/fake",
        boundary="answer",
        stream="resid_post",
        layer=1,
        token_index=-1,
    ).shape == (4,)

    resumed = runner.execute(dataset, config)
    assert len(resumed) == 2
    assert model.generate_calls == 2


def test_runner_executes_mixed_reasoning_probe_parameterizations(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="3/4", control_positional_bias=True
        ),
        FixedSubsetQuestion([[1]]),
        (
            XVsYPosteriorProbe(x=1, y=2, reasoning_budget=0),
            XVsYPosteriorProbe(x=1, y=2, reasoning_budget=3),
        ),
        TokenizerBinding(tokenizer),
    ).generate(num_question_sets=1)
    results = QwenRunner(
        model=FakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="mixed_budgets", batch_size=4),
    )

    assert len(results) == 8
    zero_budget = [row for row in results if row["reasoning_budget"] == 0]
    positive_budget = [row for row in results if row["reasoning_budget"] == 3]
    assert len(zero_budget) == len(positive_budget) == 4
    assert all(row["answer_messages"][-1]["content"] == "ANSWER:" for row in zero_budget)
    assert all(
        row["answer_messages"][-1]["content"] == "1\nANSWER:"
        for row in positive_budget
    )
    assert all(
        row["answer_messages"][-1]["role"] == "assistant"
        and "Now provide only" not in row["answer_serialized_prompt"]
        for row in positive_budget
    )
    assert all(row["strict_answer_compliance"] is True for row in results)


def test_runner_reorients_positional_metrics_to_canonical_candidates(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="3/4", control_positional_bias=True
        ),
        FixedSubsetQuestion([[1]]),
        XVsYPosteriorProbe(x=1, y=2, reasoning_budget=0),
        TokenizerBinding(tokenizer),
    ).generate(num_question_sets=1)
    results = QwenRunner(
        model=FakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="positional", batch_size=2),
    )

    assert len(results) == 4
    assert [row["model_choice"] for row in results] == ["X", "Y", "X", "Y"]
    assert all(row["model_choice_candidate"] == "1" for row in results)
    assert all(row["model_choice_canonical"] == "C1" for row in results)
    assert all(row["candidate_1_sequence_log_probability"] == pytest.approx(
        row["sequence_log_probabilities"]["1"]
    ) for row in results)
    assert all(row["candidate_2_sequence_log_probability"] == pytest.approx(
        row["sequence_log_probabilities"]["2"]
    ) for row in results)
    assert all(row["candidate_1_minus_candidate_2_logit"] == pytest.approx(2) for row in results)
    for pattern_index in range(2):
        first, second = results[pattern_index * 2 : pattern_index * 2 + 2]
        assert first["x_minus_y_logit"] == pytest.approx(2)
        assert second["x_minus_y_logit"] == pytest.approx(-2)


def test_oom_recursively_splits_batches(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning_budget=0)
    model = FakeModel(oom_batches=True)
    runner = QwenRunner(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    results = runner.execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="oom", batch_size=2),
    )
    assert len(results) == 2
    assert results.manifest["oom_retries"] == 1
    assert sorted(results.manifest["effective_batch_sizes"]) == [1, 1]


def test_every_decode_position_teacher_forced_shapes(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning_budget=0)
    runner = QwenRunner(model=FakeModel(), tokenizer=tokenizer, device=torch.device("cpu"))
    results = runner.execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id="decode",
            batch_size=2,
            capture=CaptureSpec(
                logits_boundaries=("answer",),
                streams=("resid_post",),
                layers=(-1,),
                every_decode_position=True,
            ),
        ),
    )
    tensors = load_file(tmp_path / "runs/decode/logits" / f"{results[0]['row_id']}.safetensors")
    assert tensors["answer.logits"].shape == (1, 300)
    activations = load_file(
        tmp_path / "runs/decode/activations" / f"{results[0]['row_id']}.safetensors"
    )
    assert activations["answer.resid_post.layer_1"].shape == (1, 4)


def test_all_prompt_token_capture_removes_padding_per_row(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning_budget=0)
    runner = QwenRunner(model=FakeModel(), tokenizer=tokenizer, device=torch.device("cpu"))
    results = runner.execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id="all_prompt_tokens",
            batch_size=2,
            capture=CaptureSpec(
                logits_boundaries=("answer",),
                streams=("resid_pre", "token_mixer_out", "mlp_out", "resid_post"),
                layers="all",
                tokens="all",
                every_decode_position=False,
            ),
        ),
    )
    for row in results:
        activations = load_file(
            tmp_path / "runs/all_prompt_tokens/activations" / f"{row['row_id']}.safetensors"
        )
        assert activations["answer.resid_pre.layer_0"].shape == (
            len(row["answer_input_ids"]),
            4,
        )
        assert activations["answer.token_mixer_out.layer_1"].shape == (
            len(row["answer_input_ids"]),
            4,
        )
        logits = load_file(
            tmp_path / "runs/all_prompt_tokens/logits" / f"{row['row_id']}.safetensors"
        )
        assert logits["answer.logits"].shape == (300,)


class ConstructionOnlyTokenizer(FakeTokenizer):
    chat_template = "construction-only"
    name_or_path = "construction-only"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        del enable_thinking
        text = "".join(f"[{item['role']}]{item['content']}" for item in messages)
        if add_generation_prompt:
            text += "[assistant]"
        return self.encode(text) if tokenize else text


def test_runner_reserializes_with_runtime_tokenizer_and_persists_exact_tokens(
    tmp_path: Path,
) -> None:
    construction_tokenizer = ConstructionOnlyTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(n=2, k=1, r_values="3/4"),
        FixedSubsetQuestion([[1]]),
        XVsYPosteriorProbe(x=1, y=2, reasoning_budget=0),
        TokenizerBinding(construction_tokenizer),
    ).generate(num_question_sets=1)
    dataset.save(tmp_path)
    runtime_tokenizer = FakeTokenizer()
    result = QwenRunner(
        model=FakeModel(), tokenizer=runtime_tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="runtime_tokenizer", batch_size=2),
    )[0]

    assert result["serialized_prompt"].startswith("[system]") is False
    assert result["serialized_prompt"].startswith("[user]")
    assert result["answer_serialized_prompt"].startswith("<user>")
    assert result["answer_serialized_prompt"] != result["serialized_prompt"]
    assert result["answer_input_ids"] == runtime_tokenizer.encode(
        result["answer_serialized_prompt"]
    )
    assert "The decision value must be exactly one of: 1 or 2." in result[
        "answer_serialized_prompt"
    ]
    assert result["runtime_tokenizer_template_fingerprint"] != result[
        "tokenizer_template_fingerprint"
    ]


def test_numeric_choice_parser_uses_whole_surfaces() -> None:
    assert parse_model_choice("Final answer: 2", allow_same=False, x_surface="2", y_surface="7") == "X"
    assert parse_model_choice("Final answer: 7", allow_same=False, x_surface="2", y_surface="7") == "Y"
    assert parse_model_choice("Final answer: 27", allow_same=False, x_surface="2", y_surface="7") is None
    assert parse_model_choice(
        "2", allow_same=False, x_surface="2", y_surface="7", strict=True
    ) == "X"
    assert parse_model_choice(
        "ANSWER: 7", allow_same=False, x_surface="2", y_surface="7", strict=True
    ) == "Y"
    assert parse_model_choice(
        "ANSWER: 2 because it agrees",
        allow_same=False,
        x_surface="2",
        y_surface="7",
        strict=True,
    ) is None
    assert parse_model_choice(
        "SAME", allow_same=False, x_surface="2", y_surface="7", strict=True
    ) is None
    assert parse_model_choice(
        "ANSWER: SAME",
        allow_same=True,
        x_surface="2",
        y_surface="7",
        strict=True,
    ) == "SAME"


class VerboseAnswerFakeModel(FakeModel):
    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens, kwargs
        self.generate_calls += 1
        suffix = torch.tensor(
            [[ord(character) + 3 for character in "1 because"] + [2]] * len(input_ids)
        )
        return torch.cat([input_ids, suffix], dim=1)


def test_zero_budget_rejects_an_explanatory_answer(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning_budget=0)
    results = QwenRunner(
        model=VerboseAnswerFakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="verbose_zero", batch_size=2),
    )
    assert all(row["assistant_completion"] == "ANSWER:1 because" for row in results)
    assert all(row["model_choice"] is None for row in results)
    assert all(row["strict_answer_compliance"] is False for row in results)
    assert all(row["zero_reasoning_compliance"] is False for row in results)


class ReasonThenSameFakeModel(FakeModel):
    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens, kwargs
        self.generate_calls += 1
        text = "brief work" if self.generate_calls == 1 else "SAME"
        suffix = torch.tensor(
            [[ord(character) + 3 for character in text] + [2]] * len(input_ids)
        )
        return torch.cat([input_ids, suffix], dim=1)


@pytest.mark.parametrize("call_layout", ["conversation", "replay_user"])
def test_allow_same_with_reasoning_works_for_both_layouts(
    tmp_path: Path, call_layout: str
) -> None:
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(n=2, k=1, r_values="1/2"),
        FixedSubsetQuestion([[1]]),
        XVsYPosteriorProbe(
            x=1,
            y=2,
            reasoning_budget=3,
            allow_same=True,
            call_layout=call_layout,
        ),
        TokenizerBinding(tokenizer),
    ).generate(num_question_sets=1)
    model = ReasonThenSameFakeModel()
    results = QwenRunner(
        model=model, tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id=f"same_{call_layout}",
            batch_size=2,
        ),
    )
    assert all(row["normative_comparison"] == "SAME" for row in results)
    assert all(row["ground_truth_choice"] == "SAME" for row in results)
    assert all(row["model_choice"] == "SAME" for row in results)
    assert all(row["assistant_completion"] == "ANSWER:SAME" for row in results)
    assert all(row["strict_answer_compliance"] is True for row in results)
    assert all(row["posterior_correct"] is True for row in results)
    assert all(row["call_layout"] == call_layout for row in results)


class ThinkingMarkerFakeModel(FakeModel):
    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens, kwargs
        self.generate_calls += 1
        text = "<think>visible reasoning</think>" if self.generate_calls == 1 else "1"
        suffix = torch.tensor([self._encode(text) + [2]] * len(input_ids))
        return torch.cat([input_ids, suffix], dim=1)

    @staticmethod
    def _encode(text):
        return [ord(character) + 3 for character in text]


def test_reasoning_completion_removes_stray_native_thinking_markers(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning_budget=3)
    result = QwenRunner(
        model=ThinkingMarkerFakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="thinking_markers", batch_size=2),
    )[0]
    assert result["reasoning_completion"]["text"] == "visible reasoning"
    assert result["reasoning_completion"]["raw_text"] == "<think>visible reasoning</think>"
    assert result["reasoning_completion"]["thinking_markers_removed"] is True
    assert "<think>" not in result["enforced_reasoning"]
    assert result["generation_settings"]["enable_thinking"] is False
