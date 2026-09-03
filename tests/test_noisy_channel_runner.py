from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

import mats_experiments.noisy_channel_bayesian.runner as runner_module
from mats_experiments.noisy_channel_bayesian import (
    CaptureSpec,
    ExecutionConfig,
    FixedSubsetQuestion,
    MetricSpec,
    NoisyChannelBayesianEnvironment,
    QwenRunner,
    SGLangMTPConfig,
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


class GenerateLogitsFakeModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.forward_calls += 1
        return super().forward(input_ids, attention_mask, use_cache)

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens
        self.generate_calls += 1
        suffix = torch.tensor([[ord("1") + 3, 2]] * len(input_ids))
        sequences = torch.cat([input_ids, suffix], dim=1)
        if not kwargs.get("output_logits"):
            return sequences
        logits = torch.zeros(len(input_ids), 300)
        logits[:, ord("1") + 3] = 3
        logits[:, ord("2") + 3] = 1
        return SimpleNamespace(sequences=sequences, logits=(logits,))


class FinalLogitOnlyFakeModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.logits_to_keep_calls = []

    def forward(
        self, input_ids, attention_mask, use_cache=False, logits_to_keep=0
    ):
        self.logits_to_keep_calls.append(logits_to_keep)
        output = super().forward(input_ids, attention_mask, use_cache)
        if logits_to_keep:
            output.logits = output.logits[:, -logits_to_keep:, :]
        return output


def make_dataset(tmp_path: Path, *, reasoning: bool = False):
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(n=2, k=1, r_values="3/4"),
        FixedSubsetQuestion([[1]]),
        XVsYPosteriorProbe(x=1, y=2, reasoning=reasoning),
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


@pytest.mark.parametrize(
    ("text", "reasoning", "leading", "trailing"),
    [
        ("ANSWER: 1\n", False, " ", "\n"),
        ("work\nANSWER:\n 1\n", True, "\n ", "\n"),
    ],
)
def test_completion_contract_treats_answer_whitespace_as_semantic_not_exact(
    text: str, reasoning: bool, leading: str, trailing: str
) -> None:
    parsed = runner_module.parse_completion_contract(
        text,
        reasoning=reasoning,
        allow_same=False,
        x_surface="1",
        y_surface="2",
    )

    assert parsed["model_choice"] == "X"
    assert parsed["semantic_answer_compliance"] is True
    assert parsed["strict_answer_compliance"] is True
    assert parsed["exact_answer_format_compliance"] is False
    assert parsed["answer_leading_whitespace"] == leading
    assert parsed["answer_trailing_whitespace"] == trailing
    assert parsed["answer_value_surface"] == "1"


def test_reasoning_without_answer_marker_remains_a_compliance_failure() -> None:
    parsed = runner_module.parse_completion_contract(
        "reasoning that never finishes",
        reasoning=True,
        allow_same=False,
        x_surface="1",
        y_surface="2",
    )

    assert parsed["model_choice"] is None
    assert parsed["semantic_answer_compliance"] is False
    assert parsed["compliance_break"] == "missing_answer_marker"


class WhitespaceTerminalAnswerFakeModel(FakeModel):
    def __init__(self, *, reasoning: bool):
        super().__init__()
        self.reasoning = reasoning

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens, kwargs
        self.generate_calls += 1
        text = "work ANSWER: \n1\n" if self.reasoning else " \n1\n"
        suffix = torch.tensor(
            [self._encode(text) + [self.generation_config.eos_token_id]] * len(input_ids)
        )
        return torch.cat([input_ids, suffix], dim=1)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        logits = torch.zeros(*input_ids.shape, 300)
        immediately_before_candidate = input_ids.eq(ord("\n") + 3)
        logits[..., ord("1") + 3] = torch.where(
            immediately_before_candidate, 9.0, 1.0
        )
        logits[..., ord("2") + 3] = torch.where(
            immediately_before_candidate, 4.0, 8.0
        )
        return SimpleNamespace(logits=logits)

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [ord(character) + 3 for character in text]


class DistinctAssistantEndFakeModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.generation_config.eos_token_id = 299
        self.received_eos_token_ids: list[int] = []

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens
        self.generate_calls += 1
        self.received_eos_token_ids = [int(value) for value in kwargs["eos_token_id"]]
        suffix = torch.tensor(
            [[ord("1") + 3, 2, ord("\n") + 3, 299]] * len(input_ids)
        )
        return torch.cat([input_ids, suffix], dim=1)


def test_generation_stops_at_tokenizer_or_model_eos(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=False)
    model = DistinctAssistantEndFakeModel()
    results = QwenRunner(
        model=model, tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="assistant_end", batch_size=2),
    )

    assert set(model.received_eos_token_ids) == {2, 299}
    assert all(row["generated_token_ids"] == tokenizer.encode("1") for row in results)
    assert all(row["completion"]["terminal_stop_token_id"] == 2 for row in results)
    assert all(row["completion"]["reached_eos"] is True for row in results)


@pytest.mark.parametrize("reasoning", [False, True])
def test_logits_are_measured_after_whitespace_immediately_before_candidate(
    tmp_path: Path, reasoning: bool
) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=reasoning)
    results = QwenRunner(
        model=WhitespaceTerminalAnswerFakeModel(reasoning=reasoning),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    ).execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id=f"whitespace_boundary_{reasoning}",
            batch_size=2,
            metrics=MetricSpec(sequence_scores=False),
            capture=CaptureSpec(
                logits_boundaries=("answer",),
                logits_scope="answer_surfaces",
            ),
        ),
    )

    generated_prefix = "work ANSWER: \n" if reasoning else " \n"
    for row in results:
        assert row["semantic_answer_compliance"] is True
        assert row["exact_answer_format_compliance"] is False
        assert row["answer_leading_whitespace"] == " \n"
        assert row["answer_trailing_whitespace"] == "\n"
        assert row["answer_boundary_generated_token_count"] == len(
            tokenizer.encode(generated_prefix)
        )
        assert row["answer_leading_whitespace_token_ids"] == tokenizer.encode(" \n")
        assert row["answer_value_generated_token_ids"] == tokenizer.encode("1")
        assert row["answer_trailing_whitespace_token_ids"] == tokenizer.encode("\n")
        assert row["answer_surface_raw_logits"] == {"1": 9.0, "2": 4.0}
        assert row["candidate_1_answer_surface"] == "1"
        assert row["candidate_2_answer_surface"] == "2"
        assert row["candidate_1_answer_token_ids"] == tokenizer.encode("1")
        assert row["candidate_2_answer_token_ids"] == tokenizer.encode("2")


def test_single_generation_teacher_forced_capture_scoring_and_resume(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path)
    model = FakeModel()
    runner = QwenRunner(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    config = ExecutionConfig(
        experiment_dir=tmp_path,
        run_id="fake",
        batch_size=2,
        capture=CaptureSpec(
            logits_boundaries=("answer",),
            streams=("resid_pre", "token_mixer_out", "mlp_out", "resid_post"),
            layers="all",
            tokens="last",
        ),
    )
    results = runner.execute(dataset, config)
    assert len(results) == 2
    assert model.generate_calls == 1
    assert all(row["model_choice"] == "X" for row in results)
    assert all(row["model_choice_surface"] == "1" for row in results)
    assert all(row["assistant_completion"] == "ANSWER:1" for row in results)
    assert all(row["strict_answer_compliance"] is True for row in results)
    assert all(row["generation_messages"][-1]["role"] == "user" for row in results)
    assert all(
        row["generation_messages"][-1]["content"].endswith("ANSWER:")
        for row in results
    )
    assert all(
        row["generation_serialized_prompt"].endswith("ANSWER:<assistant>")
        for row in results
    )
    assert all(row["generated_token_ids"] == tokenizer.encode("1") for row in results)
    assert all(row["full_completion"] == "ANSWER:1" for row in results)
    assert all(
        row["teacher_forced_input_ids"]
        == row["generation_input_ids"] + row["generated_sequence_token_ids"]
        for row in results
    )
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
    assert all(
        row["answer_boundary_source"] == "assistant_role_after_user_marker"
        for row in results
    )
    assert all(row["answer_boundary_generated_token_count"] == 0 for row in results)
    tensors = load_file(tmp_path / "runs/fake/activations" / f"{results[0]['row_id']}.safetensors")
    assert "answer.resid_pre.layer_0" in tensors
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
    assert model.generate_calls == 1


def test_runner_executes_mixed_reasoning_probe_parameterizations(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="3/4", control_positional_bias=True
        ),
        FixedSubsetQuestion([[1]]),
        (
            XVsYPosteriorProbe(x=1, y=2, reasoning=False),
            XVsYPosteriorProbe(x=1, y=2, reasoning=True),
        ),
        TokenizerBinding(tokenizer),
    ).generate(num_question_sets=1)
    results = QwenRunner(
        model=FakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="mixed_reasoning", batch_size=4),
    )

    assert len(results) == 8
    reasoning_off = [row for row in results if row["reasoning"] is False]
    reasoning_on = [row for row in results if row["reasoning"] is True]
    assert len(reasoning_off) == len(reasoning_on) == 4
    assert all(row["full_completion"] == "ANSWER:1" for row in reasoning_off)
    assert all(row["strict_answer_compliance"] is True for row in reasoning_off)
    assert all(row["full_completion"] == "1" for row in reasoning_on)
    assert all(row["strict_answer_compliance"] is False for row in reasoning_on)
    assert all(row["compliance_break"] == "missing_answer_marker" for row in reasoning_on)
    assert all(row["answer_boundary_generated_token_count"] is None for row in reasoning_on)


def test_runner_routes_all_continuous_completions_through_sglang_mtp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(n=2, k=1, r_values="3/4"),
        FixedSubsetQuestion([[1]]),
        (
            XVsYPosteriorProbe(x=1, y=2, reasoning=False),
            XVsYPosteriorProbe(x=1, y=2, reasoning=True),
        ),
        TokenizerBinding(tokenizer),
    ).generate(num_question_sets=1)
    dataset.save(tmp_path)
    model = FakeModel()
    runner = QwenRunner(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    calls = []

    def fake_mtp(items, *, batch_size, max_new_tokens, generation_kwargs, config):
        calls.append((items, batch_size, max_new_tokens, generation_kwargs, config))
        runner._generation_backend_metadata = {
            "backend": "sglang_native_mtp",
            "algorithm": "NEXTN",
        }
        results = {}
        for row_id, prompt in items:
            text = "1" if "ANSWER:<assistant>" in prompt else "work ANSWER:1"
            results[row_id] = {
                "text": text,
                "token_ids": tokenizer.encode(text),
                "completion_length": len(tokenizer.encode(text)),
                "reached_eos": False,
                "hit_token_cap": False,
                "effective_batch_size": len(items),
                "generation_backend": "sglang_native_mtp",
                "mtp_metrics": {"spec_accept_rate": 1.0},
            }
        return results

    monkeypatch.setattr(runner, "_generate_completion_with_sglang_mtp", fake_mtp)
    results = runner.execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id="mtp_continuous",
            batch_size=2,
            completion_mtp=SGLangMTPConfig(enabled=True),
        ),
    )

    assert len(calls) == 1
    assert len(calls[0][0]) == len(dataset)
    assert model.generate_calls == 0
    assert all(row["strict_answer_compliance"] is True for row in results)
    assert all(
        row["completion"]["generation_backend"] == "sglang_native_mtp"
        for row in results
    )
    assert all(
        row["generation_settings"]["generation_backend"]["algorithm"] == "NEXTN"
        for row in results
    )


def test_runner_reorients_positional_metrics_to_canonical_candidates(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(
            n=2, k=1, r_values="3/4", control_positional_bias=True
        ),
        FixedSubsetQuestion([[1]]),
        XVsYPosteriorProbe(x=1, y=2, reasoning=False),
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
    assert all(row["candidate_1_answer_surface"] == "1" for row in results)
    assert all(row["candidate_2_answer_surface"] == "2" for row in results)
    assert all(
        row["candidate_1_answer_token_ids"] == tokenizer.encode("1") for row in results
    )
    assert all(
        row["candidate_2_answer_token_ids"] == tokenizer.encode("2") for row in results
    )
    assert all(row["candidate_1_minus_candidate_2_logit"] == pytest.approx(2) for row in results)
    for pattern_index in range(2):
        first, second = results[pattern_index * 2 : pattern_index * 2 + 2]
        assert first["x_minus_y_logit"] == pytest.approx(2)
        assert second["x_minus_y_logit"] == pytest.approx(-2)


def test_oom_recursively_splits_batches(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=False)
    model = FakeModel(oom_batches=True)
    runner = QwenRunner(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    results = runner.execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="oom", batch_size=2),
    )
    assert len(results) == 2
    assert results.manifest["oom_retries"] == 1
    assert sorted(results.manifest["effective_batch_sizes"]) == [1, 1]


def test_answer_surface_logits_reuse_generation_without_capture_forward(
    tmp_path: Path,
) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=False)
    model = GenerateLogitsFakeModel()
    results = QwenRunner(
        model=model, tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id="surface_logits",
            batch_size=2,
            capture=CaptureSpec(
                logits_boundaries=("answer",),
                logits_scope="answer_surfaces",
            ),
            metrics=MetricSpec(sequence_scores=False),
        ),
    )

    assert model.generate_calls == 1
    # Boundary capture is teacher-forced over the exact emitted sequence.
    assert model.forward_calls == 1
    assert all("logit_path" not in row for row in results)
    for row in results:
        assert row["answer_surface_raw_logits"] == {"1": 3.0, "2": 1.0}
        assert get_answer_surface_logits(row, tmp_path / "runs/surface_logits") == {
            "1": 3.0,
            "2": 1.0,
        }


def test_boundary_capture_projects_only_final_prompt_position(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=False)
    model = FinalLogitOnlyFakeModel()
    results = QwenRunner(
        model=model, tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id="final_logits",
            batch_size=2,
            capture=CaptureSpec(logits_boundaries=("answer",)),
            metrics=MetricSpec(sequence_scores=False),
        ),
    )

    assert model.logits_to_keep_calls == [0]
    logits = load_file(
        tmp_path / "runs/final_logits/logits" / f"{results[0]['row_id']}.safetensors"
    )
    assert logits["answer.logits"].shape == (300,)


def test_results_checkpoint_is_periodic_not_per_minibatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = FakeTokenizer()
    dataset = TranscriptDatasetGenerator(
        NoisyChannelBayesianEnvironment(n=2, k=3, r_values="3/4"),
        FixedSubsetQuestion([[1], [2], [1]]),
        XVsYPosteriorProbe(x=1, y=2, reasoning=False),
        TokenizerBinding(tokenizer),
    ).generate(num_question_sets=1)
    dataset.save(tmp_path)
    results_writes = []
    original_write = runner_module._atomic_write_text

    def track_write(path, text):
        if Path(path).name == "results.jsonl":
            results_writes.append(len(text))
        return original_write(path, text)

    monkeypatch.setattr(runner_module, "_atomic_write_text", track_write)
    results = QwenRunner(
        model=FakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id="periodic_checkpoint",
            batch_size=2,
            checkpoint_every_batches=3,
            metrics=MetricSpec(sequence_scores=False),
        ),
    )

    assert len(results) == 8
    assert len(results_writes) == 1


def test_every_decode_position_teacher_forced_shapes(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=False)
    runner = QwenRunner(model=FakeModel(), tokenizer=tokenizer, device=torch.device("cpu"))
    results = runner.execute(
        dataset,
        ExecutionConfig(
            experiment_dir=tmp_path,
            run_id="decode",
            batch_size=2,
            capture=CaptureSpec(
                logits_boundaries=("answer",),
                streams=("resid_pre", "token_mixer_out", "mlp_out", "resid_post"),
                layers="all",
                every_decode_position=True,
            ),
        ),
    )
    tensors = load_file(tmp_path / "runs/decode/logits" / f"{results[0]['row_id']}.safetensors")
    assert tensors["answer.logits"].shape == (300,)
    activations = load_file(
        tmp_path / "runs/decode/activations" / f"{results[0]['row_id']}.safetensors"
    )
    for stream in ("resid_pre", "token_mixer_out", "mlp_out", "resid_post"):
        for layer in range(2):
            assert activations[f"answer.{stream}.layer_{layer}"].shape == (2, 4)


def test_all_prompt_token_capture_removes_padding_per_row(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=False)
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
                logit_tokens="all",
                every_decode_position=False,
            ),
        ),
    )
    for row in results:
        activations = load_file(
            tmp_path / "runs/all_prompt_tokens/activations" / f"{row['row_id']}.safetensors"
        )
        assert activations["answer.resid_pre.layer_0"].shape == (
            len(row["teacher_forced_input_ids"]),
            4,
        )
        assert activations["answer.token_mixer_out.layer_1"].shape == (
            len(row["teacher_forced_input_ids"]),
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
        XVsYPosteriorProbe(x=1, y=2, reasoning=False),
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
    assert result["generation_serialized_prompt"].startswith("<user>")
    assert result["generation_serialized_prompt"] != result["serialized_prompt"]
    assert result["generation_input_ids"] == runtime_tokenizer.encode(
        result["generation_serialized_prompt"]
    )
    assert "The decision value must be exactly one of: 1 or 2." in result[
        "generation_serialized_prompt"
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


def test_reasoning_off_rejects_an_explanatory_answer(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=False)
    results = QwenRunner(
        model=VerboseAnswerFakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="verbose_zero", batch_size=2),
    )
    assert all(row["assistant_completion"] == "ANSWER:1 because" for row in results)
    assert all(row["model_choice"] is None for row in results)
    assert all(row["strict_answer_compliance"] is False for row in results)
    assert all(row["no_reasoning_compliance"] is False for row in results)


class ReasonThenSameFakeModel(FakeModel):
    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens, kwargs
        self.generate_calls += 1
        text = "brief work ANSWER:SAME"
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
            reasoning=True,
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
    assert all(row["assistant_completion"] == "brief work ANSWER:SAME" for row in results)
    assert all(row["strict_answer_compliance"] is True for row in results)
    assert all(row["posterior_correct"] is True for row in results)
    assert all(row["call_layout"] == call_layout for row in results)


class ThinkingMarkerFakeModel(FakeModel):
    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        del attention_mask, max_new_tokens, kwargs
        self.generate_calls += 1
        text = "<think>visible reasoning</think> ANSWER:1"
        suffix = torch.tensor([self._encode(text) + [2]] * len(input_ids))
        return torch.cat([input_ids, suffix], dim=1)

    @staticmethod
    def _encode(text):
        return [ord(character) + 3 for character in text]


def test_reasoning_completion_is_stored_without_repair(tmp_path: Path) -> None:
    dataset, tokenizer = make_dataset(tmp_path, reasoning=True)
    result = QwenRunner(
        model=ThinkingMarkerFakeModel(), tokenizer=tokenizer, device=torch.device("cpu")
    ).execute(
        dataset,
        ExecutionConfig(experiment_dir=tmp_path, run_id="thinking_markers", batch_size=2),
    )[0]
    assert result["completion"]["text"] == "<think>visible reasoning</think> ANSWER:1"
    assert result["full_completion"] == "<think>visible reasoning</think> ANSWER:1"
    assert result["generated_token_ids"] == result["completion"]["token_ids"]
    assert result["strict_answer_compliance"] is True
    assert result["reasoning_length_tokens"] == len(
        tokenizer.encode("<think>visible reasoning</think> ANSWER:")
    )
    assert result["generation_settings"]["enable_thinking"] is False
