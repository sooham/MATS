#!/usr/bin/env python3
"""Build notebook 16: sparse semantic residual capture, probes, and residual patching."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "16_noisy_channel_bayesian_activation_patching.ipynb"


def md(source: str, cell_id: str):
    cell = nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


def code(source: str, cell_id: str):
    cell = nbf.v4.new_code_cell(dedent(source).strip() + "\n")
    cell["id"] = cell_id
    return cell


cells = [
    md(
        r"""
        # Experiment 2: agreement, reliability, and posterior activations

        This notebook is the activation-focused successor to notebook 13. It keeps notebook 13's
        raw set-membership task and visible-reasoning contrast, but uses $N=9$, $K=3$, ten
        symmetric interior reliabilities, 64 seeded question schedules, and one candidate
        presentation order. The resulting design has

        $$64\text{ schedules}\times 8\text{ report patterns}\times
        10\text{ reliabilities}\times2\text{ reasoning modes}=10{,}240\text{ rows}. $$

        Ties in agreement counts are retained. `allow_same=False` means that `SAME` is not offered
        as an answer token; it does **not** remove tie stimuli.

        Every expensive stage has an explicit gate, and every gate is initially false. Building the
        notebook does not run inference, train probes, open the locked test, or patch activations.

        Reasoning-off and reasoning-on are separate capture/probe datasets. They share the same
        schedule-level split, so a held-out schedule cannot appear in either mode's training set.
        The activation cache stores one stream—`resid_post`—at every language layer but only at a
        preregistered set of semantically aligned prompt locations and the terminal answer tail.
        """,
        "title-and-scope",
    ),
    code(
        r"""
        from __future__ import annotations

        import gc
        import hashlib
        import json
        import logging
        import math
        import os
        import re
        import shutil
        import sys
        import tempfile
        from collections import Counter, defaultdict
        from datetime import datetime, timezone
        from fractions import Fraction
        from itertools import combinations
        from pathlib import Path

        # torchao probes optional native extensions during some Transformers imports. The local
        # experiment stack does not need those extensions.
        os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")
        logging.getLogger("torchao").setLevel(logging.ERROR)

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Markdown, display
        from safetensors import SafetensorError, safe_open
        from safetensors.torch import load_file, save_file
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.decomposition import PCA
        from sklearn.metrics import (
            balanced_accuracy_score,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.model_selection import GridSearchCV, GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from transformers import AutoConfig, AutoProcessor

        REPO_ROOT = next(
            path for path in (Path.cwd(), *Path.cwd().parents)
            if (path / "pyproject.toml").exists()
        )
        sys.path.insert(0, str(REPO_ROOT))

        from mats_experiments.noisy_channel_bayesian import (
            CaptureSpec,
            ExecutionConfig,
            MetricSpec,
            ModelConfig,
            NoisyChannelBayesianEnvironment,
            QwenRunner,
            RandomSubsetQuestion,
            SGLangMTPConfig,
            SystemPrompt,
            TokenizerBinding,
            TranscriptDataset,
            TranscriptDatasetGenerator,
            XVsYPosteriorProbe,
            answer_patterns,
        )
        """,
        "imports",
    ),
    md(
        r"""
        ## Configuration

        Every experimental, storage, split, probe, visualization, and patching parameter is in this
        cell. Native Qwen thinking remains disabled in every condition: `reasoning=True` means
        ordinary visible assistant text before `ANSWER:`, as in notebook 13.

        Reasoning-off has a 64-token watchdog. Reasoning-on has a 4,096-token watchdog so visible
        reasoning has enough room to reach the required terminal answer. Only selected activation
        positions are persisted, so the longer generation cap does not multiply disk usage.
        """,
        "configuration-explanation",
    ),
    code(
        r"""
        # ------------------------------- factorial -------------------------------
        MODEL_ID = "Qwen/Qwen3.5-9B"
        MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        N = 9
        K = 3
        R_VALUES = (
            0.05, 0.15, 0.25, 0.35, 0.45,
            0.55, 0.65, 0.75, 0.85, 0.95,
        )
        R_EXACT_VALUES = tuple(str(Fraction(str(value))) for value in R_VALUES)
        REASONING_VALUES = (False, True)
        NUM_QUESTION_SETS = 64
        NUM_ANSWER_PATTERNS = 8
        CONTROL_POSITIONAL_BIAS = False
        ALLOW_SAME = False                 # tie rows stay; SAME is not an output option

        X = 2
        Y = 7
        SUBSET_SIZE = 4
        SEED = 20260905
        SYSTEM_PROMPT_TEXT = (
            "You are a Bayesian reasoner. Follow the user's game rules exactly. "
            "Use plaintext only. Do not use Markdown, headings, bullets, tables, code blocks, "
            "HTML, or any other formatting."
        )

        # --------------------------- capture and storage --------------------------
        ACTIVATION_STREAM = "resid_post"   # exactly one saved activation stream
        ACTIVATION_LAYERS = "all"          # all 32 language layers
        ACTIVATION_TOKENS = "row_selected" # semantic prompt sites plus final answer tail
        NUM_LAYERS = 32
        HIDDEN_SIZE = 4096
        ACTIVATION_BYTES_PER_ELEMENT = 2    # BF16
        MAX_COMPLETION_TOKENS_BY_REASONING = {False: 64, True: 4096}
        ANSWER_TAIL_TOKENS = 16
        ACTIVATION_STORAGE_BUDGET_GIB = 160.0
        ACTIVATION_BUDGET_SAFETY_FRACTION = 0.90
        MIN_FREE_DISK_AFTER_CAPTURE_GIB = 25.0

        MODEL_DTYPE = "auto"
        DEVICE_MAP = None
        LOCAL_FILES_ONLY = True
        ENABLE_MTP = False
        COMPLETION_BATCH_SIZE = 4
        CAPTURE_BATCH_SIZE = 1
        SCORE_BATCH_SIZE = 4
        CAPTURE_CHECKPOINT_ROWS = 8       # persist a contiguous result prefix every 8 rows

        EXPERIMENT_ROOT = (
            REPO_ROOT / "artifacts" / "noisy_channel_bayesian_experiment_2_activation_patching"
        )
        DATASET_RUN_ID = "qwen35_9b_n9_k3_r10_s64_selected_tokens_v2"
        MODE_NAMES = {False: "reasoning_off", True: "reasoning_on"}
        RUN_IDS = {
            reasoning: f"{DATASET_RUN_ID}_{MODE_NAMES[reasoning]}"
            for reasoning in REASONING_VALUES
        }

        # ------------------------------ split policy ------------------------------
        TEST_SCHEDULE_COUNT = 8
        SPLIT_SEARCH_CANDIDATES = 50_000
        PREFER_ALL_TIE_CELLS_IN_TEST = True

        # ------------------------------- probe plan -------------------------------
        # Each entry is a semantically aligned span; multi-token spans are mean pooled.
        PROBE_SITES = (
            "domain_boundary",
            "reliability_rule_boundary",
            "reliability_r_value",
            "reliability_one_minus_r_value",
            "report_1_answer",
            "report_2_answer",
            "report_3_answer",
            "observation_question_boundary",
            "candidate_1_value",
            "candidate_2_value",
            "candidate_question_boundary",
            "assistant_turn_boundary",
            "final_prompt",
            "answer_line",
        )
        CONTINUOUS_PROBE_TARGETS = (
            "delta_a", "gain", "gain_abs", "z_bayes", "z_heuristic",
        )
        BINARY_PROBE_TARGETS = ("reliability_sign",)
        PROBE_LAYERS = tuple(range(NUM_LAYERS))
        RIDGE_ALPHAS = tuple(float(value) for value in np.logspace(-3, 6, 10))
        RIDGE_SOLVER = "lsqr"
        RIDGE_TOL = 1e-5
        RIDGE_MAX_ITER = 10_000
        LOGISTIC_C_VALUES = tuple(float(value) for value in np.logspace(-4, 4, 9))
        PROBE_CV_FOLDS = 5
        PROBE_N_JOBS = 8                  # prevents many copies of 4096-wide matrices
        PROBE_TOP_K_LAYERS = 3            # selected using training CV only
        MIN_GENERATED_SITE_COVERAGE = 0.95
        MIN_PROBE_TRAIN_ROWS = HIDDEN_SIZE + 1
        PROBE_RUN_ID = "ridge_logistic_split_reasoning_selected_sites_v2"

        # ------------------------- probe visualization plan -----------------------
        # Full layer sweeps use grouped validation scores, never in-sample R^2.
        PROBE_PLOT_TARGETS = CONTINUOUS_PROBE_TARGETS
        PROBE_TRANSFER_TARGETS = ("z_bayes",)
        PROBE_TRANSFER_LAYERS = (0, 8, 16, 24, 31)
        PROBE_FIGURE_DPI = 160

        # ------------------------- whole-residual patching ------------------------
        PATCH_SITE = "final_prompt"
        PATCH_RELIABILITY_PAIRS = (("1/20", "19/20"), ("1/4", "3/4"))
        PATCH_MAX_DIRECTIONS = 64
        BRIDGE_LOGIT_TOLERANCE = 0.05
        PATCH_RUN_ID = "whole_residual_reliability_interchange_v2"

        # -------------------------- explicit execution gates ----------------------
        RUN_GPU_CAPTURE = False
        LOAD_COMPLETED_CAPTURE = False
        RUN_PROBE_TRAINING = False
        RUN_LOCKED_TEST_EVALUATION = False
        RUN_PROBE_VISUALIZATIONS = False
        RUN_PROBE_TRANSFER_ANALYSIS = False
        RUN_ACTIVATION_PATCHING = False
        """,
        "configuration",
    ),
    code(
        r"""
        def model_key(model_id: str) -> str:
            readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
            digest = hashlib.sha256(model_id.encode()).hexdigest()[:8]
            return f"{readable}_{digest}"


        MODEL_ROOT = EXPERIMENT_ROOT / model_key(MODEL_ID)
        DATASET_ROOT = MODEL_ROOT / "datasets" / DATASET_RUN_ID
        RUN_DIRS = {
            reasoning: MODEL_ROOT / "runs" / RUN_IDS[reasoning]
            for reasoning in REASONING_VALUES
        }
        SPLIT_ROOT = MODEL_ROOT / "splits" / DATASET_RUN_ID
        PROBE_ROOT = MODEL_ROOT / "probes" / PROBE_RUN_ID
        PATCH_ROOT = MODEL_ROOT / "patches" / PATCH_RUN_ID

        ROWS_PER_SCHEDULE_PER_REASONING = NUM_ANSWER_PATTERNS * len(R_VALUES)
        EXPECTED_ROWS_PER_REASONING = NUM_QUESTION_SETS * ROWS_PER_SCHEDULE_PER_REASONING
        EXPECTED_TEST_ROWS_PER_REASONING = (
            TEST_SCHEDULE_COUNT * ROWS_PER_SCHEDULE_PER_REASONING
        )
        EXPECTED_TRAIN_ROWS_PER_REASONING = (
            EXPECTED_ROWS_PER_REASONING - EXPECTED_TEST_ROWS_PER_REASONING
        )
        EXPECTED_TEST_ROWS = EXPECTED_TEST_ROWS_PER_REASONING * len(REASONING_VALUES)
        EXPECTED_TRAIN_ROWS = EXPECTED_TRAIN_ROWS_PER_REASONING * len(REASONING_VALUES)
        EXPECTED_ROWS = (
            NUM_QUESTION_SETS
            * NUM_ANSWER_PATTERNS
            * len(R_VALUES)
            * len(REASONING_VALUES)
        )
        EXPECTED_VARIANTS_PER_BASE_TRANSCRIPT = len(R_VALUES) * len(REASONING_VALUES)
        ALL_AGREEMENT_CELLS = tuple(
            (a1, a2) for a1 in range(K + 1) for a2 in range(K + 1)
        )
        TIE_CELLS = frozenset((value, value) for value in range(K + 1))

        assert NUM_ANSWER_PATTERNS == 2**K == 8
        assert EXPECTED_ROWS_PER_REASONING == 5120
        assert EXPECTED_TRAIN_ROWS_PER_REASONING == 4480
        assert EXPECTED_TEST_ROWS_PER_REASONING == 640
        assert EXPECTED_ROWS == 10240
        assert EXPECTED_TEST_ROWS == (
            TEST_SCHEDULE_COUNT
            * NUM_ANSWER_PATTERNS
            * EXPECTED_VARIANTS_PER_BASE_TRANSCRIPT
        ) == 1280
        assert EXPECTED_TRAIN_ROWS == EXPECTED_ROWS - EXPECTED_TEST_ROWS == 8960
        assert len(ALL_AGREEMENT_CELLS) == (K + 1) ** 2 == 16
        assert len(R_VALUES) == 10
        assert tuple(Fraction(value) for value in R_EXACT_VALUES) == tuple(
            Fraction(numerator, 20) for numerator in range(1, 20, 2)
        )
        assert CONTROL_POSITIONAL_BIAS is False
        assert ACTIVATION_STREAM == "resid_post"
        assert ACTIVATION_TOKENS == "row_selected" and ACTIVATION_LAYERS == "all"
        assert CAPTURE_CHECKPOINT_ROWS > 0
        if RUN_GPU_CAPTURE and LOAD_COMPLETED_CAPTURE:
            raise ValueError("Choose either fresh/resumed capture or loading, not both.")
        if RUN_ACTIVATION_PATCHING and RUN_GPU_CAPTURE:
            raise ValueError("Capture and causal patching must run in separate model-loading passes.")

        capture_spec = CaptureSpec(
            logits_boundaries=("answer",),
            logits_scope="answer_surfaces",
            streams=(ACTIVATION_STREAM,),
            layers=ACTIVATION_LAYERS,
            tokens=ACTIVATION_TOKENS,
            every_decode_position=False,
        )
        metric_spec = MetricSpec(sequence_scores=False)

        print({
            "model": MODEL_ID,
            "rows": EXPECTED_ROWS,
            "rows_per_reasoning_mode": EXPECTED_ROWS_PER_REASONING,
            "agreement_cells": len(ALL_AGREEMENT_CELLS),
            "train_rows_per_reasoning_mode": EXPECTED_TRAIN_ROWS_PER_REASONING,
            "test_rows_per_reasoning_mode": EXPECTED_TEST_ROWS_PER_REASONING,
            "activation_streams": capture_spec.streams,
            "activation_layers": capture_spec.layers,
            "activation_tokens": capture_spec.tokens,
            "gpu_capture_enabled": RUN_GPU_CAPTURE,
        })
        """,
        "derived-configuration",
    ),
    md(
        r"""
        ## Generate labels and choose a leakage-safe split

        The atomic sampling unit is a complete question schedule. A schedule contains eight
        exhaustive report patterns, and every base transcript is replayed at all ten
        reliabilities and both reasoning settings. Splitting individual rows would put nearly
        identical replays in train and test.

        The same eight complete schedules are held out in both reasoning modes. Selection uses
        labels only—never activations, logits, completions, or model accuracy. A deterministic
        50,000-subset search followed by one-swap local improvement ranks candidates by:

        1. fewest missing tie cells;
        2. smallest L1 distance between test and full-dataset agreement-cell proportions;
        3. smallest worst-cell deviation;
        4. greatest agreement-cell coverage;
        5. schedule indices as a deterministic tie-breaker.

        Once selected, all 20 reliability/reasoning variants of every base transcript stay in the
        same partition. Thus no question schedule can cross from either test dataset into either
        training dataset. The exact split and its audit are persisted before inference.
        """,
        "split-rationale",
    ),
    md(
        r"""
        ### What the probe labels mean

        For one transcript, let $a_1,a_2\in\{0,1,2,3\}$ be the numbers of source reports that
        candidate 1 and candidate 2 predict correctly. The notebook derives these labels without
        consulting Qwen:

        $$\Delta a=a_1-a_2,\qquad g(r)=\log\frac{r}{1-r},\qquad
        z_{\mathrm{Bayes}}=\Delta a\,g(r).$$

        `gain_abs` is $|g(r)|$, `reliability_sign` is $\mathbb{1}[r>1/2]$, and
        `z_heuristic` is $\Delta a|g(r)|$. The last target deliberately removes the Bayesian sign
        reversal below $r=1/2$, so comparing it with `z_bayes` asks whether the representation
        contains the correct signed computation or merely "more agreement means more evidence."

        Each target gets a separate probe. Tie rows have $\Delta a=z_{\mathrm{Bayes}}=
        z_{\mathrm{heuristic}}=0$, while their gain and reliability-sign labels remain informative.
        This is why retaining all four tie constructions helps disentangle evidence balance from
        reliability.
        """,
        "target-definitions",
    ),
    code(
        r"""
        def atomic_write_json(path: Path, payload: object) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                    handle.write("\n")
                os.replace(temporary_name, path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)


        def atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                os.replace(temporary_name, path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)


        def row_reliability_exact(row: dict[str, object]) -> str:
            values = tuple(str(value) for value in row["reliabilities_exact"])
            assert len(values) == K and len(set(values)) == 1
            return values[0]


        def base_transcript_key(row: dict[str, object]) -> tuple[int, int]:
            return int(row["question_set_index"]), int(row["answer_pattern_index"])


        def agreement_cell(row: dict[str, object]) -> tuple[int, int]:
            return (
                int(row["total_agreement_candidate_1"]),
                int(row["total_agreement_candidate_2"]),
            )


        def target_fields(row: dict[str, object]) -> dict[str, float | int]:
            reliability = Fraction(row_reliability_exact(row))
            assert 0 < reliability < 1 and reliability != Fraction(1, 2)
            gain = math.log(float(reliability / (1 - reliability)))
            delta_a = (
                int(row["total_agreement_candidate_1"])
                - int(row["total_agreement_candidate_2"])
            )
            return {
                "agreement_c1": int(row["total_agreement_candidate_1"]),
                "agreement_c2": int(row["total_agreement_candidate_2"]),
                "delta_a": float(delta_a),
                "gain": float(gain),
                "gain_abs": float(abs(gain)),
                "reliability_sign": int(reliability > Fraction(1, 2)),
                "z_bayes": float(delta_a * gain),
                "z_heuristic": float(delta_a * abs(gain)),
            }


        def token_offsets(
            *, text: str, input_ids: list[int], tokenizer
        ) -> list[tuple[int, int]]:
            encoded = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            encoded_ids = [int(value) for value in encoded["input_ids"]]
            if encoded_ids != input_ids:
                raise ValueError("Offset-tokenization IDs differ from runtime chat-template IDs.")
            return [(int(start), int(end)) for start, end in encoded["offset_mapping"]]


        def prompt_token_sites(row: dict[str, object], tokenizer) -> dict[str, list[int]]:
            text = str(row["serialized_prompt"])
            input_ids = [int(value) for value in row["input_ids"]]
            offsets = token_offsets(text=text, input_ids=input_ids, tokenizer=tokenizer)

            def indices_for_span(start: int, end: int) -> list[int]:
                indices = [
                    index for index, (token_start, token_end) in enumerate(offsets)
                    if token_end > start and token_start < end
                ]
                if not indices:
                    raise ValueError(f"No tokens overlap character span [{start}, {end}).")
                return indices

            domain = re.search(r"^DOMAIN:.*\.$", text, flags=re.MULTILINE)
            reliability = re.search(
                r"^The observed SOURCE report equals.*\.$", text, flags=re.MULTILINE
            )
            reliability_values = re.search(
                r"\br=([0-9.]+).*?\b1-r=([0-9.]+)",
                reliability.group(0) if reliability else "",
            )
            report_matches = list(re.finditer(
                r"^SOURCE reported (YES|NO)\.$", text, flags=re.MULTILINE
            ))
            if len(report_matches) != K:
                raise ValueError(f"Expected {K} observed-report markers, found {len(report_matches)}.")
            question_boundary = re.search(r"\n\nQUESTION:\n", text)
            candidate_question = re.search(
                r"Given all observations, which has larger posterior probability: "
                r"s=(\d+) or s=(\d+)\?\n",
                text,
            )
            assistant_turn = re.search(
                r"<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n$",
                text,
            )
            if any(match is None for match in (
                domain, reliability, reliability_values, question_boundary,
                candidate_question, assistant_turn,
            )):
                raise ValueError("A preregistered semantic token boundary was not found.")
            assert domain and reliability and reliability_values
            assert question_boundary and candidate_question and assistant_turn

            reliability_base = reliability.start()
            r_assignment_start = reliability_base + reliability_values.start(0)
            r_assignment_end = reliability_base + reliability_values.end(1)
            one_minus_start = reliability_base + reliability_values.start(0) + (
                reliability_values.group(0).index("1-r=")
            )
            one_minus_end = reliability_base + reliability_values.end(2)
            sites = {
                # Period/newline neighborhoods are represented as small mean-pooled spans.
                "domain_boundary": indices_for_span(domain.end() - 2, domain.end() + 1),
                "reliability_rule_boundary": indices_for_span(
                    reliability.end() - 2, reliability.end() + 1
                ),
                "reliability_r_value": indices_for_span(r_assignment_start, r_assignment_end),
                "reliability_one_minus_r_value": indices_for_span(one_minus_start, one_minus_end),
                **{
                    f"report_{index}_answer": indices_for_span(
                        match.start(1), match.end(1)
                    )
                    for index, match in enumerate(report_matches, start=1)
                },
                "observation_question_boundary": indices_for_span(
                    question_boundary.start(), question_boundary.end()
                ),
                "candidate_1_value": indices_for_span(
                    candidate_question.start(1) - len("s="), candidate_question.end(1)
                ),
                "candidate_2_value": indices_for_span(
                    candidate_question.start(2) - len("s="), candidate_question.end(2)
                ),
                "candidate_question_boundary": indices_for_span(
                    candidate_question.end() - 2, candidate_question.end()
                ),
                "assistant_turn_boundary": indices_for_span(
                    assistant_turn.start(), assistant_turn.end()
                ),
                "final_prompt": [len(input_ids) - 1],
            }
            assert set(sites) == set(PROBE_SITES) - {"answer_line"}
            assert all(
                positions and all(0 <= position < len(input_ids) for position in positions)
                for positions in sites.values()
            )
            return sites


        def unique_base_rows(rows: list[dict[str, object]]) -> dict[tuple[int, int], dict[str, object]]:
            grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
            for row in rows:
                grouped[base_transcript_key(row)].append(row)
            result = {}
            for key, variants in grouped.items():
                if len(variants) != EXPECTED_VARIANTS_PER_BASE_TRANSCRIPT:
                    raise ValueError(f"Base transcript {key} has {len(variants)} variants.")
                if {
                    (row_reliability_exact(row), bool(row["reasoning"])) for row in variants
                } != set(
                    (reliability, reasoning)
                    for reliability in R_EXACT_VALUES
                    for reasoning in REASONING_VALUES
                ):
                    raise ValueError(f"Base transcript {key} is missing a paired factorial cell.")
                invariant_fields = (
                    "membership_sets", "observed_reports", "candidate_1", "candidate_2",
                    "total_agreement_candidate_1", "total_agreement_candidate_2",
                )
                reference = variants[0]
                assert all(
                    row[field] == reference[field]
                    for row in variants for field in invariant_fields
                )
                result[key] = reference
            return result


        def choose_test_schedules(rows: list[dict[str, object]]) -> tuple[tuple[int, ...], dict[str, object]]:
            base_rows = unique_base_rows(rows)
            schedule_counts = {
                schedule: Counter(
                    agreement_cell(row)
                    for (row_schedule, _), row in base_rows.items()
                    if row_schedule == schedule
                )
                for schedule in range(NUM_QUESTION_SETS)
            }
            global_counts = sum(schedule_counts.values(), Counter())
            assert sum(global_counts.values()) == NUM_QUESTION_SETS * NUM_ANSWER_PATTERNS

            full_total = sum(global_counts.values())

            def evaluate(selected: tuple[int, ...]):
                test_counts = sum((schedule_counts[index] for index in selected), Counter())
                test_total = sum(test_counts.values())
                missing_ties = sorted(TIE_CELLS - set(test_counts))
                deviations = [
                    abs(test_counts[cell] / test_total - global_counts[cell] / full_total)
                    for cell in ALL_AGREEMENT_CELLS
                ]
                score = (
                    len(missing_ties),
                    sum(deviations),
                    max(deviations),
                    -len(test_counts),
                    selected,
                )
                return score, test_counts, missing_ties

            rng = np.random.default_rng(SEED + 1)
            sampled = {
                tuple(sorted(int(value) for value in rng.choice(
                    NUM_QUESTION_SETS, size=TEST_SCHEDULE_COUNT, replace=False
                )))
                for _ in range(SPLIT_SEARCH_CANDIDATES)
            }
            sampled.add(tuple(range(TEST_SCHEDULE_COUNT)))
            globally_available_ties = TIE_CELLS & set(global_counts)
            if PREFER_ALL_TIE_CELLS_IN_TEST:
                # If each tie cell occurs anywhere, one witness schedule per cell gives a
                # cover of size at most four; fill deterministically to eight schedules.
                tie_cover = set()
                covered_ties = set()
                for tie_cell in sorted(globally_available_ties):
                    if tie_cell in covered_ties:
                        continue
                    witness = next(
                        schedule for schedule in range(NUM_QUESTION_SETS)
                        if schedule_counts[schedule][tie_cell] > 0
                    )
                    tie_cover.add(witness)
                    covered_ties.update(
                        TIE_CELLS & set(schedule_counts[witness])
                    )
                tie_cover.update(
                    schedule for schedule in range(NUM_QUESTION_SETS)
                    if len(tie_cover) < TEST_SCHEDULE_COUNT
                )
                sampled.add(tuple(sorted(tie_cover)))
            selected = min(sampled, key=lambda candidate: evaluate(candidate)[0])

            # Deterministic one-swap refinement of the best sampled subset.
            while True:
                current_score = evaluate(selected)[0]
                selected_set = set(selected)
                neighbors = (
                    tuple(sorted(selected_set - {removed} | {added}))
                    for removed in selected
                    for added in range(NUM_QUESTION_SETS)
                    if added not in selected_set
                )
                improved = min(neighbors, key=lambda candidate: evaluate(candidate)[0])
                if evaluate(improved)[0] >= current_score:
                    break
                selected = improved

            score, test_counts, missing_ties = evaluate(selected)
            audit = {
                "reasoning_modes_share_schedule_partition": True,
                "random_subsets_requested": SPLIT_SEARCH_CANDIDATES,
                "unique_subsets_evaluated": len(sampled),
                "selection_rule": [
                    "fewest_missing_tie_cells",
                    "minimum_l1_cell_proportion_distance",
                    "minimum_worst_cell_proportion_deviation",
                    "maximum_number_of_covered_cells",
                    "lexicographic_schedule_indices",
                ],
                "selected_test_schedules": list(selected),
                "score": {
                    "missing_tie_cell_count": score[0],
                    "l1_cell_proportion_distance": score[1],
                    "worst_cell_proportion_deviation": score[2],
                    "covered_cell_count": -score[3],
                },
                "missing_tie_cells": [list(cell) for cell in missing_ties],
                "globally_available_tie_cells": [
                    list(cell) for cell in sorted(globally_available_ties)
                ],
                "globally_unavailable_tie_cells": [
                    list(cell) for cell in sorted(TIE_CELLS - globally_available_ties)
                ],
                "test_base_cell_counts": {
                    f"{cell[0]},{cell[1]}": test_counts[cell] for cell in ALL_AGREEMENT_CELLS
                },
                "all_base_cell_counts": {
                    f"{cell[0]},{cell[1]}": global_counts[cell] for cell in ALL_AGREEMENT_CELLS
                },
            }
            return selected, audit
        """,
        "split-and-label-helpers",
    ),
    code(
        r"""
        # Tokenizer loading and dataset construction are CPU-only. No Qwen weights are loaded here.
        hf_config = AutoConfig.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            local_files_only=LOCAL_FILES_ONLY,
        )
        hf_text_config = getattr(hf_config, "text_config", hf_config)
        if (
            int(hf_text_config.num_hidden_layers) != NUM_LAYERS
            or int(hf_text_config.hidden_size) != HIDDEN_SIZE
            or hf_text_config.dtype != torch.bfloat16
        ):
            raise ValueError(
                "The checkpoint text configuration does not match the layer/width/BF16 "
                "assumptions used by the activation-storage preflight."
            )
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            local_files_only=LOCAL_FILES_ONLY,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer_binding = TokenizerBinding(tokenizer, enable_thinking=False)

        environments = tuple(
            NoisyChannelBayesianEnvironment(
                n=N,
                k=K,
                r_values=reliability,
                control_positional_bias=CONTROL_POSITIONAL_BIAS,
            )
            for reliability in R_VALUES
        )
        probes = tuple(
            XVsYPosteriorProbe(
                x=X,
                y=Y,
                reasoning=reasoning,
                allow_same=ALLOW_SAME,
                call_layout="conversation",
            )
            for reasoning in REASONING_VALUES
        )

        raw_dataset = TranscriptDatasetGenerator(
            environment=environments,
            question=RandomSubsetQuestion(
                subset_size=SUBSET_SIZE, replacement=False, sort=True
            ),
            probe=probes,
            tokenizer_binding=tokenizer_binding,
            system_prompt=SystemPrompt(SYSTEM_PROMPT_TEXT),
            seed=SEED,
        ).generate(num_question_sets=NUM_QUESTION_SETS)
        assert len(raw_dataset) == EXPECTED_ROWS
        assert raw_dataset.manifest["presentations_per_scenario"] == 1
        assert raw_dataset.manifest["control_positional_bias"] is False

        raw_rows = [dict(row) for row in raw_dataset]
        selected_test_schedules, split_audit = choose_test_schedules(raw_rows)
        test_schedule_set = set(selected_test_schedules)

        enriched_rows = []
        for source in raw_rows:
            row = dict(source)
            row.update(target_fields(row))
            row["base_transcript_id"] = (
                f"schedule_{int(row['question_set_index']):02d}_"
                f"pattern_{int(row['answer_pattern_index']):02d}"
            )
            row["split"] = (
                "test" if int(row["question_set_index"]) in test_schedule_set else "train"
            )
            row["reasoning_mode"] = MODE_NAMES[bool(row["reasoning"])]
            row["prompt_token_sites"] = prompt_token_sites(row, tokenizer)
            prompt_capture_indices = sorted({
                position
                for positions in row["prompt_token_sites"].values()
                for position in positions
            })
            row["activation_token_selector"] = [
                *prompt_capture_indices,
                *range(-ANSWER_TAIL_TOKENS, 0),
            ]
            enriched_rows.append(row)

        # The two prompt variants intentionally have equal token length for every paired
        # schedule/pattern/reliability cell. Semantic sites are still located independently:
        # equal total length does not imply that internal locations share absolute indices.
        paired_prompt_lengths: dict[tuple[int, int, str], dict[bool, int]] = defaultdict(dict)
        for row in enriched_rows:
            pair_key = (
                int(row["question_set_index"]),
                int(row["answer_pattern_index"]),
                row_reliability_exact(row),
            )
            paired_prompt_lengths[pair_key][bool(row["reasoning"])] = len(row["input_ids"])
        if len(paired_prompt_lengths) != NUM_QUESTION_SETS * NUM_ANSWER_PATTERNS * len(R_VALUES):
            raise ValueError("Prompt-length audit has the wrong number of paired cells.")
        prompt_length_mismatches = {
            key: lengths for key, lengths in paired_prompt_lengths.items()
            if set(lengths) != set(REASONING_VALUES)
            or lengths[False] != lengths[True]
        }
        if prompt_length_mismatches:
            examples = list(prompt_length_mismatches.items())[:5]
            raise ValueError(
                "Reasoning-off/on prompts are not token-length matched; examples="
                f"{examples}"
            )

        train_rows = [row for row in enriched_rows if row["split"] == "train"]
        test_rows = [row for row in enriched_rows if row["split"] == "test"]
        assert len(train_rows) == EXPECTED_TRAIN_ROWS
        assert len(test_rows) == EXPECTED_TEST_ROWS
        assert {int(row["question_set_index"]) for row in train_rows}.isdisjoint(
            int(row["question_set_index"]) for row in test_rows
        )
        assert {str(row["base_transcript_id"]) for row in train_rows}.isdisjoint(
            str(row["base_transcript_id"]) for row in test_rows
        )
        for reasoning in REASONING_VALUES:
            mode_train = [row for row in train_rows if bool(row["reasoning"]) is reasoning]
            mode_test = [row for row in test_rows if bool(row["reasoning"]) is reasoning]
            assert len(mode_train) == EXPECTED_TRAIN_ROWS_PER_REASONING
            assert len(mode_test) == EXPECTED_TEST_ROWS_PER_REASONING
            assert {int(row["question_set_index"]) for row in mode_train}.isdisjoint(
                int(row["question_set_index"]) for row in mode_test
            )
        if PREFER_ALL_TIE_CELLS_IN_TEST:
            globally_available_ties = TIE_CELLS & {
                agreement_cell(row) for row in enriched_rows
            }
            assert globally_available_ties <= {
                agreement_cell(row) for row in test_rows
            }

        # Full prompt prefixes must also be distinct across the schedule split.
        train_prompt_hashes = {
            hashlib.sha256(bytes().join(int(token).to_bytes(4, "little") for token in row["input_ids"])).hexdigest()
            for row in train_rows
        }
        test_prompt_hashes = {
            hashlib.sha256(bytes().join(int(token).to_bytes(4, "little") for token in row["input_ids"])).hexdigest()
            for row in test_rows
        }
        assert train_prompt_hashes.isdisjoint(test_prompt_hashes)

        dataset_manifest = {
            **raw_dataset.manifest,
            "notebook": "16_noisy_channel_bayesian_activation_patching.ipynb",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "expected_row_count": EXPECTED_ROWS,
            "split_policy": split_audit,
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "row_count_per_reasoning_mode": EXPECTED_ROWS_PER_REASONING,
            "train_row_count_per_reasoning_mode": EXPECTED_TRAIN_ROWS_PER_REASONING,
            "test_row_count_per_reasoning_mode": EXPECTED_TEST_ROWS_PER_REASONING,
            "reasoning_modes_share_schedule_partition": True,
            "paired_reasoning_prompts_have_equal_token_length": True,
            "activation_capture": {
                "stream": ACTIVATION_STREAM,
                "layers": ACTIVATION_LAYERS,
                "tokens": ACTIVATION_TOKENS,
                "num_layers": NUM_LAYERS,
                "hidden_size": HIDDEN_SIZE,
                "dtype": "bfloat16",
            },
        }
        dataset = TranscriptDataset(
            enriched_rows, manifest=dataset_manifest, experiment_dir=DATASET_ROOT
        )
        dataset.save(DATASET_ROOT)
        datasets_by_reasoning = {}
        for reasoning in REASONING_VALUES:
            mode_rows = [
                dict(row) for row in enriched_rows if bool(row["reasoning"]) is reasoning
            ]
            mode_root = DATASET_ROOT / MODE_NAMES[reasoning]
            mode_manifest = {
                **dataset_manifest,
                "reasoning": reasoning,
                "reasoning_mode": MODE_NAMES[reasoning],
                "expected_row_count": EXPECTED_ROWS_PER_REASONING,
                "train_row_count": EXPECTED_TRAIN_ROWS_PER_REASONING,
                "test_row_count": EXPECTED_TEST_ROWS_PER_REASONING,
            }
            mode_dataset = TranscriptDataset(
                mode_rows, manifest=mode_manifest, experiment_dir=mode_root
            )
            mode_dataset.save(mode_root)
            datasets_by_reasoning[reasoning] = mode_dataset

        dataset_split_fingerprint = hashlib.sha256(json.dumps(
            [(str(row["row_id"]), str(row["split"])) for row in enriched_rows],
            separators=(",", ":"),
        ).encode()).hexdigest()
        probe_plan = {
            "schema_version": 1,
            "dataset_split_fingerprint": dataset_split_fingerprint,
            "activation_run_ids": {
                MODE_NAMES[reasoning]: RUN_IDS[reasoning]
                for reasoning in REASONING_VALUES
            },
            "model_revision": MODEL_REVISION,
            "activation_stream": ACTIVATION_STREAM,
            "activation_tokens": ACTIVATION_TOKENS,
            "reasoning_modes": [MODE_NAMES[value] for value in REASONING_VALUES],
            "separate_probe_per_reasoning_mode": True,
            "sites": list(PROBE_SITES),
            "continuous_targets": list(CONTINUOUS_PROBE_TARGETS),
            "binary_targets": list(BINARY_PROBE_TARGETS),
            "layers": list(PROBE_LAYERS),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "ridge_solver": RIDGE_SOLVER,
            "ridge_tolerance": RIDGE_TOL,
            "ridge_max_iterations": RIDGE_MAX_ITER,
            "logistic_c_values": list(LOGISTIC_C_VALUES),
            "cv_folds": PROBE_CV_FOLDS,
            "top_k_layers": PROBE_TOP_K_LAYERS,
            "minimum_generated_site_coverage": MIN_GENERATED_SITE_COVERAGE,
            "minimum_probe_train_rows": MIN_PROBE_TRAIN_ROWS,
        }
        PROBE_CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(
            probe_plan, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        probe_plan["probe_config_fingerprint"] = PROBE_CONFIG_FINGERPRINT
        atomic_write_json(PROBE_ROOT / "probe_plan.json", probe_plan)

        split_assignments = [
            {
                "row_id": row["row_id"],
                "base_transcript_id": row["base_transcript_id"],
                "question_set_index": row["question_set_index"],
                "answer_pattern_index": row["answer_pattern_index"],
                "reliability_exact": row_reliability_exact(row),
                "reasoning": row["reasoning"],
                "reasoning_mode": row["reasoning_mode"],
                "agreement_c1": row["agreement_c1"],
                "agreement_c2": row["agreement_c2"],
                "split": row["split"],
            }
            for row in enriched_rows
        ]
        atomic_write_json(SPLIT_ROOT / "split_manifest.json", split_audit)
        atomic_write_jsonl(SPLIT_ROOT / "split_assignments.jsonl", split_assignments)

        print({
            "dataset_rows": len(dataset),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "test_schedules": selected_test_schedules,
            "missing_test_tie_cells": split_audit["missing_tie_cells"],
            "test_cell_coverage": split_audit["score"]["covered_cell_count"],
        })
        """,
        "generate-and-split",
    ),
    md(
        r"""
        ### Split audit

        The eight held-out schedules contribute 64 base transcripts and are selected to balance all
        16 agreement cells. They cover every tie construction that occurs anywhere in the generated
        schedules (normally all four). The tables below show
        combined-mode counts after expanding each base transcript across ten reliabilities and two
        reasoning modes. The schedule partition itself is identical in both modes.
        """,
        "split-audit-explanation",
    ),
    code(
        r"""
        def expanded_cell_table(rows: list[dict[str, object]], split: str) -> pd.DataFrame:
            counts = Counter(agreement_cell(row) for row in rows if row["split"] == split)
            return pd.DataFrame(
                [[counts[(a1, a2)] for a2 in range(K + 1)] for a1 in range(K + 1)],
                index=pd.Index(range(K + 1), name="agreement C1"),
                columns=pd.Index(range(K + 1), name="agreement C2"),
            )


        train_cell_table = expanded_cell_table(enriched_rows, "train")
        test_cell_table = expanded_cell_table(enriched_rows, "test")
        display(Markdown("#### Training rows"))
        display(train_cell_table)
        display(Markdown("#### Test rows"))
        display(test_cell_table)

        tie_test_counts = {
            cell: int(test_cell_table.loc[cell[0], cell[1]]) for cell in sorted(TIE_CELLS)
        }
        globally_available_ties = TIE_CELLS & {
            agreement_cell(row) for row in enriched_rows
        }
        assert all(tie_test_counts[cell] > 0 for cell in globally_available_ties)
        print("Test tie-cell counts:", tie_test_counts)
        """,
        "display-split-audit",
    ),
    md(
        r"""
        ## Storage preflight

        One BF16 residual vector costs $4096\times2$ bytes. Across 32 layers this is 256 KiB per
        stored token per row. The sparse selector stores only the union of the semantic prompt
        spans plus the last `ANSWER_TAIL_TOKENS` completion positions. The reasoning-on generation
        may contain up to 4,096 tokens, but unselected activations are never written to disk.

        The preflight uses every row's requested selector size as a conservative upper bound;
        prompt/tail overlaps are de-duplicated during capture. It runs before constructing `QwenRunner`
        and refuses to load the model if either the 90%-of-budget payload limit or the free-disk
        reserve would be violated.

        Safetensors headers and JSON manifests are not included in the tensor-payload formula, which
        is why only 90% of the nominal 160 GiB budget is assignable to tensor payloads.
        """,
        "storage-rationale",
    ),
    code(
        r"""
        GIB = 1024**3
        selected_token_counts = [
            len(row["activation_token_selector"]) for row in enriched_rows
        ]
        worst_case_token_total = sum(selected_token_counts)
        bytes_per_token_per_row = NUM_LAYERS * HIDDEN_SIZE * ACTIVATION_BYTES_PER_ELEMENT
        worst_case_activation_bytes = worst_case_token_total * bytes_per_token_per_row
        worst_case_activation_gib = worst_case_activation_bytes / GIB
        allowed_payload_gib = (
            ACTIVATION_STORAGE_BUDGET_GIB * ACTIVATION_BUDGET_SAFETY_FRACTION
        )
        existing_activation_bytes = sum(
            path.stat().st_size
            for run_dir in RUN_DIRS.values()
            for path in (run_dir / "activations").glob("*.safetensors")
        )
        free_disk_gib = shutil.disk_usage(EXPERIMENT_ROOT.parent).free / GIB
        # Existing files under this exact run ID are overwritten atomically on a retry, so they
        # count as reclaimable target capacity. The reserve covers one temporary row copy.
        effective_target_capacity_gib = free_disk_gib + existing_activation_bytes / GIB

        storage_preflight = {
            "minimum_prompt_tokens": min(len(row["input_ids"]) for row in enriched_rows),
            "maximum_prompt_tokens": max(len(row["input_ids"]) for row in enriched_rows),
            "mean_prompt_tokens": float(np.mean([len(row["input_ids"]) for row in enriched_rows])),
            "maximum_completion_tokens_by_reasoning": {
                MODE_NAMES[reasoning]: MAX_COMPLETION_TOKENS_BY_REASONING[reasoning]
                for reasoning in REASONING_VALUES
            },
            "minimum_selected_activation_positions": min(selected_token_counts),
            "maximum_selected_activation_positions": max(selected_token_counts),
            "mean_selected_activation_positions": float(np.mean(selected_token_counts)),
            "bytes_per_token_across_saved_layers": bytes_per_token_per_row,
            "worst_case_activation_payload_gib": worst_case_activation_gib,
            "allowed_activation_payload_gib": allowed_payload_gib,
            "existing_activation_files_gib": existing_activation_bytes / GIB,
            "free_disk_gib_before_capture": free_disk_gib,
            "effective_target_capacity_gib": effective_target_capacity_gib,
            "required_free_disk_reserve_gib": MIN_FREE_DISK_AFTER_CAPTURE_GIB,
        }
        display(storage_preflight)

        assert worst_case_activation_gib <= allowed_payload_gib, (
            "Sparse activation payload exceeds the safety-adjusted budget. Reduce selected "
            "token spans or the factorial before loading Qwen."
        )
        assert effective_target_capacity_gib >= (
            worst_case_activation_gib + MIN_FREE_DISK_AFTER_CAPTURE_GIB
        ), (
            "Insufficient current free disk for the worst-case activation payload plus reserve."
        )
        atomic_write_json(MODEL_ROOT / "storage_preflight.json", storage_preflight)
        """,
        "storage-preflight",
    ),
    md(
        r"""
        ## GPU capture — deliberately gated

        `RUN_GPU_CAPTURE=True` performs one continuous generation per row, then teacher-forces the
        exact prompt plus generated sequence and saves `resid_post` for only the selected semantic
        positions at all 32 layers. Capture batch size is one. Candidate answer logits are stored as two JSON floats;
        no full-vocabulary logit tensor is persisted.

        The custom runner is used for generation/capture because it already provides atomic
        per-row safetensors and exact answer-boundary bookkeeping. The notebook calls it on
        contiguous eight-row prefixes, making `results.jsonl` a durable prefix checkpoint. A
        fully completed run can be loaded without recomputation; after interruption, at most the
        unfinished checkpoint needs to be regenerated and recaptured. TransformerLens's
        `TransformerBridge` is used later for interventions. Before any patch is accepted, the
        notebook checks that an unpatched bridge forward reproduces the runner's candidate margin.
        """,
        "capture-rationale",
    ),
    code(
        r"""
        RESULT_IDENTITY_FIELDS = (
            "row_id",
            "question_set_index",
            "answer_pattern_index",
            "reliabilities_exact",
            "reasoning",
            "reasoning_mode",
            "split",
            "base_transcript_id",
            "agreement_c1",
            "agreement_c2",
            "delta_a",
            "gain",
            "gain_abs",
            "reliability_sign",
            "z_bayes",
            "z_heuristic",
            "serialized_prompt",
            "input_ids",
            "prompt_token_sites",
            "activation_token_selector",
        )


        def validate_saved_rows_against_dataset(
            rows: list[dict[str, object]], *, reasoning: bool, require_complete: bool
        ) -> None:
            configured_rows = list(datasets_by_reasoning[reasoning])
            if len(rows) > len(configured_rows):
                raise ValueError("Saved results are longer than the configured dataset.")
            if require_complete and len(rows) != len(configured_rows):
                raise ValueError("Saved results are not a complete configured dataset.")
            for index, (saved, configured) in enumerate(
                zip(rows, configured_rows, strict=False)
            ):
                for field in RESULT_IDENTITY_FIELDS:
                    saved_value = json.dumps(saved.get(field), sort_keys=True, allow_nan=False)
                    configured_value = json.dumps(
                        configured.get(field), sort_keys=True, allow_nan=False
                    )
                    if saved_value != configured_value:
                        raise ValueError(
                            f"Saved row {index} differs from the configured dataset at "
                            f"{field!r}; refusing to mix stale captures or split labels."
                        )


        def validate_run_manifest_configuration(
            manifest: dict[str, object], *, reasoning: bool
        ) -> None:
            if manifest.get("model", {}).get("model_name_or_path") != MODEL_ID:
                raise ValueError("Saved run belongs to a different model.")
            if manifest.get("model", {}).get("revision") != MODEL_REVISION:
                raise ValueError("Saved run belongs to a different model revision.")
            execution = manifest.get("execution", {})
            saved_capture = execution.get("capture", {})
            expected_capture = {
                "streams": list(capture_spec.streams),
                "layers": capture_spec.layers,
                "tokens": capture_spec.tokens,
            }
            actual_capture = {
                "streams": saved_capture.get("streams"),
                "layers": saved_capture.get("layers"),
                "tokens": saved_capture.get("tokens"),
            }
            if actual_capture != expected_capture:
                raise ValueError("Saved run uses a different activation-capture configuration.")
            if (
                execution.get("max_completion_tokens")
                != MAX_COMPLETION_TOKENS_BY_REASONING[reasoning]
            ):
                raise ValueError("Saved run uses a different completion-token cap.")


        def load_completed_run(reasoning: bool) -> TranscriptDataset:
            run_dir = RUN_DIRS[reasoning]
            run_manifest_path = run_dir / "run_manifest.json"
            if not run_manifest_path.exists():
                raise FileNotFoundError(
                    f"No completed run at {run_manifest_path}. Run capture first."
                )
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if manifest.get("row_count") != EXPECTED_ROWS_PER_REASONING:
                raise ValueError("Saved reasoning-mode run has the wrong row count.")
            validate_run_manifest_configuration(manifest, reasoning=reasoning)
            results_path = run_dir / str(manifest.get("results_file", "results.jsonl"))
            rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            validate_saved_rows_against_dataset(
                rows, reasoning=reasoning, require_complete=True
            )
            return TranscriptDataset(rows, manifest=manifest, experiment_dir=MODEL_ROOT)


        def completed_capture_prefix_length(reasoning: bool) -> int:
            # Validate and return the durable contiguous prefix in results.jsonl.

            run_dir = RUN_DIRS[reasoning]
            results_path = run_dir / "results.jsonl"
            if not results_path.exists():
                return 0
            rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            validate_saved_rows_against_dataset(
                rows, reasoning=reasoning, require_complete=False
            )
            run_manifest_path = run_dir / "run_manifest.json"
            if run_manifest_path.exists():
                validate_run_manifest_configuration(
                    json.loads(run_manifest_path.read_text(encoding="utf-8")),
                    reasoning=reasoning,
                )
            for row in rows:
                relative = row.get("activation_path")
                if not isinstance(relative, str) or not (run_dir / relative).is_file():
                    raise FileNotFoundError(
                        f"Checkpointed row {row['row_id']} is missing its activation file."
                    )
            return len(rows)


        def capture_checkpoint_ends(durable_rows: int) -> tuple[int, ...]:
            if not 0 <= durable_rows <= EXPECTED_ROWS_PER_REASONING:
                raise ValueError("Durable row count is outside the configured dataset.")
            ends = []
            cursor = durable_rows
            while cursor < EXPECTED_ROWS_PER_REASONING:
                cursor = min(
                    cursor + CAPTURE_CHECKPOINT_ROWS, EXPECTED_ROWS_PER_REASONING
                )
                ends.append(cursor)
            return tuple(ends)


        assert capture_checkpoint_ends(0)[0] == CAPTURE_CHECKPOINT_ROWS
        assert capture_checkpoint_ends(EXPECTED_ROWS_PER_REASONING - 1) == (
            EXPECTED_ROWS_PER_REASONING,
        )
        assert capture_checkpoint_ends(EXPECTED_ROWS_PER_REASONING) == ()


        results_by_reasoning: dict[bool, TranscriptDataset] = {}
        results: TranscriptDataset | None = None
        if RUN_GPU_CAPTURE and LOAD_COMPLETED_CAPTURE:
            raise ValueError("Choose either RUN_GPU_CAPTURE or LOAD_COMPLETED_CAPTURE, not both.")

        if RUN_GPU_CAPTURE:
            # The preflight assertions above have already run. This is the first model-loading line.
            runner = QwenRunner(ModelConfig(
                model_name_or_path=MODEL_ID,
                revision=MODEL_REVISION,
                dtype=MODEL_DTYPE,
                device_map=DEVICE_MAP,
                local_files_only=LOCAL_FILES_ONLY,
            ))
            for reasoning in REASONING_VALUES:
                mode_name = MODE_NAMES[reasoning]
                mode_dataset = datasets_by_reasoning[reasoning]
                execution = ExecutionConfig(
                    experiment_dir=MODEL_ROOT,
                    run_id=RUN_IDS[reasoning],
                    batch_size=COMPLETION_BATCH_SIZE,
                    completion_batch_size=COMPLETION_BATCH_SIZE,
                    capture_batch_size=CAPTURE_BATCH_SIZE,
                    score_batch_size=SCORE_BATCH_SIZE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS_BY_REASONING[reasoning],
                    resume=True,
                    metrics=metric_spec,
                    capture=capture_spec,
                    completion_mtp=SGLangMTPConfig(enabled=ENABLE_MTP),
                )
                durable_rows = completed_capture_prefix_length(reasoning)
                print(
                    datetime.now(timezone.utc).isoformat(),
                    f"{mode_name}: starting/resuming after {durable_rows} durable rows",
                )
                dataset_rows = list(mode_dataset)
                checkpoint_ends = capture_checkpoint_ends(durable_rows)
                mode_results = (
                    load_completed_run(reasoning)
                    if durable_rows == EXPECTED_ROWS_PER_REASONING else None
                )
                for checkpoint_end in checkpoint_ends:
                    prefix_dataset = TranscriptDataset(
                        dataset_rows[:checkpoint_end],
                        manifest=mode_dataset.manifest,
                        experiment_dir=MODEL_ROOT,
                    )
                    mode_results = prefix_dataset.execute(runner, execution)
                    if len(mode_results) != checkpoint_end:
                        raise RuntimeError(
                            "Capture checkpoint did not finalize its complete prefix."
                        )
                    print(
                        datetime.now(timezone.utc).isoformat(),
                        f"{mode_name}: durable checkpoint "
                        f"{checkpoint_end}/{EXPECTED_ROWS_PER_REASONING}",
                    )
                if mode_results is None or len(mode_results) != EXPECTED_ROWS_PER_REASONING:
                    raise RuntimeError(f"GPU capture ended without all {mode_name} rows.")
                results_by_reasoning[reasoning] = mode_results
            del runner
            gc.collect()
            torch.cuda.empty_cache()
        elif LOAD_COMPLETED_CAPTURE:
            results_by_reasoning = {
                reasoning: load_completed_run(reasoning)
                for reasoning in REASONING_VALUES
            }
        else:
            print(
                "GPU capture is paused. Set RUN_GPU_CAPTURE=True only when you are ready; "
                "after completion, use LOAD_COMPLETED_CAPTURE=True."
            )

        if len(results_by_reasoning) == len(REASONING_VALUES):
            result_by_id = {
                str(row["row_id"]): dict(row)
                for mode_results in results_by_reasoning.values()
                for row in mode_results
            }
            results = TranscriptDataset(
                [result_by_id[str(row["row_id"])] for row in enriched_rows],
                manifest={"schema_version": dataset.manifest["schema_version"]},
                experiment_dir=MODEL_ROOT,
            )
        """,
        "run-or-load-capture",
    ),
    code(
        r"""
        def validate_activation_inventory(
            result_rows: TranscriptDataset, *, reasoning: bool
        ) -> dict[str, object]:
            if len(result_rows) != EXPECTED_ROWS_PER_REASONING:
                raise ValueError("Mode-specific activation inventory has the wrong row count.")
            if {bool(row["reasoning"]) for row in result_rows} != {reasoning}:
                raise ValueError("Mode-specific activation inventory mixes reasoning modes.")
            expected_keys = {
                f"answer.{ACTIVATION_STREAM}.layer_{layer}" for layer in range(NUM_LAYERS)
            }
            total_bytes = 0
            sequence_lengths = []
            first_dtype = None
            answer_line_available_rows = 0
            for row_index, row in enumerate(result_rows):
                if row.get("generation_serialized_prompt") != row.get("serialized_prompt"):
                    raise ValueError(f"Runtime prompt text drifted for row {row['row_id']}.")
                generation_ids = [int(token) for token in row.get("generation_input_ids", [])]
                dataset_ids = [int(token) for token in row.get("input_ids", [])]
                if generation_ids != dataset_ids:
                    raise ValueError(f"Runtime prompt token IDs drifted for row {row['row_id']}.")
                teacher_forced_ids = [
                    int(token) for token in row.get("teacher_forced_input_ids", [])
                ]
                if teacher_forced_ids[: len(dataset_ids)] != dataset_ids:
                    raise ValueError(f"Teacher-forced prompt prefix drifted for row {row['row_id']}.")
                if (
                    row.get("runtime_tokenizer_template_fingerprint")
                    != tokenizer_binding.fingerprint
                ):
                    raise ValueError(f"Runtime tokenizer/template drifted for row {row['row_id']}.")
                relative = row.get("activation_path")
                if not isinstance(relative, str):
                    raise ValueError(f"Row {row['row_id']} has no activation_path.")
                path = RUN_DIRS[reasoning] / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                total_bytes += path.stat().st_size
                captured_indices = [
                    int(value) for value in row.get("activation_token_indices", [])
                ]
                if not captured_indices or len(captured_indices) != len(set(captured_indices)):
                    raise ValueError(f"Row {row['row_id']} lacks unique sparse capture indices.")
                if captured_indices != sorted(captured_indices):
                    raise ValueError(f"Row {row['row_id']} sparse indices are not canonical.")
                if not all(0 <= value < len(teacher_forced_ids) for value in captured_indices):
                    raise ValueError(f"Row {row['row_id']} has an out-of-range sparse index.")
                prompt_positions = {
                    int(position)
                    for positions in row["prompt_token_sites"].values()
                    for position in positions
                }
                if not prompt_positions <= set(captured_indices):
                    raise ValueError(f"Row {row['row_id']} is missing a semantic prompt site.")
                line_start = row.get("answer_line_generated_token_start")
                line_end = row.get("answer_line_generated_token_end")
                completion_start = row.get("teacher_forced_completion_start")
                if line_start is not None and line_end is not None and completion_start is not None:
                    answer_positions = set(range(
                        int(completion_start) + int(line_start),
                        int(completion_start) + int(line_end),
                    ))
                    if answer_positions and answer_positions <= set(captured_indices):
                        answer_line_available_rows += 1
                sequence_lengths.append(len(row["teacher_forced_input_ids"]))
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != expected_keys:
                        raise ValueError(f"Unexpected stream/layer keys in {path}.")
                    for key in expected_keys:
                        shape = tuple(handle.get_slice(key).get_shape())
                        if shape != (len(captured_indices), HIDDEN_SIZE):
                            raise ValueError(f"Unexpected shape {shape} for {path}:{key}.")
                    if row_index == 0:
                        first_dtype = str(handle.get_tensor(sorted(expected_keys)[0]).dtype)
            if first_dtype != "torch.bfloat16":
                raise ValueError(f"Expected BF16 safetensors, found {first_dtype}.")
            return {
                "row_count": len(result_rows),
                "reasoning": reasoning,
                "reasoning_mode": MODE_NAMES[reasoning],
                "stream_count": 1,
                "layer_count": NUM_LAYERS,
                "dtype": first_dtype,
                "minimum_sequence_tokens": min(sequence_lengths),
                "maximum_sequence_tokens": max(sequence_lengths),
                "minimum_captured_positions": min(
                    len(row["activation_token_indices"]) for row in result_rows
                ),
                "maximum_captured_positions": max(
                    len(row["activation_token_indices"]) for row in result_rows
                ),
                "answer_line_available_rows": answer_line_available_rows,
                "answer_line_coverage_fraction": (
                    answer_line_available_rows / EXPECTED_ROWS_PER_REASONING
                ),
                "activation_file_bytes": total_bytes,
                "activation_file_gib": total_bytes / GIB,
            }


        def save_activation_split_indices(
            result_rows: TranscriptDataset, *, reasoning: bool
        ) -> dict[str, int]:
            indices: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
            for row in result_rows:
                split = str(row["split"])
                if split not in indices:
                    raise ValueError(f"Unexpected activation split {split!r}.")
                indices[split].append({
                    "row_id": str(row["row_id"]),
                    "base_transcript_id": str(row["base_transcript_id"]),
                    "question_set_index": int(row["question_set_index"]),
                    "answer_pattern_index": int(row["answer_pattern_index"]),
                    "activation_path": str(row["activation_path"]),
                    "activation_run_id": RUN_IDS[reasoning],
                    "reasoning": reasoning,
                    "split": split,
                    "dataset_split_fingerprint": dataset_split_fingerprint,
                })
            if len(indices["train"]) != EXPECTED_TRAIN_ROWS_PER_REASONING:
                raise ValueError("Mode-specific training activation index has the wrong size.")
            if len(indices["test"]) != EXPECTED_TEST_ROWS_PER_REASONING:
                raise ValueError("Mode-specific test activation index has the wrong size.")
            train_paths = {row["activation_path"] for row in indices["train"]}
            test_paths = {row["activation_path"] for row in indices["test"]}
            if (
                len(train_paths) != EXPECTED_TRAIN_ROWS_PER_REASONING
                or len(test_paths) != EXPECTED_TEST_ROWS_PER_REASONING
            ):
                raise ValueError("Activation paths are not unique within each split.")
            if not train_paths.isdisjoint(test_paths):
                raise ValueError("An activation safetensor appears in both train and test indices.")
            train_schedules = {row["question_set_index"] for row in indices["train"]}
            test_schedules = {row["question_set_index"] for row in indices["test"]}
            if not train_schedules.isdisjoint(test_schedules):
                raise ValueError("A question schedule appears in both activation partitions.")
            for split, rows in indices.items():
                atomic_write_jsonl(
                    SPLIT_ROOT / MODE_NAMES[reasoning] / f"{split}_activations.jsonl",
                    rows,
                )
            return {split: len(rows) for split, rows in indices.items()}


        if results is not None:
            capture_inventory = {}
            for reasoning in REASONING_VALUES:
                inventory = validate_activation_inventory(
                    results_by_reasoning[reasoning], reasoning=reasoning
                )
                inventory["activation_split_rows"] = save_activation_split_indices(
                    results_by_reasoning[reasoning], reasoning=reasoning
                )
                atomic_write_json(
                    RUN_DIRS[reasoning] / "activation_inventory.json", inventory
                )
                capture_inventory[MODE_NAMES[reasoning]] = inventory
            actual_capture_gib = sum(
                float(inventory["activation_file_gib"])
                for inventory in capture_inventory.values()
            )
            if actual_capture_gib > ACTIVATION_STORAGE_BUDGET_GIB:
                raise RuntimeError("Actual sparse activation files exceed the declared budget.")
            display(capture_inventory)
        else:
            capture_inventory = None
        """,
        "capture-inventory",
    ),
    md(
        r"""
        ## Linear probes

        Qwen remains frozen. For a residual vector $h\in\mathbb{R}^{4096}$, a continuous probe is

        $$\hat y=w^\top\operatorname{standardize}(h)+b,$$

        fitted with ridge regression. At a fixed site and layer, each reasoning mode has its own
        design matrix with 4,480 training transcripts and 4,096 columns. This gives more rows than
        features, while ridge still controls multicollinearity by minimizing

        $$\lVert y-Xw-b\rVert_2^2+\alpha\lVert w\rVert_2^2.$$

        Grouped cross-validation chooses $\alpha$ from `RIDGE_ALPHAS`. The iterative `lsqr` solver
        avoids unstable direct solves of the highly correlated residual features. Reliability sign uses
        L2-regularized logistic regression and chooses inverse regularization `C`. Feature
        normalization is part of the scikit-learn pipeline, so its mean and scale are refitted
        inside every training fold rather than computed globally. The eight held-out schedules never
        participate in normalization, hyperparameter choice, layer selection, or probe fitting.

        Reasoning-off and reasoning-on probes are fit, selected, saved, and tested independently.
        No activation from one reasoning mode can enter the other mode's scaler, CV folds, or fit.

        No gradients pass through Qwen, and Qwen's weights never change. For every
        site/layer/target combination, only the probe's $w$ and $b$ are learned. This makes the
        experiment a test of linear decodability, not model fine-tuning.

        The 14 preregistered sites cover the domain and reliability sentence boundaries, both
        displayed reliability values, all three reported answers, the observation/question gap,
        both `s=value` candidate spans and the question boundary, the assistant turn boundary, the final prompt
        token, and the mean-pooled generated answer line. Only these locations were cached.

        For each site/target, the top layers are locked using grouped training-only cross-validation.
        Test evaluation is a separate switch. Probe weights, intercepts, scaler statistics, CV
        metrics, and locked layers are saved under `artifacts/.../probes/`.
        """,
        "probe-rationale",
    ),
    code(
        r"""
        def result_site_positions(row: dict[str, object], site: str) -> list[int] | None:
            if site == "answer_line":
                start = row.get("answer_line_generated_token_start")
                end = row.get("answer_line_generated_token_end")
                completion_start = row.get("teacher_forced_completion_start")
                if start is None or end is None or completion_start is None:
                    return None
                positions = list(range(
                    int(completion_start) + int(start),
                    int(completion_start) + int(end),
                ))
                return positions or None
            if site == "answer_prefix":
                value = row.get("answer_boundary_input_index")
                return [int(value)] if value is not None else None
            prompt_sites = row.get("prompt_token_sites", {})
            value = prompt_sites.get(site) if isinstance(prompt_sites, dict) else None
            if isinstance(value, list) and value:
                return [int(position) for position in value]
            if value is not None:
                return [int(value)]
            return None


        def result_site_position(row: dict[str, object], site: str) -> int | None:
            positions = result_site_positions(row, site)
            return positions[-1] if positions else None


        def activation_path(row: dict[str, object]) -> Path:
            relative = row.get("activation_path")
            if not isinstance(relative, str):
                raise ValueError(f"Row {row['row_id']} has no activation capture.")
            return RUN_DIRS[bool(row["reasoning"])] / relative


        def activation_vector(row: dict[str, object], *, site: str, layer: int) -> np.ndarray:
            positions = result_site_positions(row, site)
            if positions is None:
                raise ValueError(f"Row {row['row_id']} has no {site} position.")
            captured_indices = [int(value) for value in row["activation_token_indices"]]
            captured_lookup = {
                absolute_position: stored_position
                for stored_position, absolute_position in enumerate(captured_indices)
            }
            missing = [position for position in positions if position not in captured_lookup]
            if missing:
                raise ValueError(
                    f"Row {row['row_id']} did not capture {site} positions {missing}."
                )
            stored_positions = [captured_lookup[position] for position in positions]
            key = f"answer.{ACTIVATION_STREAM}.layer_{layer}"
            with safe_open(activation_path(row), framework="pt", device="cpu") as handle:
                vectors = handle.get_tensor(key)[stored_positions].float().numpy()
            vector = vectors.mean(axis=0)
            if vector.shape != (HIDDEN_SIZE,):
                raise ValueError(f"Unexpected activation-vector shape {vector.shape}.")
            return vector


        def rows_available_at_site(result_rows: TranscriptDataset, site: str) -> list[dict[str, object]]:
            rows = []
            for source in result_rows:
                row = dict(source)
                positions = result_site_positions(row, site)
                if positions is None:
                    continue
                captured = {
                    int(value) for value in row.get("activation_token_indices", [])
                }
                if not set(positions) <= captured:
                    if site == "answer_line":
                        continue
                    missing = sorted(set(positions) - captured)
                    raise ValueError(
                        f"Row {row['row_id']} is missing captured {site} positions {missing}."
                    )
                rows.append(row)
            train_prefix_hashes = set()
            test_prefix_hashes = set()
            for row in rows:
                positions = result_site_positions(row, site)
                assert positions
                prefix_ids = row["teacher_forced_input_ids"][: max(positions) + 1]
                digest = hashlib.sha256(
                    b"".join(int(token).to_bytes(4, "little") for token in prefix_ids)
                ).hexdigest()
                (test_prefix_hashes if row["split"] == "test" else train_prefix_hashes).add(digest)
            if not train_prefix_hashes.isdisjoint(test_prefix_hashes):
                raise ValueError(f"Exact token-prefix leakage detected at site {site}.")
            return rows


        def activation_matrix(
            rows: list[dict[str, object]], *, site: str, layer: int
        ) -> np.ndarray:
            return np.stack([
                activation_vector(row, site=site, layer=layer) for row in rows
            ]).astype(np.float32, copy=False)


        def target_array(rows: list[dict[str, object]], target: str) -> np.ndarray:
            dtype = np.int64 if target in BINARY_PROBE_TARGETS else np.float64
            return np.asarray([row[target] for row in rows], dtype=dtype)


        def probe_paths(
            reasoning: bool, site: str, target: str, layer: int
        ) -> tuple[Path, Path]:
            root = PROBE_ROOT / MODE_NAMES[reasoning] / site / target
            return (
                root / f"layer_{layer:02d}.safetensors",
                root / f"layer_{layer:02d}.json",
            )


        def probe_artifacts_match_configuration(
            reasoning: bool, site: str, target: str, layer: int
        ) -> bool:
            weights_path, metadata_path = probe_paths(reasoning, site, target, layer)
            if not weights_path.is_file() or not metadata_path.is_file():
                return False
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                with safe_open(weights_path, framework="pt", device="cpu") as handle:
                    tensor_metadata = handle.metadata() or {}
                return (
                    metadata.get("probe_config_fingerprint") == PROBE_CONFIG_FINGERPRINT
                    and tensor_metadata.get("probe_config_fingerprint")
                    == PROBE_CONFIG_FINGERPRINT
                )
            except (OSError, ValueError, json.JSONDecodeError, SafetensorError):
                return False


        def save_fitted_probe(
            *, search: GridSearchCV, reasoning: bool, site: str, target: str, layer: int,
            metadata: dict[str, object]
        ) -> None:
            weights_path, metadata_path = probe_paths(reasoning, site, target, layer)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline = search.best_estimator_
            scaler = pipeline.named_steps["scale"]
            estimator = pipeline.named_steps["model"]
            tensors = {
                "coef": torch.as_tensor(np.asarray(estimator.coef_).reshape(-1), dtype=torch.float32),
                "intercept": torch.as_tensor(np.asarray(estimator.intercept_).reshape(-1), dtype=torch.float32),
                "feature_mean": torch.as_tensor(scaler.mean_, dtype=torch.float32),
                "feature_scale": torch.as_tensor(scaler.scale_, dtype=torch.float32),
            }
            temporary = weights_path.with_suffix(".safetensors.tmp")
            save_file(
                tensors,
                temporary,
                metadata={
                    "probe_config_fingerprint": PROBE_CONFIG_FINGERPRINT,
                    "reasoning_mode": MODE_NAMES[reasoning],
                    "site": site,
                    "target": target,
                    "layer": str(layer),
                },
            )
            os.replace(temporary, weights_path)
            atomic_write_json(metadata_path, metadata)


        def fit_one_probe(
            *, X_train: np.ndarray, train_rows: list[dict[str, object]],
            reasoning: bool, site: str, target: str, layer: int
        ) -> dict[str, object]:
            y_train = target_array(train_rows, target)
            groups = np.asarray([int(row["question_set_index"]) for row in train_rows])
            unique_groups = np.unique(groups)
            if {bool(row["reasoning"]) for row in train_rows} != {reasoning}:
                raise ValueError("A probe training matrix mixes reasoning modes.")
            if len(unique_groups) != NUM_QUESTION_SETS - TEST_SCHEDULE_COUNT:
                raise ValueError("Probe training rows do not contain exactly 56 schedules.")
            folds = list(GroupKFold(n_splits=PROBE_CV_FOLDS).split(X_train, y_train, groups))

            if target in BINARY_PROBE_TARGETS:
                # L2 is LogisticRegression's default. Leaving `penalty` unset avoids the
                # scikit-learn 1.8+ deprecation of that explicit keyword.
                estimator = LogisticRegression(
                    solver="lbfgs", max_iter=2_000, random_state=SEED
                )
                parameters = {"model__C": LOGISTIC_C_VALUES}
                scoring = "roc_auc"
                kind = "logistic"
            else:
                estimator = Ridge(
                    solver=RIDGE_SOLVER, tol=RIDGE_TOL, max_iter=RIDGE_MAX_ITER
                )
                parameters = {"model__alpha": RIDGE_ALPHAS}
                scoring = "r2"
                kind = "ridge"

            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("model", estimator),
            ])
            search = GridSearchCV(
                pipeline,
                param_grid=parameters,
                scoring=scoring,
                cv=folds,
                n_jobs=PROBE_N_JOBS,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            record = {
                "reasoning": reasoning,
                "reasoning_mode": MODE_NAMES[reasoning],
                "site": site,
                "target": target,
                "layer": layer,
                "kind": kind,
                "selection_metric": scoring,
                "best_cv_score": float(search.best_score_),
                "best_params": {
                    key: float(value) for key, value in search.best_params_.items()
                },
                "train_row_count": len(train_rows),
                "train_schedule_ids": sorted(int(value) for value in unique_groups),
                "test_rows_used": 0,
                "feature_count": int(X_train.shape[1]),
                "probe_config_fingerprint": PROBE_CONFIG_FINGERPRINT,
            }
            save_fitted_probe(
                search=search, reasoning=reasoning, site=site, target=target,
                layer=layer, metadata=record
            )
            return record


        def read_jsonl_if_present(path: Path) -> list[dict[str, object]]:
            if not path.exists():
                return []
            return [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        """,
        "probe-helpers",
    ),
    code(
        r"""
        probe_cv_path = PROBE_ROOT / "training_cv_metrics.jsonl"
        all_probe_targets = CONTINUOUS_PROBE_TARGETS + BINARY_PROBE_TARGETS
        expected_probe_keys = {
            (reasoning, site, target, layer)
            for reasoning in REASONING_VALUES
            for site in PROBE_SITES
            for target in all_probe_targets
            for layer in PROBE_LAYERS
        }
        probe_records_by_key = {}
        for row in read_jsonl_if_present(probe_cv_path):
            key = (
                bool(row["reasoning"]), str(row["site"]),
                str(row["target"]), int(row["layer"]),
            )
            if (
                key in expected_probe_keys
                and row.get("probe_config_fingerprint") == PROBE_CONFIG_FINGERPRINT
                and probe_artifacts_match_configuration(*key)
            ):
                probe_records_by_key[key] = row

        if RUN_PROBE_TRAINING:
            if results is None:
                raise RuntimeError("Load or run the completed activation capture first.")
            for reasoning in REASONING_VALUES:
                mode_results = results_by_reasoning[reasoning]
                for site in PROBE_SITES:
                    site_rows = rows_available_at_site(mode_results, site)
                    site_train_rows = [row for row in site_rows if row["split"] == "train"]
                    required_rows = max(
                        MIN_PROBE_TRAIN_ROWS,
                        math.ceil(MIN_GENERATED_SITE_COVERAGE * EXPECTED_TRAIN_ROWS_PER_REASONING),
                    )
                    if len(site_train_rows) < required_rows:
                        raise RuntimeError(
                            f"{MODE_NAMES[reasoning]}/{site} has {len(site_train_rows)} usable "
                            f"training rows; at least {required_rows} are required."
                        )
                    if len(site_train_rows) < EXPECTED_TRAIN_ROWS_PER_REASONING:
                        print(
                            f"{MODE_NAMES[reasoning]}/{site}: "
                            f"{EXPECTED_TRAIN_ROWS_PER_REASONING - len(site_train_rows)} rows "
                            "lack a usable generated answer boundary and are excluded."
                        )
                    for layer in PROBE_LAYERS:
                        missing_targets = [
                            target for target in all_probe_targets
                            if (reasoning, site, target, layer) not in probe_records_by_key
                        ]
                        if not missing_targets:
                            continue
                        print(
                            datetime.now(timezone.utc).isoformat(),
                            MODE_NAMES[reasoning], site, layer, missing_targets,
                        )
                        X_train = activation_matrix(site_train_rows, site=site, layer=layer)
                        for target in missing_targets:
                            record = fit_one_probe(
                                X_train=X_train,
                                train_rows=site_train_rows,
                                reasoning=reasoning,
                                site=site,
                                target=target,
                                layer=layer,
                            )
                            probe_records_by_key[(reasoning, site, target, layer)] = record
                            atomic_write_jsonl(
                                probe_cv_path,
                                sorted(
                                    probe_records_by_key.values(),
                                    key=lambda row: (
                                        bool(row["reasoning"]), str(row["site"]),
                                        str(row["target"]), int(row["layer"]),
                                    ),
                                ),
                            )
                        del X_train
                        gc.collect()
        else:
            print("Probe training is paused; set RUN_PROBE_TRAINING=True after capture.")

        probe_cv_records = list(probe_records_by_key.values())
        print("Completed training-only probe records:", len(probe_cv_records))
        """,
        "train-probes",
    ),
    md(
        r"""
        ### Lock layers, then open the test set

        Layer locking is deterministic and uses only `best_cv_score` from the 56 training
        schedules in each reasoning mode. The test switch should remain false until every planned training probe exists
        and `locked_layers.json` has been written. Test metrics include $R^2$, MAE, RMSE,
        correlation, and calibration for regressions; and AUROC, balanced accuracy, and log loss
        for reliability sign. Each mode's probe is evaluated only on that mode's 640 test rows.
        """,
        "test-evaluation-rationale",
    ),
    code(
        r"""
        expected_probe_count = len(expected_probe_keys)
        locked_layers_path = PROBE_ROOT / "locked_layers.json"
        locked_layers: dict[str, dict[str, dict[str, list[int]]]] | None = None

        if len(probe_cv_records) == expected_probe_count:
            locked_layers = {}
            for reasoning in REASONING_VALUES:
                mode_name = MODE_NAMES[reasoning]
                locked_layers[mode_name] = {}
                for site in PROBE_SITES:
                    locked_layers[mode_name][site] = {}
                    for target in CONTINUOUS_PROBE_TARGETS + BINARY_PROBE_TARGETS:
                        candidates = [
                            row for row in probe_cv_records
                            if bool(row["reasoning"]) is reasoning
                            and row["site"] == site and row["target"] == target
                        ]
                        if len(candidates) != len(PROBE_LAYERS):
                            raise ValueError(
                                f"Incomplete layer sweep for {mode_name}/{site}/{target}."
                            )
                        ranked = sorted(
                            candidates,
                            key=lambda row: (-float(row["best_cv_score"]), int(row["layer"])),
                        )
                        locked_layers[mode_name][site][target] = [
                            int(row["layer"]) for row in ranked[:PROBE_TOP_K_LAYERS]
                        ]
            atomic_write_json(locked_layers_path, {
                "probe_config_fingerprint": PROBE_CONFIG_FINGERPRINT,
                "layers": locked_layers,
            })
            display(locked_layers)
        elif locked_layers_path.exists():
            saved_lock = json.loads(locked_layers_path.read_text(encoding="utf-8"))
            if saved_lock.get("probe_config_fingerprint") == PROBE_CONFIG_FINGERPRINT:
                locked_layers = saved_lock["layers"]
            else:
                print("Ignoring locked_layers.json from a different probe configuration.")
        else:
            print(
                f"Layer locking waits for {expected_probe_count} training probes; "
                f"currently have {len(probe_cv_records)}."
            )


        def load_probe_prediction(
            X: np.ndarray, *, reasoning: bool, site: str, target: str, layer: int
        ) -> np.ndarray:
            if not probe_artifacts_match_configuration(reasoning, site, target, layer):
                raise ValueError("Probe weights or metadata do not match the locked configuration.")
            weights_path, _ = probe_paths(reasoning, site, target, layer)
            tensors = load_file(weights_path)
            mean = tensors["feature_mean"].numpy()
            scale = tensors["feature_scale"].numpy()
            coef = tensors["coef"].numpy()
            intercept = float(tensors["intercept"].reshape(-1)[0])
            score = ((X - mean) / scale) @ coef + intercept
            if target in BINARY_PROBE_TARGETS:
                score = np.where(
                    score >= 0,
                    1 / (1 + np.exp(-score)),
                    np.exp(score) / (1 + np.exp(score)),
                )
            return np.asarray(score)


        def continuous_metrics(
            y: np.ndarray, prediction: np.ndarray
        ) -> dict[str, float | None]:
            correlation = (
                float(np.corrcoef(y, prediction)[0, 1])
                if np.std(y) > 0 and np.std(prediction) > 0 else None
            )
            calibration_slope, calibration_intercept = (
                np.polyfit(prediction, y, 1) if np.std(prediction) > 0 else (None, None)
            )
            return {
                "r2": float(r2_score(y, prediction)),
                "mae": float(mean_absolute_error(y, prediction)),
                "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
                "pearson": correlation,
                "calibration_slope_y_on_prediction": (
                    float(calibration_slope) if calibration_slope is not None else None
                ),
                "calibration_intercept_y_on_prediction": (
                    float(calibration_intercept) if calibration_intercept is not None else None
                ),
            }


        def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
            prediction = (probability >= 0.5).astype(int)
            return {
                "auroc": float(roc_auc_score(y, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
                "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            }


        test_metric_rows: list[dict[str, object]] = []
        test_availability_rows: list[dict[str, object]] = []
        if RUN_LOCKED_TEST_EVALUATION:
            if results is None or locked_layers is None:
                raise RuntimeError("Completed captures and locked training-only layers are required.")
            cache: dict[
                tuple[bool, str, int], tuple[list[dict[str, object]], np.ndarray]
            ] = {}
            for reasoning in REASONING_VALUES:
                mode_name = MODE_NAMES[reasoning]
                mode_results = results_by_reasoning[reasoning]
                for site, targets in locked_layers[mode_name].items():
                    available = rows_available_at_site(mode_results, site)
                    selected_rows = [row for row in available if row["split"] == "test"]
                    test_availability_rows.append({
                        "reasoning": reasoning,
                        "reasoning_mode": mode_name,
                        "site": site,
                        "available_test_rows": len(selected_rows),
                        "expected_test_rows": EXPECTED_TEST_ROWS_PER_REASONING,
                        "coverage_fraction": (
                            len(selected_rows) / EXPECTED_TEST_ROWS_PER_REASONING
                        ),
                        "analysis_role": (
                            "confirmatory"
                            if len(selected_rows) == EXPECTED_TEST_ROWS_PER_REASONING
                            else "availability_limited_exploratory"
                        ),
                        "probe_config_fingerprint": PROBE_CONFIG_FINGERPRINT,
                    })
                    for target, layers in targets.items():
                        for layer in layers:
                            key = (reasoning, site, int(layer))
                            if key not in cache:
                                cache[key] = (
                                    selected_rows,
                                    activation_matrix(
                                        selected_rows, site=site, layer=int(layer)
                                    ),
                                )
                            metric_rows, X_test = cache[key]
                            y_test = target_array(metric_rows, target)
                            prediction = load_probe_prediction(
                                X_test, reasoning=reasoning, site=site,
                                target=target, layer=int(layer),
                            )
                            metrics = (
                                binary_metrics(y_test, prediction)
                                if target in BINARY_PROBE_TARGETS
                                else continuous_metrics(y_test, prediction)
                            )
                            test_metric_rows.append({
                                "reasoning": reasoning,
                                "reasoning_mode": mode_name,
                                "site": site,
                                "target": target,
                                "layer": int(layer),
                                "subset": "all",
                                "row_count": len(metric_rows),
                                "site_available_test_rows": len(metric_rows),
                                "site_coverage_fraction": (
                                    len(metric_rows) / EXPECTED_TEST_ROWS_PER_REASONING
                                ),
                                "probe_config_fingerprint": PROBE_CONFIG_FINGERPRINT,
                                **metrics,
                            })
            atomic_write_jsonl(PROBE_ROOT / "locked_test_metrics.jsonl", test_metric_rows)
            atomic_write_jsonl(
                PROBE_ROOT / "locked_test_availability.jsonl", test_availability_rows
            )
            display(pd.DataFrame(test_availability_rows))
            display(pd.DataFrame(test_metric_rows))
        else:
            print("The held-out test remains sealed: RUN_LOCKED_TEST_EVALUATION=False.")
        """,
        "lock-and-evaluate",
    ),
    md(
        r"""
        ## Probe geometry and information-transport visualizations

        Write each fitted probe as

        $$\hat y_{\ell,p}=w_{\ell,p}^{\mathsf T}h_{\ell,p}+b_{\ell,p},$$

        where $\ell$ is the residual-stream layer and $p$ is a configured token site. The full
        layer/location maps below use **grouped validation** $R^2$ from held-out training schedules;
        they never use in-sample training $R^2$. Each separately locked 640-row mode-specific test
        set is shown
        only after `RUN_LOCKED_TEST_EVALUATION` has deliberately opened it, and only at the layers
        selected before opening the test.

        All 14 sites are swept independently in each reasoning mode. Figures and transfer matrices
        therefore carry an explicit mode label; no plot pools the two probe datasets.

        Ridge stores coefficients in standardized-feature coordinates. For geometric comparisons
        in the shared residual-stream basis, the code converts them back to the exact raw-space
        direction $w_{\rm raw}=w_{\rm standardized}/\sigma$. Cross-layer and cross-location tests
        apply the complete source affine readout—including its source mean, scale, and intercept—to
        the destination activations. Re-standardizing at the destination would test a different
        readout and would not establish transport of the same code.

        The generated `answer_line` site may be availability-limited if a completion does not obey
        the answer contract. Training aborts if fewer than the configured coverage threshold—or
        fewer than 4,097 rows—remain. Test coverage is always written beside its metrics.
        """,
        "probe-visualization-rationale",
    ),
    code(
        r"""
        FIGURE_ROOT = PROBE_ROOT / "figures"
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


        def figure_slug(value: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


        def save_and_display_figure(fig, name: str) -> Path:
            path = FIGURE_ROOT / f"{figure_slug(name)}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=PROBE_FIGURE_DPI, bbox_inches="tight")
            display(fig)
            plt.close(fig)
            return path


        def probe_affine_parameters(
            *, reasoning: bool, site: str, target: str, layer: int
        ) -> dict[str, np.ndarray | float]:
            if not probe_artifacts_match_configuration(reasoning, site, target, layer):
                raise ValueError(
                    f"Missing or incompatible probe: "
                    f"{MODE_NAMES[reasoning]}/{site}/{target}/layer_{layer:02d}"
                )
            weights_path, _ = probe_paths(reasoning, site, target, layer)
            tensors = load_file(weights_path)
            mean = tensors["feature_mean"].numpy().astype(np.float64, copy=False)
            scale = tensors["feature_scale"].numpy().astype(np.float64, copy=False)
            standardized_coef = tensors["coef"].numpy().astype(np.float64, copy=False)
            intercept = float(tensors["intercept"].reshape(-1)[0])
            raw_direction = standardized_coef / scale
            raw_intercept = intercept - float(mean @ raw_direction)
            return {
                "mean": mean,
                "scale": scale,
                "standardized_coef": standardized_coef,
                "intercept": intercept,
                "raw_direction": raw_direction,
                "raw_intercept": raw_intercept,
            }


        def normalized_raw_direction(
            *, reasoning: bool, site: str, target: str, layer: int
        ) -> np.ndarray:
            direction = np.asarray(probe_affine_parameters(
                reasoning=reasoning, site=site, target=target, layer=layer
            )["raw_direction"])
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(
                    f"Degenerate probe direction: "
                    f"{MODE_NAMES[reasoning]}/{site}/{target}/layer_{layer:02d}"
                )
            return direction / norm


        def grouped_validation_matrix(reasoning: bool, target: str) -> np.ndarray:
            matrix = np.full((len(PROBE_LAYERS), len(PROBE_SITES)), np.nan)
            layer_to_row = {int(layer): index for index, layer in enumerate(PROBE_LAYERS)}
            site_to_column = {site: index for index, site in enumerate(PROBE_SITES)}
            for row in probe_cv_records:
                if (
                    bool(row["reasoning"]) is reasoning
                    and row["target"] == target
                    and row["site"] in site_to_column
                ):
                    matrix[layer_to_row[int(row["layer"])], site_to_column[str(row["site"])]] = (
                        float(row["best_cv_score"])
                    )
            if np.isnan(matrix).any():
                raise ValueError(
                    f"Incomplete grouped-validation matrix for "
                    f"{MODE_NAMES[reasoning]}/{target}."
                )
            return matrix


        def draw_matrix(
            matrix: np.ndarray,
            *,
            xlabels: list[str],
            ylabels: list[str],
            title: str,
            colorbar_label: str,
            vmin: float | None = None,
            vmax: float | None = None,
            figsize: tuple[float, float] = (8.0, 6.0),
        ):
            fig, ax = plt.subplots(figsize=figsize)
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#e6e6e6")
            image = ax.imshow(
                np.ma.masked_invalid(matrix), aspect="auto", origin="lower",
                interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
            )
            ax.set_xticks(np.arange(len(xlabels)), labels=xlabels, rotation=35, ha="right")
            y_step = max(1, len(ylabels) // 8)
            y_indices = np.arange(0, len(ylabels), y_step)
            ax.set_yticks(y_indices, labels=[ylabels[index] for index in y_indices])
            ax.set_xlabel("test location" if "transfer" in title.lower() else "token location")
            ax.set_ylabel("source/train layer" if "transfer" in title.lower() else "layer")
            ax.set_title(title)
            fig.colorbar(image, ax=ax, label=colorbar_label)
            return fig, ax
        """,
        "probe-visualization-helpers",
    ),
    code(
        r"""
        # 1–4, 7, and 8: training-only grouped-validation performance and probe geometry.
        if RUN_PROBE_VISUALIZATIONS:
            if len(probe_cv_records) != expected_probe_count:
                raise RuntimeError("Complete the configured probe sweep before plotting it.")

            for reasoning in REASONING_VALUES:
                mode_name = MODE_NAMES[reasoning]
                for target in PROBE_PLOT_TARGETS:
                    validation = grouped_validation_matrix(reasoning, target)

                    # 1. Layer × location grouped-validation heatmap.
                    fig, _ = draw_matrix(
                        validation,
                        xlabels=list(PROBE_SITES),
                        ylabels=[str(layer) for layer in PROBE_LAYERS],
                        title=f"{mode_name} · {target}: grouped schedule-validation R²",
                        colorbar_label="validation R²",
                        vmin=min(0.0, float(np.nanmin(validation))),
                        vmax=1.0,
                    )
                    save_and_display_figure(
                        fig, f"01_validation_heatmap_{mode_name}_{target}"
                    )

                    # 2. Probe performance versus layer.
                    fig, ax = plt.subplots(figsize=(11, 5.2))
                    for site_index, site in enumerate(PROBE_SITES):
                        ax.plot(
                            PROBE_LAYERS, validation[:, site_index],
                            marker="o", ms=2.5, label=site,
                        )
                    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
                    ax.set(
                        xlabel="layer", ylabel="grouped validation R²",
                        title=f"{mode_name} · {target}: decodability by layer",
                    )
                    ax.legend(ncol=2, fontsize=8)
                    ax.grid(alpha=0.2)
                    save_and_display_figure(
                        fig, f"02_validation_by_layer_{mode_name}_{target}"
                    )

                    # 3. Best available location and its score at each layer.
                    best_location_index = np.nanargmax(validation, axis=1)
                    best_score = validation[np.arange(len(PROBE_LAYERS)), best_location_index]
                    fig, (score_ax, site_ax) = plt.subplots(
                        2, 1, figsize=(10, 7), sharex=True,
                        gridspec_kw={"height_ratios": [2, 1]},
                    )
                    score_ax.plot(PROBE_LAYERS, best_score, marker="o", ms=3)
                    score_ax.axhline(0, color="black", lw=0.8, alpha=0.5)
                    score_ax.set(
                        ylabel="max validation R²",
                        title=f"{mode_name} · {target}: best location at each layer",
                    )
                    score_ax.grid(alpha=0.2)
                    site_ax.scatter(
                        PROBE_LAYERS, best_location_index,
                        c=PROBE_LAYERS, cmap="viridis", s=24,
                    )
                    site_ax.set_yticks(np.arange(len(PROBE_SITES)), labels=PROBE_SITES)
                    site_ax.set(xlabel="layer", ylabel="argmax location")
                    site_ax.grid(axis="x", alpha=0.2)
                    save_and_display_figure(
                        fig, f"03_best_location_by_layer_{mode_name}_{target}"
                    )

                    # 4. Pairwise cosine similarity, one compact panel per location.
                    n_columns = 3
                    n_rows = math.ceil(len(PROBE_SITES) / n_columns)
                    fig, axes = plt.subplots(
                        n_rows, n_columns, figsize=(13, 4.1 * n_rows), squeeze=False
                    )
                    for site_index, site in enumerate(PROBE_SITES):
                        directions = np.stack([
                            normalized_raw_direction(
                                reasoning=reasoning, site=site,
                                target=target, layer=int(layer),
                            )
                            for layer in PROBE_LAYERS
                        ])
                        similarity = directions @ directions.T
                        ax = axes.flat[site_index]
                        image = ax.imshow(
                            similarity, origin="lower", vmin=-1, vmax=1, cmap="coolwarm"
                        )
                        ax.set(xlabel="probe layer", ylabel="probe layer", title=site)
                        ticks = np.arange(0, len(PROBE_LAYERS), 8)
                        ax.set_xticks(ticks, labels=[PROBE_LAYERS[i] for i in ticks])
                        ax.set_yticks(ticks, labels=[PROBE_LAYERS[i] for i in ticks])
                    for ax in axes.flat[len(PROBE_SITES):]:
                        ax.set_visible(False)
                    fig.colorbar(image, ax=list(axes.flat), label="cosine similarity", shrink=0.65)
                    fig.suptitle(
                        f"{mode_name} · {target}: residual-basis direction similarity"
                    )
                    save_and_display_figure(
                        fig, f"04_direction_cosine_{mode_name}_{target}"
                    )

                    # 7. PCA trajectory of every normalized layer/location direction.
                    direction_rows = []
                    for site in PROBE_SITES:
                        for layer in PROBE_LAYERS:
                            direction_rows.append((
                                site, int(layer),
                                normalized_raw_direction(
                                    reasoning=reasoning, site=site,
                                    target=target, layer=int(layer),
                                ),
                            ))
                    direction_matrix = np.stack([row[2] for row in direction_rows])
                    pca = PCA(n_components=2)
                    coordinates = pca.fit_transform(direction_matrix)
                    fig, ax = plt.subplots(figsize=(8.5, 7.0))
                    for site in PROBE_SITES:
                        mask = np.asarray([row[0] == site for row in direction_rows])
                        layers = np.asarray([row[1] for row in direction_rows])[mask]
                        order = np.argsort(layers)
                        points = coordinates[mask][order]
                        layers = layers[order]
                        ax.plot(points[:, 0], points[:, 1], alpha=0.55, label=site)
                        scatter = ax.scatter(
                            points[:, 0], points[:, 1], c=layers,
                            cmap="viridis", s=26,
                        )
                    ax.set(
                        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                        title=f"{mode_name} · {target}: PCA of raw-space probe directions",
                    )
                    ax.legend(ncol=2, fontsize=8)
                    fig.colorbar(scatter, ax=ax, label="layer")
                    save_and_display_figure(
                        fig, f"07_direction_pca_{mode_name}_{target}"
                    )

                    # 8. Standardized-coordinate and raw-space norms.
                    fig, (standard_ax, raw_ax) = plt.subplots(
                        2, 1, figsize=(11, 8), sharex=True
                    )
                    for site in PROBE_SITES:
                        parameters = [
                            probe_affine_parameters(
                                reasoning=reasoning, site=site,
                                target=target, layer=int(layer),
                            )
                            for layer in PROBE_LAYERS
                        ]
                        standard_ax.plot(PROBE_LAYERS, [
                            np.linalg.norm(np.asarray(item["standardized_coef"]))
                            for item in parameters
                        ], label=site)
                        raw_ax.plot(PROBE_LAYERS, [
                            np.linalg.norm(np.asarray(item["raw_direction"]))
                            for item in parameters
                        ], label=site)
                    standard_ax.set(
                        ylabel="‖w‖₂ after feature standardization",
                        title=f"{mode_name} · {target}: probe norms",
                    )
                    raw_ax.set(xlabel="layer", ylabel="‖w raw‖₂")
                    standard_ax.legend(ncol=2, fontsize=8)
                    for ax in (standard_ax, raw_ax):
                        ax.grid(alpha=0.2)
                    save_and_display_figure(
                        fig, f"08_probe_norm_{mode_name}_{target}"
                    )
        else:
            print("Probe visualizations are paused: RUN_PROBE_VISUALIZATIONS=False.")
        """,
        "probe-validation-and-geometry-plots",
    ),
    code(
        r"""
        # 1 (locked-test supplement) and 9. These appear only after the test was opened once.
        locked_test_metrics_path = PROBE_ROOT / "locked_test_metrics.jsonl"
        locked_test_records = read_jsonl_if_present(locked_test_metrics_path)
        locked_test_records = [
            row for row in locked_test_records
            if row.get("probe_config_fingerprint") == PROBE_CONFIG_FINGERPRINT
        ]

        if RUN_PROBE_VISUALIZATIONS and locked_test_records:
            if results is None or locked_layers is None:
                raise RuntimeError("Loaded captures and locked layers are required for scatterplots.")
            for reasoning in REASONING_VALUES:
                mode_name = MODE_NAMES[reasoning]
                mode_results = results_by_reasoning[reasoning]
                for target in PROBE_PLOT_TARGETS:
                    matrix = np.full((len(PROBE_LAYERS), len(PROBE_SITES)), np.nan)
                    for row in locked_test_records:
                        if (
                            bool(row["reasoning"]) is reasoning
                            and row["target"] == target
                            and row["subset"] == "all"
                        ):
                            matrix[
                                int(row["layer"]), PROBE_SITES.index(str(row["site"]))
                            ] = float(row["r2"])
                    fig, _ = draw_matrix(
                        matrix,
                        xlabels=list(PROBE_SITES),
                        ylabels=[str(layer) for layer in PROBE_LAYERS],
                        title=(
                            f"{mode_name} · {target}: locked-test R² "
                            "(blank = layer not preselected)"
                        ),
                        colorbar_label="locked-test R²",
                        vmin=min(0.0, float(np.nanmin(matrix))),
                        vmax=max(1.0, float(np.nanmax(matrix))),
                    )
                    save_and_display_figure(
                        fig, f"01_locked_test_sparse_heatmap_{mode_name}_{target}"
                    )

                    n_columns = 3
                    n_rows = math.ceil(len(PROBE_SITES) / n_columns)
                    fig, axes = plt.subplots(
                        n_rows, n_columns, figsize=(13, 4.0 * n_rows), squeeze=False
                    )
                    for site_index, site in enumerate(PROBE_SITES):
                        ax = axes.flat[site_index]
                        layer = int(locked_layers[mode_name][site][target][0])
                        rows = [
                            row for row in rows_available_at_site(mode_results, site)
                            if row["split"] == "test"
                        ]
                        X_test = activation_matrix(rows, site=site, layer=layer)
                        observed = target_array(rows, target)
                        predicted = load_probe_prediction(
                            X_test, reasoning=reasoning, site=site,
                            target=target, layer=layer,
                        )
                        ax.scatter(observed, predicted, s=18, alpha=0.65)
                        limits = [
                            min(float(observed.min()), float(predicted.min())),
                            max(float(observed.max()), float(predicted.max())),
                        ]
                        ax.plot(limits, limits, color="black", ls="--", lw=1)
                        metrics = continuous_metrics(observed, predicted)
                        pearson_text = (
                            f"{metrics['pearson']:.3f}"
                            if metrics["pearson"] is not None else "undefined"
                        )
                        ax.set(
                            xlabel="true y", ylabel="predicted y",
                            title=(
                                f"{site} · L{layer}\n"
                                f"n={len(rows)}, R²={metrics['r2']:.3f}, r={pearson_text}"
                            ),
                        )
                        ax.grid(alpha=0.15)
                    for ax in axes.flat[len(PROBE_SITES):]:
                        ax.set_visible(False)
                    fig.suptitle(f"{mode_name} · {target}: locked-test predictions")
                    save_and_display_figure(
                        fig, f"09_locked_test_scatter_{mode_name}_{target}"
                    )
        elif RUN_PROBE_VISUALIZATIONS:
            print("Locked-test heatmaps and scatterplots remain sealed until test evaluation runs.")
        """,
        "probe-locked-test-plots",
    ),
    md(
        r"""
        ### Cross-layer and cross-location transfer

        These are preregistered here before opening the test, but are separately gated because they
        load many activation matrices. A matrix cell uses a probe trained at the row's source layer
        or location and applies that exact affine readout to the column's destination activations.
        Thus the diagonal is ordinary held-out decoding and off-diagonal structure measures whether
        a common linear code persists or moves. These matrices are exploratory multiple comparisons
        and must not change the already locked layers used for causal patching.
        """,
        "probe-transfer-rationale",
    ),
    code(
        r"""
        # 5, 6, and the projection-variance companion to 8.
        if RUN_PROBE_TRANSFER_ANALYSIS:
            if not locked_test_records:
                raise RuntimeError(
                    "Cross-location/layer transfer is gated until locked test evaluation has run."
                )
            if results is None:
                raise RuntimeError("Load the completed activation capture first.")

            projection_rows = []
            for reasoning in REASONING_VALUES:
                mode_name = MODE_NAMES[reasoning]
                mode_results = results_by_reasoning[reasoning]
                for target in PROBE_TRANSFER_TARGETS:
                    for site in PROBE_SITES:
                        rows = [
                            row for row in rows_available_at_site(mode_results, site)
                            if row["split"] == "test"
                        ]
                        observed = target_array(rows, target)
                        destination_matrices = {
                            int(layer): activation_matrix(
                                rows, site=site, layer=int(layer)
                            )
                            for layer in PROBE_LAYERS
                        }
                        source_parameters = [
                            probe_affine_parameters(
                                reasoning=reasoning, site=site,
                                target=target, layer=int(layer),
                            )
                            for layer in PROBE_LAYERS
                        ]
                        directions = np.stack([
                            np.asarray(item["raw_direction"])
                            for item in source_parameters
                        ], axis=1)
                        intercepts = np.asarray([
                            float(item["raw_intercept"]) for item in source_parameters
                        ])
                        transfer = np.full(
                            (len(PROBE_LAYERS), len(PROBE_LAYERS)), np.nan
                        )
                        for destination_index, destination_layer in enumerate(PROBE_LAYERS):
                            predictions = (
                                destination_matrices[int(destination_layer)] @ directions
                                + intercepts
                            )
                            for source_index, _ in enumerate(PROBE_LAYERS):
                                transfer[source_index, destination_index] = r2_score(
                                    observed, predictions[:, source_index]
                                )
                            diagonal_prediction = predictions[:, destination_index]
                            projection_rows.append({
                                "reasoning": reasoning,
                                "reasoning_mode": mode_name,
                                "site": site,
                                "target": target,
                                "layer": int(destination_layer),
                                "row_count": len(rows),
                                "projection_variance": float(np.var(diagonal_prediction)),
                                "projection_standard_deviation": float(
                                    np.std(diagonal_prediction)
                                ),
                                "probe_config_fingerprint": PROBE_CONFIG_FINGERPRINT,
                            })
                        np.savez_compressed(
                            FIGURE_ROOT
                            / (
                                f"cross_layer_{figure_slug(mode_name)}_"
                                f"{figure_slug(target)}_{figure_slug(site)}.npz"
                            ),
                            matrix=transfer,
                            layers=np.asarray(PROBE_LAYERS),
                        )
                        fig, ax = draw_matrix(
                            transfer,
                            xlabels=[str(layer) for layer in PROBE_LAYERS],
                            ylabels=[str(layer) for layer in PROBE_LAYERS],
                            title=(
                                f"{mode_name} · {target} · {site}: "
                                f"cross-layer transfer (n={len(rows)})"
                            ),
                            colorbar_label="held-out R²",
                            vmin=max(-1.0, float(np.nanpercentile(transfer, 5))),
                            vmax=min(
                                1.0,
                                max(0.0, float(np.nanpercentile(transfer, 95))),
                            ),
                            figsize=(7.4, 6.4),
                        )
                        ax.set_xlabel("test layer")
                        ax.set_ylabel("train/source layer")
                        save_and_display_figure(
                            fig, f"05_cross_layer_{mode_name}_{target}_{site}"
                        )

                    # Cross-location transfer uses the same rows at every destination.
                    common_rows = [
                        row for row in mode_results
                        if row["split"] == "test"
                        and all(
                            result_site_position(row, site) is not None
                            for site in PROBE_SITES
                        )
                    ]
                    if not common_rows:
                        raise RuntimeError(
                            f"No complete-site test rows for {mode_name}/{target}."
                        )
                    observed = target_array(common_rows, target)
                    for layer in PROBE_TRANSFER_LAYERS:
                        destination_matrices = {
                            site: activation_matrix(
                                common_rows, site=site, layer=int(layer)
                            )
                            for site in PROBE_SITES
                        }
                        transfer = np.full(
                            (len(PROBE_SITES), len(PROBE_SITES)), np.nan
                        )
                        for source_index, source_site in enumerate(PROBE_SITES):
                            parameters = probe_affine_parameters(
                                reasoning=reasoning, site=source_site,
                                target=target, layer=int(layer),
                            )
                            direction = np.asarray(parameters["raw_direction"])
                            offset = float(parameters["raw_intercept"])
                            for destination_index, destination_site in enumerate(PROBE_SITES):
                                prediction = (
                                    destination_matrices[destination_site] @ direction + offset
                                )
                                transfer[source_index, destination_index] = r2_score(
                                    observed, prediction
                                )
                        np.savez_compressed(
                            FIGURE_ROOT
                            / (
                                f"cross_location_{figure_slug(mode_name)}_"
                                f"{figure_slug(target)}_layer_{int(layer):02d}.npz"
                            ),
                            matrix=transfer,
                            sites=np.asarray(PROBE_SITES),
                            layer=int(layer),
                        )
                        fig, ax = draw_matrix(
                            transfer,
                            xlabels=list(PROBE_SITES),
                            ylabels=list(PROBE_SITES),
                            title=(
                                f"{mode_name} · {target} · L{int(layer)}: "
                                f"cross-location transfer (n={len(common_rows)})"
                            ),
                            colorbar_label="held-out R²",
                            vmin=max(-1.0, float(np.nanmin(transfer))),
                            vmax=min(1.0, max(0.0, float(np.nanmax(transfer)))),
                            figsize=(8.0, 7.0),
                        )
                        ax.set_xlabel("test location")
                        ax.set_ylabel("train/source location")
                        save_and_display_figure(
                            fig,
                            f"06_cross_location_{mode_name}_{target}_layer_{int(layer):02d}",
                        )

            atomic_write_jsonl(PROBE_ROOT / "projection_variance.jsonl", projection_rows)
            projection_frame = pd.DataFrame(projection_rows)
            for reasoning in REASONING_VALUES:
                mode_name = MODE_NAMES[reasoning]
                for target in PROBE_TRANSFER_TARGETS:
                    fig, ax = plt.subplots(figsize=(11, 5.2))
                    selected = projection_frame[
                        (projection_frame["reasoning"] == reasoning)
                        & (projection_frame["target"] == target)
                    ]
                    for site in PROBE_SITES:
                        site_rows = selected[selected["site"] == site].sort_values("layer")
                        ax.plot(
                            site_rows["layer"],
                            site_rows["projection_standard_deviation"],
                            marker="o", ms=3, label=site,
                        )
                    ax.set(
                        xlabel="layer", ylabel="held-out std(wᵀh + b)",
                        title=(
                            f"{mode_name} · {target}: variance of fitted probe projection"
                        ),
                    )
                    ax.legend(ncol=2, fontsize=8)
                    ax.grid(alpha=0.2)
                    save_and_display_figure(
                        fig, f"08_projection_std_{mode_name}_{target}"
                    )
        else:
            print("Transfer matrices are paused: RUN_PROBE_TRANSFER_ANALYSIS=False.")
        """,
        "probe-transfer-plots",
    ),
    md(
        r"""
        ## Whole-residual reliability interchange

        A successful probe is only evidence of decodability. This first causal screen patches the
        complete `resid_post` vector at the final prompt token between test rows that have identical
        questions, reports, answer pattern, and reasoning condition but paired reliabilities.
        Directions are tested both ways. The 64-patch budget is explicitly stratified over held-out
        schedule, reasoning mode, reliability pair, and direction. Within each stratum, examples
        are spread over the ordered agreement cells. This prevents a sorted-list subsample from
        accidentally retaining only one reliability magnitude or one direction.

        Patching uses TransformerLens 3's `TransformerBridge`, not the deprecated
        `HookedTransformer.from_pretrained` path. The hook name is resolved from the live bridge
        rather than assumed. Only held-out schedules are patched. The code truncates the
        teacher-forced input at the answer boundary and asks Qwen for only the last-position logits,
        reducing transient GPU memory.

        Before intervention, several bridge margins must reproduce the margins saved by the native
        capture runner within `BRIDGE_LOGIT_TOLERANCE`. A mismatch aborts rather than silently
        combining incompatible hook conventions.
        """,
        "patching-rationale",
    ),
    code(
        r"""
        def canonical_saved_margin(row: dict[str, object]) -> float:
            values = row.get("answer_surface_raw_logits")
            if not isinstance(values, dict):
                raise ValueError(f"Row {row['row_id']} has no answer-boundary surface logits.")
            return float(values[str(row["candidate_1"])]) - float(values[str(row["candidate_2"])])


        def bridge_resid_post_hook_name(bridge, layer: int) -> str:
            candidates = (
                f"blocks.{layer}.hook_resid_post",
                f"blocks.{layer}.hook_out",
            )
            for name in candidates:
                if name in bridge.hook_dict:
                    return name
            nearby = sorted(
                name for name in bridge.hook_dict
                if name.startswith(f"blocks.{layer}.") and ("resid" in name or name.endswith("hook_out"))
            )
            raise KeyError(f"No residual-post hook for layer {layer}; nearby hooks={nearby}")


        def bridge_margin(
            bridge, row: dict[str, object], *, hook: tuple[str, object] | None = None
        ) -> float:
            boundary = result_site_position(row, "answer_prefix")
            if boundary is None:
                raise ValueError(f"Row {row['row_id']} lacks an answer boundary.")
            input_ids = torch.tensor(
                [row["teacher_forced_input_ids"][: boundary + 1]],
                dtype=torch.long,
                device="cuda",
            )
            attention_mask = torch.ones_like(input_ids)
            kwargs = {
                "attention_mask": attention_mask,
                "return_type": "logits",
                "prepend_bos": False,
                "use_cache": False,
                "logits_to_keep": 1,
            }
            with torch.inference_mode():
                if hook is None:
                    logits = bridge(input_ids, **kwargs)
                else:
                    logits = bridge.run_with_hooks(input_ids, fwd_hooks=[hook], **kwargs)
            token_1 = int(row["candidate_1_answer_token_ids"][0])
            token_2 = int(row["candidate_2_answer_token_ids"][0])
            if len(row["candidate_1_answer_token_ids"]) != 1 or len(row["candidate_2_answer_token_ids"]) != 1:
                raise ValueError("Patching margin requires single-token candidate surfaces.")
            return float((logits[0, -1, token_1] - logits[0, -1, token_2]).float().cpu())


        def reliability_patch_directions(result_rows: TranscriptDataset) -> list[tuple[dict, dict]]:
            test = [dict(row) for row in result_rows if row["split"] == "test"]
            index = {
                (
                    int(row["question_set_index"]),
                    int(row["answer_pattern_index"]),
                    bool(row["reasoning"]),
                    row_reliability_exact(row),
                ): row
                for row in test
                if result_site_position(row, "answer_prefix") is not None
            }
            strata: dict[tuple[int, bool, int, int], list[tuple[dict, dict]]] = defaultdict(list)
            for schedule in sorted(test_schedule_set):
                for pattern in range(NUM_ANSWER_PATTERNS):
                    for reasoning in REASONING_VALUES:
                        for pair_index, (low, high) in enumerate(PATCH_RELIABILITY_PAIRS):
                            low_row = index.get((schedule, pattern, reasoning, low))
                            high_row = index.get((schedule, pattern, reasoning, high))
                            if low_row is not None and high_row is not None:
                                strata[(schedule, reasoning, pair_index, 0)].append((low_row, high_row))
                                strata[(schedule, reasoning, pair_index, 1)].append((high_row, low_row))

            expected_strata = {
                (schedule, reasoning, pair_index, direction)
                for schedule in test_schedule_set
                for reasoning in REASONING_VALUES
                for pair_index in range(len(PATCH_RELIABILITY_PAIRS))
                for direction in (0, 1)
            }
            if set(strata) != expected_strata:
                missing = sorted(expected_strata - set(strata))
                raise ValueError(
                    "Cannot construct a balanced patch design because answer-boundary rows "
                    f"are missing from strata {missing}."
                )
            if not len(strata) <= PATCH_MAX_DIRECTIONS <= sum(map(len, strata.values())):
                raise ValueError(
                    "PATCH_MAX_DIRECTIONS must cover every patch stratum without exceeding "
                    "the available paired directions."
                )

            base_quota, remainder = divmod(PATCH_MAX_DIRECTIONS, len(strata))
            directions = []
            for stratum_index, key in enumerate(sorted(strata)):
                candidates = sorted(strata[key], key=lambda pair: (
                    float(pair[0]["delta_a"]),
                    int(pair[0]["agreement_c1"]),
                    int(pair[0]["agreement_c2"]),
                    int(pair[0]["answer_pattern_index"]),
                ))
                quota = base_quota + int(stratum_index < remainder)
                if quota > len(candidates):
                    raise ValueError(f"Patch stratum {key} has only {len(candidates)} candidates.")
                if quota == 1:
                    selected_indices = [len(candidates) // 2]
                else:
                    selected_indices = [
                        round(index * (len(candidates) - 1) / (quota - 1))
                        for index in range(quota)
                    ]
                directions.extend(candidates[index] for index in selected_indices)

            assert len(directions) == PATCH_MAX_DIRECTIONS
            assert all(base["split"] == donor["split"] == "test" for base, donor in directions)
            return directions


        if RUN_ACTIVATION_PATCHING:
            if results is None or locked_layers is None:
                raise RuntimeError("Completed captures and locked training-only layers are required.")
            from transformer_lens.model_bridge import TransformerBridge

            patch_layers_by_mode = {
                mode_name: locked_layers[mode_name][PATCH_SITE]["z_bayes"]
                for mode_name in MODE_NAMES.values()
            }
            directions = reliability_patch_directions(results)
            patch_configuration = {
                "schema_version": 1,
                "probe_config_fingerprint": PROBE_CONFIG_FINGERPRINT,
                "patch_run_id": PATCH_RUN_ID,
                "site": PATCH_SITE,
                "layers_by_reasoning_mode": {
                    mode_name: [int(layer) for layer in layers]
                    for mode_name, layers in patch_layers_by_mode.items()
                },
                "reliability_pairs": [list(pair) for pair in PATCH_RELIABILITY_PAIRS],
                "directions": [
                    [str(base["row_id"]), str(donor["row_id"])] for base, donor in directions
                ],
            }
            PATCH_CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(
                patch_configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()).hexdigest()
            patch_design = {
                **patch_configuration,
                "patch_config_fingerprint": PATCH_CONFIG_FINGERPRINT,
                "row_count": len(directions),
                "test_schedules": sorted(test_schedule_set),
                "reasoning_counts": dict(Counter(str(base["reasoning"]) for base, _ in directions)),
                "directed_reliability_counts": dict(Counter(
                    f"{row_reliability_exact(base)}->{row_reliability_exact(donor)}"
                    for base, donor in directions
                )),
                "delta_a_counts": dict(Counter(str(base["delta_a"]) for base, _ in directions)),
            }
            atomic_write_json(PATCH_ROOT / "patch_design.json", patch_design)
            display(patch_design)
            bridge = TransformerBridge.boot_transformers(
                MODEL_ID,
                revision=MODEL_REVISION,
                device="cuda",
                dtype=torch.bfloat16,
                tokenizer=tokenizer,
                trust_remote_code=False,
            )
            all_patch_layers = sorted({
                int(layer)
                for layers in patch_layers_by_mode.values()
                for layer in layers
            })
            print("Resolved patch hooks:", {
                layer: bridge_resid_post_hook_name(bridge, layer)
                for layer in all_patch_layers
            })

            # Compatibility gate: cover each reasoning mode and base reliability before patching.
            compatibility_rows = []
            compatibility_strata = set()
            for base, _ in directions:
                key = (bool(base["reasoning"]), row_reliability_exact(base))
                if key not in compatibility_strata:
                    compatibility_strata.add(key)
                    compatibility_rows.append(base)
            patched_reliabilities = {
                reliability
                for pair in PATCH_RELIABILITY_PAIRS
                for reliability in pair
            }
            assert len(compatibility_rows) == (
                len(REASONING_VALUES) * len(patched_reliabilities)
            )
            for row in compatibility_rows:
                difference = abs(bridge_margin(bridge, row) - canonical_saved_margin(row))
                if difference > BRIDGE_LOGIT_TOLERANCE:
                    raise RuntimeError(
                        f"TransformerBridge/native margin mismatch {difference:.4f} exceeds tolerance."
                    )

            patch_results_path = PATCH_ROOT / "results.jsonl"
            completed_patches = {
                (row["base_row_id"], row["donor_row_id"], int(row["layer"])): row
                for row in read_jsonl_if_present(patch_results_path)
                if row.get("patch_config_fingerprint") == PATCH_CONFIG_FINGERPRINT
            }
            baseline_margins: dict[str, float] = {}
            for base, donor in directions:
                base_id, donor_id = str(base["row_id"]), str(donor["row_id"])
                if base_id not in baseline_margins:
                    baseline_margins[base_id] = bridge_margin(bridge, base)
                patch_layers = patch_layers_by_mode[MODE_NAMES[bool(base["reasoning"])]]
                for layer in patch_layers:
                    patch_key = (base_id, donor_id, int(layer))
                    if patch_key in completed_patches:
                        continue
                    donor_vector = torch.from_numpy(
                        activation_vector(donor, site=PATCH_SITE, layer=int(layer))
                    ).to(device="cuda", dtype=torch.bfloat16)
                    base_position = result_site_position(base, PATCH_SITE)
                    assert base_position is not None

                    def replace_residual(activation, hook, *, vector=donor_vector, position=base_position):
                        del hook
                        patched = activation.clone()
                        patched[0, position, :] = vector
                        return patched

                    hook_name = bridge_resid_post_hook_name(bridge, int(layer))
                    patched_margin = bridge_margin(
                        bridge, base, hook=(hook_name, replace_residual)
                    )
                    record = {
                        "base_row_id": base_id,
                        "donor_row_id": donor_id,
                        "split": "test",
                        "site": PATCH_SITE,
                        "layer": int(layer),
                        "reasoning": bool(base["reasoning"]),
                        "question_set_index": int(base["question_set_index"]),
                        "answer_pattern_index": int(base["answer_pattern_index"]),
                        "delta_a": float(base["delta_a"]),
                        "base_reliability": row_reliability_exact(base),
                        "donor_reliability": row_reliability_exact(donor),
                        "base_z_bayes": float(base["z_bayes"]),
                        "counterfactual_z_bayes": float(donor["z_bayes"]),
                        "baseline_margin": baseline_margins[base_id],
                        "patched_margin": patched_margin,
                        "margin_change": patched_margin - baseline_margins[base_id],
                        "patch_config_fingerprint": PATCH_CONFIG_FINGERPRINT,
                    }
                    completed_patches[patch_key] = record
                    atomic_write_jsonl(
                        patch_results_path,
                        sorted(
                            completed_patches.values(),
                            key=lambda row: (
                                str(row["base_row_id"]), str(row["donor_row_id"]), int(row["layer"])
                            ),
                        ),
                    )
            del bridge
            gc.collect()
            torch.cuda.empty_cache()
            display(pd.DataFrame(completed_patches.values()))
        else:
            print("Activation patching is paused: RUN_ACTIVATION_PATCHING=False.")
        """,
        "whole-residual-patching",
    ),
    md(
        r"""
        ## Execution order and interpretation

        1. Leave all gates false and run the notebook once. Confirm 10,240 total rows, the
           4,480/640 split inside each reasoning mode, paired prompt-length equality, the available
           test tie cells, and the storage preflight.
        2. When the GPU is free, set `RUN_GPU_CAPTURE=True`. Leave probe and patch switches false.
           Do not change factorial or capture settings under the same mode-specific `RUN_IDS`. A completed run is
           reusable; after interruption, rerunning this gate validates and resumes the durable
           contiguous prefix in eight-row checkpoints.
        3. After capture completes, set `RUN_GPU_CAPTURE=False` and
           `LOAD_COMPLETED_CAPTURE=True`; validate the inventory.
        4. Set `RUN_PROBE_TRAINING=True`. This reads only training schedules and writes weights and
           grouped-CV scores. Allow it to finish and inspect `locked_layers.json`.
        5. Set `RUN_PROBE_TRAINING=False` and `RUN_PROBE_VISUALIZATIONS=True` to render the full
           training-schedule validation and probe-geometry plots without opening the test.
        6. Freeze the locked layers, then set `RUN_LOCKED_TEST_EVALUATION=True` once. Keep transfer
           analysis false on that first pass so the confirmatory table is written before exploratory
           matrices are inspected.
        7. Set `RUN_PROBE_TRANSFER_ANALYSIS=True` only after the locked test table exists.
        8. Only after the decoding result is locked, set `RUN_ACTIVATION_PATCHING=True` for the
           held-out, bidirectional whole-residual reliability swaps.

        A high held-out $z^*$ probe does not establish mechanism. The causal question is whether
        replacing reliability-conditioned state moves the answer margin toward the donor's exact
        counterfactual in both directions and at both reliability magnitudes. Report raw margin
        changes and failures as well as successes.
        """,
        "execution-checklist",
    ),
    code(
        r"""
        final_status = {
            "dataset_ready": len(dataset) == EXPECTED_ROWS,
            "split_ready": (
                len(train_rows) == EXPECTED_TRAIN_ROWS
                and len(test_rows) == EXPECTED_TEST_ROWS
            ),
            "paired_prompt_lengths_equal": not prompt_length_mismatches,
            "test_schedules": list(selected_test_schedules),
            "all_tie_cells_in_test": TIE_CELLS <= {agreement_cell(row) for row in test_rows},
            "storage_preflight_passed": worst_case_activation_gib <= allowed_payload_gib,
            "capture_loaded": results is not None,
            "probe_training_records": len(probe_cv_records),
            "layers_locked": locked_layers is not None,
            "gpu_execution_requested": any((RUN_GPU_CAPTURE, RUN_ACTIVATION_PATCHING)),
        }
        display(final_status)
        """,
        "final-status",
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (MATS CUDA)",
            "language": "python",
            "name": "mats-cuda",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
