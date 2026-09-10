"""Build notebook 22: agreement-grid all-token/all-layer residual probes."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "notebooks" / "22_noisy_channel_bayesian_agreement_all_token_probes.ipynb"


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
        # Agreement-grid probes at every token and layer

        This notebook generates the agreement-balanced transcript design from notebook 21,
        captures Qwen3.5-4B `resid_post` activations at **every teacher-forced token and every
        language layer**, trains leakage-safe linear probes, evaluates them on held-out complete
        transcripts, and plots token-by-layer result maps.

        For the first and second candidates **by presentation order**, it probes:

        1. $g(r)=\ln(r/(1-r))$;
        2. $-g(r)$;
        3. $\Delta a=a_{\rm first}-a_{\rm second}$;
        4. $-\Delta a=a_{\rm second}-a_{\rm first}$;
        5. $z^*=\Delta a\,g(r)$;
        6. the length-$K$ first-candidate agreement vector;
        7. the length-$K$ second-candidate agreement vector;
        8. their length-$2K$ concatenation.

        Exact multi-output ridge solves are batched over token positions on CUDA. Because ridge
        loss is separable over output columns, each batched solve is mathematically the same as
        fitting every requested readout independently with the shared fixed regularization, while
        training all 17 scalar output components concurrently on the NVIDIA GPU.

        Expensive stages are gated and initially disabled. Run dataset generation first, inspect
        the storage preflight, then enable capture, CUDA probe training, the final symmetry
        analysis, and Hugging Face publication in separate passes.
        """,
        "title-and-scope",
    ),
    md(
        r"""
        ## Leakage contract

        The split unit is a complete sampled question/report schedule—not an individual row. All
        reliabilities, both candidate orders, and every selected reasoning mode for that schedule
        stay in one partition. The split ID is a content hash of the membership sets, observed
        reports, and canonical candidates, so an accidentally duplicated schedule cannot cross
        partitions under a different numeric index.

        With the default two schedules per agreement cell, one entire schedule from every one of
        the 64 cells is held out. Scaling and probe fitting use training rows only. The ridge alpha
        is configured in advance rather than selected on the test set. Test activations are used
        only for the reported held-out metrics and the explicitly post-hoc symmetric-reliability
        comparison at the end.
        """,
        "leakage-contract",
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
        from pathlib import Path

        os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")
        logging.getLogger("torchao").setLevel(logging.ERROR)

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from huggingface_hub import HfApi
        from IPython.display import Markdown, display
        from safetensors import SafetensorError, safe_open
        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoProcessor

        REPO_ROOT = next(
            path for path in (Path.cwd(), *Path.cwd().parents)
            if (path / "pyproject.toml").exists()
        )
        sys.path.insert(0, str(REPO_ROOT))

        from mats_experiments.noisy_channel_bayesian import (
            AgreementSubsetQuestion,
            AgreementTranscriptDatasetGenerator,
            CaptureSpec,
            ExecutionConfig,
            MetricSpec,
            ModelConfig,
            NoisyChannelBayesianEnvironment,
            QwenRunner,
            SGLangMTPConfig,
            SystemPrompt,
            TokenizerBinding,
            TranscriptDataset,
            XVsYPosteriorProbe,
        )
        """,
        "imports",
    ),
    md(
        r"""
        ## Configuration

        `MODEL_NAME_OR_PATH`, `REASONING_VALUES`, and every notebook-21 dataset parameter are
        configurable here. `NUM_QUESTION_SETS=2` means two schedules per agreement-vector cell,
        not two schedules total. The default symmetric reliability values let the gain probe learn
        both sign and magnitude while supporting the final $r\leftrightarrow1-r$ comparison.

        Capturing all tokens is storage-intensive. Reasoning-off is the default; selecting
        reasoning-on increases the upper bound by its completion-token allowance. The preflight
        refuses to start model capture if the configured budget or free-space reserve would be
        violated.
        """,
        "configuration-rationale",
    ),
    code(
        r"""
        # -------------------------- model and factorial --------------------------
        MODEL_NAME_OR_PATH = str(REPO_ROOT / "models" / "Qwen--Qwen3.5-4B")
        MODEL_REVISION = None
        MODEL_DTYPE = "auto"
        DEVICE_MAP = None
        LOCAL_FILES_ONLY = True

        N = 8
        K = 3
        X = 2
        Y = 7
        SUBSET_SIZE = 4
        NUM_QUESTION_SETS = 2  # Per exact (X vector, Y vector) agreement cell.
        R_VALUES = (0.1, 0.3, 0.7, 0.9)
        REASONING_VALUES = (False,)  # Use (True,) or (False, True) deliberately.
        CONTROL_POSITIONAL_BIAS = True
        ALLOW_SAME = False
        SEED = 20260909
        SYSTEM_PROMPT_TEXT = (
            "You are a Bayesian reasoner. Follow the user's game rules exactly. "
            "Use plaintext only. Do not use Markdown, headings, bullets, tables, code blocks, "
            "HTML, or any other formatting."
        )

        # ------------------------ leakage-safe split plan ------------------------
        TEST_GROUPS_PER_AGREEMENT_CELL = 1

        # --------------------- activation capture and storage --------------------
        ACTIVATION_STREAM = "resid_post"
        ACTIVATION_LAYERS = "all"
        ACTIVATION_TOKENS = "all"
        ACTIVATION_BYTES_PER_ELEMENT = 2  # Qwen BF16 residuals.
        MAX_COMPLETION_TOKENS_BY_REASONING = {False: 64, True: 512}
        ACTIVATION_STORAGE_BUDGET_GIB = 120.0
        PROBE_WEIGHT_STORAGE_BUDGET_GIB = 8.0
        PROBE_GPU_WORKSPACE_BUDGET_GIB = 8.0
        MIN_FREE_DISK_AFTER_CAPTURE_GIB = 20.0

        ENABLE_MTP = False
        COMPLETION_BATCH_SIZE = 8
        CAPTURE_BATCH_SIZE = 1
        SCORE_BATCH_SIZE = 8
        CHECKPOINT_EVERY_BATCHES = 8

        # ------------------------------ probe plan -------------------------------
        RIDGE_ALPHA = 100.0  # Fixed before test evaluation; no test-set tuning.
        PROBE_TRAIN_DEVICE = "cuda:0"  # Training refuses CPU/MPS by design.
        PROBE_INFERENCE_DEVICE = "auto"  # CUDA first; MPS fallback is allowed.
        ALLOW_MPS_FOR_PROBE_INFERENCE = True
        TOKEN_BLOCK_SIZE = 32  # Positions and all targets are solved concurrently.
        MIN_TRAIN_ROWS_PER_TOKEN = 64
        MIN_TEST_ROWS_PER_TOKEN = 32
        MAX_ANALYSIS_TOKEN_POSITION = None  # None means every captured absolute position.
        FIGURE_DPI = 150

        # ----------------------- Hugging Face probe storage ---------------------
        HF_PROBE_REPO_ID = None  # Example: "your-user/qwen35-4b-agreement-probes".
        HF_PROBE_REPO_PRIVATE = True
        HF_PROBE_REVISION = "main"
        HF_UPLOAD_NUM_WORKERS = 8

        # -------------------------- explicit stage gates -------------------------
        RUN_GPU_CAPTURE = False
        LOAD_COMPLETED_CAPTURE = False
        RUN_PROBE_TRAINING = False
        RUN_SYMMETRIC_RELIABILITY_ANALYSIS = False
        RUN_UPLOAD_PROBES_TO_HF = False

        EXPERIMENT_ROOT = REPO_ROOT / "artifacts" / "agreement_all_token_probes"
        """,
        "configuration",
    ),
    code(
        r"""
        TARGET_WIDTHS = {
            "gain": 1,
            "negative_gain": 1,
            "agreement_first_minus_second": 1,
            "agreement_second_minus_first": 1,
            "z_star": 1,
            "agreement_first_vector": K,
            "agreement_second_vector": K,
            "agreement_concatenated_vector": 2 * K,
        }
        TARGET_SYMMETRY = {
            "gain": "sign_flip",
            "negative_gain": "sign_flip",
            "agreement_first_minus_second": "invariant",
            "agreement_second_minus_first": "invariant",
            "z_star": "sign_flip",
            "agreement_first_vector": "invariant",
            "agreement_second_vector": "invariant",
            "agreement_concatenated_vector": "invariant",
        }
        TARGET_SLICES = {}
        target_cursor = 0
        for target_name, target_width in TARGET_WIDTHS.items():
            TARGET_SLICES[target_name] = slice(target_cursor, target_cursor + target_width)
            target_cursor += target_width
        JOINT_TARGET_WIDTH = target_cursor
        VECTOR_TARGETS = frozenset(
            target for target, width in TARGET_WIDTHS.items() if width > 1
        )

        config_for_fingerprint = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "model_revision": MODEL_REVISION,
            "n": N,
            "k": K,
            "x": X,
            "y": Y,
            "subset_size": SUBSET_SIZE,
            "num_question_sets_per_cell": NUM_QUESTION_SETS,
            "reliability_values": [str(Fraction(str(value))) for value in R_VALUES],
            "reasoning_values": list(REASONING_VALUES),
            "control_positional_bias": CONTROL_POSITIONAL_BIAS,
            "allow_same": ALLOW_SAME,
            "seed": SEED,
            "system_prompt": SYSTEM_PROMPT_TEXT,
            "test_groups_per_cell": TEST_GROUPS_PER_AGREEMENT_CELL,
            "activation_stream": ACTIVATION_STREAM,
            "activation_layers": ACTIVATION_LAYERS,
            "activation_tokens": ACTIVATION_TOKENS,
            "max_completion_tokens": {
                str(reasoning): MAX_COMPLETION_TOKENS_BY_REASONING[reasoning]
                for reasoning in REASONING_VALUES
            },
            "ridge_alpha": RIDGE_ALPHA,
            "probe_solver": "exact_batched_cuda_dual_cholesky",
            "probe_train_device": PROBE_TRAIN_DEVICE,
            "token_block_size": TOKEN_BLOCK_SIZE,
            "minimum_train_rows_per_token": MIN_TRAIN_ROWS_PER_TOKEN,
            "minimum_test_rows_per_token": MIN_TEST_ROWS_PER_TOKEN,
            "maximum_analysis_token_position": MAX_ANALYSIS_TOKEN_POSITION,
            "targets": TARGET_WIDTHS,
        }
        CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(
            config_for_fingerprint, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        EXPERIMENT_DIR = EXPERIMENT_ROOT / CONFIG_FINGERPRINT[:12]
        DATASET_ROOT = EXPERIMENT_DIR / "dataset"
        RUN_DIRS = {
            reasoning: EXPERIMENT_DIR / "runs" / (
                "reasoning_on" if reasoning else "reasoning_off"
            )
            for reasoning in REASONING_VALUES
        }
        PROBE_ROOT = EXPERIMENT_DIR / "probes"
        FIGURE_ROOT = PROBE_ROOT / "figures"
        SYMMETRY_ROOT = PROBE_ROOT / "symmetric_reliability"
        MODE_NAMES = {False: "reasoning_off", True: "reasoning_on"}

        if not REASONING_VALUES or len(set(REASONING_VALUES)) != len(REASONING_VALUES):
            raise ValueError("REASONING_VALUES must be a non-empty tuple without duplicates.")
        if any(type(value) is not bool for value in REASONING_VALUES):
            raise TypeError("REASONING_VALUES may contain only booleans.")
        if not CONTROL_POSITIONAL_BIAS:
            print(
                "CONTROL_POSITIONAL_BIAS=False: first/second probes still work, but only the "
                "configured X-then-Y presentation is present."
            )
        if NUM_QUESTION_SETS <= TEST_GROUPS_PER_AGREEMENT_CELL:
            raise ValueError(
                "NUM_QUESTION_SETS must exceed TEST_GROUPS_PER_AGREEMENT_CELL so every "
                "agreement cell occurs in both train and test."
            )
        if torch.device(PROBE_TRAIN_DEVICE).type != "cuda":
            raise ValueError("PROBE_TRAIN_DEVICE must be a CUDA device; CPU training is disabled.")
        if RIDGE_ALPHA <= 0:
            raise ValueError("RIDGE_ALPHA must be positive for the dual Cholesky solve.")
        if TOKEN_BLOCK_SIZE < 1:
            raise ValueError("TOKEN_BLOCK_SIZE must be positive.")
        reliability_fractions = tuple(Fraction(str(value)) for value in R_VALUES)
        if len(set(reliability_fractions)) != len(reliability_fractions):
            raise ValueError("R_VALUES contains a duplicate exact reliability.")
        if any(not 0 < value < 1 for value in reliability_fractions):
            raise ValueError("Every reliability must lie strictly between zero and one.")
        if any(1 - value not in reliability_fractions for value in reliability_fractions):
            raise ValueError("R_VALUES must be closed under r -> 1-r for symmetry analysis.")

        AGREEMENT_PATTERN_COUNT = 2**K
        AGREEMENT_CELL_COUNT = 4**K
        QUESTION_SET_COUNT = AGREEMENT_CELL_COUNT * NUM_QUESTION_SETS
        PRESENTATION_COUNT = 2 if CONTROL_POSITIONAL_BIAS else 1
        EXPECTED_ROWS = (
            len(R_VALUES)
            * len(REASONING_VALUES)
            * QUESTION_SET_COUNT
            * PRESENTATION_COUNT
        )
        print({
            "model": MODEL_NAME_OR_PATH,
            "reasoning_values": REASONING_VALUES,
            "agreement_grid": (AGREEMENT_PATTERN_COUNT, AGREEMENT_PATTERN_COUNT),
            "question_sets_per_cell": NUM_QUESTION_SETS,
            "expected_rows": EXPECTED_ROWS,
            "joint_probe_output_width": JOINT_TARGET_WIDTH,
            "experiment_dir": str(EXPERIMENT_DIR),
            "config_fingerprint": CONFIG_FINGERPRINT,
        })
        """,
        "derived-configuration",
    ),
    md(
        r"""
        ## Generate labels and make the sealed split

        This stage loads only the Qwen processor/tokenizer, not model weights. Each row receives
        the eight target values plus two identities:

        - `base_transcript_id` includes the exact schedule, observed reports, and presentation
          order, while excluding reliability and reasoning;
        - `split_group_id` is even stricter for leakage control: it groups both presentation orders
          of the same schedule and also excludes reliability and reasoning.

        Within each agreement-vector cell, complete content-hash groups are ranked deterministically
        and the configured number are assigned to test. No activations, generations, logits, or
        behavioral outcomes participate in the split.
        """,
        "generation-and-split-rationale",
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


        def content_id(payload: object) -> str:
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            return hashlib.sha256(encoded.encode()).hexdigest()[:24]


        def row_reliability(row: dict[str, object]) -> Fraction:
            values = tuple(Fraction(str(value)) for value in row["reliabilities_exact"])
            if len(values) != K or len(set(values)) != 1:
                raise ValueError("Expected one shared exact reliability across K questions.")
            return values[0]


        def agreement_target_fields(row: dict[str, object]) -> dict[str, object]:
            reliability = row_reliability(row)
            gain = math.log(float(reliability / (1 - reliability)))
            first_vector = [int(value) for value in row["agreement_x_by_question"]]
            second_vector = [int(value) for value in row["agreement_y_by_question"]]
            if len(first_vector) != K or len(second_vector) != K:
                raise ValueError("Candidate agreement vectors have the wrong width.")
            delta = sum(first_vector) - sum(second_vector)
            return {
                "gain": float(gain),
                "negative_gain": float(-gain),
                "agreement_first_minus_second": float(delta),
                "agreement_second_minus_first": float(-delta),
                "z_star": float(delta * gain),
                "agreement_first_vector": first_vector,
                "agreement_second_vector": second_vector,
                "agreement_concatenated_vector": first_vector + second_vector,
            }


        def split_identity_fields(row: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
            schedule = {
                "membership_sets": row["membership_sets"],
                "observed_reports": row["observed_reports"],
                "candidate_1": row["candidate_1"],
                "candidate_2": row["candidate_2"],
            }
            transcript = {
                **schedule,
                "presentation_order": row["presentation_order"],
                "candidate_value_order": row["candidate_value_order"],
            }
            return schedule, transcript


        def joint_target_matrix(rows: list[dict[str, object]]) -> np.ndarray:
            blocks = []
            for target, width in TARGET_WIDTHS.items():
                block = np.asarray([row[target] for row in rows], dtype=np.float64)
                if width == 1:
                    block = block.reshape(-1, 1)
                if block.shape != (len(rows), width):
                    raise ValueError(
                        f"Target {target!r} has shape {block.shape}, expected {(len(rows), width)}."
                    )
                blocks.append(block)
            result = np.concatenate(blocks, axis=1)
            if result.shape != (len(rows), JOINT_TARGET_WIDTH):
                raise ValueError("Joint target matrix has the wrong shape.")
            return result
        """,
        "label-and-persistence-helpers",
    ),
    code(
        r"""
        hf_config = AutoConfig.from_pretrained(
            MODEL_NAME_OR_PATH,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            local_files_only=LOCAL_FILES_ONLY,
        )
        text_config = getattr(hf_config, "text_config", hf_config)
        NUM_LAYERS = int(text_config.num_hidden_layers)
        HIDDEN_SIZE = int(text_config.hidden_size)
        if "4B" not in MODEL_NAME_OR_PATH and "4b" not in MODEL_NAME_OR_PATH:
            print("Warning: the requested default is Qwen3.5-4B; MODEL_NAME_OR_PATH was changed.")

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME_OR_PATH,
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
        raw_dataset = AgreementTranscriptDatasetGenerator(
            environment=environments,
            question=AgreementSubsetQuestion(subset_size=SUBSET_SIZE, sort=True),
            probe=probes,
            tokenizer_binding=tokenizer_binding,
            system_prompt=SystemPrompt(SYSTEM_PROMPT_TEXT),
            seed=SEED,
        ).generate(num_question_sets=NUM_QUESTION_SETS)
        if len(raw_dataset) != EXPECTED_ROWS:
            raise ValueError(f"Generated {len(raw_dataset)} rows, expected {EXPECTED_ROWS}.")

        enriched_rows = []
        for source in raw_dataset:
            row = dict(source)
            row.update(agreement_target_fields(row))
            schedule_identity, transcript_identity = split_identity_fields(row)
            row["split_group_id"] = content_id(schedule_identity)
            row["base_transcript_id"] = content_id(transcript_identity)
            row["reasoning_mode"] = MODE_NAMES[bool(row["reasoning"])]
            enriched_rows.append(row)

        groups_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in enriched_rows:
            cell = (str(row["agreement_x_pattern"]), str(row["agreement_y_pattern"]))
            groups_by_cell[cell].add(str(row["split_group_id"]))
        if len(groups_by_cell) != AGREEMENT_CELL_COUNT:
            raise ValueError("The generated dataset does not cover the complete agreement grid.")

        test_group_ids = set()
        for cell, group_ids in sorted(groups_by_cell.items()):
            if len(group_ids) < NUM_QUESTION_SETS:
                raise ValueError(
                    f"Agreement cell {cell} contains duplicate question/report schedules. "
                    "Change SEED or increase NUM_QUESTION_SETS before splitting."
                )
            ranked = sorted(
                group_ids,
                key=lambda group_id: hashlib.sha256(
                    f"{SEED}:split:{cell}:{group_id}".encode()
                ).hexdigest(),
            )
            test_group_ids.update(ranked[:TEST_GROUPS_PER_AGREEMENT_CELL])

        for row in enriched_rows:
            row["split"] = (
                "test" if str(row["split_group_id"]) in test_group_ids else "train"
            )

        train_rows = [row for row in enriched_rows if row["split"] == "train"]
        test_rows = [row for row in enriched_rows if row["split"] == "test"]
        train_groups = {str(row["split_group_id"]) for row in train_rows}
        test_groups = {str(row["split_group_id"]) for row in test_rows}
        train_transcripts = {str(row["base_transcript_id"]) for row in train_rows}
        test_transcripts = {str(row["base_transcript_id"]) for row in test_rows}
        train_row_ids = {str(row["row_id"]) for row in train_rows}
        test_row_ids = {str(row["row_id"]) for row in test_rows}
        if not train_groups.isdisjoint(test_groups):
            raise ValueError("A complete question/report schedule crosses train and test.")
        if not train_transcripts.isdisjoint(test_transcripts):
            raise ValueError("A base transcript crosses train and test.")
        if not train_row_ids.isdisjoint(test_row_ids):
            raise ValueError("A concrete dataset row crosses train and test.")

        split_cell_counts = {
            split: Counter(
                (row["agreement_x_pattern"], row["agreement_y_pattern"])
                for row in enriched_rows if row["split"] == split
            )
            for split in ("train", "test")
        }
        if any(len(counts) != AGREEMENT_CELL_COUNT for counts in split_cell_counts.values()):
            raise ValueError("Both train and test must retain every agreement-vector cell.")

        prompt_hashes_by_split = {
            split: {
                hashlib.sha256(str(row["serialized_prompt"]).encode()).hexdigest()
                for row in enriched_rows if row["split"] == split
            }
            for split in ("train", "test")
        }
        if not prompt_hashes_by_split["train"].isdisjoint(prompt_hashes_by_split["test"]):
            raise ValueError("An exact serialized prompt appears in both train and test.")

        dataset_manifest = {
            **raw_dataset.manifest,
            "notebook": "22_noisy_channel_bayesian_agreement_all_token_probes.ipynb",
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "model_revision": MODEL_REVISION,
            "config_fingerprint": CONFIG_FINGERPRINT,
            "target_widths": TARGET_WIDTHS,
            "joint_target_width": JOINT_TARGET_WIDTH,
            "split_unit": "content_hash_of_question_report_schedule",
            "split_excludes_reliability_reasoning_and_presentation_order": True,
            "test_groups_per_agreement_cell": TEST_GROUPS_PER_AGREEMENT_CELL,
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_split_group_count": len(train_groups),
            "test_split_group_count": len(test_groups),
            "all_agreement_cells_in_both_partitions": True,
            "activation_capture": {
                "stream": ACTIVATION_STREAM,
                "layers": ACTIVATION_LAYERS,
                "tokens": ACTIVATION_TOKENS,
                "num_layers": NUM_LAYERS,
                "hidden_size": HIDDEN_SIZE,
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
            mode_dataset = TranscriptDataset(
                mode_rows,
                manifest={
                    **dataset_manifest,
                    "reasoning": reasoning,
                    "reasoning_mode": MODE_NAMES[reasoning],
                    "row_count_for_mode": len(mode_rows),
                },
                experiment_dir=mode_root,
            )
            mode_dataset.save(mode_root)
            datasets_by_reasoning[reasoning] = mode_dataset

        atomic_write_json(EXPERIMENT_DIR / "configuration.json", config_for_fingerprint)
        atomic_write_json(EXPERIMENT_DIR / "split_audit.json", {
            "config_fingerprint": CONFIG_FINGERPRINT,
            "train_split_group_ids": sorted(train_groups),
            "test_split_group_ids": sorted(test_groups),
            "train_base_transcript_count": len(train_transcripts),
            "test_base_transcript_count": len(test_transcripts),
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_cell_counts": {
                f"{x_pattern}|{y_pattern}": count
                for (x_pattern, y_pattern), count in sorted(split_cell_counts["train"].items())
            },
            "test_cell_counts": {
                f"{x_pattern}|{y_pattern}": count
                for (x_pattern, y_pattern), count in sorted(split_cell_counts["test"].items())
            },
        })
        print({
            "dataset_rows": len(dataset),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_schedule_groups": len(train_groups),
            "test_schedule_groups": len(test_groups),
            "agreement_cells_in_each_split": AGREEMENT_CELL_COUNT,
            "dataset_path": str(DATASET_ROOT / "dataset.jsonl"),
        })
        """,
        "generate-label-and-split",
    ),
    code(
        r"""
        def split_grid(split: str) -> pd.DataFrame:
            counts = split_cell_counts[split]
            patterns = list(dataset.manifest["agreement_patterns"])
            return pd.DataFrame(
                [[counts[(x_pattern, y_pattern)] for y_pattern in patterns]
                 for x_pattern in patterns],
                index=pd.Index(patterns, name="first canonical X agreement pattern"),
                columns=pd.Index(patterns, name="canonical Y agreement pattern"),
            )


        display(Markdown("### Training rows per exact agreement-vector cell"))
        display(split_grid("train"))
        display(Markdown("### Held-out test rows per exact agreement-vector cell"))
        display(split_grid("test"))
        display({
            "train/test split groups disjoint": train_groups.isdisjoint(test_groups),
            "train/test base transcripts disjoint": train_transcripts.isdisjoint(test_transcripts),
            "train/test row IDs disjoint": train_row_ids.isdisjoint(test_row_ids),
            "train/test serialized prompts disjoint": prompt_hashes_by_split["train"].isdisjoint(
                prompt_hashes_by_split["test"]
            ),
        })
        """,
        "display-split-audit",
    ),
    md(
        r"""
        ## Storage preflight

        The activation upper bound charges every row for its full prompt plus its configured
        maximum completion, at every layer, in BF16. The probe-weight bound charges a separate
        joint readout, feature mean, and feature scale for every possible token/layer location in
        float32. These are deliberately conservative bounds; actual completions may be shorter.
        """,
        "storage-preflight-rationale",
    ),
    code(
        r"""
        GIB = 1024**3
        activation_upper_bound_bytes = 0
        max_token_upper_bound_by_reasoning = {}
        for reasoning in REASONING_VALUES:
            mode_rows = list(datasets_by_reasoning[reasoning])
            maximum = max(
                len(row["input_ids"]) + MAX_COMPLETION_TOKENS_BY_REASONING[reasoning]
                for row in mode_rows
            )
            max_token_upper_bound_by_reasoning[MODE_NAMES[reasoning]] = maximum
            activation_upper_bound_bytes += sum(
                (len(row["input_ids"]) + MAX_COMPLETION_TOKENS_BY_REASONING[reasoning])
                * NUM_LAYERS * HIDDEN_SIZE * ACTIVATION_BYTES_PER_ELEMENT
                for row in mode_rows
            )
        activation_upper_bound_gib = activation_upper_bound_bytes / GIB
        probe_weight_upper_bound_bytes = sum(
            max_token_upper_bound_by_reasoning[MODE_NAMES[reasoning]]
            * NUM_LAYERS
            * (JOINT_TARGET_WIDTH * HIDDEN_SIZE + 2 * HIDDEN_SIZE + JOINT_TARGET_WIDTH)
            * 4
            for reasoning in REASONING_VALUES
        )
        probe_weight_upper_bound_gib = probe_weight_upper_bound_bytes / GIB
        maximum_train_rows = max(
            sum(row["split"] == "train" for row in datasets_by_reasoning[reasoning])
            for reasoning in REASONING_VALUES
        )
        maximum_test_rows = max(
            sum(row["split"] == "test" for row in datasets_by_reasoning[reasoning])
            for reasoning in REASONING_VALUES
        )
        probe_gpu_workspace_upper_bound_bytes = TOKEN_BLOCK_SIZE * 4 * (
            (3 * maximum_train_rows + 2 * maximum_test_rows) * HIDDEN_SIZE
            + 3 * maximum_train_rows**2
            + 2 * maximum_train_rows * JOINT_TARGET_WIDTH
            + JOINT_TARGET_WIDTH * HIDDEN_SIZE
        )
        probe_gpu_workspace_upper_bound_gib = probe_gpu_workspace_upper_bound_bytes / GIB
        EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
        free_disk_gib = shutil.disk_usage(EXPERIMENT_ROOT).free / GIB
        storage_preflight = {
            "rows": len(dataset),
            "layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "all_tokens_captured": ACTIVATION_TOKENS == "all",
            "maximum_token_upper_bound_by_reasoning": max_token_upper_bound_by_reasoning,
            "activation_upper_bound_gib": activation_upper_bound_gib,
            "activation_budget_gib": ACTIVATION_STORAGE_BUDGET_GIB,
            "probe_weight_upper_bound_gib": probe_weight_upper_bound_gib,
            "probe_weight_budget_gib": PROBE_WEIGHT_STORAGE_BUDGET_GIB,
            "probe_gpu_workspace_upper_bound_gib": probe_gpu_workspace_upper_bound_gib,
            "probe_gpu_workspace_budget_gib": PROBE_GPU_WORKSPACE_BUDGET_GIB,
            "concurrent_token_positions": TOKEN_BLOCK_SIZE,
            "free_disk_gib": free_disk_gib,
            "minimum_free_disk_after_capture_gib": MIN_FREE_DISK_AFTER_CAPTURE_GIB,
        }
        display(storage_preflight)
        atomic_write_json(EXPERIMENT_DIR / "storage_preflight.json", storage_preflight)

        storage_checks_pass = (
            activation_upper_bound_gib <= ACTIVATION_STORAGE_BUDGET_GIB
            and probe_weight_upper_bound_gib <= PROBE_WEIGHT_STORAGE_BUDGET_GIB
            and probe_gpu_workspace_upper_bound_gib <= PROBE_GPU_WORKSPACE_BUDGET_GIB
            and free_disk_gib
            >= activation_upper_bound_gib
            + probe_weight_upper_bound_gib
            + MIN_FREE_DISK_AFTER_CAPTURE_GIB
        )
        if RUN_GPU_CAPTURE and not storage_checks_pass:
            raise RuntimeError(
                "All-token capture failed its storage preflight. Reduce the factorial or "
                "completion allowance, or deliberately revise the storage budgets."
            )
        if (
            RUN_PROBE_TRAINING
            and probe_gpu_workspace_upper_bound_gib > PROBE_GPU_WORKSPACE_BUDGET_GIB
        ):
            raise RuntimeError(
                "Concurrent CUDA probe training exceeds PROBE_GPU_WORKSPACE_BUDGET_GIB; "
                "reduce TOKEN_BLOCK_SIZE deliberately."
            )
        if not storage_checks_pass:
            print("Storage preflight does not currently authorize RUN_GPU_CAPTURE=True.")
        """,
        "storage-preflight",
    ),
    md(
        r"""
        ## Capture all residual tokens at all layers

        `CaptureSpec(tokens="all", layers="all")` stores the full teacher-forced sequence for
        each row, including the generated completion. Runs are separated by reasoning mode so
        their completion limits and artifacts cannot mix. The runner's normal atomic row files,
        checkpointing, and resume behavior are retained.
        """,
        "capture-rationale",
    ),
    code(
        r"""
        capture_spec = CaptureSpec(
            streams=(ACTIVATION_STREAM,),
            layers=ACTIVATION_LAYERS,
            tokens=ACTIVATION_TOKENS,
            logits_boundaries=(),
            every_decode_position=False,
        )
        metric_spec = MetricSpec(sequence_scores=False)

        RESULT_IDENTITY_FIELDS = (
            "row_id",
            "question_set_index",
            "membership_sets",
            "observed_reports",
            "candidate_value_order",
            "presentation_order",
            "reliabilities_exact",
            "reasoning",
            "split",
            "split_group_id",
            "base_transcript_id",
            *TARGET_WIDTHS,
            "serialized_prompt",
            "input_ids",
        )


        def validate_result_rows(
            rows: list[dict[str, object]], *, reasoning: bool, require_complete: bool
        ) -> None:
            configured = list(datasets_by_reasoning[reasoning])
            if len(rows) > len(configured) or (require_complete and len(rows) != len(configured)):
                raise ValueError("Saved result count does not match the configured mode dataset.")
            for index, (saved, expected) in enumerate(zip(rows, configured, strict=False)):
                for field in RESULT_IDENTITY_FIELDS:
                    if json.dumps(saved.get(field), sort_keys=True) != json.dumps(
                        expected.get(field), sort_keys=True
                    ):
                        raise ValueError(
                            f"Saved result {index} differs at {field!r}; refusing stale capture."
                        )


        def load_completed_capture(reasoning: bool) -> TranscriptDataset:
            run_dir = RUN_DIRS[reasoning]
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"No completed capture manifest at {manifest_path}.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
                raise ValueError("Saved run has a different activation-capture specification.")
            if manifest.get("model", {}).get("model_name_or_path") != MODEL_NAME_OR_PATH:
                raise ValueError("Saved run belongs to a different model.")
            results_path = run_dir / str(manifest.get("results_file", "results.jsonl"))
            rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            validate_result_rows(rows, reasoning=reasoning, require_complete=True)
            for row in rows:
                relative = row.get("activation_path")
                if not isinstance(relative, str) or not (run_dir / relative).is_file():
                    raise FileNotFoundError(f"Missing activation for row {row['row_id']}.")
            return TranscriptDataset(rows, manifest=manifest, experiment_dir=run_dir)


        results_by_reasoning: dict[bool, TranscriptDataset] = {}
        if RUN_GPU_CAPTURE and LOAD_COMPLETED_CAPTURE:
            raise ValueError("Choose RUN_GPU_CAPTURE or LOAD_COMPLETED_CAPTURE, not both.")
        if RUN_GPU_CAPTURE:
            runner = QwenRunner(ModelConfig(
                model_name_or_path=MODEL_NAME_OR_PATH,
                revision=MODEL_REVISION,
                dtype=MODEL_DTYPE,
                device_map=DEVICE_MAP,
                local_files_only=LOCAL_FILES_ONLY,
            ))
            for reasoning in REASONING_VALUES:
                execution = ExecutionConfig(
                    experiment_dir=EXPERIMENT_DIR,
                    run_id=MODE_NAMES[reasoning],
                    batch_size=COMPLETION_BATCH_SIZE,
                    completion_batch_size=COMPLETION_BATCH_SIZE,
                    capture_batch_size=CAPTURE_BATCH_SIZE,
                    score_batch_size=SCORE_BATCH_SIZE,
                    checkpoint_every_batches=CHECKPOINT_EVERY_BATCHES,
                    max_completion_tokens=MAX_COMPLETION_TOKENS_BY_REASONING[reasoning],
                    resume=True,
                    metrics=metric_spec,
                    capture=capture_spec,
                    completion_mtp=SGLangMTPConfig(enabled=ENABLE_MTP),
                )
                print(datetime.now(timezone.utc).isoformat(), "capturing", MODE_NAMES[reasoning])
                results_by_reasoning[reasoning] = datasets_by_reasoning[reasoning].execute(
                    runner, execution
                )
                validate_result_rows(
                    list(results_by_reasoning[reasoning]),
                    reasoning=reasoning,
                    require_complete=True,
                )
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif LOAD_COMPLETED_CAPTURE:
            results_by_reasoning = {
                reasoning: load_completed_capture(reasoning)
                for reasoning in REASONING_VALUES
            }
        else:
            print(
                "Capture is paused. After the storage audit passes, set RUN_GPU_CAPTURE=True; "
                "on later runs use LOAD_COMPLETED_CAPTURE=True."
            )
        """,
        "capture-or-load",
    ),
    code(
        r"""
        def activation_path(row: dict[str, object]) -> Path:
            relative = row.get("activation_path")
            if not isinstance(relative, str):
                raise ValueError(f"Row {row['row_id']} has no activation_path.")
            return RUN_DIRS[bool(row["reasoning"])] / relative


        def activation_key(layer: int) -> str:
            return f"answer.{ACTIVATION_STREAM}.layer_{layer}"


        activation_lengths_by_reasoning: dict[bool, dict[str, int]] = {}
        max_positions_by_reasoning: dict[bool, int] = {}
        if results_by_reasoning:
            expected_keys = {activation_key(layer) for layer in range(NUM_LAYERS)}
            for reasoning, mode_results in results_by_reasoning.items():
                lengths = {}
                for row_index, row in enumerate(mode_results):
                    path = activation_path(row)
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        keys = set(handle.keys())
                        if not expected_keys <= keys:
                            missing = sorted(expected_keys - keys)
                            raise ValueError(f"Activation file {path} is missing keys {missing[:3]}.")
                        shapes = {
                            tuple(handle.get_slice(key).get_shape()) for key in expected_keys
                        }
                    if len(shapes) != 1:
                        raise ValueError(f"Layers in {path} have inconsistent activation shapes.")
                    shape = next(iter(shapes))
                    if len(shape) != 2 or shape[1] != HIDDEN_SIZE:
                        raise ValueError(f"Unexpected activation shape {shape} in {path}.")
                    expected_length = len(row["teacher_forced_input_ids"])
                    if shape[0] != expected_length:
                        raise ValueError(
                            f"Captured length {shape[0]} differs from teacher-forced length "
                            f"{expected_length} for row {row['row_id']}."
                        )
                    lengths[str(row["row_id"])] = int(shape[0])
                    if (row_index + 1) % 128 == 0 or row_index + 1 == len(mode_results):
                        print(
                            datetime.now(timezone.utc).isoformat(),
                            MODE_NAMES[reasoning],
                            f"validated {row_index + 1}/{len(mode_results)} activation files",
                        )
                activation_lengths_by_reasoning[reasoning] = lengths
                maximum = max(lengths.values())
                if MAX_ANALYSIS_TOKEN_POSITION is not None:
                    maximum = min(maximum, int(MAX_ANALYSIS_TOKEN_POSITION) + 1)
                max_positions_by_reasoning[reasoning] = maximum
            display({
                MODE_NAMES[reasoning]: {
                    "rows": len(results_by_reasoning[reasoning]),
                    "minimum_tokens": min(activation_lengths_by_reasoning[reasoning].values()),
                    "maximum_tokens": max(activation_lengths_by_reasoning[reasoning].values()),
                    "analyzed_positions": max_positions_by_reasoning[reasoning],
                }
                for reasoning in results_by_reasoning
            })
        else:
            print("Activation inventory waits for a completed or loaded capture.")
        """,
        "activation-inventory",
    ),
    md(
        r"""
        ## Train and evaluate every token-by-layer probe

        Absolute teacher-forced token positions are used. For variable-length completions, a
        location is fit on every row that reaches that position; coverage is recorded explicitly.
        Positions below the configured minimum train or test count are retained in the output as
        `insufficient_rows` and appear as missing cells rather than silently disappearing.

        Each token block opens every row's safetensor once per layer. It then pads the available
        training rows, standardizes them on CUDA, and solves all positions in the block and all 17
        output components with one batched dual-Cholesky ridge operation. There is deliberately no
        CPU or MPS training fallback. A block-level weight file and metrics JSONL are written
        atomically, so an interrupted sweep can resume without repeating completed blocks. Feature
        scaling is fit on training activations only. Held-out activations never affect the scaler
        or coefficients.
        """,
        "probe-training-rationale",
    ),
    code(
        r"""
        def load_activation_block(
            rows: list[dict[str, object]], *, layer: int, positions: tuple[int, ...]
        ) -> dict[int, tuple[list[dict[str, object]], np.ndarray]]:
            selected_rows: dict[int, list[dict[str, object]]] = {
                position: [] for position in positions
            }
            selected_vectors: dict[int, list[np.ndarray]] = {
                position: [] for position in positions
            }
            key = activation_key(layer)
            for row in rows:
                with safe_open(activation_path(row), framework="pt", device="cpu") as handle:
                    tensor = handle.get_tensor(key)
                    available = [position for position in positions if position < tensor.shape[0]]
                    if not available:
                        continue
                    vectors = tensor[available].float().numpy()
                for vector_index, position in enumerate(available):
                    selected_rows[position].append(row)
                    selected_vectors[position].append(vectors[vector_index])
            result = {}
            for position in positions:
                vectors = selected_vectors[position]
                matrix = (
                    np.stack(vectors).astype(np.float32, copy=False)
                    if vectors else np.empty((0, HIDDEN_SIZE), dtype=np.float32)
                )
                result[position] = selected_rows[position], matrix
            return result


        def finite_component_r2(y: np.ndarray, prediction: np.ndarray) -> list[float | None]:
            values = []
            for component in range(y.shape[1]):
                target = y[:, component]
                predicted = prediction[:, component]
                denominator = float(np.sum((target - target.mean()) ** 2))
                values.append(
                    float(1 - np.sum((target - predicted) ** 2) / denominator)
                    if denominator > 0 else None
                )
            return values


        def finite_component_pearson(
            y: np.ndarray, prediction: np.ndarray
        ) -> list[float | None]:
            values = []
            for component in range(y.shape[1]):
                target = y[:, component]
                predicted = prediction[:, component]
                values.append(
                    float(np.corrcoef(target, predicted)[0, 1])
                    if np.std(target) > 0 and np.std(predicted) > 0 else None
                )
            return values


        def target_metrics(
            y: np.ndarray, prediction: np.ndarray, *, binary_vector: bool
        ) -> dict[str, object]:
            component_r2 = finite_component_r2(y, prediction)
            component_pearson = finite_component_pearson(y, prediction)
            finite_r2 = [value for value in component_r2 if value is not None]
            finite_pearson = [value for value in component_pearson if value is not None]
            metrics: dict[str, object] = {
                "r2": float(np.mean(finite_r2)) if finite_r2 else None,
                "mae": float(np.mean(np.abs(y - prediction))),
                "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
                "pearson": float(np.mean(finite_pearson)) if finite_pearson else None,
                "component_r2": component_r2,
                "component_pearson": component_pearson,
            }
            if binary_vector:
                predicted_bits = (prediction >= 0.5).astype(np.int64)
                target_bits = y.astype(np.int64)
                metrics.update({
                    "bit_accuracy": float(np.mean(predicted_bits == target_bits)),
                    "exact_vector_accuracy": float(
                        np.mean(np.all(predicted_bits == target_bits, axis=1))
                    ),
                })
            return metrics


        def require_cuda_probe_training_device() -> torch.device:
            device = torch.device(PROBE_TRAIN_DEVICE)
            if device.type != "cuda":
                raise ValueError("Probe training is CUDA-only by configuration.")
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "RUN_PROBE_TRAINING=True requires a CUDA GPU; no CPU fallback is allowed."
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Configured CUDA device {device} is not available.")
            return device


        def padded_gpu_batch(
            matrices: list[np.ndarray], *, width: int, device: torch.device
        ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
            if not matrices:
                raise ValueError("Cannot construct an empty concurrent probe batch.")
            row_counts = [len(matrix) for matrix in matrices]
            maximum = max(row_counts)
            values = torch.zeros(
                (len(matrices), maximum, width), dtype=torch.float32, device=device
            )
            valid = torch.zeros(
                (len(matrices), maximum, 1), dtype=torch.float32, device=device
            )
            for batch_index, matrix in enumerate(matrices):
                if matrix.shape != (row_counts[batch_index], width):
                    raise ValueError(
                        f"Concurrent batch matrix has shape {matrix.shape}, expected "
                        f"{(row_counts[batch_index], width)}."
                    )
                count = row_counts[batch_index]
                values[batch_index, :count] = torch.as_tensor(
                    matrix, dtype=torch.float32, device=device
                )
                valid[batch_index, :count] = 1.0
            return values, valid, row_counts


        def atomic_save_safetensors(
            path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
        ) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.unlink(missing_ok=True)
            save_file({key: value.contiguous() for key, value in tensors.items()}, temporary,
                      metadata=metadata)
            os.replace(temporary, path)


        def block_paths(
            reasoning: bool, layer: int, block_start: int
        ) -> tuple[Path, Path]:
            stem = f"tokens_{block_start:05d}_{block_start + TOKEN_BLOCK_SIZE - 1:05d}"
            root = PROBE_ROOT / MODE_NAMES[reasoning] / f"layer_{layer:02d}"
            return root / f"{stem}.safetensors", root / f"{stem}.metrics.jsonl"


        def block_is_complete(reasoning: bool, layer: int, block_start: int) -> bool:
            weights_path, metrics_path = block_paths(reasoning, layer, block_start)
            if not weights_path.is_file() or not metrics_path.is_file():
                return False
            try:
                with safe_open(weights_path, framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
                metric_rows = [
                    json.loads(line)
                    for line in metrics_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                return (
                    metadata.get("config_fingerprint") == CONFIG_FINGERPRINT
                    and metric_rows
                    and all(
                        row.get("config_fingerprint") == CONFIG_FINGERPRINT
                        for row in metric_rows
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError, SafetensorError):
                return False


        def fit_probe_block(
            *, reasoning: bool, layer: int, positions: tuple[int, ...]
        ) -> None:
            if not positions:
                raise ValueError("A probe block must contain at least one token position.")
            result_rows = [dict(row) for row in results_by_reasoning[reasoning]]
            block = load_activation_block(result_rows, layer=layer, positions=positions)
            metric_rows = []
            fit_items = []

            for position in positions:
                available_rows, X_all = block[position]
                if any(row["split"] not in ("train", "test") for row in available_rows):
                    raise ValueError("Every available activation row must be train or test.")
                train_mask = np.asarray(
                    [row["split"] == "train" for row in available_rows], dtype=bool
                )
                test_mask = ~train_mask
                position_train_rows = [
                    row for row, selected in zip(available_rows, train_mask, strict=True)
                    if selected
                ]
                position_test_rows = [
                    row for row, selected in zip(available_rows, test_mask, strict=True)
                    if selected
                ]
                position_train_groups = {
                    str(row["split_group_id"]) for row in position_train_rows
                }
                position_test_groups = {
                    str(row["split_group_id"]) for row in position_test_rows
                }
                if not position_train_groups.isdisjoint(position_test_groups):
                    raise ValueError(f"Transcript leakage at layer={layer}, token={position}.")

                enough_rows = (
                    len(position_train_rows) >= MIN_TRAIN_ROWS_PER_TOKEN
                    and len(position_test_rows) >= MIN_TEST_ROWS_PER_TOKEN
                )
                if not enough_rows:
                    for target, width in TARGET_WIDTHS.items():
                        metric_rows.append({
                            "config_fingerprint": CONFIG_FINGERPRINT,
                            "reasoning": reasoning,
                            "reasoning_mode": MODE_NAMES[reasoning],
                            "layer": layer,
                            "token_position": position,
                            "target": target,
                            "target_width": width,
                            "status": "insufficient_rows",
                            "available_row_count": len(available_rows),
                            "train_row_count": len(position_train_rows),
                            "test_row_count": len(position_test_rows),
                            "train_split_group_count": len(position_train_groups),
                            "test_split_group_count": len(position_test_groups),
                        })
                    continue

                fit_items.append({
                    "position": position,
                    "available_row_count": len(available_rows),
                    "train_rows": position_train_rows,
                    "test_rows": position_test_rows,
                    "train_group_count": len(position_train_groups),
                    "test_group_count": len(position_test_groups),
                    "X_train": X_all[train_mask].astype(np.float32, copy=False),
                    "X_test": X_all[test_mask].astype(np.float32, copy=False),
                    "y_train": joint_target_matrix(position_train_rows).astype(
                        np.float32, copy=False
                    ),
                    "y_test": joint_target_matrix(position_test_rows).astype(
                        np.float32, copy=False
                    ),
                })

            fitted_positions = [int(item["position"]) for item in fit_items]
            coefficients_array = np.empty(
                (0, JOINT_TARGET_WIDTH, HIDDEN_SIZE), dtype=np.float32
            )
            intercepts_array = np.empty((0, JOINT_TARGET_WIDTH), dtype=np.float32)
            feature_means_array = np.empty((0, HIDDEN_SIZE), dtype=np.float32)
            feature_scales_array = np.empty((0, HIDDEN_SIZE), dtype=np.float32)

            if fit_items:
                for item in fit_items:
                    if not all(np.isfinite(item[name]).all() for name in (
                        "X_train", "X_test", "y_train", "y_test"
                    )):
                        raise FloatingPointError(
                            f"Non-finite probe input at layer={layer}, "
                            f"token={item['position']}."
                        )
                device = require_cuda_probe_training_device()
                X_train, train_valid, train_counts = padded_gpu_batch(
                    [item["X_train"] for item in fit_items],
                    width=HIDDEN_SIZE,
                    device=device,
                )
                X_test, test_valid, test_counts = padded_gpu_batch(
                    [item["X_test"] for item in fit_items],
                    width=HIDDEN_SIZE,
                    device=device,
                )
                y_train, _, _ = padded_gpu_batch(
                    [item["y_train"] for item in fit_items],
                    width=JOINT_TARGET_WIDTH,
                    device=device,
                )

                train_denominator = train_valid.sum(dim=1).clamp_min(1.0)
                feature_mean = (X_train * train_valid).sum(dim=1) / train_denominator
                X_train_centered = (X_train - feature_mean[:, None, :]) * train_valid
                feature_variance = (
                    X_train_centered.square().sum(dim=1) / train_denominator
                )
                feature_scale = torch.sqrt(feature_variance)
                feature_scale = torch.where(
                    feature_scale > 1e-6,
                    feature_scale,
                    torch.ones_like(feature_scale),
                )
                X_train_scaled = (
                    X_train_centered / feature_scale[:, None, :]
                ) * train_valid
                target_mean = (y_train * train_valid).sum(dim=1) / train_denominator
                y_train_centered = (y_train - target_mean[:, None, :]) * train_valid

                gram = torch.bmm(X_train_scaled, X_train_scaled.transpose(1, 2))
                gram.diagonal(dim1=-2, dim2=-1).add_(RIDGE_ALPHA)
                cholesky, info = torch.linalg.cholesky_ex(gram)
                if torch.any(info != 0).item():
                    failed = torch.nonzero(info != 0).flatten().tolist()
                    raise FloatingPointError(
                        f"CUDA Cholesky failed for concurrent positions {failed}."
                    )
                dual = torch.cholesky_solve(y_train_centered, cholesky)
                coefficients = torch.bmm(
                    X_train_scaled.transpose(1, 2), dual
                ).transpose(1, 2)
                intercepts = target_mean.squeeze(1)

                X_test_scaled = (
                    (X_test - feature_mean[:, None, :]) / feature_scale[:, None, :]
                ) * test_valid
                train_prediction = (
                    torch.bmm(X_train_scaled, coefficients.transpose(1, 2))
                    + intercepts[:, None, :]
                )
                test_prediction = (
                    torch.bmm(X_test_scaled, coefficients.transpose(1, 2))
                    + intercepts[:, None, :]
                )
                numerical_outputs = (
                    feature_mean,
                    feature_scale,
                    coefficients,
                    intercepts,
                    train_prediction,
                    test_prediction,
                )
                if (
                    not all(torch.isfinite(value).all().item() for value in numerical_outputs)
                    or torch.any(feature_scale <= 0).item()
                ):
                    raise FloatingPointError(
                        f"Non-finite CUDA probe fit at layer={layer}, "
                        f"tokens={fitted_positions}."
                    )

                train_prediction_array = train_prediction.detach().cpu().numpy()
                test_prediction_array = test_prediction.detach().cpu().numpy()
                coefficients_array = coefficients.detach().cpu().numpy()
                intercepts_array = intercepts.detach().cpu().numpy()
                feature_means_array = feature_mean.detach().cpu().numpy()
                feature_scales_array = feature_scale.detach().cpu().numpy()

                for fit_index, item in enumerate(fit_items):
                    position = int(item["position"])
                    y_train_array = item["y_train"]
                    y_test_array = item["y_test"]
                    position_train_prediction = train_prediction_array[
                        fit_index, :train_counts[fit_index]
                    ]
                    position_test_prediction = test_prediction_array[
                        fit_index, :test_counts[fit_index]
                    ]
                    for target, width in TARGET_WIDTHS.items():
                        target_slice = TARGET_SLICES[target]
                        train_metrics = target_metrics(
                            y_train_array[:, target_slice],
                            position_train_prediction[:, target_slice],
                            binary_vector=target in VECTOR_TARGETS,
                        )
                        test_metrics = target_metrics(
                            y_test_array[:, target_slice],
                            position_test_prediction[:, target_slice],
                            binary_vector=target in VECTOR_TARGETS,
                        )
                        metric_rows.append({
                            "config_fingerprint": CONFIG_FINGERPRINT,
                            "reasoning": reasoning,
                            "reasoning_mode": MODE_NAMES[reasoning],
                            "layer": layer,
                            "token_position": position,
                            "target": target,
                            "target_width": width,
                            "status": "fitted",
                            "ridge_alpha": RIDGE_ALPHA,
                            "solver": "exact_batched_cuda_dual_cholesky",
                            "concurrent_position_count": len(fit_items),
                            "training_device": str(device),
                            "training_device_name": torch.cuda.get_device_name(device),
                            "available_row_count": item["available_row_count"],
                            "train_row_count": len(item["train_rows"]),
                            "test_row_count": len(item["test_rows"]),
                            "train_split_group_count": item["train_group_count"],
                            "test_split_group_count": item["test_group_count"],
                            **{f"train_{key}": value for key, value in train_metrics.items()},
                            **{f"test_{key}": value for key, value in test_metrics.items()},
                        })

            weights_path, metrics_path = block_paths(reasoning, layer, positions[0])
            fitted_count = len(fitted_positions)
            tensors = {
                "token_positions": torch.as_tensor(fitted_positions, dtype=torch.int64),
                "coef": torch.as_tensor(coefficients_array),
                "intercept": torch.as_tensor(intercepts_array),
                "feature_mean": torch.as_tensor(feature_means_array),
                "feature_scale": torch.as_tensor(feature_scales_array),
            }
            atomic_save_safetensors(weights_path, tensors, {
                "config_fingerprint": CONFIG_FINGERPRINT,
                "reasoning_mode": MODE_NAMES[reasoning],
                "layer": str(layer),
                "block_start": str(positions[0]),
                "fitted_position_count": str(fitted_count),
                "solver": "exact_batched_cuda_dual_cholesky",
                "training_device": PROBE_TRAIN_DEVICE,
                "concurrent_output_width": str(JOINT_TARGET_WIDTH),
                "target_slices": json.dumps({
                    target: [value.start, value.stop]
                    for target, value in TARGET_SLICES.items()
                }, sort_keys=True),
            })
            atomic_write_jsonl(metrics_path, metric_rows)
            del block
            gc.collect()
        """,
        "probe-helpers",
    ),
    code(
        r"""
        if RUN_PROBE_TRAINING:
            if not results_by_reasoning:
                raise RuntimeError("Completed activation captures are required for probe training.")
            training_device = require_cuda_probe_training_device()
            training_properties = torch.cuda.get_device_properties(training_device)
            display({
                "probe_training_device": str(training_device),
                "probe_training_device_name": training_properties.name,
                "probe_training_device_memory_gib": training_properties.total_memory / GIB,
                "solver": "exact_batched_cuda_dual_cholesky",
                "concurrent_token_positions_per_block": TOKEN_BLOCK_SIZE,
                "concurrent_output_components": JOINT_TARGET_WIDTH,
                "cpu_training_fallback": False,
            })
            for reasoning in REASONING_VALUES:
                maximum = max_positions_by_reasoning[reasoning]
                for layer in range(NUM_LAYERS):
                    for block_start in range(0, maximum, TOKEN_BLOCK_SIZE):
                        positions = tuple(
                            range(block_start, min(block_start + TOKEN_BLOCK_SIZE, maximum))
                        )
                        if block_is_complete(reasoning, layer, block_start):
                            continue
                        print(
                            datetime.now(timezone.utc).isoformat(),
                            MODE_NAMES[reasoning],
                            f"layer={layer}/{NUM_LAYERS - 1}",
                            f"tokens={positions[0]}..{positions[-1]}",
                        )
                        fit_probe_block(
                            reasoning=reasoning, layer=layer, positions=positions
                        )
        else:
            print("Probe training is paused; set RUN_PROBE_TRAINING=True after capture.")


        def load_all_probe_metrics() -> list[dict[str, object]]:
            records = []
            for path in sorted(PROBE_ROOT.glob("reasoning_*/layer_*/*.metrics.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("config_fingerprint") == CONFIG_FINGERPRINT:
                        records.append(row)
            return records


        probe_metric_rows = load_all_probe_metrics()
        expected_metric_rows = (
            sum(max_positions_by_reasoning.values())
            * NUM_LAYERS
            * len(TARGET_WIDTHS)
            if max_positions_by_reasoning else 0
        )
        probe_sweep_complete = (
            expected_metric_rows > 0 and len(probe_metric_rows) == expected_metric_rows
        )
        print({
            "probe_metric_rows": len(probe_metric_rows),
            "expected_when_complete": expected_metric_rows,
            "completed_fraction": (
                len(probe_metric_rows) / expected_metric_rows if expected_metric_rows else 0.0
            ),
            "probe_sweep_complete": probe_sweep_complete,
        })
        """,
        "train-probes",
    ),
    md(
        r"""
        ## Held-out token-by-layer results

        Each panel below is one requested probe. A pixel is held-out $R^2$ at one absolute token
        position and one residual-stream layer; no test row was used for scaling or fitting.
        Vector-probe $R^2$ is the mean over nonconstant output components. Missing pixels are
        positions that did not meet the preregistered train/test coverage thresholds or blocks that
        have not yet completed. A companion coverage map makes variable-length completion support
        visible.
        """,
        "main-results-rationale",
    ),
    code(
        r"""
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


        def result_matrix(
            frame: pd.DataFrame, *, reasoning: bool, target: str, metric: str
        ) -> np.ndarray:
            maximum = max_positions_by_reasoning[reasoning]
            matrix = np.full((NUM_LAYERS, maximum), np.nan, dtype=np.float64)
            selected = frame[
                (frame["reasoning"] == reasoning)
                & (frame["target"] == target)
                & (frame["status"] == "fitted")
            ]
            for row in selected.itertuples(index=False):
                value = getattr(row, metric, None)
                if value is not None and np.isfinite(value):
                    matrix[int(row.layer), int(row.token_position)] = float(value)
            return matrix


        def plot_probe_maps(frame: pd.DataFrame, *, reasoning: bool) -> None:
            fig, axes = plt.subplots(4, 2, figsize=(20, 18), constrained_layout=True)
            prompt_boundary = int(np.median([
                int(row["teacher_forced_completion_start"])
                for row in results_by_reasoning[reasoning]
            ]))
            for axis, target in zip(axes.flat, TARGET_WIDTHS, strict=True):
                matrix = result_matrix(
                    frame, reasoning=reasoning, target=target, metric="test_r2"
                )
                image = axis.imshow(
                    matrix,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=-0.25,
                    vmax=1.0,
                )
                axis.axvline(prompt_boundary - 0.5, color="white", linestyle="--", linewidth=0.8)
                axis.set(
                    title=f"{target} (width={TARGET_WIDTHS[target]})",
                    xlabel="absolute teacher-forced token position",
                    ylabel="layer",
                )
                fig.colorbar(image, ax=axis, shrink=0.78, label="held-out R²")
            fig.suptitle(
                f"Qwen3.5-4B agreement probes — {MODE_NAMES[reasoning]} — all tokens × layers",
                fontsize=16,
            )
            path = FIGURE_ROOT / f"held_out_r2_{MODE_NAMES[reasoning]}.png"
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.show()

            coverage = np.full(
                (NUM_LAYERS, max_positions_by_reasoning[reasoning]), np.nan
            )
            selected = frame[
                (frame["reasoning"] == reasoning)
                & (frame["target"] == "gain")
            ]
            for row in selected.itertuples(index=False):
                coverage[int(row.layer), int(row.token_position)] = int(row.test_row_count)
            fig, axis = plt.subplots(figsize=(18, 5), constrained_layout=True)
            image = axis.imshow(
                coverage, origin="lower", aspect="auto", interpolation="nearest", cmap="magma"
            )
            axis.axvline(prompt_boundary - 0.5, color="white", linestyle="--", linewidth=0.8)
            axis.set(
                title=f"Held-out row coverage — {MODE_NAMES[reasoning]}",
                xlabel="absolute teacher-forced token position",
                ylabel="layer",
            )
            fig.colorbar(image, ax=axis, label="available held-out rows")
            fig.savefig(
                FIGURE_ROOT / f"test_coverage_{MODE_NAMES[reasoning]}.png",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
            plt.show()


        if probe_metric_rows:
            probe_metrics = pd.DataFrame(probe_metric_rows)
            display(
                probe_metrics[probe_metrics["status"] == "fitted"]
                .groupby(["reasoning_mode", "target"], as_index=False)
                .agg(
                    fitted_token_layer_cells=("test_r2", "count"),
                    median_test_r2=("test_r2", "median"),
                    maximum_test_r2=("test_r2", "max"),
                    median_test_mae=("test_mae", "median"),
                )
            )
            for reasoning in REASONING_VALUES:
                if reasoning in results_by_reasoning:
                    plot_probe_maps(probe_metrics, reasoning=reasoning)
        else:
            probe_metrics = pd.DataFrame()
            print("Token-by-layer figures wait for completed probe blocks.")
        """,
        "main-probe-results",
    ),
    md(
        r"""
        ## Post-hoc matched symmetric-reliability comparison

        This section deliberately comes after the complete probe-training and held-out result
        sections. It uses only held-out schedules and pairs rows whose question sets, observed
        reports, canonical candidates, presentation order, and reasoning mode are identical while
        reliability changes from $r$ to $1-r$.

        Under that flip, the true labels for `gain`, `negative_gain`, and `z_star` change sign.
        Both agreement-difference labels and all three agreement-vector labels remain invariant.
        For each all-token/all-layer probe, the analysis applies all output components together on
        CUDA (or MPS when explicitly allowed) and reports the corresponding sign-flip or invariance
        error, plus the raw residual cosine similarity, relative L2 change, and token-ID match
        fraction. Prompt positions are aligned exactly; completion positions are positional
        comparisons and the token-ID match map reveals where generated continuations diverge.
        """,
        "symmetric-reliability-rationale",
    ),
    code(
        r"""
        def symmetric_reliability_pairs() -> tuple[tuple[Fraction, Fraction], ...]:
            values = set(reliability_fractions)
            return tuple(sorted(
                (value, 1 - value) for value in values if value < Fraction(1, 2)
            ))


        def matched_symmetric_rows(
            reasoning: bool, low: Fraction, high: Fraction
        ) -> list[tuple[dict[str, object], dict[str, object]]]:
            rows = [
                dict(row) for row in results_by_reasoning[reasoning]
                if row["split"] == "test" and row_reliability(row) in (low, high)
            ]
            grouped: dict[tuple[int, str], dict[Fraction, dict[str, object]]] = defaultdict(dict)
            for row in rows:
                key = (int(row["question_set_index"]), str(row["presentation_order"]))
                reliability = row_reliability(row)
                if reliability in grouped[key]:
                    raise ValueError(f"Duplicate symmetric-reliability row for {key}.")
                grouped[key][reliability] = row
            if not grouped or any(set(pair) != {low, high} for pair in grouped.values()):
                raise ValueError("Symmetric reliability matching is incomplete.")
            invariant_fields = (
                "question_set_index",
                "membership_sets",
                "observed_reports",
                "candidate_1",
                "candidate_2",
                "candidate_value_order",
                "presentation_order",
                "reasoning",
                "split",
                "split_group_id",
                "base_transcript_id",
                "agreement_first_minus_second",
                "agreement_second_minus_first",
                "agreement_first_vector",
                "agreement_second_vector",
                "agreement_concatenated_vector",
            )
            result = []
            for key in sorted(grouped):
                low_row, high_row = grouped[key][low], grouped[key][high]
                for field in invariant_fields:
                    if low_row[field] != high_row[field]:
                        raise ValueError(f"Matched rows differ at invariant field {field!r}.")
                if not math.isclose(float(low_row["gain"]), -float(high_row["gain"]), abs_tol=1e-12):
                    raise ValueError("Matched gains are not exact sign flips.")
                if not math.isclose(float(low_row["z_star"]), -float(high_row["z_star"]), abs_tol=1e-12):
                    raise ValueError("Matched z_star labels are not exact sign flips.")
                result.append((low_row, high_row))
            return result


        def load_probe_block(
            reasoning: bool, layer: int, block_start: int
        ) -> dict[str, np.ndarray]:
            weights_path, _ = block_paths(reasoning, layer, block_start)
            if not block_is_complete(reasoning, layer, block_start):
                raise FileNotFoundError(f"Incomplete probe block {weights_path}.")
            with safe_open(weights_path, framework="pt", device="cpu") as handle:
                return {key: handle.get_tensor(key).numpy() for key in handle.keys()}


        def resolve_probe_inference_device() -> torch.device:
            if PROBE_INFERENCE_DEVICE == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda:0")
                if (
                    ALLOW_MPS_FOR_PROBE_INFERENCE
                    and torch.backends.mps.is_available()
                ):
                    return torch.device("mps")
                raise RuntimeError(
                    "Probe inference requires CUDA or the explicitly allowed MPS fallback."
                )
            device = torch.device(PROBE_INFERENCE_DEVICE)
            if device.type == "cpu":
                raise ValueError("CPU probe inference is disabled; select CUDA or MPS.")
            if device.type == "mps" and not ALLOW_MPS_FOR_PROBE_INFERENCE:
                raise ValueError("MPS probe inference was not enabled.")
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("The configured CUDA inference device is unavailable.")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("The configured MPS inference device is unavailable.")
            return device


        def paired_activation_block(
            pairs: list[tuple[dict[str, object], dict[str, object]]],
            *,
            layer: int,
            positions: tuple[int, ...],
        ) -> dict[int, tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]]:
            low_rows_by_position = {position: [] for position in positions}
            high_rows_by_position = {position: [] for position in positions}
            low_vectors = {position: [] for position in positions}
            high_vectors = {position: [] for position in positions}
            key = activation_key(layer)
            for low_row, high_row in pairs:
                with safe_open(activation_path(low_row), framework="pt", device="cpu") as low_handle:
                    low_tensor = low_handle.get_tensor(key)
                with safe_open(activation_path(high_row), framework="pt", device="cpu") as high_handle:
                    high_tensor = high_handle.get_tensor(key)
                available = [
                    position for position in positions
                    if position < low_tensor.shape[0] and position < high_tensor.shape[0]
                ]
                if not available:
                    continue
                low_array = low_tensor[available].float().numpy()
                high_array = high_tensor[available].float().numpy()
                for array_index, position in enumerate(available):
                    low_rows_by_position[position].append(low_row)
                    high_rows_by_position[position].append(high_row)
                    low_vectors[position].append(low_array[array_index])
                    high_vectors[position].append(high_array[array_index])
            result = {}
            for position in positions:
                if not low_vectors[position]:
                    continue
                result[position] = (
                    low_rows_by_position[position],
                    high_rows_by_position[position],
                    np.stack(low_vectors[position]).astype(np.float32, copy=False),
                    np.stack(high_vectors[position]).astype(np.float32, copy=False),
                )
            return result


        def symmetry_block_path(
            reasoning: bool, low: Fraction, layer: int, block_start: int
        ) -> Path:
            pair_name = f"{low.numerator}_of_{low.denominator}__{(1-low).numerator}_of_{(1-low).denominator}"
            return (
                SYMMETRY_ROOT / MODE_NAMES[reasoning] / pair_name / f"layer_{layer:02d}"
                / f"tokens_{block_start:05d}_{block_start + TOKEN_BLOCK_SIZE - 1:05d}.jsonl"
            )


        def analyze_symmetry_block(
            *,
            reasoning: bool,
            low: Fraction,
            high: Fraction,
            layer: int,
            positions: tuple[int, ...],
            pairs: list[tuple[dict[str, object], dict[str, object]]],
        ) -> None:
            output_path = symmetry_block_path(reasoning, low, layer, positions[0])
            if output_path.is_file():
                existing = [
                    json.loads(line)
                    for line in output_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if existing and all(
                    row.get("config_fingerprint") == CONFIG_FINGERPRINT for row in existing
                ):
                    return
            activation_pairs = paired_activation_block(
                pairs, layer=layer, positions=positions
            )
            weights = load_probe_block(reasoning, layer, positions[0])
            inference_device = resolve_probe_inference_device()
            weight_lookup = {
                int(position): index
                for index, position in enumerate(weights["token_positions"])
            }
            output_rows = []
            for position in positions:
                if position not in activation_pairs or position not in weight_lookup:
                    continue
                low_rows, high_rows, X_low, X_high = activation_pairs[position]
                weight_index = weight_lookup[position]
                mean = weights["feature_mean"][weight_index]
                scale = weights["feature_scale"][weight_index]
                coef = weights["coef"][weight_index]
                intercept = weights["intercept"][weight_index]
                if (
                    not all(np.isfinite(value).all() for value in (
                        X_low, X_high, mean, scale, coef, intercept
                    ))
                    or np.any(scale <= 0)
                ):
                    raise FloatingPointError(
                        f"Invalid symmetry input at layer={layer}, token={position}."
                    )
                X_low_device = torch.as_tensor(
                    X_low, dtype=torch.float32, device=inference_device
                )
                X_high_device = torch.as_tensor(
                    X_high, dtype=torch.float32, device=inference_device
                )
                mean_device = torch.as_tensor(
                    mean, dtype=torch.float32, device=inference_device
                )
                scale_device = torch.as_tensor(
                    scale, dtype=torch.float32, device=inference_device
                )
                coef_device = torch.as_tensor(
                    coef, dtype=torch.float32, device=inference_device
                )
                intercept_device = torch.as_tensor(
                    intercept, dtype=torch.float32, device=inference_device
                )
                prediction_low = (
                    ((X_low_device - mean_device) / scale_device) @ coef_device.T
                    + intercept_device
                ).cpu().numpy()
                prediction_high = (
                    ((X_high_device - mean_device) / scale_device) @ coef_device.T
                    + intercept_device
                ).cpu().numpy()
                if not all(np.isfinite(value).all() for value in (
                    prediction_low, prediction_high
                )):
                    raise FloatingPointError(
                        f"Non-finite symmetry prediction at layer={layer}, token={position}."
                    )
                target_low = joint_target_matrix(low_rows)
                target_high = joint_target_matrix(high_rows)
                norms_low = np.linalg.norm(X_low, axis=1)
                norms_high = np.linalg.norm(X_high, axis=1)
                raw_cosine = np.sum(X_low * X_high, axis=1) / np.maximum(
                    norms_low * norms_high, 1e-12
                )
                relative_l2 = np.linalg.norm(X_high - X_low, axis=1) / np.maximum(
                    0.5 * (norms_low + norms_high), 1e-12
                )
                token_matches = [
                    int(low_row["teacher_forced_input_ids"][position]
                        == high_row["teacher_forced_input_ids"][position])
                    for low_row, high_row in zip(low_rows, high_rows, strict=True)
                ]
                common = {
                    "config_fingerprint": CONFIG_FINGERPRINT,
                    "reasoning": reasoning,
                    "reasoning_mode": MODE_NAMES[reasoning],
                    "low_reliability_exact": str(low),
                    "high_reliability_exact": str(high),
                    "layer": layer,
                    "token_position": position,
                    "matched_pair_count": len(low_rows),
                    "raw_activation_cosine": float(np.mean(raw_cosine)),
                    "raw_activation_relative_l2": float(np.mean(relative_l2)),
                    "token_id_match_fraction": float(np.mean(token_matches)),
                    "probe_inference_device": str(inference_device),
                }
                for target, symmetry in TARGET_SYMMETRY.items():
                    target_slice = TARGET_SLICES[target]
                    low_prediction = prediction_low[:, target_slice]
                    high_prediction = prediction_high[:, target_slice]
                    low_truth = target_low[:, target_slice]
                    high_truth = target_high[:, target_slice]
                    predicted_symmetry_residual = (
                        low_prediction + high_prediction
                        if symmetry == "sign_flip"
                        else high_prediction - low_prediction
                    )
                    true_symmetry_residual = (
                        low_truth + high_truth
                        if symmetry == "sign_flip"
                        else high_truth - low_truth
                    )
                    if not np.allclose(true_symmetry_residual, 0.0, atol=1e-12):
                        raise ValueError(f"True labels violate {symmetry} for {target}.")
                    change_error = (
                        (high_prediction - low_prediction)
                        - (high_truth - low_truth)
                    )
                    output_rows.append({
                        **common,
                        "target": target,
                        "target_width": TARGET_WIDTHS[target],
                        "expected_symmetry": symmetry,
                        "probe_symmetry_mae": float(
                            np.mean(np.abs(predicted_symmetry_residual))
                        ),
                        "predicted_change_rmse_vs_true": float(
                            np.sqrt(np.mean(change_error**2))
                        ),
                        "low_prediction_mae": float(
                            np.mean(np.abs(low_prediction - low_truth))
                        ),
                        "high_prediction_mae": float(
                            np.mean(np.abs(high_prediction - high_truth))
                        ),
                    })
            atomic_write_jsonl(output_path, output_rows)
        """,
        "symmetry-helpers",
    ),
    code(
        r"""
        if RUN_SYMMETRIC_RELIABILITY_ANALYSIS:
            if not results_by_reasoning or not probe_metric_rows:
                raise RuntimeError(
                    "Completed captures and all-token probe artifacts are required."
                )
            for reasoning in REASONING_VALUES:
                maximum = max_positions_by_reasoning[reasoning]
                for low, high in symmetric_reliability_pairs():
                    pairs = matched_symmetric_rows(reasoning, low, high)
                    print(
                        MODE_NAMES[reasoning], str(low), "<->", str(high),
                        "held-out matched transcripts:", len(pairs),
                    )
                    display({
                        "reasoning_mode": MODE_NAMES[reasoning],
                        "low_reliability": str(low),
                        "high_reliability": str(high),
                        "question_set_index": pairs[0][0]["question_set_index"],
                        "membership_sets": pairs[0][0]["membership_sets"],
                        "observed_reports": pairs[0][0]["observed_reports"],
                        "candidate_value_order": pairs[0][0]["candidate_value_order"],
                        "presentation_order": pairs[0][0]["presentation_order"],
                        "agreement_first_vector": pairs[0][0]["agreement_first_vector"],
                        "agreement_second_vector": pairs[0][0]["agreement_second_vector"],
                        "gain_low_high": [pairs[0][0]["gain"], pairs[0][1]["gain"]],
                        "z_star_low_high": [pairs[0][0]["z_star"], pairs[0][1]["z_star"]],
                    })
                    for layer in range(NUM_LAYERS):
                        for block_start in range(0, maximum, TOKEN_BLOCK_SIZE):
                            positions = tuple(range(
                                block_start, min(block_start + TOKEN_BLOCK_SIZE, maximum)
                            ))
                            if not block_is_complete(reasoning, layer, block_start):
                                raise RuntimeError(
                                    f"Probe block missing at layer={layer}, token={block_start}."
                                )
                            analyze_symmetry_block(
                                reasoning=reasoning,
                                low=low,
                                high=high,
                                layer=layer,
                                positions=positions,
                                pairs=pairs,
                            )
        else:
            print(
                "Symmetric reliability analysis is paused. Enable it only after the main "
                "all-token probe sweep is complete."
            )


        def load_symmetry_metrics() -> list[dict[str, object]]:
            rows = []
            for path in sorted(SYMMETRY_ROOT.glob("reasoning_*/*/layer_*/*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row.get("config_fingerprint") == CONFIG_FINGERPRINT:
                            rows.append(row)
            return rows


        symmetry_metric_rows = load_symmetry_metrics()
        if symmetry_metric_rows:
            symmetry_metrics = pd.DataFrame(symmetry_metric_rows)
            display(
                symmetry_metrics.groupby(
                    ["reasoning_mode", "low_reliability_exact", "target"], as_index=False
                ).agg(
                    token_layer_cells=("probe_symmetry_mae", "count"),
                    median_probe_symmetry_mae=("probe_symmetry_mae", "median"),
                    median_change_rmse=("predicted_change_rmse_vs_true", "median"),
                    median_raw_activation_cosine=("raw_activation_cosine", "median"),
                    median_raw_activation_relative_l2=("raw_activation_relative_l2", "median"),
                )
            )
            for (reasoning_mode, low_reliability), selected in symmetry_metrics.groupby(
                ["reasoning_mode", "low_reliability_exact"]
            ):
                maximum = int(selected["token_position"].max()) + 1
                fig, axes = plt.subplots(5, 2, figsize=(21, 21), constrained_layout=True)
                first = selected.drop_duplicates(["layer", "token_position"])
                raw_specs = (
                    ("raw_activation_cosine", "raw residual cosine", "coolwarm", -1.0, 1.0),
                    ("token_id_match_fraction", "token-ID match fraction", "viridis", 0.0, 1.0),
                )
                for axis, (metric, title, cmap, vmin, vmax) in zip(
                    axes.flat[:2], raw_specs, strict=True
                ):
                    matrix = np.full((NUM_LAYERS, maximum), np.nan)
                    for row in first.itertuples(index=False):
                        matrix[int(row.layer), int(row.token_position)] = float(
                            getattr(row, metric)
                        )
                    image = axis.imshow(
                        matrix, origin="lower", aspect="auto", interpolation="nearest",
                        cmap=cmap, vmin=vmin, vmax=vmax,
                    )
                    axis.set(title=title, xlabel="token position", ylabel="layer")
                    fig.colorbar(image, ax=axis, shrink=0.76)
                for axis, target in zip(axes.flat[2:], TARGET_WIDTHS, strict=True):
                    target_rows = selected[selected["target"] == target]
                    matrix = np.full((NUM_LAYERS, maximum), np.nan)
                    for row in target_rows.itertuples(index=False):
                        matrix[int(row.layer), int(row.token_position)] = math.log10(
                            float(row.probe_symmetry_mae) + 1e-6
                        )
                    image = axis.imshow(
                        matrix, origin="lower", aspect="auto", interpolation="nearest",
                        cmap="magma",
                    )
                    axis.set(
                        title=f"{target}: {TARGET_SYMMETRY[target]} error",
                        xlabel="token position",
                        ylabel="layer",
                    )
                    fig.colorbar(image, ax=axis, shrink=0.76, label="log10 MAE")
                high_reliability = str(1 - Fraction(low_reliability))
                fig.suptitle(
                    f"Matched reliability flip {low_reliability} ↔ {high_reliability} — "
                    f"{reasoning_mode}",
                    fontsize=16,
                )
                figure_path = (
                    SYMMETRY_ROOT
                    / f"symmetry_maps_{reasoning_mode}_{low_reliability.replace('/', '_of_')}.png"
                )
                fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
                plt.show()
        else:
            symmetry_metrics = pd.DataFrame()
            print("No symmetric-reliability comparison artifacts are loaded yet.")
        """,
        "run-and-display-symmetry-analysis",
    ),
    md(
        r"""
        ## Publish probe weights to Hugging Face

        The final optional stage publishes the fitted safetensor blocks, held-out metrics,
        figures, symmetry results, and a self-describing model card. Activations and raw
        transcripts are outside `PROBE_ROOT` and are never uploaded. `upload_large_folder` is
        resumable and parallelized for the many token-block artifacts. The gate is disabled by
        default: set `HF_PROBE_REPO_ID`, authenticate with the normal Hugging Face mechanism, and
        enable `RUN_UPLOAD_PROBES_TO_HF` only after the complete probe sweep has been verified.
        """,
        "huggingface-upload-rationale",
    ),
    code(
        r"""
        def prepare_probe_model_card() -> Path:
            metadata_root = PROBE_ROOT / "metadata"
            metadata_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(metadata_root / "configuration.json", config_for_fingerprint)
            atomic_write_json(metadata_root / "dataset_manifest.json", dataset.manifest)
            atomic_write_json(metadata_root / "publication_manifest.json", {
                "config_fingerprint": CONFIG_FINGERPRINT,
                "base_model": MODEL_NAME_OR_PATH,
                "target_widths": TARGET_WIDTHS,
                "target_slices": {
                    target: [target_slice.start, target_slice.stop]
                    for target, target_slice in TARGET_SLICES.items()
                },
                "solver": "exact_batched_cuda_dual_cholesky",
                "training_device_required": PROBE_TRAIN_DEVICE,
                "training_rows": len(train_rows),
                "held_out_rows": len(test_rows),
                "train_test_transcript_groups_disjoint": True,
                "expected_probe_metric_rows": expected_metric_rows,
                "actual_probe_metric_rows": len(probe_metric_rows),
            })
            model_card = f'''---
        library_name: pytorch
        base_model: {MODEL_NAME_OR_PATH}
        tags:
        - mechanistic-interpretability
        - linear-probe
        - qwen3.5
        ---

        # Qwen3.5 agreement-grid residual-stream probes

        Configuration fingerprint: `{CONFIG_FINGERPRINT}`

        These safetensors contain exact ridge readouts for all absolute teacher-forced token
        positions and all {NUM_LAYERS} language layers. The eight logical targets have widths
        `{json.dumps(TARGET_WIDTHS, sort_keys=True)}` and are stored as a joint
        {JOINT_TARGET_WIDTH}-component output with slices recorded in each block's metadata.

        Training used batched dual Cholesky on CUDA (`{PROBE_TRAIN_DEVICE}`), with token positions
        and all target components solved concurrently. Complete question/report schedules,
        reliabilities, reasoning variants, and candidate presentation orders were assigned as one
        split group; no transcript group occurs in both train and held-out test data.

        See `metadata/configuration.json`, `metadata/dataset_manifest.json`, and
        `metadata/publication_manifest.json` for reconstruction details. Raw activations and
        transcript rows are intentionally not included.
        '''
            path = PROBE_ROOT / "README.md"
            path.write_text(model_card, encoding="utf-8")
            return path


        huggingface_publication = None
        if RUN_UPLOAD_PROBES_TO_HF:
            if not probe_sweep_complete:
                raise RuntimeError(
                    "Refusing Hugging Face upload until every expected probe metric row exists."
                )
            if not isinstance(HF_PROBE_REPO_ID, str) or not HF_PROBE_REPO_ID.strip():
                raise ValueError("Set a non-empty HF_PROBE_REPO_ID before enabling upload.")
            model_card_path = prepare_probe_model_card()
            api = HfApi()
            repo_url = api.create_repo(
                repo_id=HF_PROBE_REPO_ID,
                repo_type="model",
                private=HF_PROBE_REPO_PRIVATE,
                exist_ok=True,
            )
            api.upload_large_folder(
                repo_id=HF_PROBE_REPO_ID,
                repo_type="model",
                revision=HF_PROBE_REVISION,
                folder_path=PROBE_ROOT,
                allow_patterns=[
                    "README.md",
                    "metadata/**",
                    "reasoning_*/**/*.safetensors",
                    "reasoning_*/**/*.metrics.jsonl",
                    "figures/**",
                    "symmetric_reliability/**",
                ],
                num_workers=HF_UPLOAD_NUM_WORKERS,
            )
            huggingface_publication = {
                "repo_id": HF_PROBE_REPO_ID,
                "repo_url": str(repo_url),
                "revision": HF_PROBE_REVISION,
                "model_card": str(model_card_path),
            }
            display(huggingface_publication)
        else:
            print(
                "Hugging Face publication is paused. Configure HF_PROBE_REPO_ID and enable "
                "RUN_UPLOAD_PROBES_TO_HF only after reviewing the completed local sweep."
            )
        """,
        "publish-probes-to-huggingface",
    ),
    code(
        r"""
        final_status = {
            "config_fingerprint": CONFIG_FINGERPRINT,
            "dataset_rows": len(dataset),
            "leakage_audit_passed": (
                train_groups.isdisjoint(test_groups)
                and train_transcripts.isdisjoint(test_transcripts)
                and train_row_ids.isdisjoint(test_row_ids)
            ),
            "capture_modes_loaded": [
                MODE_NAMES[reasoning] for reasoning in results_by_reasoning
            ],
            "probe_metric_rows": len(probe_metric_rows),
            "expected_probe_metric_rows": expected_metric_rows,
            "probe_sweep_complete": probe_sweep_complete,
            "symmetry_metric_rows": len(symmetry_metric_rows),
            "huggingface_publication": huggingface_publication,
            "dataset_root": str(DATASET_ROOT),
            "probe_root": str(PROBE_ROOT),
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
