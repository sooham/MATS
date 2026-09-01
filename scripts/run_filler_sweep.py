#!/usr/bin/env python3
"""Score token-exact filler counts for the N=8 candidate-2-versus-7 probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mats_experiments.filler_token_probe import (
    FILLER_SURFACES,
    FINAL_PREFIX,
    n8_target_surface,
    prefilled_input_ids,
    render_n8_messages,
    verify_single_token,
)
from mats_experiments.model_loading import load_qwen
from mats_experiments.raw_reasoning_probe import load_candidate_decisions, normative_absolute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path("artifacts/minimal_noisy_source/transcripts.jsonl"),
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen--Qwen3.5-4B")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repeats", default="0")
    parser.add_argument("--fillers", default=",".join(FILLER_SURFACES))
    parser.add_argument("--f-min", type=int, default=0)
    parser.add_argument("--f-max", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        help="Use Accelerate device mapping and cap this process's GPU weight allocation.",
    )
    parser.add_argument("--max-cpu-memory-gib", type=float, default=700)
    parser.add_argument(
        "--gptq-backend",
        default="gptq_torch",
        help="GPTQModel backend value (default: gptq_torch).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["filler"]), int(row["filler_count"]))].append(row)
    conditions = []
    for (filler, filler_count), group in sorted(grouped.items()):
        non_ties = [row for row in group if row["normative_absolute"] != "tie"]
        target_two = [row for row in group if row["normative_absolute"] == 2]
        target_seven = [row for row in group if row["normative_absolute"] == 7]
        conditions.append(
            {
                "filler": filler,
                "filler_count": filler_count,
                "n": len(group),
                "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
                "candidate_only_accuracy": (
                    sum(bool(row["correct"]) for row in non_ties) / len(non_ties)
                    if non_ties
                    else None
                ),
                "candidate_2_accuracy": (
                    sum(bool(row["correct"]) for row in target_two) / len(target_two)
                    if target_two
                    else None
                ),
                "candidate_7_accuracy": (
                    sum(bool(row["correct"]) for row in target_seven) / len(target_seven)
                    if target_seven
                    else None
                ),
                "prediction_counts": dict(
                    sorted(Counter(str(row["predicted_absolute"]) for row in group).items())
                ),
            }
        )
    return {"records": len(rows), "conditions": conditions}


def main() -> None:
    args = parse_args()
    if args.f_min < 0 or args.f_max < args.f_min:
        raise ValueError("Require 0 <= f-min <= f-max.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    fillers = parse_csv(args.fillers)
    unknown = set(fillers) - set(FILLER_SURFACES)
    if unknown:
        raise ValueError(f"Unknown fillers: {sorted(unknown)}")
    repeats = [int(value) for value in parse_csv(args.repeats)]
    decisions = load_candidate_decisions(args.transcripts, repeats=repeats)
    if not decisions:
        raise RuntimeError("No decisions matched the requested repeats.")

    existing: list[dict[str, object]] = []
    if args.output.exists() and not args.overwrite:
        existing = [json.loads(line) for line in args.output.read_text().splitlines()]
    completed = {
        (str(row["example_id"]), str(row["filler"]), int(row["filler_count"]))
        for row in existing
    }

    import torch
    loaded = load_qwen(
        model_path=args.model_path,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        gptq_backend=args.gptq_backend,
    )
    model = loaded.model
    processor = loaded.processor
    device = loaded.input_device
    tokenizer = processor.tokenizer
    filler_ids = {
        name: verify_single_token(tokenizer, FILLER_SURFACES[name]) for name in fillers
    }
    final_prefix_ids = tokenizer(FINAL_PREFIX, add_special_tokens=False)["input_ids"]
    if not final_prefix_ids:
        raise ValueError("FINAL prefix tokenization was empty.")
    answer_ids = {
        surface: verify_single_token(tokenizer, surface) for surface in ("2", "7", "=")
    }
    print(f"Model load: {json.dumps(loaded.metadata, sort_keys=True)}", flush=True)

    jobs = []
    for filler_count in range(args.f_min, args.f_max + 1):
        for filler in fillers:
            for decision in decisions:
                key = (decision.example_id, filler, filler_count)
                if key not in completed:
                    jobs.append((decision, filler, filler_count))
    print(
        f"Scoring {len(jobs)} pending records: {len(decisions)} decisions, "
        f"fillers={fillers}, F={args.f_min}..{args.f_max}",
        flush=True,
    )

    rows = list(existing)
    write_jsonl(args.output, rows)
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        sequences = []
        messages_batch = []
        for decision, filler, filler_count in batch:
            messages = render_n8_messages(decision)
            messages_batch.append(messages)
            sequences.append(
                prefilled_input_ids(
                    processor=processor,
                    messages=messages,
                    filler_token_id=filler_ids[filler],
                    filler_count=filler_count,
                    final_prefix_ids=final_prefix_ids,
                )
            )
        max_length = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), max_length), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, -len(sequence) :] = torch.tensor(sequence, device=device)
            attention_mask[index, -len(sequence) :] = 1
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :].float()
        option_id_tensor = torch.tensor(
            [answer_ids[surface] for surface in ("2", "7", "=")],
            device=logits.device,
        )
        option_logits = logits.index_select(-1, option_id_tensor).cpu()
        option_log_probs = option_logits.log_softmax(-1)
        batch_rows = []
        for (decision, filler, filler_count), messages, sequence, values, log_probs in zip(
            batch,
            messages_batch,
            sequences,
            option_logits.tolist(),
            option_log_probs.tolist(),
        ):
            surfaces = ("2", "7", "=")
            predicted_surface = surfaces[max(range(3), key=values.__getitem__)]
            predicted_absolute: object = (
                "tie" if predicted_surface == "=" else int(predicted_surface)
            )
            target_absolute = normative_absolute(decision)
            target_surface = n8_target_surface(decision)
            batch_rows.append(
                {
                    "example_id": decision.example_id,
                    "repeat": decision.repeat,
                    "reliability": float(decision.reliability),
                    "n": decision.n,
                    "k": len(decision.observations),
                    "policy": decision.policy,
                    "candidates": [decision.left_candidate, decision.right_candidate],
                    "filler": filler,
                    "filler_surface": FILLER_SURFACES[filler],
                    "filler_token_id": filler_ids[filler],
                    "filler_count": filler_count,
                    "final_prefix": FINAL_PREFIX,
                    "final_prefix_token_ids": final_prefix_ids,
                    "input_token_count": len(sequence),
                    "answer_token_ids": answer_ids,
                    "answer_logits": dict(zip(surfaces, map(float, values))),
                    "conditional_answer_log_probs": dict(
                        zip(surfaces, map(float, log_probs))
                    ),
                    "predicted_surface": predicted_surface,
                    "predicted_absolute": predicted_absolute,
                    "normative_surface": target_surface,
                    "normative_absolute": target_absolute,
                    "correct": predicted_absolute == target_absolute,
                    "enable_thinking": False,
                    "single_call": True,
                }
            )
        rows.extend(batch_rows)
        append_jsonl(args.output, batch_rows)
        if start == 0 or (start + len(batch)) % (args.batch_size * 20) == 0:
            print(f"completed {start + len(batch)}/{len(jobs)}", flush=True)

    rows.sort(key=lambda row: (str(row["filler"]), int(row["filler_count"]), str(row["example_id"])))
    write_jsonl(args.output, rows)
    result_summary = summary(rows)
    manifest = {
        "model": loaded.metadata,
        "scope": "immediate one-token answer after token-exact filler and FINAL prefix",
        "game": {"n": 8, "k": 3, "policy": "random_memoryless", "candidates": [2, 7]},
        "split": {"repeats": repeats},
        "protocol": {
            "fillers": {
                name: {"surface": FILLER_SURFACES[name], "token_id": filler_ids[name]}
                for name in fillers
            },
            "f_min": args.f_min,
            "f_max": args.f_max,
            "final_prefix": FINAL_PREFIX,
            "final_prefix_token_ids": final_prefix_ids,
            "answer_token_ids": answer_ids,
            "enable_thinking": False,
            "generated_reasoning_tokens": 0,
            "batch_size": args.batch_size,
        },
        "input_contract": (
            "public rules, raw questions/reports, candidates; no evaluator-derived evidence"
        ),
        "summary": result_summary,
    }
    manifest_path = args.manifest or args.output.with_name(
        args.output.stem.replace("results", "manifest") + ".json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "conditions": len(result_summary["conditions"])}, indent=2))


if __name__ == "__main__":
    main()
