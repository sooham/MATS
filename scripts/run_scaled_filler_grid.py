#!/usr/bin/env python3
"""Run the frozen immediate filler probe over every requested N×r cell."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mats_experiments.filler_token_probe import (
    FILLER_SURFACES,
    FINAL_PREFIX,
    generate_scaled_decisions,
    prefilled_input_ids,
    render_scaled_messages,
    scaled_target_surface,
    verify_single_token,
)

DEFAULT_NS = "12,16,20,24,28,32,64,128"
DEFAULT_RS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen--Qwen3.5-4B")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--n-values", default=DEFAULT_NS)
    parser.add_argument("--reliabilities", default=DEFAULT_RS)
    parser.add_argument("--examples-per-cell", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=20260831)
    parser.add_argument("--filler", choices=sorted(FILLER_SURFACES), default="underscore")
    parser.add_argument("--filler-count", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def make_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["n"]), float(row["reliability"]))].append(row)
    cells = []
    for (n, reliability), group in sorted(grouped.items()):
        non_ties = [row for row in group if row["normative_answer"] != "tie"]
        target_first = [
            row
            for row in non_ties
            if row["normative_answer"] == row["first_candidate"]
        ]
        target_second = [
            row
            for row in non_ties
            if row["normative_answer"] == row["second_candidate"]
        ]
        target_low = [
            row
            for row in non_ties
            if row["normative_answer"] == min(row["candidates"])
        ]
        target_high = [
            row
            for row in non_ties
            if row["normative_answer"] == max(row["candidates"])
        ]

        def accuracy(subset: list[dict[str, object]]) -> float | None:
            return (
                sum(bool(row["correct"]) for row in subset) / len(subset)
                if subset
                else None
            )

        cells.append(
            {
                "n": n,
                "reliability": reliability,
                "records": len(group),
                "accuracy": accuracy(group),
                "candidate_only_accuracy": accuracy(non_ties),
                "first_target_accuracy": accuracy(target_first),
                "second_target_accuracy": accuracy(target_second),
                "lower_target_accuracy": accuracy(target_low),
                "higher_target_accuracy": accuracy(target_high),
                "normative_counts": dict(
                    sorted(Counter(str(row["normative_role"]) for row in group).items())
                ),
                "prediction_counts": dict(
                    sorted(Counter(str(row["predicted_surface"]) for row in group).items())
                ),
            }
        )
    return {"records": len(rows), "cells": cells}


def main() -> None:
    args = parse_args()
    n_values = [int(value) for value in parse_csv(args.n_values)]
    reliabilities = [Fraction(value) for value in parse_csv(args.reliabilities)]
    decisions = generate_scaled_decisions(
        n_values=n_values,
        reliabilities=reliabilities,
        examples_per_cell=args.examples_per_cell,
        base_seed=args.base_seed,
    )
    expected = len(n_values) * len(reliabilities) * args.examples_per_cell
    if len(decisions) != expected:
        raise AssertionError(f"Expected {expected} decisions, got {len(decisions)}.")

    existing: list[dict[str, object]] = []
    if args.output.exists() and not args.overwrite:
        existing = [json.loads(line) for line in args.output.read_text().splitlines()]
    completed = {str(row["example_id"]) for row in existing}
    pending = [decision for decision in decisions if decision.example_id not in completed]

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    filler_id = verify_single_token(tokenizer, FILLER_SURFACES[args.filler])
    final_prefix_ids = tokenizer(FINAL_PREFIX, add_special_tokens=False)["input_ids"]
    answer_ids = {
        surface: verify_single_token(tokenizer, surface) for surface in ("X", "Y", "=")
    }
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()

    print(
        f"Scoring {len(pending)} pending examples across {len(n_values)} N values and "
        f"{len(reliabilities)} reliabilities.",
        flush=True,
    )
    rows = list(existing)
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        messages_batch = [render_scaled_messages(decision) for decision in batch]
        sequences = [
            prefilled_input_ids(
                processor=processor,
                messages=messages,
                filler_token_id=filler_id,
                filler_count=args.filler_count,
                final_prefix_ids=final_prefix_ids,
            )
            for messages in messages_batch
        ]
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
        option_ids = torch.tensor(
            [answer_ids[surface] for surface in ("X", "Y", "=")], device=device
        )
        option_logits = logits.index_select(-1, option_ids).cpu()
        option_log_probs = option_logits.log_softmax(-1)
        for decision, sequence, values, log_probs in zip(
            batch, sequences, option_logits.tolist(), option_log_probs.tolist()
        ):
            answer_surfaces = ("X", "Y", "=")
            predicted_surface = answer_surfaces[max(range(3), key=values.__getitem__)]
            surface_to_answer = {
                decision.first_alias: decision.first_candidate,
                decision.second_alias: decision.second_candidate,
                "=": "tie",
            }
            predicted_answer = surface_to_answer[predicted_surface]
            target_surface = scaled_target_surface(decision)
            target = decision.normative_answer
            if target == "tie":
                normative_role = "tie"
            elif target == decision.first_candidate:
                normative_role = "first"
            else:
                normative_role = "second"
            rows.append(
                {
                    "example_id": decision.example_id,
                    "n": decision.n,
                    "k": len(decision.observations),
                    "policy": "random_memoryless",
                    "reliability": float(decision.reliability),
                    "replicate": decision.replicate,
                    "candidates": sorted(
                        [decision.first_candidate, decision.second_candidate]
                    ),
                    "first_candidate": decision.first_candidate,
                    "second_candidate": decision.second_candidate,
                    "first_alias": decision.first_alias,
                    "second_alias": decision.second_alias,
                    "normative_answer": target,
                    "normative_role": normative_role,
                    "normative_surface": target_surface,
                    "predicted_answer": predicted_answer,
                    "predicted_surface": predicted_surface,
                    "correct": predicted_answer == target,
                    "answer_logits": dict(zip(answer_surfaces, map(float, values))),
                    "conditional_answer_log_probs": dict(
                        zip(answer_surfaces, map(float, log_probs))
                    ),
                    "input_token_count": len(sequence),
                    "enable_thinking": False,
                    "single_call": True,
                }
            )
        write_jsonl(args.output, rows)
        if start == 0 or (start + len(batch)) % (args.batch_size * 20) == 0:
            print(f"completed {start + len(batch)}/{len(pending)}", flush=True)

    rows.sort(key=lambda row: (int(row["n"]), float(row["reliability"]), int(row["replicate"])))
    write_jsonl(args.output, rows)
    result_summary = make_summary(rows)
    manifest = {
        "model": args.model_path.name,
        "scope": "complete N-by-reliability immediate filler grid",
        "grid": {
            "n_values": n_values,
            "reliabilities": list(map(float, reliabilities)),
            "examples_per_cell": args.examples_per_cell,
            "cells": len(n_values) * len(reliabilities),
            "base_seed": args.base_seed,
        },
        "game": {"k": 3, "policy": "random_memoryless", "subset_size": "N/2"},
        "protocol": {
            "filler": args.filler,
            "filler_surface": FILLER_SURFACES[args.filler],
            "filler_token_id": filler_id,
            "filler_count": args.filler_count,
            "final_prefix": FINAL_PREFIX,
            "final_prefix_token_ids": final_prefix_ids,
            "answer_token_ids": answer_ids,
            "enable_thinking": False,
            "generated_reasoning_tokens": 0,
            "position_control": "exact 2x2 balance of low/high presentation and X/Y alias mapping per cell",
        },
        "summary": result_summary,
    }
    manifest_path = args.manifest or args.output.with_name(
        args.output.stem.replace("results", "manifest") + ".json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "cells": len(result_summary["cells"])}, indent=2))


if __name__ == "__main__":
    main()
