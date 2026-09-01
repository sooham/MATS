#!/usr/bin/env python3
"""Run one honest Qwen continuation per noisy-source candidate example."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mats_experiments.model_loading import load_qwen
from mats_experiments.qwen_scoring import generate_responses_batch
from mats_experiments.raw_reasoning_probe import (
    CandidateDecision,
    load_candidate_decisions,
    normative_absolute,
)
from mats_experiments.single_call_candidate import (
    VARIANTS,
    candidate_order,
    parse_candidate_number,
    render_messages,
)


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
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--batch-by-reliability",
        action="store_true",
        help=(
            "Group equal-reliability examples before batching. This changes only "
            "execution order and is useful when generation lengths differ by reliability."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the variant generation cap (intended for diagnostic smoke runs).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable the checkpoint's native thinking chat-template mode. The "
            "--max-new-tokens value remains the total continuation cap."
        ),
    )
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
    parser.add_argument(
        "--per-reliability-limit",
        type=int,
        help="Deterministically retain the first N examples at each reliability.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_repeats(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _balanced_limit(
    decisions: list[CandidateDecision], limit: int | None
) -> list[CandidateDecision]:
    if limit is None:
        return decisions
    if limit < 1:
        raise ValueError("--per-reliability-limit must be positive.")
    counts: Counter[object] = Counter()
    selected = []
    for decision in decisions:
        if counts[decision.reliability] < limit:
            selected.append(decision)
            counts[decision.reliability] += 1
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["reliability"])].append(row)
    predictions = Counter(str(row["predicted_absolute"]) for row in rows)
    return {
        "records": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "parse_rate": sum(bool(row["parse_success"]) for row in rows) / len(rows),
        "prediction_counts": dict(sorted(predictions.items())),
        "by_reliability": {
            str(reliability): {
                "n": len(group),
                "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
                "parse_rate": sum(bool(row["parse_success"]) for row in group) / len(group),
            }
            for reliability, group in sorted(grouped.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite.")
    repeats = _parse_repeats(args.repeats)
    decisions = _balanced_limit(
        load_candidate_decisions(args.transcripts, repeats=repeats),
        args.per_reliability_limit,
    )
    if args.batch_by_reliability:
        decisions.sort(key=lambda decision: (float(decision.reliability), decision.example_id))
    if not decisions:
        raise RuntimeError("No candidate decisions matched the requested split.")
    variant = VARIANTS[args.variant]
    max_new_tokens = (
        variant.max_new_tokens
        if args.max_new_tokens is None
        else args.max_new_tokens
    )
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")

    loaded = load_qwen(
        model_path=args.model_path,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        gptq_backend=args.gptq_backend,
    )
    model = loaded.model
    processor = loaded.processor
    device = loaded.input_device
    print(f"Model load: {json.dumps(loaded.metadata, sort_keys=True)}", flush=True)

    rows: list[dict[str, object]] = []
    for start in range(0, len(decisions), args.batch_size):
        batch = decisions[start : start + args.batch_size]
        messages_batch = [render_messages(decision, variant=variant) for decision in batch]
        responses = generate_responses_batch(
            model=model,
            processor=processor,
            device=device,
            messages_batch=messages_batch,
            enable_thinking=args.enable_thinking,
            max_new_tokens=max_new_tokens,
        )
        for decision, messages, response in zip(batch, messages_batch, responses):
            generated_text = str(response["generated_text"])
            thinking_completed = (
                not args.enable_thinking or "</think>" in generated_text
            )
            answer_text = (
                generated_text.rsplit("</think>", 1)[-1]
                if thinking_completed
                else ""
            )
            predicted = parse_candidate_number(
                answer_text,
                candidates=(decision.left_candidate, decision.right_candidate),
            )
            target = normative_absolute(decision)
            rows.append(
                {
                    "example_id": decision.example_id,
                    "repeat": decision.repeat,
                    "reliability": float(decision.reliability),
                    "reliability_exact": (
                        f"{decision.reliability.numerator}/{decision.reliability.denominator}"
                    ),
                    "n": decision.n,
                    "k": len(decision.observations),
                    "policy": decision.policy,
                    "candidates": [decision.left_candidate, decision.right_candidate],
                    "presentation_order": list(candidate_order(decision)),
                    "variant": args.variant,
                    "enable_thinking": args.enable_thinking,
                    "max_new_tokens": max_new_tokens,
                    "messages": messages,
                    "generated_text": generated_text,
                    "generated_token_count": response["generated_token_count"],
                    "reached_eos": response["reached_eos"],
                    "hit_max_new_tokens": response["hit_max_new_tokens"],
                    "thinking_completed": thinking_completed,
                    "predicted_absolute": predicted,
                    "normative_absolute": target,
                    "parse_success": predicted is not None,
                    "correct": predicted == target,
                }
            )
        _write_jsonl(args.output, rows)
        print(f"completed {len(rows)}/{len(decisions)}", flush=True)

    summary = _summarize(rows)
    manifest = {
        "model": loaded.metadata,
        "scope": "candidate-only, exactly one Qwen continuation per transcript",
        "game": {"n": 8, "k": 3, "policy": "random_memoryless"},
        "split": {
            "repeats": repeats,
            "per_reliability_limit": args.per_reliability_limit,
            "batch_by_reliability": args.batch_by_reliability,
        },
        "variant": {
            "name": args.variant,
            **variant.__dict__,
            "effective_max_new_tokens": max_new_tokens,
            "enable_thinking": args.enable_thinking,
        },
        "input_contract": {
            "visible": "public rules, raw membership questions/reports, candidate identities",
            "withheld": (
                "truthful answers, channel coins, evaluator memberships/counts, likelihoods, "
                "posteriors, and target labels"
            ),
            "orientation": "SHA-256(example_id), independent of scoring target",
        },
        "summary": summary,
    }
    manifest_path = args.manifest or args.output.with_name(
        args.output.stem.replace("results", "manifest") + ".json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
