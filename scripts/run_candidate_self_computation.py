"""Reproduce the multi-call Qwen candidate self-computation probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mats_experiments.candidate_self_computation import (
    atomic_membership_prompt,
    choose_qwen_count,
    count_adjudication_prompt,
    endpoint_count_final_prompt,
    interior_final_prompt,
    membership_audit,
    parse_count,
    parse_final_label,
    parse_yes_no,
    semantic_from_label,
    truth_report_pairs,
    visible_count_prompt,
)
from mats_experiments.qwen_scoring import generate_responses_batch
from mats_experiments.raw_reasoning_probe import load_candidate_decisions


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
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def parse_repeats(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def generate_texts(
    *, model: object, processor: object, device: object, prompts: Sequence[str], batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        responses = generate_responses_batch(
            model=model,
            processor=processor,
            device=device,
            messages_batch=[[{"role": "user", "content": prompt}] for prompt in batch],
            enable_thinking=False,
            max_new_tokens=max_new_tokens,
        )
        texts.extend(str(response["generated_text"]) for response in responses)
        print(f"  generated {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)
    return texts


def split_name(repeat: int) -> str:
    if repeat in {0, 1, 2, 3}:
        return "development"
    if repeat in {8, 9, 10, 11}:
        return "validation"
    if repeat in {12, 13, 14, 15}:
        return "test"
    if repeat in {4, 5, 6, 7}:
        return "replication"
    return "other"


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    repeats = parse_repeats(args.repeats)
    decisions = load_candidate_decisions(args.transcripts, repeats=repeats)
    if args.limit is not None:
        decisions = decisions[: args.limit]
    if not decisions:
        raise RuntimeError("No candidate decisions matched the requested filters.")

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Candidate self-computation requires CUDA.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()

    membership_jobs: list[tuple[str, int, int, str]] = []
    for decision in decisions:
        for candidate in (decision.left_candidate, decision.right_candidate):
            for observation in decision.observations:
                membership_jobs.append(
                    (
                        decision.example_id,
                        candidate,
                        observation.question,
                        atomic_membership_prompt(candidate, observation.subset),
                    )
                )
    print(f"Atomic membership stage: {len(membership_jobs)} calls", flush=True)
    membership_texts = generate_texts(
        model=model,
        processor=processor,
        device=device,
        prompts=[job[3] for job in membership_jobs],
        batch_size=args.batch_size,
        max_new_tokens=24,
    )
    memberships: dict[tuple[str, int], list[object]] = {}
    membership_generations: dict[tuple[str, int], list[str]] = {}
    for (example_id, candidate, _question, _prompt), text in zip(
        membership_jobs, membership_texts
    ):
        memberships.setdefault((example_id, candidate), []).append(parse_yes_no(text))
        membership_generations.setdefault((example_id, candidate), []).append(text)

    count_jobs: list[tuple[str, int, bool, str]] = []
    pairs_by_candidate: dict[tuple[str, int], list[tuple[int, object, str]]] = {}
    for decision in decisions:
        reports = [observation.report for observation in decision.observations]
        for candidate in (decision.left_candidate, decision.right_candidate):
            key = (decision.example_id, candidate)
            pairs = truth_report_pairs(memberships[key], reports)  # type: ignore[arg-type]
            pairs_by_candidate[key] = pairs  # type: ignore[assignment]
            for reverse in (False, True):
                count_jobs.append(
                    (
                        decision.example_id,
                        candidate,
                        reverse,
                        visible_count_prompt(pairs, reverse=reverse),  # type: ignore[arg-type]
                    )
                )
    print(f"Visible count stage: {len(count_jobs)} calls", flush=True)
    count_texts = generate_texts(
        model=model,
        processor=processor,
        device=device,
        prompts=[job[3] for job in count_jobs],
        batch_size=args.batch_size,
        max_new_tokens=80,
    )
    count_drafts: dict[tuple[str, int], dict[str, object]] = {}
    for (example_id, candidate, reverse, _prompt), text in zip(count_jobs, count_texts):
        count_drafts.setdefault((example_id, candidate), {})[
            "reverse" if reverse else "forward"
        ] = {"text": text, "count": parse_count(text)}

    adjudication_jobs: list[tuple[str, int, str]] = []
    for key, drafts in count_drafts.items():
        forward = drafts["forward"]
        reverse = drafts["reverse"]
        if forward["count"] is None or forward["count"] != reverse["count"]:  # type: ignore[index]
            adjudication_jobs.append(
                (
                    key[0],
                    key[1],
                    count_adjudication_prompt(
                        forward_draft=str(forward["text"]),  # type: ignore[index]
                        reverse_draft=str(reverse["text"]),  # type: ignore[index]
                        pairs=pairs_by_candidate[key],  # type: ignore[arg-type]
                    ),
                )
            )
    adjudications: dict[tuple[str, int], dict[str, object]] = {}
    if adjudication_jobs:
        print(f"Count adjudication stage: {len(adjudication_jobs)} calls", flush=True)
        texts = generate_texts(
            model=model,
            processor=processor,
            device=device,
            prompts=[job[2] for job in adjudication_jobs],
            batch_size=args.batch_size,
            max_new_tokens=192,
        )
        for (example_id, candidate, _prompt), text in zip(adjudication_jobs, texts):
            adjudications[(example_id, candidate)] = {
                "text": text,
                "count": parse_count(text),
            }

    qwen_counts: dict[tuple[str, int], int | None] = {}
    for key, drafts in count_drafts.items():
        qwen_counts[key] = choose_qwen_count(
            forward=drafts["forward"]["count"],  # type: ignore[index,arg-type]
            reverse=drafts["reverse"]["count"],  # type: ignore[index,arg-type]
            adjudicated=adjudications.get(key, {}).get("count"),  # type: ignore[arg-type]
        )

    final_prompts = []
    for decision in decisions:
        first_key = (decision.example_id, decision.left_candidate)
        second_key = (decision.example_id, decision.right_candidate)
        if decision.reliability in {0, 1}:
            special_count = 3 if decision.reliability == 1 else 0
            final_prompts.append(
                endpoint_count_final_prompt(
                    first_count=qwen_counts[first_key],
                    second_count=qwen_counts[second_key],
                    special_count=special_count,
                )
            )
        else:
            final_prompts.append(
                interior_final_prompt(
                    first_count=qwen_counts[first_key],
                    second_count=qwen_counts[second_key],
                    r=float(decision.reliability),
                )
            )
    print(f"Final comparison stage: {len(final_prompts)} calls", flush=True)
    final_texts = generate_texts(
        model=model,
        processor=processor,
        device=device,
        prompts=final_prompts,
        batch_size=args.batch_size,
        max_new_tokens=320,
    )

    records = []
    for decision, final_prompt, final_text in zip(decisions, final_prompts, final_texts):
        left_key = (decision.example_id, decision.left_candidate)
        right_key = (decision.example_id, decision.right_candidate)
        label = parse_final_label(final_text)
        predicted = semantic_from_label(label)
        subsets = [observation.subset for observation in decision.observations]
        left_audit = membership_audit(
            candidate=decision.left_candidate,
            subsets=subsets,
            qwen_truths=memberships[left_key],  # type: ignore[arg-type]
        )
        right_audit = membership_audit(
            candidate=decision.right_candidate,
            subsets=subsets,
            qwen_truths=memberships[right_key],  # type: ignore[arg-type]
        )
        records.append(
            {
                "example_id": decision.example_id,
                "split": split_name(decision.repeat),
                "repeat": decision.repeat,
                "reliability": float(decision.reliability),
                "left_candidate": decision.left_candidate,
                "right_candidate": decision.right_candidate,
                "normative_semantic_choice": decision.normative_choice,
                "predicted_semantic_choice": predicted,
                "parsed": predicted is not None,
                "correct": predicted == decision.normative_choice,
                "qwen_memberships": {
                    str(decision.left_candidate): memberships[left_key],
                    str(decision.right_candidate): memberships[right_key],
                },
                "membership_generations": {
                    str(decision.left_candidate): membership_generations[left_key],
                    str(decision.right_candidate): membership_generations[right_key],
                },
                "membership_answers_correct_audit": left_audit["correct"]
                + right_audit["correct"],
                "membership_answers_total": left_audit["total"] + right_audit["total"],
                "count_drafts": {
                    str(decision.left_candidate): count_drafts[left_key],
                    str(decision.right_candidate): count_drafts[right_key],
                },
                "adjudications": {
                    str(candidate): adjudications.get((decision.example_id, candidate))
                    for candidate in (decision.left_candidate, decision.right_candidate)
                },
                "qwen_match_counts": {
                    str(decision.left_candidate): qwen_counts[left_key],
                    str(decision.right_candidate): qwen_counts[right_key],
                },
                "endpoint_positive_checks": None,
                "final_prompt": final_prompt,
                "generation": final_text,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    summary = {
        "records": len(records),
        "accuracy": sum(bool(record["correct"]) for record in records) / len(records),
        "parse_rate": sum(bool(record["parsed"]) for record in records) / len(records),
        "membership_accuracy_audit": (
            sum(int(record["membership_answers_correct_audit"]) for record in records)
            / sum(int(record["membership_answers_total"]) for record in records)
        ),
        "by_reliability": {
            str(reliability): (
                sum(bool(row["correct"]) for row in records if row["reliability"] == reliability)
                / sum(row["reliability"] == reliability for row in records)
            )
            for reliability in sorted({float(row["reliability"]) for row in records})
        },
    }
    manifest = {
        "model": args.model_path.name,
        "scope": "candidate-only multi-call Qwen self-computation reproduction",
        "final_readout": "A/B/C comparison from Qwen-generated counts; 320-token cap",
        "repeats": repeats,
        "input_contract": {
            "initial_evidence": "raw membership sets and SOURCE reports",
            "intermediates": "all task-dependent memberships and counts are generated by Qwen",
            "evaluator": "formats, routes, parses, and audits; audit values are never fed back",
        },
        "summary": summary,
    }
    manifest_path = args.manifest or args.output.with_name(
        args.output.stem.replace("results", "manifest") + ".json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
