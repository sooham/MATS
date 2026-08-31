"""Run generated raw-evidence candidate probes against local Qwen3.5-4B."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mats_experiments.qwen_scoring import generate_responses_batch
from mats_experiments.raw_reasoning_probe import (
    CandidateDecision,
    Orientation,
    PromptStyle,
    aggregate_orientation_pair,
    load_candidate_decisions,
    render_raw_candidate_prompt,
    score_generated_response,
)


@dataclass(frozen=True)
class Variant:
    style: PromptStyle
    enable_thinking: bool
    max_new_tokens: int


VARIANTS = {
    "silent_no_thinking": Variant("silent", False, 64),
    "structured_no_thinking": Variant("structured", False, 160),
    "structured_likelihood_no_thinking": Variant("structured_likelihood", False, 192),
    "atomic_likelihood_no_thinking": Variant("atomic_likelihood", False, 256),
    "compact_no_thinking": Variant("compact", False, 512),
    "direct_no_thinking": Variant("direct", False, 768),
    "audit_no_thinking": Variant("audit", False, 768),
    "direct_thinking": Variant("direct", True, 2048),
    "audit_thinking": Variant("audit", True, 2048),
    "compact_thinking": Variant("compact", True, 2048),
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    parser.add_argument(
        "--variants",
        default=(
            "silent_no_thinking,structured_no_thinking,"
            "structured_likelihood_no_thinking,compact_no_thinking"
        ),
    )
    parser.add_argument("--orientations", default="forward")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_existing(path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str, str]]]:
    if not path.exists():
        return [], set()
    records = []
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    keys = {
        (str(row["example_id"]), str(row["variant"]), str(row["orientation"]))
        for row in records
    }
    return records, keys


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[(str(record["variant"]), float(record["reliability"]))].append(record)
    by_variant_reliability = []
    for (variant, reliability), rows in sorted(groups.items()):
        by_variant_reliability.append(
            {
                "variant": variant,
                "reliability": reliability,
                "n": len(rows),
                "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
                "parse_rate": sum(bool(row["parse_success"]) for row in rows) / len(rows),
            }
        )

    pairs: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        pairs[(str(record["example_id"]), str(record["variant"]))].append(record)
    paired = [
        aggregate_orientation_pair(rows)
        for rows in pairs.values()
        if {row["orientation"] for row in rows} == {"forward", "reverse"}
    ]
    return {
        "by_variant_reliability": by_variant_reliability,
        "orientation_pairs": len(paired),
        "paired_accuracy": (
            sum(bool(row["correct"]) for row in paired) / len(paired) if paired else None
        ),
        "orientation_consistency": (
            sum(bool(row["orientation_consistent"]) for row in paired) / len(paired)
            if paired
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    repeats = [int(value) for value in parse_csv(args.repeats)]
    variant_names = parse_csv(args.variants)
    orientations = parse_csv(args.orientations)
    unknown_variants = set(variant_names) - set(VARIANTS)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")
    if not set(orientations) <= {"forward", "reverse"}:
        raise ValueError("Orientations must be forward and/or reverse.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")

    decisions = load_candidate_decisions(args.transcripts, repeats=repeats)
    if args.limit is not None:
        decisions = decisions[: args.limit]
    if not decisions:
        raise RuntimeError("No candidate decisions matched the requested filters.")

    if args.overwrite:
        existing_records, completed = [], set()
    else:
        existing_records, completed = _load_existing(args.output)

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("This model-backed experiment requires CUDA.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()

    records = list(existing_records)
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        jobs: list[tuple[CandidateDecision, Orientation, str]] = []
        for decision in decisions:
            for orientation_value in orientations:
                orientation: Orientation = orientation_value  # type: ignore[assignment]
                key = (decision.example_id, variant_name, orientation)
                if key in completed:
                    continue
                prompt = render_raw_candidate_prompt(
                    decision, style=variant.style, orientation=orientation
                )
                jobs.append((decision, orientation, prompt))

        print(
            f"{variant_name}: {len(jobs)} pending responses "
            f"(thinking={variant.enable_thinking}, max_new_tokens={variant.max_new_tokens})",
            flush=True,
        )
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start : start + args.batch_size]
            responses = generate_responses_batch(
                model=model,
                processor=processor,
                device=device,
                messages_batch=[[{"role": "user", "content": prompt}] for _, _, prompt in batch],
                enable_thinking=variant.enable_thinking,
                max_new_tokens=variant.max_new_tokens,
            )
            for (decision, orientation, prompt), response in zip(batch, responses):
                generated_text = str(response["generated_text"])
                score = score_generated_response(
                    decision, orientation=orientation, generated_text=generated_text
                )
                records.append(
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
                        "left_candidate": decision.left_candidate,
                        "right_candidate": decision.right_candidate,
                        "normative_semantic_choice": decision.normative_choice,
                        "variant": variant_name,
                        "prompt_style": variant.style,
                        "enable_thinking": variant.enable_thinking,
                        "orientation": orientation,
                        "prompt": prompt,
                        "generated_text": generated_text,
                        **score,
                    }
                )
                completed.add((decision.example_id, variant_name, orientation))
            _write_jsonl(args.output, records)
            print(f"  completed {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)

    manifest_path = args.manifest or args.output.with_name(
        args.output.stem.replace("results", "manifest") + ".json"
    )
    manifest = {
        "model": args.model_path.name,
        "input_contract": {
            "visible_to_model": [
                "uniform prior and domain size",
                "stated channel reliability",
                "three raw membership sets and SOURCE YES/NO reports",
                "the two candidate names",
                "generic reasoning/format instructions",
            ],
            "forbidden": [
                "evaluator-derived memberships",
                "match counts or other sufficient statistics",
                "likelihoods or posterior values",
                "target-derived corrections",
            ],
        },
        "protocol": {
            "repeats": repeats,
            "variants": {name: asdict(VARIANTS[name]) for name in variant_names},
            "orientations": orientations,
            "batch_size": args.batch_size,
            "decisions": len(decisions),
            "records": len(records),
            "greedy_generation": True,
        },
        "summary": _summary(records),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
