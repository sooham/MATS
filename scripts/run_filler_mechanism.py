#!/usr/bin/env python3
"""Run alias-swap causal controls and a layerwise filler logit lens."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mats_experiments.filler_token_probe import (
    FILLER_SURFACES,
    FINAL_PREFIX,
    ScaledDecision,
    generate_scaled_decisions,
    prefilled_input_ids,
    render_scaled_messages,
    scaled_target_surface,
    verify_single_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen--Qwen3.5-4B")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--reliabilities", default="0.1,0.3,0.7,0.9")
    parser.add_argument("--examples-per-reliability", type=int, default=4)
    parser.add_argument("--filler-counts", default="0,1,2,4,8,11,16,32,64,100")
    parser.add_argument("--fillers", default="underscore,period")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def swapped_aliases(decision: ScaledDecision) -> ScaledDecision:
    return replace(
        decision,
        first_alias=decision.second_alias,
        second_alias=decision.first_alias,
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def select_base_decisions(
    *, n: int, reliabilities: list[Fraction], examples_per_reliability: int
) -> list[ScaledDecision]:
    bank = generate_scaled_decisions(
        n_values=[n],
        reliabilities=reliabilities,
        examples_per_cell=32,
    )
    selected = []
    for reliability in reliabilities:
        candidates = [
            decision
            for decision in bank
            if decision.reliability == reliability
            and decision.normative_answer != "tie"
        ]
        if len(candidates) < examples_per_reliability:
            raise RuntimeError(f"Not enough non-tie decisions at r={reliability}.")
        selected.extend(candidates[:examples_per_reliability])
    return selected


def summarize(rows: list[dict[str, object]], final_layer: int) -> dict[str, object]:
    final_rows = [row for row in rows if row["layer"] == final_layer]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in final_rows:
        grouped[(str(row["filler"]), int(row["filler_count"]))].append(row)
    conditions = []
    for (filler, filler_count), group in sorted(grouped.items()):
        pairs: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            pairs[str(row["example_id"])].append(row)
        complete_pairs = [pair for pair in pairs.values() if len(pair) == 2]
        conditions.append(
            {
                "filler": filler,
                "filler_count": filler_count,
                "records": len(group),
                "surface_accuracy": sum(bool(row["correct"]) for row in group) / len(group),
                "prediction_counts": dict(
                    sorted(Counter(str(row["predicted_surface"]) for row in group).items())
                ),
                "alias_swap_semantic_consistency": (
                    sum(pair[0]["predicted_answer"] == pair[1]["predicted_answer"] for pair in complete_pairs)
                    / len(complete_pairs)
                ),
            }
        )
    return {"layer_records": len(rows), "final_conditions": conditions}


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite.")
    reliabilities = [Fraction(value) for value in parse_csv(args.reliabilities)]
    filler_counts = [int(value) for value in parse_csv(args.filler_counts)]
    fillers = parse_csv(args.fillers)
    unknown = set(fillers) - set(FILLER_SURFACES)
    if unknown:
        raise ValueError(f"Unknown fillers: {sorted(unknown)}")
    base_decisions = select_base_decisions(
        n=args.n,
        reliabilities=reliabilities,
        examples_per_reliability=args.examples_per_reliability,
    )
    jobs = []
    for filler_count in filler_counts:
        for filler in fillers:
            for decision in base_decisions:
                jobs.append((decision, "original", filler, filler_count))
                jobs.append((swapped_aliases(decision), "swapped", filler, filler_count))

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    filler_ids = {
        filler: verify_single_token(tokenizer, FILLER_SURFACES[filler])
        for filler in fillers
    }
    final_prefix_ids = tokenizer(FINAL_PREFIX, add_special_tokens=False)["input_ids"]
    answer_ids = {
        surface: verify_single_token(tokenizer, surface) for surface in ("X", "Y", "=")
    }
    option_id_tensor = torch.tensor(
        [answer_ids[surface] for surface in ("X", "Y", "=")], device=device
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()
    final_norm = model.model.language_model.norm
    final_layer = len(model.model.language_model.layers)

    print(
        f"Scoring {len(jobs)} inputs with {final_layer + 1} logit-lens depths.",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        messages_batch = [render_scaled_messages(job[0]) for job in batch]
        sequences = [
            prefilled_input_ids(
                processor=processor,
                messages=messages,
                filler_token_id=filler_ids[filler],
                filler_count=filler_count,
                final_prefix_ids=final_prefix_ids,
            )
            for messages, (_, _, filler, filler_count) in zip(messages_batch, batch)
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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                logits_to_keep=1,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None or len(hidden_states) != final_layer + 1:
                raise RuntimeError("Unexpected hidden-state structure.")
            layer_option_logits = []
            for layer, hidden in enumerate(hidden_states):
                final_position = hidden[:, -1, :]
                lens_hidden = final_position if layer == final_layer else final_norm(final_position)
                logits = model.lm_head(lens_hidden).float().index_select(-1, option_id_tensor)
                layer_option_logits.append(logits.cpu())

        for batch_index, (decision, alias_condition, filler, filler_count) in enumerate(batch):
            surface_to_answer = {
                decision.first_alias: decision.first_candidate,
                decision.second_alias: decision.second_candidate,
                "=": "tie",
            }
            target_surface = scaled_target_surface(decision)
            for layer, layer_logits in enumerate(layer_option_logits):
                values = layer_logits[batch_index].tolist()
                surfaces = ("X", "Y", "=")
                predicted_surface = surfaces[max(range(3), key=values.__getitem__)]
                predicted_answer = surface_to_answer[predicted_surface]
                rows.append(
                    {
                        "example_id": decision.example_id,
                        "reliability": float(decision.reliability),
                        "alias_condition": alias_condition,
                        "first_candidate": decision.first_candidate,
                        "second_candidate": decision.second_candidate,
                        "first_alias": decision.first_alias,
                        "second_alias": decision.second_alias,
                        "normative_answer": decision.normative_answer,
                        "normative_surface": target_surface,
                        "filler": filler,
                        "filler_token_id": filler_ids[filler],
                        "filler_count": filler_count,
                        "layer": layer,
                        "answer_logits": dict(zip(surfaces, map(float, values))),
                        "predicted_surface": predicted_surface,
                        "predicted_answer": predicted_answer,
                        "correct": predicted_answer == decision.normative_answer,
                        "enable_thinking": False,
                    }
                )
        if start == 0 or (start + len(batch)) % (args.batch_size * 20) == 0:
            print(f"completed {start + len(batch)}/{len(jobs)}", flush=True)

    write_jsonl(args.output, rows)
    result_summary = summarize(rows, final_layer)
    manifest = {
        "model": args.model_path.name,
        "scope": "causal alias swap and layerwise filler logit lens",
        "selection": {
            "n": args.n,
            "reliabilities": list(map(float, reliabilities)),
            "examples_per_reliability": args.examples_per_reliability,
            "base_examples": len(base_decisions),
        },
        "protocol": {
            "fillers": {
                filler: {"surface": FILLER_SURFACES[filler], "token_id": filler_ids[filler]}
                for filler in fillers
            },
            "filler_counts": filler_counts,
            "alias_conditions": ["original", "swapped"],
            "layers": list(range(final_layer + 1)),
            "answer_token_ids": answer_ids,
            "enable_thinking": False,
            "generated_reasoning_tokens": 0,
        },
        "summary": result_summary,
    }
    manifest_path = args.manifest or args.output.with_name(
        args.output.stem.replace("results", "manifest") + ".json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"layer_records": len(rows), "conditions": len(result_summary["final_conditions"])}, indent=2))


if __name__ == "__main__":
    main()
