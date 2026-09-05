"""Check whether CPU inference can satisfy notebook 16's saved-logit compatibility gate."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['TORCHAO_FORCE_SKIP_LOADING_SO_FILES'] = '1'

import json
import time
import argparse
import types
from pathlib import Path
import torch
from transformers import Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/noisy_channel_bayesian_experiment_2_activation_patching/Qwen_Qwen3.5-9B_4c87a623'
parser = argparse.ArgumentParser()
parser.add_argument('--bf16-linear-fp32-accumulation', action='store_true')
parser.add_argument('--replay-capture-forward', action='store_true',
                    help='Use the complete saved sequence and full ConditionalGeneration forward, as the original capture does.')
args = parser.parse_args()
precision = 'bf16_linear_fp32_accumulation' if args.bf16_linear_fp32_accumulation else 'float32'
scope = '_full_capture_forward' if args.replay_capture_forward else ''
OUT = BASE / f'analyses/completed_reasoning_off_20260905/cpu_patch_parity_{precision}{scope}.json'
torch.set_num_threads(8)
assert not torch.cuda.is_available()
rows = [json.loads(line) for line in (BASE / 'runs/qwen35_9b_n9_k3_r10_s64_selected_tokens_v2_reasoning_off/results.jsonl').read_text().splitlines()]
selected = [next(r for r in rows if r['split'] == 'test' and r['reliabilities_exact'][0] == rel and r['delta_a'] != 0)
            for rel in ['1/20', '19/20', '1/4', '3/4']]
print('Loading separate CPU model in float32; GPU hidden; eight low-priority threads', flush=True)
start = time.monotonic()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    ROOT / 'models/Qwen--Qwen3.5-9B', local_files_only=True,
    dtype=torch.float32, device_map='cpu', attn_implementation='sdpa').eval()
assert all(p.device.type == 'cpu' for p in model.parameters())
if args.bf16_linear_fp32_accumulation:
    # AVX2 CPUs lack fast BF16 GEMM. Keep BF16 activations and weights exactly,
    # performing linear accumulation in FP32 and rounding each linear output to BF16.
    # This is an explicitly labeled CPU implementation; saved-state parity still gates use.
    model.bfloat16()
    def cpu_linear(module, x):
        return torch.nn.functional.linear(x.float(), module.weight, module.bias).to(x.dtype)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.float()
            module.forward = types.MethodType(cpu_linear, module)
    # Numerical unit check against native BF16 linear at a small tractable shape.
    generator = torch.Generator().manual_seed(42)
    x_check = torch.randn(7, 29, generator=generator).bfloat16()
    w_check = torch.randn(13, 29, generator=generator).bfloat16()
    torch.testing.assert_close(torch.nn.functional.linear(x_check, w_check),
        torch.nn.functional.linear(x_check.float(), w_check.float()).bfloat16(), rtol=0, atol=0)
    print('BF16 linear accumulation check passed exactly', flush=True)
print('Loaded in', time.monotonic() - start, 'seconds', flush=True)
records = []
for row in selected:
    start = time.monotonic()
    boundary = row['answer_boundary_input_index']
    ids = torch.tensor([row['teacher_forced_input_ids'] if args.replay_capture_forward
                        else row['teacher_forced_input_ids'][:boundary + 1]])
    with torch.inference_mode():
        candidates = [row['candidate_1_answer_token_ids'][0], row['candidate_2_answer_token_ids'][0]]
        if args.replay_capture_forward:
            full = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            logits = full[0, boundary, candidates].clone()
            del full
        else:
            h = model.model.language_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[:, -1, :]
            logits = torch.nn.functional.linear(h.float(), model.lm_head.weight[candidates])[0]
            if args.bf16_linear_fp32_accumulation:
                logits = logits.bfloat16()
    margin = float(logits[0] - logits[1])
    saved = row['answer_surface_raw_logits'][str(row['candidate_1'])] - row['answer_surface_raw_logits'][str(row['candidate_2'])]
    record = {'row_id': row['row_id'], 'reliability': row['reliabilities_exact'][0], 'cpu_logits': logits.float().tolist(), 'cpu_margin': margin,
              'saved_margin': saved, 'absolute_difference': abs(margin - saved), 'seconds': time.monotonic() - start}
    records.append(record)
    print(record, flush=True)
    payload = {'device': 'cpu', 'dtype': precision, 'saved_capture_dtype': 'bfloat16',
               'replays_full_capture_forward': args.replay_capture_forward,
               'scored_position': 'answer_boundary_input_index',
               'tolerance_from_notebook16': .05, 'records': records,
               'complete': len(records) == len(selected),
               'compatible': len(records) == len(selected) and all(r['absolute_difference'] <= .05 for r in records),
               'gpu_used': False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    if not args.bf16_linear_fp32_accumulation and not args.replay_capture_forward:
        (OUT.parent / 'cpu_patch_parity.json').write_text(json.dumps(payload, indent=2) + '\n')
print('CPU parity check complete', flush=True)
