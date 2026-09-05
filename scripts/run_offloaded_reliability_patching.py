"""Memory-capped GPU replay of notebook 16 while independent CPU probe fits continue.

No existing process is contacted or changed. All weights stay on CPU between module
calls; only the calling process's CUDA allocator is capped. A short idle/headroom
preflight and checks on the original GPU owners protect the live run's resources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT/'artifacts/noisy_channel_bayesian_experiment_2_activation_patching/Qwen_Qwen3.5-9B_4c87a623'
ANALYSIS = MODEL_ROOT/'analyses/completed_reasoning_off_20260905'
OUT = ANALYSIS/'offloaded_reliability_interchange'


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def query_gpu():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(gpu) != 1:
        raise RuntimeError('This runner is scoped to the single verified GPU.')
    total, free, utilization = [int(v.strip()) for v in gpu[0].split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                         '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    owners = {}
    for line in processes:
        if line.strip():
            pid, memory = line.split(',')
            owners[int(pid.strip())] = int(memory.strip())
    return {'total_mib': total, 'free_mib': free, 'utilization_percent': utilization, 'owners': owners}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Execute the frozen patch design after compatibility passes')
    args = parser.parse_args()
    before = query_gpu()
    if before['free_mib'] < 11*1024:
        raise RuntimeError(f'Insufficient free memory for isolated capped replay: {before}')
    for _ in range(3):
        sample = query_gpu()
        if sample['utilization_percent'] > 1 or sample['free_mib'] < 11*1024:
            raise RuntimeError('GPU compute is active or headroom changed; replay was not started.')
        time.sleep(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['TORCHAO_FORCE_SKIP_LOADING_SO_FILES'] = '1'
    import sys
    sys.path.insert(0, str(ROOT/'notebooks'))
    import numpy as np
    import torch
    from accelerate import cpu_offload
    from safetensors import safe_open
    from transformers import Qwen3_5ForConditionalGeneration
    from patching_exploration import make_reliability_design

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(6144 / before['total_mib'])
    run_root = MODEL_ROOT/'runs/qwen35_9b_n9_k3_r10_s64_selected_tokens_v2_reasoning_off'
    rows = [json.loads(line) for line in (run_root/'results.jsonl').read_text().splitlines()]
    by_id = {r['row_id']: r for r in rows}
    assert len(rows) == 5120 and all(r['reasoning'] is False for r in rows)
    snapshot = json.loads((ANALYSIS/'snapshot.json').read_text())
    design = make_reliability_design(rows, snapshot, OUT)
    setup = {'cpu_offload': True, 'dtype': 'bfloat16', 'cuda_allocator_limit_mib': 6144,
             'before': before, 'existing_processes_modified': False,
             'capture_replay': 'full saved sequence; score at answer_boundary_input_index',
             'design_sha256': design['design_sha256'], 'pid': os.getpid()}
    atomic_json(OUT/'execution_setup.json', setup)

    def owner_guard():
        now = query_gpu()
        original = before['owners']
        unexpected = set(now['owners']) - set(original) - {os.getpid()}
        growth = {pid: now['owners'][pid] - memory for pid, memory in original.items()
                  if pid in now['owners'] and now['owners'][pid] > memory + 128}
        if unexpected or growth:
            raise RuntimeError(f'Another GPU workload changed; stopping our replay: {unexpected=}, {growth=}')

    print('Loading weights onto CPU; GPU allocator capped at 6 GiB', flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        ROOT/'models/Qwen--Qwen3.5-9B', local_files_only=True, dtype=torch.bfloat16,
        device_map='cpu', attn_implementation='sdpa').eval()
    cpu_offload(model, execution_device=torch.device('cuda:0'), offload_buffers=True,
                preload_module_classes=['Qwen3_5GatedDeltaNet'])
    layers = model.model.language_model.layers

    def saved_margin(row):
        values = row['answer_surface_raw_logits']
        return values[str(row['candidate_1'])] - values[str(row['candidate_2'])]

    def position(row, site='final_prompt'):
        value = row['prompt_token_sites'][site]
        if isinstance(value, list):
            assert len(value) == 1
            return value[0]
        return value

    def vector(row, layer, site='final_prompt'):
        index = row['activation_token_indices'].index(position(row, site))
        with safe_open(run_root/row['activation_path'], framework='pt', device='cpu') as handle:
            return handle.get_tensor(f'answer.resid_post.layer_{layer}')[index].clone()

    def forward(row, *, layer=None, donor_vector=None, site='final_prompt'):
        owner_guard()
        handle = None
        if layer is not None:
            def replace(module, inputs, output):
                del module, inputs
                patched = output.clone()
                patched[0, position(row, site)] = donor_vector.to(output.device, dtype=output.dtype)
                return patched
            handle = layers[layer].register_forward_hook(replace)
        try:
            ids = torch.tensor([row['teacher_forced_input_ids']], device='cuda')
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                boundary = row['answer_boundary_input_index']
                a, b = row['candidate_1_answer_token_ids'][0], row['candidate_2_answer_token_ids'][0]
                selected = logits[0, boundary, [a,b]].float().cpu()
            del logits, ids
            return float(selected[0]-selected[1])
        finally:
            if handle is not None:
                handle.remove()

    # Check every reliability in the frozen intervention design before patching.
    parity = []
    for reliability in ['1/20', '19/20', '1/4', '3/4']:
        row = next(r for r in rows if r['split']=='test' and r['reliabilities_exact'][0]==reliability and r['delta_a'] != 0)
        start = time.monotonic()
        replayed = forward(row)
        parity.append({'row_id': row['row_id'], 'reliability': reliability,
                       'saved_margin': saved_margin(row), 'replayed_margin': replayed,
                       'absolute_difference': abs(replayed-saved_margin(row)),
                       'seconds': time.monotonic()-start})
        print(parity[-1], flush=True)
        atomic_json(OUT/'compatibility.json', {'records': parity, 'complete': len(parity)==4,
                    'compatible': len(parity)==4 and all(r['absolute_difference'] <= .05 for r in parity),
                    'tolerance': .05, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                    'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2})
    if any(record['absolute_difference'] > .05 for record in parity):
        print('Original compatibility gate failed; no patches executed.', flush=True)
        return
    if not args.run:
        print('Compatibility passed; rerun with --run for the frozen patch design.', flush=True)
        return

    result_path = OUT/'results.jsonl'
    completed = [json.loads(line) for line in result_path.read_text().splitlines()] if result_path.exists() else []
    assert all(r['design_sha256']==design['design_sha256'] for r in completed)
    keys = {(r['recipient_row_id'],r['donor_row_id'],r['layer'],r['condition']) for r in completed}
    baseline = {}
    for pair_index, pair in enumerate(design['directions']):
        recipient, donor = by_id[pair['recipient_row_id']], by_id[pair['donor_row_id']]
        recipient_id = recipient['row_id']
        if recipient_id not in baseline:
            baseline[recipient_id] = forward(recipient)
            if abs(baseline[recipient_id]-saved_margin(recipient)) > .05:
                raise RuntimeError(f'Recipient baseline does not reproduce capture: {recipient_id}')
        for layer in design['layers']:
            for condition, source in [('self_patch',recipient),('reliability_interchange',donor)]:
                key = (recipient_id,donor['row_id'],layer,condition)
                if key in keys:
                    continue
                patched = forward(recipient, layer=layer, donor_vector=vector(source,layer))
                if condition=='self_patch' and abs(patched-baseline[recipient_id]) > .05:
                    raise RuntimeError('Saved-state self-patch changes the baseline beyond tolerance.')
                record = {**pair, 'layer': layer, 'site': design['site'], 'condition': condition,
                          'baseline_margin': baseline[recipient_id], 'patched_margin': patched,
                          'margin_change': patched-baseline[recipient_id],
                          'design_sha256': design['design_sha256'], 'reasoning': False,
                          'backend': 'native_bf16_gpu_with_cpu_weight_offload'}
                completed.append(record)
                keys.add(key)
                temporary = result_path.with_suffix('.jsonl.tmp')
                temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in completed))
                temporary.replace(result_path)
        print(f'Completed directed pair {pair_index+1}/{len(design["directions"])}; {len(completed)} records', flush=True)
    atomic_json(OUT/'status.json', {'complete': True, 'records': len(completed),
                'interchange_records': sum(r['condition']=='reliability_interchange' for r in completed),
                'self_patch_records': sum(r['condition']=='self_patch' for r in completed),
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
                'after': query_gpu(), 'existing_processes_modified': False})


if __name__ == '__main__':
    main()
