"""Native BF16, memory-capped GPU execution of frozen agreement-matched patches.

Experimental variants are batched only if their unpatched and saved-self baselines
pass the original 0.05 capture gate. Otherwise that recipient uses singleton replay.
No existing training process or notebook kernel is contacted or modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'notebooks'))
from run_offloaded_reliability_patching import query_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['discovery','evaluation'], required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cached-suffix', action='store_true',
                        help='Reuse exact full-sequence residuals before the patched layer; validate against full native forwards')
    args = parser.parse_args()
    assert 2 <= args.batch_size <= 4
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', TORCHAO_FORCE_SKIP_LOADING_SO_FILES='1')
    from agreement_matched_patching import OUT, RUN_ROOT, SITES, digest, eligible, key, label, load_rows, margin, positions, prepare, records, select_locations, write_json
    protocol = prepare()
    rows = load_rows()
    by_id = {r['row_id']: r for r in rows}
    output = OUT/args.phase
    output.mkdir(parents=True, exist_ok=True)
    if (output/'status.json').exists() and json.loads((output/'status.json').read_text()).get('complete'):
        print('This frozen phase is already complete; no GPU model loaded.', flush=True)
        return
    if args.phase == 'discovery':
        cohorts = [('discovery_grid_train', protocol['discovery_pairs'])]
        locations = [{'site': site, 'layer': layer} for site in SITES for layer in range(32)]
    else:
        locations = select_locations(protocol)['locations']
        cohorts = [('grid_train_followup', protocol['grid_followup_pairs']), ('totals_heldout', protocol['heldout_totals_pairs'])]

    # Jobs are keyed by recipient and vector source, avoiding repeated self/control
    # evaluations when a recipient occurs in multiple cross-pairs.
    jobs, edges = {}, []
    matching_groups = defaultdict(list)
    for row in rows:
        if eligible(row):
            for tier in ['totals','grid']:
                matching_groups[tier,row['split'],key(row,tier)].append(row)
    def add_job(recipient, site, layer, condition, sources, weights=None):
        job = {'recipient_id': recipient, 'site': site, 'layer': layer, 'condition': condition,
               'sources': sources, 'weights': weights}
        jid = digest(job)[:24]
        jobs[jid] = {**job, 'job_id': jid}
        return jid
    for cohort, pairs in cohorts:
        for pair in pairs:
            for direction, recipient_id, donor_id in [('denoising',pair['corrupt_id'],pair['clean_id']), ('noising',pair['clean_id'],pair['corrupt_id'])]:
                recipient = by_id[recipient_id]
                for loc in locations:
                    site, layer = loc['site'], loc['layer']
                    specs = [('cross_patch', site, [donor_id], None), ('self_patch', site, [recipient_id], None)]
                    if args.phase == 'evaluation':
                        group_key=key(recipient,pair['tier'])
                        group = matching_groups[pair['tier'],recipient['split'],group_key]
                        controls = sorted([r for r in group if r['row_id'] != recipient_id and label(r, True) == label(recipient, True)], key=lambda r: digest([recipient_id,r['row_id']])) if label(recipient, True) else []
                        control_id = controls[0]['row_id'] if controls else recipient_id
                        train_group = matching_groups[pair['tier'],'train',group_key]
                        assert train_group, 'Missing matched training mean'
                        train_ids = sorted(r['row_id'] for r in train_group)
                        specs += [('same_label_resampled' if controls else 'resampled_self_fallback', site, [control_id], None),
                                  ('training_group_mean', site, train_ids, [1/len(train_ids)]*len(train_ids)),
                                  ('off_target', 'domain_boundary', [donor_id], None)]
                    for condition, patch_site, sources, weights in specs:
                        jid = add_job(recipient_id, patch_site, layer, condition, sources, weights)
                        edges.append({**pair, 'cohort': cohort, 'direction': direction, 'recipient_id': recipient_id,
                                      'donor_id': donor_id, 'site': site, 'patch_site': patch_site,
                                      'layer': layer, 'condition': condition, 'job_id': jid})
    manifest = {'protocol_sha256': protocol['protocol_sha256'], 'phase': args.phase,
                'jobs': list(jobs.values()), 'edges': edges}
    manifest['design_sha256'] = digest(manifest)
    design_path = output/'execution_design.json'
    if design_path.exists():
        assert json.loads(design_path.read_text()) == manifest
    else:
        write_json(design_path, manifest)
    print(f'Frozen {args.phase}: {len(jobs)} unique forwards, {len(edges)} pair-condition records', flush=True)

    before = query_gpu()
    for _ in range(3):
        sample = query_gpu()
        if sample['utilization_percent'] > 1 or sample['free_mib'] < 11*1024:
            raise RuntimeError('Live GPU work/headroom prevents replay; no model loaded.')
        time.sleep(1)
    import torch
    from accelerate import cpu_offload
    from safetensors import safe_open
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(6144/before['total_mib'])
    setup = {'pid': os.getpid(), 'before': before, 'allocator_cap_mib': 6144, 'cpu_weight_offload': True,
             'batch_size_requested': args.batch_size, 'existing_processes_modified': False,
             'cached_suffix': args.cached_suffix,
             'design_sha256': manifest['design_sha256']}
    write_json(output/'execution_setup.json', setup)

    def guard():
        state = query_gpu()
        unexpected = set(state['owners']) - set(before['owners']) - {os.getpid()}
        growth = {pid:state['owners'][pid]-mem for pid,mem in before['owners'].items()
                  if pid in state['owners'] and state['owners'][pid] > mem+128}
        if unexpected or growth:
            raise RuntimeError(f'Other GPU workload changed; stopping only this replay: {unexpected=}, {growth=}')

    print('Loading separate CPU-offloaded native BF16 model; 6 GiB CUDA cap', flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(ROOT/'models/Qwen--Qwen3.5-9B',
        local_files_only=True, dtype=torch.bfloat16, device_map='cpu', attn_implementation='sdpa').eval()
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen--Qwen3.5-9B', local_files_only=True)
    cpu_offload(model, execution_device=torch.device('cuda:0'), offload_buffers=True,
                preload_module_classes=['Qwen3_5GatedDeltaNet'])
    layers = model.model.language_model.layers
    if args.cached_suffix:
        # The output matrix fits inside the existing cap; retain it to avoid
        # transferring it again for every suffix replay. No other process changes.
        from accelerate.hooks import remove_hook_from_module
        cpu_head = model.lm_head._hf_hook.weights_map['weight']
        remove_hook_from_module(model.lm_head)
        model.lm_head.weight = torch.nn.Parameter(cpu_head.to('cuda'), requires_grad=False)
    layer_cache, call_kwargs = {}, {}

    @lru_cache(maxsize=8192)
    def vector(row_id, site, layer):
        row = by_id[row_id]
        index = row['activation_token_indices'].index(positions(row,site))
        with safe_open(RUN_ROOT/row['activation_path'], framework='pt', device='cpu') as handle:
            return handle.get_tensor(f'answer.resid_post.layer_{layer}')[index].clone()

    def decode_logits(row, logits):
        at = logits[:,row['answer_boundary_input_index'],:]
        a,b = row['candidate_1_answer_token_ids'][0],row['candidate_2_answer_token_ids'][0]
        selected = at[:,[a,b]].float().cpu()
        top = at.argmax(-1).cpu().tolist()
        return [{'margin':float(v[0]-v[1]),'candidate_logits':v.tolist(),'top_token_id':t,
                 'top_token':tokenizer.decode([t]),
                 'top_token_correct':t==(a if row['z_bayes']>0 else b)} for v,t in zip(selected,top)]

    def forward(row, variants, capture=False):
        guard()
        handles = []
        by_layer = defaultdict(list)
        for i, job in enumerate(variants):
            if job is not None:
                by_layer[job['layer']].append((i,job))
        for layer, patches in by_layer.items():
            def replace(module, inputs, hidden, patches=patches):
                patched = hidden.clone()
                for i, job in patches:
                    if job['weights'] is None:
                        donor = vector(job['sources'][0],job['site'],job['layer'])
                    else:
                        donor = sum(vector(s,job['site'],job['layer']).float()*w for s,w in zip(job['sources'],job['weights']))
                    patched[i,positions(row,job['site'])] = donor.to(hidden.device,dtype=hidden.dtype)
                return patched
            handles.append(layers[layer].register_forward_hook(replace))
        if capture:
            assert variants == [None]
            layer_cache.clear()
            call_kwargs.clear()
            for index, module in enumerate(layers):
                def capture_kwargs(module, inputs, kwargs, index=index):
                    assert kwargs.get('past_key_values') is None
                    call_kwargs[index] = kwargs.copy()
                def capture_output(module, inputs, hidden, index=index):
                    layer_cache[index] = hidden.detach().clone()
                handles.append(module.register_forward_pre_hook(capture_kwargs, with_kwargs=True))
                handles.append(module.register_forward_hook(capture_output))
        try:
            ids = torch.tensor([row['teacher_forced_input_ids']]*len(variants), device='cuda')
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                values = decode_logits(row,logits)
            del logits, ids
            return values
        finally:
            for handle in handles: handle.remove()

    def suffix_forward(row, job=None, layer=0):
        guard()
        layer = job['layer'] if job is not None else layer
        with torch.inference_mode():
            hidden = layer_cache[layer].clone()
            if job is not None:
                if job['weights'] is None:
                    donor = vector(job['sources'][0],job['site'],layer)
                else:
                    donor = sum(vector(s,job['site'],layer).float()*w for s,w in zip(job['sources'],job['weights']))
                hidden[0,positions(row,job['site'])] = donor.to(hidden.device,dtype=hidden.dtype)
            for index in range(layer+1,len(layers)):
                hidden = layers[index](hidden,**call_kwargs[index])
            hidden = model.model.language_model.norm(hidden)
            logits = model.lm_head(hidden[:,0:,:])
            result = decode_logits(row,logits)[0]
        return result

    saved = {r['job_id']:r for r in records(output/'job_results.jsonl')}
    assert all(r['design_sha256']==manifest['design_sha256'] for r in saved.values())
    baseline_records = records(output/'baseline_audit.jsonl')
    grouped = defaultdict(list)
    for jid,job in jobs.items():
        if jid not in saved: grouped[job['recipient_id']].append(job)
    started = time.monotonic()
    def write_lines(path, values):
        temp = path.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values))
        temp.replace(path)
    def save_job(job, value, baseline, mode):
        if job['condition']=='self_patch' and abs(value['margin']-baseline['margin'])>.05:
            raise RuntimeError('Saved self-state does not reproduce fresh baseline')
        saved[job['job_id']] = {**job, 'baseline':baseline, 'patched':value,
            'margin_change':value['margin']-baseline['margin'],
            'truth_signed_change':(value['margin']-baseline['margin'])*(1 if by_id[job['recipient_id']]['z_bayes']>0 else -1),
            'backend':mode, 'design_sha256':manifest['design_sha256']}
    try:
        for recipient_index,(row_id,tasks) in enumerate(grouped.items()):
            row = by_id[row_id]
            single = forward(row,[None],capture=args.cached_suffix)[0]
            assert abs(single['margin']-margin(row)) <= .05, f'Singleton saved baseline mismatch: {row_id}'
            if args.cached_suffix:
                checks=[]
                prechecked_patches={}
                for layer in [0,15,31]:
                    cached=suffix_forward(row,layer=layer)
                    assert cached == single, f'Cached unpatched suffix differs from full native forward: {row_id}, {layer}'
                    checks.append({'layer':layer,'unpatched_exact':True})
                # Cross-check a real patched suffix against full native replay
                # at each tested site for every recipient, before accepting jobs.
                for site in SITES:
                    check_job=next((j for j in tasks if j['condition']=='cross_patch' and j['site']==site),None)
                    if check_job is not None:
                        full=forward(row,[check_job])[0]
                        cached=suffix_forward(row,check_job)
                        assert full == cached, f'Cached patched suffix differs from full native forward: {check_job}'
                        prechecked_patches[check_job['job_id']]=cached
                        checks.append({'site':site,'layer':check_job['layer'],'cross_patch_exact':True})
                baseline_records.append({'row_id':row_id,'saved_margin':margin(row),'single':single,
                    'batch_baselines':[],'batch_compatible':False,'batch_and_self_compatible':False,
                    'replay_mode':'exact_native_cached_suffix','suffix_checks':checks})
                write_lines(output/'baseline_audit.jsonl',baseline_records)
                for job in tasks:
                    value=prechecked_patches.get(job['job_id'])
                    if value is None:
                        value=suffix_forward(row,job)
                    save_job(job,value,single,'native_bf16_exact_cached_suffix')
                    write_lines(output/'job_results.jsonl',saved.values())
                write_json(output/'status.json',{'complete':False,'jobs_done':len(saved),'jobs_expected':len(jobs),
                    'recipients_done_this_run':recipient_index+1,'recipients_this_run':len(grouped),
                    'seconds_this_run':time.monotonic()-started})
                print(f'{args.phase}: recipient {recipient_index+1}/{len(grouped)}, jobs {len(saved)}/{len(jobs)}, exact_cached_suffix=True, seconds={time.monotonic()-started:.1f}',flush=True)
                continue
            batch = forward(row,[None]*args.batch_size)
            batch_ok = all(abs(v['margin']-single['margin'])<=.05 for v in batch)
            audit = {'row_id':row_id, 'saved_margin':margin(row), 'single':single,
                     'batch_baselines':batch, 'batch_compatible':batch_ok}
            # Validate all saved self states in the same batched shapes before accepting any cross patches.
            self_jobs = [t for t in tasks if t['condition']=='self_patch']
            if batch_ok:
                for start in range(0,len(self_jobs),args.batch_size-1):
                    chunk = self_jobs[start:start+args.batch_size-1]
                    variants = [None]+chunk+[None]*(args.batch_size-1-len(chunk))
                    values = forward(row,variants)
                    if any(abs(v['margin']-single['margin'])>.05 for v in values):
                        batch_ok=False
                        break
            audit['batch_and_self_compatible'] = batch_ok
            baseline_records.append(audit)
            write_lines(output/'baseline_audit.jsonl',baseline_records)
            if batch_ok:
                for start in range(0,len(tasks),args.batch_size-1):
                    chunk = tasks[start:start+args.batch_size-1]
                    values = forward(row,[None]+chunk+[None]*(args.batch_size-1-len(chunk)))
                    assert abs(values[0]['margin']-single['margin'])<=.05
                    for job,value in zip(chunk,values[1:]):save_job(job,value,values[0],'native_bf16_batch_with_exact_capture_gate')
                    write_lines(output/'job_results.jsonl',saved.values())
            else:
                for job in tasks:
                    save_job(job,forward(row,[job])[0],single,'native_bf16_singleton_fallback')
                    write_lines(output/'job_results.jsonl',saved.values())
            write_json(output/'status.json',{'complete':False,'jobs_done':len(saved),'jobs_expected':len(jobs),
                       'recipients_done_this_run':recipient_index+1,'recipients_this_run':len(grouped),
                       'seconds_this_run':time.monotonic()-started})
            print(f'{args.phase}: recipient {recipient_index+1}/{len(grouped)}, jobs {len(saved)}/{len(jobs)}, batch={batch_ok}, seconds={time.monotonic()-started:.1f}',flush=True)
        assert set(saved)==set(jobs)
        expanded=[]
        for edge in edges:
            job = saved[edge['job_id']]
            recipient=by_id[edge['recipient_id']]
            truth=1 if recipient['z_bayes']>0 else -1
            expanded.append({**edge, 'baseline_margin':job['baseline']['margin'],'patched_margin':job['patched']['margin'],
                'margin_change':job['margin_change'],'truth_signed_change':job['truth_signed_change'],
                'baseline_correct':job['baseline']['margin']*truth>0,'patched_correct':job['patched']['margin']*truth>0,
                'baseline_top_token_correct':job['baseline']['top_token_correct'],'patched_top_token_correct':job['patched']['top_token_correct'],
                'baseline_top_token':job['baseline']['top_token'],'patched_top_token':job['patched']['top_token'],
                'backend':job['backend'],'design_sha256':manifest['design_sha256']})
        write_lines(output/'results.jsonl',expanded)
        write_json(output/'status.json',{'complete':True,'jobs_done':len(saved),'jobs_expected':len(jobs),
            'records':len(expanded),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'after':query_gpu(),
            'existing_processes_modified':False,'seconds_this_run':time.monotonic()-started})
        print('COMPLETE',args.phase,len(saved),'unique jobs',flush=True)
    except BaseException as error:
        write_json(output/'status.json',{'complete':False,'jobs_done':len(saved),'jobs_expected':len(jobs),
                   'error':str(error),'error_type':type(error).__name__,'pid':os.getpid()})
        raise


if __name__ == '__main__':
    main()
