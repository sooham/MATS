"""Independent design, source-data, replay and notebook audit for notebook 20."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
from agreement_matched_patching import OUT,RUN_ROOT,FIXED,load_rows,key,label,margin,positions,write_json


def read_lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def check_hash(payload,field):
    copy=payload.copy()
    expected=copy.pop(field)
    assert hashlib.sha256(json.dumps(copy,sort_keys=True).encode()).hexdigest()==expected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    rows=load_rows();by_id={r['row_id']:r for r in rows}
    training_sources=defaultdict(set)
    for row in rows:
        if row['reasoning'] is False and row['reliabilities'][0]<.5 and row['delta_a']!=0 and row['split']=='train':
            for tier in ['totals','grid']:
                training_sources[tier,key(row,tier)].add(row['row_id'])
    protocol=json.loads((OUT/'protocol.json').read_text())
    check_hash(protocol,'protocol_sha256')
    assert hashlib.sha256((RUN_ROOT/'results.jsonl').read_bytes()).hexdigest()==protocol['results_sha256']
    discovery_ids={p[k] for p in protocol['discovery_pairs'] for k in ['clean_id','corrupt_id']}
    result={'protocol_sha256':protocol['protocol_sha256'],'source_results_unchanged':True,'phases':{}}
    control_ids={p[k] for cohort in ['grid_followup_pairs','heldout_totals_pairs'] for p in protocol[cohort] for k in ['clean_id','corrupt_id']}
    domain_prefixes={tuple(by_id[rid]['teacher_forced_input_ids'][:positions(by_id[rid],'domain_boundary')+1]) for rid in control_ids}
    result['off_target_control_distinct_input_prefixes']=len(domain_prefixes)
    for cohort in ['discovery_pairs','grid_followup_pairs','heldout_totals_pairs']:
        for pair in protocol[cohort]:
            clean,corrupt=by_id[pair['clean_id']],by_id[pair['corrupt_id']]
            assert all(clean[f]==corrupt[f] for f in FIXED)
            assert clean['reasoning'] is False and clean['reliabilities'][0]<.5
            assert clean['z_bayes']==corrupt['z_bayes']!=0
            assert key(clean,pair['tier'])==key(corrupt,pair['tier'])
            assert clean['agreement_c1']==corrupt['agreement_c1'] and clean['agreement_c2']==corrupt['agreement_c2']
            assert label(clean)=='clean' and label(corrupt)=='corrupt'
            assert clean['split']==corrupt['split']==('test' if cohort=='heldout_totals_pairs' else 'train')
            assert pair['strict_boundary_pair']==(label(clean,True)=='clean' and label(corrupt,True)=='corrupt')
            if cohort=='grid_followup_pairs':
                assert clean['row_id'] not in discovery_ids and corrupt['row_id'] not in discovery_ids
    for phase in ['discovery','evaluation']:
        root=OUT/phase
        if not (root/'execution_design.json').exists():
            result['phases'][phase]={'complete':False,'status':'not_started'}
            continue
        design=json.loads((root/'execution_design.json').read_text())
        check_hash(design,'design_sha256')
        assert design['protocol_sha256']==protocol['protocol_sha256']
        jobs={j['job_id']:j for j in design['jobs']}
        assert set(jobs)=={e['job_id'] for e in design['edges']}
        if phase=='discovery':
            cohorts=[('discovery_grid_train',protocol['discovery_pairs'])]
            locations=[(s,layer) for s in protocol['sites'] for layer in range(32)]
            conditions=['cross_patch','self_patch']
        else:
            cohorts=[('grid_train_followup',protocol['grid_followup_pairs']),('totals_heldout',protocol['heldout_totals_pairs'])]
            lock=json.loads((OUT/'locked_locations.json').read_text())
            locations=[(loc['site'],loc['layer']) for loc in lock['locations']]
            conditions=['cross_patch','self_patch','same_label_resampled','training_group_mean','off_target']
        expected={(cohort,p['pair_id'],direction,site,layer,condition)
                  for cohort,pairs in cohorts for p in pairs for direction in ['denoising','noising']
                  for site,layer in locations for condition in conditions}
        observed={(e['cohort'],e['pair_id'],e['direction'],e['site'],e['layer'],
                   'same_label_resampled' if e['condition']=='resampled_self_fallback' else e['condition']) for e in design['edges']}
        assert expected==observed and len(expected)==len(design['edges']), 'Design must cover the full frozen protocol'
        measured=read_lines(root/'job_results.jsonl') if (root/'job_results.jsonl').exists() else []
        actual={j['job_id']:j for j in measured}
        assert len(actual)==len(measured) and set(actual)<=set(jobs)
        max_saved_difference,max_self_change=0.,0.
        for jid,record in actual.items():
            job=jobs[jid]
            assert all(record[k]==v for k,v in job.items())
            assert record['design_sha256']==design['design_sha256']
            row=by_id[job['recipient_id']]
            assert positions(row,job['site'])<=row['answer_boundary_input_index']
            assert positions(row,job['site']) in row['activation_token_indices']
            baseline,patched=record['baseline'],record['patched']
            max_saved_difference=max(max_saved_difference,abs(baseline['margin']-margin(row)))
            assert abs(baseline['margin']-margin(row))<=.05
            assert patched['margin']-baseline['margin']==record['margin_change']
            assert record['truth_signed_change']==record['margin_change']*np.sign(row['z_bayes'])
            for value in [baseline,patched]:
                assert value['candidate_logits'][0]-value['candidate_logits'][1]==value['margin']
                winner_token=row['candidate_1_answer_token_ids'][0] if row['z_bayes']>0 else row['candidate_2_answer_token_ids'][0]
                assert value['top_token_correct']==(value['top_token_id']==winner_token)
            if job['condition']=='self_patch':
                max_self_change=max(max_self_change,abs(record['margin_change']))
                assert abs(record['margin_change'])<=.05
        for edge in design['edges']:
            recipient,donor=by_id[edge['recipient_id']],by_id[edge['donor_id']]
            job=jobs[edge['job_id']]
            assert key(recipient,edge['tier'])==key(donor,edge['tier'])
            assert recipient['split']==donor['split']
            assert job['recipient_id']==recipient['row_id'] and job['layer']==edge['layer']
            assert job['site']==edge['patch_site']
            if edge['condition'] in ['cross_patch','off_target']:
                assert job['sources']==[donor['row_id']] and job['weights'] is None
                assert job['site']==('domain_boundary' if edge['condition']=='off_target' else edge['site'])
            elif edge['condition'] in ['self_patch','resampled_self_fallback']:
                assert job['sources']==[recipient['row_id']] and job['weights'] is None
            elif edge['condition']=='same_label_resampled':
                source=by_id[job['sources'][0]]
                assert source['row_id']!=recipient['row_id']
                assert key(source,edge['tier'])==key(recipient,edge['tier'])
                assert source['split']==recipient['split'] and label(source,True)==label(recipient,True) is not None
            elif edge['condition']=='training_group_mean':
                assert abs(sum(job['weights'])-1)<1e-12
                assert len(job['weights'])==len(job['sources'])
                assert all(abs(w-1/len(job['sources']))<1e-12 for w in job['weights'])
                expected_sources=training_sources[edge['tier'],key(recipient,edge['tier'])]
                assert set(job['sources'])==expected_sources
                for source in job['sources']:
                    assert by_id[source]['split']=='train' and key(by_id[source],edge['tier'])==key(recipient,edge['tier'])
            else:raise AssertionError(edge['condition'])
        state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'complete':False}
        if state.get('complete'):
            assert set(actual)==set(jobs)
            expanded=read_lines(root/'results.jsonl')
            assert len(expanded)==len(design['edges'])
            assert len({(e['cohort'],e['pair_id'],e['direction'],e['site'],e['layer'],e['condition']) for e in expanded})==len(expanded)
            for measured_edge,edge in zip(expanded,design['edges']):
                assert all(measured_edge[k]==v for k,v in edge.items())
                job=actual[edge['job_id']]
                assert measured_edge['baseline_margin']==job['baseline']['margin']
                assert measured_edge['patched_margin']==job['patched']['margin']
                assert measured_edge['truth_signed_change']==job['truth_signed_change']
        result['phases'][phase]={'complete':state.get('complete',False),'expected_jobs':len(jobs),
            'measured_jobs':len(actual),'pair_condition_records':len(design['edges']),
            'max_saved_margin_difference':max_saved_difference,'max_self_patch_change':max_self_change,
            'conditions':dict(Counter(e['condition'] for e in design['edges']))}
    if all(p['complete'] for p in result['phases'].values()):
        discovery=read_lines(OUT/'discovery/results.jsonl')
        lock=json.loads((OUT/'locked_locations.json').read_text())
        assert lock['protocol_sha256']==protocol['protocol_sha256']
        assert lock['discovery_results_sha256']==hashlib.sha256((OUT/'discovery/results.jsonl').read_bytes()).hexdigest()
        scores=defaultdict(list)
        for row in discovery:
            if row['condition']=='cross_patch' and row['layer']<31:
                scores[row['site'],row['layer']].append(row['truth_signed_change']*(1 if row['direction']=='denoising' else -1))
        for location in lock['locations']:
            best=min(((-float(np.mean(values)),layer) for (site,layer),values in scores.items() if site==location['site']))
            assert location['layer']==best[1] and abs(location['discovery_score']+best[0])<1e-12
        result['training_only_location_selection_verified']=True
    result['notebooks']={}
    for name in ['18_noisy_channel_completed_reasoning_off_probes.ipynb',
                 '19_noisy_channel_low_reliability_patching.ipynb',
                 '20_noisy_channel_agreement_matched_patching.ipynb']:
        nb=nbformat.read(ROOT/'notebooks'/name,as_version=4);nbformat.validate(nb)
        cells=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        result['notebooks'][name]={'executed_code_cells':len(cells),'png_outputs':sum('image/png' in o.get('data',{}) for c in cells for o in c.outputs)}
        if name.startswith('20_') and all(p['complete'] for p in result['phases'].values()):
            displayed_states={}
            for output in cells[-1].outputs:
                text=output.get('data',{}).get('text/plain')
                if not text:continue
                try:value=ast.literal_eval(text)
                except (SyntaxError,ValueError):continue
                if isinstance(value,dict):displayed_states.update(value)
            for phase in ['discovery','evaluation']:
                assert displayed_states[f'{phase}/status.json']['complete'], 'Notebook must display final, not stale partial, outputs'
            assert result['notebooks'][name]['png_outputs']>=2
    snapshot=json.loads((OUT.parent/'completed_reasoning_off_20260905/snapshot.json').read_text())
    assert hashlib.sha256((ROOT/'notebooks/16_noisy_channel_bayesian_activation_patching.ipynb').read_bytes()).hexdigest()==snapshot['source_notebook_sha256_at_snapshot']
    result['original_notebook_unchanged']=True
    result['complete']=all(p['complete'] for p in result['phases'].values())
    print(json.dumps(result,indent=2))
    if args.require_complete:
        assert result['complete'], 'Experiment still incomplete'
        write_json(OUT/'final_audit.json',result)


if __name__=='__main__':main()
