"""Frozen, same-reliability agreement-matched clean/corrupt patch experiments."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT/'artifacts/noisy_channel_bayesian_experiment_2_activation_patching/Qwen_Qwen3.5-9B_4c87a623'
RUN_ROOT = MODEL_ROOT/'runs/qwen35_9b_n9_k3_r10_s64_selected_tokens_v2_reasoning_off'
OUT = MODEL_ROOT/'analyses/agreement_matched_reasoning_off_20260905'
SITES = ['final_prompt', 'answer_prefix']
FIXED = ['reliabilities_exact', 'presentation_order', 'candidate_1', 'candidate_2', 'reasoning', 'n', 'k']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_rows():
    rows = [json.loads(s) for s in (RUN_ROOT/'results.jsonl').read_text().splitlines() if s.strip()]
    assert len(rows) == 5120 and all(r['reasoning'] is False for r in rows)
    return rows


def margin(row):
    return row['answer_surface_raw_logits'][str(row['candidate_1'])] - row['answer_surface_raw_logits'][str(row['candidate_2'])]


def eligible(row):
    return row['reasoning'] is False and row['reliabilities'][0] < .5 and row['delta_a'] != 0


def key(row, tier):
    fields = FIXED + (['agreement_c1', 'agreement_c2'] if tier == 'totals' else
                     ['agreement_candidate_1_by_question', 'agreement_candidate_2_by_question'])
    assert tier in ['totals', 'grid']
    return json.dumps([row[f] for f in fields], sort_keys=True)


def label(row, strict=False):
    correct = bool(row['canonical_posterior_correct'])
    if strict and not ((margin(row)*row['z_bayes'] > 0) if correct else (margin(row)*row['z_bayes'] < 0)):
        return None
    return 'clean' if correct else 'corrupt'


def enumerate_pairs(rows, tier, split, strict=False, exclude=()):
    excluded = set(exclude)
    groups = defaultdict(list)
    for row in rows:
        if eligible(row) and row['row_id'] not in excluded and (split == 'all' or row['split'] == split):
            groups[key(row, tier)].append(row)
    pairs = []
    for group_key, group in sorted(groups.items()):
        clean = [r for r in group if label(r, strict) == 'clean']
        corrupt = [r for r in group if label(r, strict) == 'corrupt']
        for good in clean:
            for bad in corrupt:
                assert good['z_bayes'] == bad['z_bayes']
                assert good['agreement_c1'] == bad['agreement_c1'] and good['agreement_c2'] == bad['agreement_c2']
                pair = {'pair_id': digest([tier, good['row_id'], bad['row_id']])[:20],
                        'tier': tier, 'split': split, 'group_id': digest(group_key)[:20],
                        'clean_id': good['row_id'], 'corrupt_id': bad['row_id'],
                        'reliability': good['reliabilities_exact'][0], 'winner': good['canonical_ground_truth_choice'],
                        'agreement_c1': good['agreement_c1'], 'agreement_c2': good['agreement_c2'],
                        'clean_schedule': good['question_set_index'], 'corrupt_schedule': bad['question_set_index'],
                        'strict_boundary_pair': label(good, True) == 'clean' and label(bad, True) == 'corrupt',
                        'same_grid': key(good, 'grid') == key(bad, 'grid'),
                        'same_questions': good['membership_sets'] == bad['membership_sets'],
                        'same_reports': good['observed_reports'] == bad['observed_reports']}
                pairs.append(pair)
    return sorted(pairs, key=lambda p: p['pair_id'])


def prepare():
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for tier in ['totals', 'grid']:
        for split in ['train', 'test', 'all']:
            for strict in [False, True]:
                pairs = enumerate_pairs(rows, tier, split, strict)
                records.append({'tier': tier, 'split': split, 'labels': 'emitted_and_strict_boundary' if strict else 'emitted',
                                'pairs': len(pairs), 'mixed_groups': len({p['group_id'] for p in pairs}),
                                'unique_clean': len({p['clean_id'] for p in pairs}),
                                'unique_corrupt': len({p['corrupt_id'] for p in pairs}),
                                'schedules': len({p[k] for p in pairs for k in ['clean_schedule','corrupt_schedule']}),
                                'winners': sorted({p['winner'] for p in pairs})})
    pd.DataFrame(records).to_csv(OUT/'matching_counts.csv', index=False)
    # One strict grid-matched training pair per reliability × Bayesian winner.
    # Stable hashes break ties without looking at patched outcomes or margin size.
    strata = defaultdict(list)
    for pair in enumerate_pairs(rows, 'grid', 'train', True):
        strata[pair['reliability'], pair['winner']].append(pair)
    discovery = [min(group, key=lambda p: p['pair_id']) for _, group in sorted(strata.items())]
    discovery_ids = {p[k] for p in discovery for k in ['clean_id','corrupt_id']}
    # Independent rows for the remaining strict-grid training follow-up.
    grid_groups = defaultdict(list)
    for pair in enumerate_pairs(rows, 'grid', 'train', True, discovery_ids):
        grid_groups[pair['group_id']].append(pair)
    grid_followup = [min(group, key=lambda p: p['pair_id']) for _, group in sorted(grid_groups.items())]
    test_pairs = enumerate_pairs(rows, 'totals', 'test', False)
    protocol = {'version': 1, 'user_clarification': 'questions and observed answers may differ; reliability, ordering and reasoning off fixed',
                'results_sha256': hashlib.sha256((RUN_ROOT/'results.jsonl').read_bytes()).hexdigest(),
                'fixed_fields': FIXED, 'tiers': {'totals': 'both individual agreement totals equal', 'grid': 'both per-question agreement bit vectors equal'},
                'sites': SITES, 'discovery_layers': list(range(32)),
                'discovery_pairs': discovery, 'grid_followup_pairs': grid_followup, 'heldout_totals_pairs': test_pairs,
                'selection': 'For each site choose layer 0..30 maximizing mean bidirectional truth-signed margin effect on discovery pairs; lower layer breaks ties. Layer 31 is excluded as a final readout/structural control.',
                'metrics': ['raw and truth-signed margin change', 'candidate-boundary accuracy', 'full-vocabulary next-token correctness', 'repair and noising damage rates'],
                'controls': ['fresh unpatched baseline in every batch', 'saved self-patch', 'same-label matched resampled donor', 'label-agnostic training matched-group mean', 'off-target domain-boundary donor patch'],
                'limitations': ['grid follow-up uses original training split, with rows disjoint from discovery',
                                'held-out totals pairs are outcome-selected and dependent, not an unbiased accuracy estimate',
                                'strict held-out boundary pairs have only C2 as Bayesian winner',
                                'no reliability, identity, or order generalization',
                                'answer-prefix whole-state transfer can copy a decision without repairing Bayesian computation',
                                'scores use teacher-forced answer prefix; complete patched responses are not regenerated']}
    protocol['protocol_sha256'] = digest(protocol)
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Refusing to overwrite a changed frozen protocol'
    else:
        write_json(path, protocol)
    for name in ['discovery_pairs','grid_followup_pairs','heldout_totals_pairs']:
        pd.DataFrame(protocol[name]).to_csv(OUT/(name+'.csv'), index=False)
    return protocol


def positions(row, site):
    if site == 'answer_prefix':
        return row['answer_boundary_input_index']
    values = row['prompt_token_sites'][site]
    return values[-1] if isinstance(values, list) else values


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def select_locations(protocol):
    status = json.loads((OUT/'discovery/status.json').read_text())
    assert status['complete']
    frame = pd.DataFrame(records(OUT/'discovery/results.jsonl'))
    cross = frame[frame.condition == 'cross_patch'].copy()
    cross['directional_effect'] = cross.truth_signed_change * np.where(cross.direction == 'denoising', 1, -1)
    table = cross.groupby(['site','layer'], as_index=False).agg(score=('directional_effect','mean'), records=('directional_effect','size'))
    assert len(table) == 64 and table.records.eq(2*len(protocol['discovery_pairs'])).all()
    locations = []
    for site in SITES:
        best = table[(table.site == site) & (table.layer < 31)].sort_values(['score','layer'], ascending=[False,True]).iloc[0]
        locations.append({'site': site, 'layer': int(best.layer), 'discovery_score': float(best.score)})
    lock = {'protocol_sha256': protocol['protocol_sha256'], 'locations': locations,
            'discovery_results_sha256': hashlib.sha256((OUT/'discovery/results.jsonl').read_bytes()).hexdigest(),
            'selection': protocol['selection'], 'test_patch_results_used': False}
    path = OUT/'locked_locations.json'
    if path.exists():
        assert json.loads(path.read_text()) == lock
    else:
        write_json(path, lock)
    table.to_csv(OUT/'discovery_layer_scores.csv', index=False)
    return lock


def summarize_phase(phase):
    """Read only completed checkpoints; incomplete plots are explicitly labeled."""
    import matplotlib.pyplot as plt
    root = OUT/phase
    state = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'complete':False}
    if not (root/'execution_design.json').exists():
        return state, pd.DataFrame(), pd.DataFrame()
    design = json.loads((root/'execution_design.json').read_text())
    jobs = {r['job_id']:r for r in records(root/'job_results.jsonl')}
    pair_jobs=defaultdict(set)
    for edge in design['edges']:
        pair_jobs[edge['cohort'],edge['pair_id']].add(edge['job_id'])
    available=set(jobs)
    state['cohort_progress']={}
    for cohort in sorted({cohort for cohort,_ in pair_jobs}):
        groups=[ids for (name,_),ids in pair_jobs.items() if name==cohort]
        expected=set().union(*groups)
        state['cohort_progress'][cohort]={'jobs_done':len(expected & available),
            'jobs_expected':len(expected),'pairs_complete':sum(ids<=available for ids in groups),
            'pairs_expected':len(groups)}
    by_id = {r['row_id']:r for r in load_rows()}
    expanded=[]
    for edge in design['edges']:
        if edge['job_id'] not in jobs: continue
        job=jobs[edge['job_id']]
        truth=1 if by_id[edge['recipient_id']]['z_bayes']>0 else -1
        expanded.append({**edge,'baseline_margin':job['baseline']['margin'],'patched_margin':job['patched']['margin'],
                         'baseline_truth_signed_margin':job['baseline']['margin']*truth,
                         'patched_truth_signed_margin':job['patched']['margin']*truth,
                         'truth_signed_change':job['truth_signed_change'],'margin_change':job['margin_change'],
                         'baseline_correct':job['baseline']['margin']*truth>0,
                         'patched_correct':job['patched']['margin']*truth>0,
                         'baseline_top_token_correct':job['baseline']['top_token_correct'],
                         'patched_top_token_correct':job['patched']['top_token_correct']})
    frame=pd.DataFrame(expanded)
    if frame.empty: return state, frame, pd.DataFrame()
    frame['repair']=~frame.baseline_correct & frame.patched_correct
    frame['damage']=frame.baseline_correct & ~frame.patched_correct
    frame['full_vocab_repair']=~frame.baseline_top_token_correct & frame.patched_top_token_correct
    frame['full_vocab_damage']=frame.baseline_top_token_correct & ~frame.patched_top_token_correct
    frame['choice_changed']=np.sign(frame.baseline_margin)!=np.sign(frame.patched_margin)
    # Sensitivity analysis, not used for location selection: require a margin
    # buffer on both sides of the decision boundary instead of a near-tie flip.
    frame['buffered_repair']=(frame.baseline_truth_signed_margin<=-.25)&(frame.patched_truth_signed_margin>=.25)
    frame['buffered_damage']=(frame.baseline_truth_signed_margin>=.25)&(frame.patched_truth_signed_margin<=-.25)
    frame['directional_effect']=frame.truth_signed_change*np.where(frame.direction=='denoising',1,-1)
    keys=['cohort','site','layer','direction','condition']
    summary=[]
    for scope,subset in [('all_emitted_pairs',frame),('strict_boundary_pairs',frame[frame.strict_boundary_pair])]:
        for group_key,group in subset.groupby(keys):
            recipients=group.groupby('recipient_id').agg(effect=('truth_signed_change','mean'),accuracy=('patched_correct','mean'))
            summary.append({**dict(zip(keys,group_key)), 'label_scope':scope,'pairs':len(group),
                            'unique_recipients':len(recipients),'recipient_schedules':group.apply(lambda r:r.corrupt_schedule if r.direction=='denoising' else r.clean_schedule,axis=1).nunique(),
                            'baseline_accuracy':group.baseline_correct.mean(),'patched_accuracy':group.patched_correct.mean(),
                            'repair_count':int(group.repair.sum()),'damage_count':int(group.damage.sum()),
                            'buffered_repair_count':int(group.buffered_repair.sum()),
                            'buffered_damage_count':int(group.buffered_damage.sum()),
                            'full_vocab_repair_count':int(group.full_vocab_repair.sum()),
                            'full_vocab_damage_count':int(group.full_vocab_damage.sum()),
                            'repair_rate_among_baseline_incorrect':float(group.repair.sum()/(~group.baseline_correct).sum()) if (~group.baseline_correct).any() else None,
                            'damage_rate_among_baseline_correct':float(group.damage.sum()/group.baseline_correct.sum()) if group.baseline_correct.any() else None,
                            'choice_changes':int(group.choice_changed.sum()),
                            'mean_truth_signed_change':group.truth_signed_change.mean(),
                            'mean_directional_effect':group.directional_effect.mean(),
                            'recipient_weighted_truth_signed_change':recipients.effect.mean(),
                            'recipient_weighted_patched_accuracy':recipients.accuracy.mean(),
                            'baseline_full_vocab_next_token_accuracy':group.baseline_top_token_correct.mean(),
                            'patched_full_vocab_next_token_accuracy':group.patched_top_token_correct.mean(),
                            'run_complete':state.get('complete',False)})
    summary=pd.DataFrame(summary)
    summary.to_csv(root/'summary.csv',index=False)
    frame.to_csv(root/'scored_pairs.csv',index=False)
    if phase=='evaluation':
        contrast_keys=['cohort','pair_id','recipient_id','site','layer','direction','strict_boundary_pair']
        paired=frame.pivot(index=contrast_keys,columns='condition',values='directional_effect')
        contrasts=[]
        if 'cross_patch' in paired:
            for control in ['same_label_resampled','training_group_mean','off_target','self_patch']:
                if control not in paired: continue
                selected=paired[['cross_patch',control]].dropna().copy()
                selected['cross_minus_control']=selected.cross_patch-selected[control]
                selected=selected.reset_index()
                for scope,subset in [('all_emitted_pairs',selected),('strict_boundary_pairs',selected[selected.strict_boundary_pair])]:
                    for group_key,group in subset.groupby(['cohort','site','layer','direction']):
                        contrasts.append({**dict(zip(['cohort','site','layer','direction'],group_key)),
                            'control':control,'label_scope':scope,'pairs':len(group),
                            'mean_cross_minus_control':group.cross_minus_control.mean(),
                            'recipient_weighted_cross_minus_control':group.groupby('recipient_id').cross_minus_control.mean().mean()})
        pd.DataFrame(contrasts).to_csv(root/'paired_control_contrasts.csv',index=False)
    prefix='Complete' if state.get('complete') else 'Partial'
    cross=summary[(summary.condition=='cross_patch') & (summary.label_scope=='strict_boundary_pairs')]
    if phase=='discovery':
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
        for ax,site in zip(axes,SITES):
            for direction,g in cross[cross.site==site].groupby('direction'):
                g=g.sort_values('layer')
                ax.plot(g.layer,g.mean_directional_effect,label=direction)
            ax.axhline(0,color='black',lw=.7)
            ax.axvline(31,color='gray',ls=':',label='readout/structural control')
            ax.set(title=site,xlabel='Patched residual-post layer',ylabel='Directional margin effect')
            ax.legend()
        fig.suptitle(f'{prefix}: exact-grid training discovery; positive = toward correct / incorrect by patch direction')
    else:
        fig,axes=plt.subplots(2,2,figsize=(13,8),squeeze=False)
        for row_index,cohort in enumerate(['grid_train_followup','totals_heldout']):
            for col,site in enumerate(SITES):
                ax=axes[row_index,col]
                selected=summary[(summary.cohort==cohort)&(summary.site==site)&(summary.label_scope=='strict_boundary_pairs')]
                conditions=['cross_patch','same_label_resampled','training_group_mean','off_target','self_patch']
                for offset,direction in [(-.18,'denoising'),(.18,'noising')]:
                    g=selected[selected.direction==direction].set_index('condition').reindex(conditions)
                    ax.bar(np.arange(len(conditions))+offset,g.mean_directional_effect,width=.36,label=direction)
                ax.set_xticks(np.arange(len(conditions)),['cross','same-label','mean','off-target','self'],rotation=25)
                ax.axhline(0,color='black',lw=.7)
                ax.set(title=f'{cohort}: {site}',ylabel='Directional margin effect')
                ax.legend()
        fig.suptitle(f'{prefix}: frozen-location evaluation; strict boundary-consistent pairs')
    fig.tight_layout()
    fig.savefig(root/'effects.png',dpi=150,bbox_inches='tight')
    plt.show()
    plt.close(fig)
    if phase=='evaluation':
        fig,axes=plt.subplots(2,2,figsize=(13,8),squeeze=False)
        conditions=['cross_patch','same_label_resampled','training_group_mean','off_target','self_patch']
        for row_index,cohort in enumerate(['grid_train_followup','totals_heldout']):
            for col,site in enumerate(SITES):
                ax=axes[row_index,col]
                selected=summary[(summary.cohort==cohort)&(summary.site==site)&(summary.label_scope=='strict_boundary_pairs')]
                for offset,direction in [(-.18,'denoising'),(.18,'noising')]:
                    group=selected[selected.direction==direction].set_index('condition').reindex(conditions)
                    metric='full_vocab_repair_count' if direction=='denoising' else 'full_vocab_damage_count'
                    ax.bar(np.arange(len(conditions))+offset,group[metric]/group.pairs,width=.36,label=direction)
                ax.set_xticks(np.arange(len(conditions)),['cross','same-label','mean','off-target','self'],rotation=25)
                ax.set_ylim(0,1.05)
                ax.set(title=f'{cohort}: {site}',ylabel='Fraction repaired / damaged')
                ax.legend()
        fig.suptitle(f'{prefix}: full-vocabulary next-token decisions at the saved answer prefix; strict pairs')
        fig.tight_layout()
        fig.savefig(root/'choice_changes.png',dpi=150,bbox_inches='tight')
        plt.show()
        plt.close(fig)
    return state,frame,summary


if __name__ == '__main__':
    p = prepare()
    print({k: len(p[k]) for k in ['discovery_pairs','grid_followup_pairs','heldout_totals_pairs']})
