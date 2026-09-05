"""Read-only saved-state diagnostics and exact, reviewable patch designs."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from completed_probe_exploration import positions, save_figure, write_json, write_jsonl


def prospective_matching_audit(results, output):
    """Count a possible matching relaxation; do not select or patch donors."""
    rows = [r for r in results if r['reasoning'] is False and r['reliabilities'][0] < .5 and r['delta_a'] != 0]
    fields = ['reliabilities_exact', 'observed_reports', 'presentation_order',
              'candidate_1', 'candidate_2', 'reasoning',
              'agreement_candidate_1_by_question', 'agreement_candidate_2_by_question']
    records = []
    for split in ['all', 'train', 'test']:
        selected = [r for r in rows if split == 'all' or r['split'] == split]
        groups = defaultdict(list)
        for row in selected:
            groups[json.dumps([row[f] for f in fields], sort_keys=True)].append(row)
        for criterion in ['emitted_answer', 'emitted_and_strict_boundary_agree']:
            for winner in ['C1', 'C2', 'both']:
                pair_count = 0
                clean_ids, corrupt_ids, mixed_groups = set(), set(), 0
                for group in groups.values():
                    if winner != 'both' and group[0]['canonical_ground_truth_choice'] != winner:
                        continue
                    clean, corrupted = [], []
                    for row in group:
                        margin = (row['answer_surface_raw_logits'][str(row['candidate_1'])]
                                  - row['answer_surface_raw_logits'][str(row['candidate_2'])])
                        signed_margin = margin * np.sign(row['z_bayes'])
                        emitted_correct = row['canonical_posterior_correct']
                        if emitted_correct and (criterion == 'emitted_answer' or signed_margin > 0):
                            clean.append(row)
                        if not emitted_correct and (criterion == 'emitted_answer' or signed_margin < 0):
                            corrupted.append(row)
                    if not clean or not corrupted:
                        continue
                    assert len({r['z_bayes'] for r in group}) == 1
                    mixed_groups += 1
                    pair_count += len(clean) * len(corrupted)
                    clean_ids.update(r['row_id'] for r in clean)
                    corrupt_ids.update(r['row_id'] for r in corrupted)
                records.append({'split': split, 'correctness_criterion': criterion, 'bayes_winner': winner,
                                'mixed_groups': mixed_groups, 'denoising_pairs_if_relaxed': pair_count,
                                'noising_pairs_if_relaxed': pair_count,
                                'unique_clean_rows': len(clean_ids), 'unique_corrupted_rows': len(corrupt_ids),
                                'executed': False})
    frame = pd.DataFrame(records)
    frame.to_csv(output/'prospective_relaxed_matching_counts.csv', index=False)
    write_json(output/'prospective_relaxed_matching_protocol.json', {
        'status': 'proposal_only_no_donors_selected_no_patches_run',
        'preserved_fields': fields,
        'relaxed_field_requires_user_choice': 'exact question membership sets',
        'same_bayesian_pairwise_log_odds': True,
        'counts_are_dependent_pairs_not_independent_samples': True,
        'interpretation': 'Preserves each candidate evidence pattern, not just agreement totals; changing distractor set members can still change representations.',
    })
    return frame


def model_tensor(model_root, key):
    index = json.loads((model_root / 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(model_root / index[key], framework='pt', device='cpu') as handle:
        return handle.get_tensor(key)


def layerwise_margin_decomposition(results, run_root, model_root, output):
    torch.set_num_threads(2)
    rows = [r for r in results if r['split'] == 'test']
    schedules = sorted({r['question_set_index'] for r in rows})
    assert len(schedules) == 8
    config = json.loads((model_root / 'config.json').read_text())['text_config']
    norm_weight = model_tensor(model_root, 'model.language_model.norm.weight').float()
    # Slicing the two answer-token rows does not load the 9B model.
    token_ids = [rows[0]['candidate_1_answer_token_ids'][0], rows[0]['candidate_2_answer_token_ids'][0]]
    index = json.loads((model_root / 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(model_root / index['lm_head.weight'], framework='pt', device='cpu') as handle:
        head = torch.cat([handle.get_slice('lm_head.weight')[i:i+1] for i in token_ids]).float()
    margins = {site: np.empty((32, len(rows))) for site in ['final_prompt', 'answer_prefix']}
    parity = []
    for row_index, row in enumerate(rows):
        stored = {p: j for j, p in enumerate(row['activation_token_indices'])}
        final_position = positions(row, 'final_prompt')
        assert len(final_position) == 1
        site_indices = [stored[final_position[0]], stored[row['answer_boundary_input_index']]]
        with safe_open(run_root / row['activation_path'], framework='pt', device='cpu') as handle:
            for layer in range(32):
                residual = handle.get_tensor(f'answer.resid_post.layer_{layer}')[site_indices].float()
                normalized = residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + config['rms_norm_eps'])
                normalized = normalized * (1 + norm_weight)
                logits = normalized @ head.T
                diff = logits[:, 0] - logits[:, 1]
                for j, site in enumerate(margins):
                    margins[site][layer, row_index] = float(diff[j])
                if layer == 31:
                    rounded_logits = normalized[1].bfloat16() @ head.bfloat16().T
                    rounded_margin = float(rounded_logits[0] - rounded_logits[1])
                    saved = row['answer_surface_raw_logits'][str(row['candidate_1'])] - row['answer_surface_raw_logits'][str(row['candidate_2'])]
                    parity.append({'row_id': row['row_id'], 'saved_margin': saved,
                                   'cpu_head_bf16_margin': rounded_margin,
                                   'absolute_difference': abs(rounded_margin-saved),
                                   'fp32_readout_margin': float(diff[1])})
        if (row_index + 1) % 160 == 0:
            print('Saved-state CPU logit-lens readout', row_index + 1, '/', len(rows), flush=True)
    write_jsonl(output / 'head_parity_rows.jsonl', parity)
    differences = np.array([r['absolute_difference'] for r in parity])
    parity_summary = {'rows': len(rows), 'only_final_norm_and_two_head_rows_loaded': True,
                      'max_absolute_difference': float(differences.max()),
                      'mean_absolute_difference': float(differences.mean()),
                      'within_original_0_05_tolerance_fraction': float((differences <= .05).mean()),
                      'interpretation': 'Head-only reconstruction verifies saved final residual readout; it does not validate CPU full-model replay.'}
    write_json(output / 'head_parity_summary.json', parity_summary)
    lookup = {(r['question_set_index'], r['answer_pattern_index'], r['reliabilities_exact'][0]): i for i, r in enumerate(rows)}
    reliability_pairs = [('1/20','19/20'),('3/20','17/20'),('1/4','3/4'),('7/20','13/20'),('9/20','11/20')]
    rng = np.random.default_rng(20260905)
    boot = rng.integers(0, len(schedules), size=(2000, len(schedules)))
    records = []
    for site, matrix in margins.items():
        np.savez_compressed(output / f'{site}_layerwise_head_margins.npz', margins=matrix,
                            row_ids=np.array([r['row_id'] for r in rows]), layers=np.arange(32))
        for layer in range(32):
            numerators = {'even': [], 'odd': []}
            denominators = []
            for schedule in schedules:
                x, even, odd = [], [], []
                for pattern in range(8):
                    for low, high in reliability_pairs:
                        i, j = lookup[schedule, pattern, low], lookup[schedule, pattern, high]
                        x.append(rows[i]['delta_a'])
                        even.append((matrix[layer, i] + matrix[layer, j]) / 2)
                        odd.append((matrix[layer, j] - matrix[layer, i]) / 2)
                x = np.asarray(x)
                assert abs(x.mean()) < 1e-12  # exhaustive complement patterns
                denominators.append(float(x @ x))
                numerators['even'].append(float(x @ even))
                numerators['odd'].append(float(x @ odd))
            den = np.array(denominators)
            for component, values in numerators.items():
                num = np.array(values)
                estimate = num.sum() / den.sum()
                boot_den = den[boot].sum(axis=1)
                good = boot_den > 0
                distribution = num[boot].sum(axis=1)[good] / boot_den[good]
                lo, hi = np.quantile(distribution, [.025, .975])
                records.append({'site': site, 'layer': layer, 'component': component,
                                'slope': float(estimate), 'ci_low': float(lo), 'ci_high': float(hi),
                                'schedule_count': len(schedules), 'bootstrap_samples': 2000,
                                'aggregation': 'pooled across all five symmetric reliability pairs',
                                'readout': 'final norm and head in float32 on saved BF16 residuals; descriptive'})
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'even_odd_slopes_by_layer.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for ax, site in zip(axes, margins):
        for component, color in [('even', '#2369a6'), ('odd', '#d46826')]:
            selected = frame[(frame.site == site) & (frame.component == component)].sort_values('layer')
            ax.plot(selected.layer, selected.slope, color=color, label=component)
            ax.fill_between(selected.layer, selected.ci_low, selected.ci_high, color=color, alpha=.15)
        ax.axhline(0, color='black', lw=.7)
        ax.set(title=site, xlabel='Layer', ylabel='Readout margin slope on agreement difference')
        ax.legend()
    fig.suptitle('Reliability-even and reliability-odd readouts; 95% schedule bootstrap intervals')
    save_figure(fig, output / 'even_odd_slopes_by_layer.png')
    return frame, parity_summary


def make_reliability_design(results, snapshot, output):
    rows = [r for r in results if r['split'] == 'test']
    lookup = {(r['question_set_index'], r['answer_pattern_index'], r['reliabilities_exact'][0]): r for r in rows}
    strata = defaultdict(list)
    for schedule in sorted({r['question_set_index'] for r in rows}):
        for pair_index, (low, high) in enumerate([('1/20','19/20'),('1/4','3/4')]):
            for pattern in range(8):
                a, b = lookup[schedule, pattern, low], lookup[schedule, pattern, high]
                for recipient, donor, direction in [(a,b,0),(b,a,1)]:
                    assert all(recipient[f] == donor[f] for f in ['membership_sets','observed_reports','presentation_order','candidate_1','candidate_2','reasoning'])
                    strata[schedule, pair_index, direction].append((recipient, donor))
    selected = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda pair: (pair[0]['delta_a'], pair[0]['agreement_c1'], pair[0]['agreement_c2'], pair[0]['answer_pattern_index']))
        for recipient, donor in [candidates[0], candidates[-1]]:
            selected.append({'recipient_row_id': recipient['row_id'], 'donor_row_id': donor['row_id'],
                             'schedule': recipient['question_set_index'], 'answer_pattern': recipient['answer_pattern'],
                             'recipient_reliability': recipient['reliabilities_exact'][0],
                             'donor_reliability': donor['reliabilities_exact'][0], 'delta_a': recipient['delta_a'],
                             'recipient_z_bayes': recipient['z_bayes'], 'donor_z_bayes': donor['z_bayes'],
                             'non_tie': recipient['delta_a'] != 0})
    assert len(selected) == 64
    design = {'experiment': 'notebook16_reliability_interchange_separate_from_exact_followup',
              'site': 'final_prompt', 'layers': snapshot['locked_layers']['reasoning_off']['final_prompt']['z_bayes'],
              'layer_selection': 'training grouped CV only, frozen before test access',
              'directions': selected, 'row_count': len(selected), 'non_tie_directions': sum(p['non_tie'] for p in selected),
              'reasoning': False, 'gpu_used': False,
              'controls': ['unpatched', 'self_patch', 'resampled_donor_same_reliability', 'training_mean_same_reliability', 'off_target_domain_boundary'],
              'probe_config_fingerprint': snapshot['probe_config_fingerprint']}
    design['design_sha256'] = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    write_json(output / 'notebook16_reliability_patch_design.json', design)
    return design


def summarize_offloaded_interchange(analysis_root, output):
    """Summarize the separate reliability-changing experiment, including partial runs."""
    root = analysis_root/'offloaded_reliability_interchange'
    path = root/'results.jsonl'
    setup = json.loads((root/'execution_setup.json').read_text()) if (root/'execution_setup.json').exists() else {}
    compatibility = json.loads((root/'compatibility.json').read_text()) if (root/'compatibility.json').exists() else {}
    status = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'complete': False}
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if not rows:
        return {'status': status, 'setup': setup, 'compatibility': compatibility, 'records': 0}, pd.DataFrame()
    frame = pd.DataFrame(rows)
    assert not frame.reasoning.any()
    assert not frame.duplicated(['recipient_row_id','donor_row_id','layer','condition']).any()
    self_patches = frame[frame.condition == 'self_patch']
    intervention = frame[frame.condition == 'reliability_interchange'].copy()
    intervention['base_reliability_regime'] = np.where(intervention.recipient_reliability.isin(['1/20','1/4']), 'low', 'high')
    intervention['baseline_recipient_correct'] = intervention.baseline_margin * intervention.recipient_z_bayes > 0
    intervention['patched_recipient_correct'] = intervention.patched_margin * intervention.recipient_z_bayes > 0
    intervention['baseline_counterfactual_correct'] = intervention.baseline_margin * intervention.donor_z_bayes > 0
    intervention['patched_counterfactual_correct'] = intervention.patched_margin * intervention.donor_z_bayes > 0
    intervention['margin_change_toward_counterfactual'] = intervention.margin_change * np.sign(intervention.donor_z_bayes)
    intervention['choice_changed'] = np.sign(intervention.baseline_margin) != np.sign(intervention.patched_margin)
    intervention['original_task_repaired'] = ~intervention.baseline_recipient_correct & intervention.patched_recipient_correct
    intervention['original_task_harmed'] = intervention.baseline_recipient_correct & ~intervention.patched_recipient_correct
    summaries = []
    for (layer, regime), group in intervention[intervention.non_tie].groupby(['layer','base_reliability_regime']):
        summaries.append({'layer': int(layer), 'recipient_reliability_regime': regime,
                          'directed_pairs': len(group), 'schedules': int(group.schedule.nunique()),
                          'baseline_original_task_accuracy': float(group.baseline_recipient_correct.mean()),
                          'patched_original_task_accuracy': float(group.patched_recipient_correct.mean()),
                          'baseline_donor_counterfactual_accuracy': float(group.baseline_counterfactual_correct.mean()),
                          'patched_donor_counterfactual_accuracy': float(group.patched_counterfactual_correct.mean()),
                          'mean_raw_margin_change': float(group.margin_change.mean()),
                          'mean_absolute_margin_change': float(group.margin_change.abs().mean()),
                          'max_absolute_margin_change': float(group.margin_change.abs().max()),
                          'choice_changes': int(group.choice_changed.sum()),
                          'original_task_repairs': int(group.original_task_repaired.sum()),
                          'original_task_harms': int(group.original_task_harmed.sum()),
                          'mean_change_toward_donor_counterfactual': float(group.margin_change_toward_counterfactual.mean()),
                          'run_complete': status.get('complete', False)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'offloaded_interchange_summary.csv', index=False)
    intervention.to_csv(output/'offloaded_interchange_scored_rows.csv', index=False)
    audit = {'status': status, 'setup': setup, 'compatibility': compatibility, 'records': len(rows),
             'interchange_records': len(intervention), 'self_patch_records': len(self_patches),
             'max_self_patch_absolute_change': float(self_patches.margin_change.abs().max()),
             'expected_interchange_records': 192, 'expected_self_patch_records': 192,
             'non_tie_choice_changes': int(intervention.loc[intervention.non_tie, 'choice_changed'].sum()),
             'non_tie_original_task_repairs': int(intervention.loc[intervention.non_tie, 'original_task_repaired'].sum()),
             'non_tie_original_task_harms': int(intervention.loc[intervention.non_tie, 'original_task_harmed'].sum()),
             'answers_exact_same_reliability_followup': False}
    if status.get('complete'):
        assert len(intervention) == len(self_patches) == 192
        assert intervention.groupby('layer').size().eq(64).all()
        assert intervention.groupby('layer').non_tie.sum().eq(56).all()
    write_json(output/'offloaded_interchange_audit.json', audit)
    state_path = output/'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update({
        'gpu_used': True,
        'notebook_kernel_uses_gpu': False,
        'reliability_interchange_measured': len(intervention) > 0,
        'reliability_interchange_complete': status.get('complete', False),
        'exact_same_reliability_rescue_measured': False,
        'causal_rescue_measured': False,
        'causal_rescue_field_scope': 'exact same-reliability clean/incorrect follow-up',
        'full_experiment2_causal_controls_complete': False,
    })
    write_json(state_path, state)
    if not summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for regime, group in summary.groupby('recipient_reliability_regime'):
            group = group.sort_values('layer')
            axes[0].plot(group.layer, group.mean_absolute_margin_change, 'o-', label=f'{regime}-r recipient')
            axes[1].plot(group.layer, group.patched_donor_counterfactual_accuracy - group.baseline_donor_counterfactual_accuracy,
                         'o-', label=f'{regime}-r recipient')
        axes[0].set(xlabel='Patched layer (training-CV-selected)', ylabel='Mean absolute C1−C2 margin change')
        axes[1].set(xlabel='Patched layer', ylabel='Change in donor-counterfactual accuracy')
        for ax in axes:
            ax.axhline(0, color='black', lw=.7)
            ax.legend()
        prefix = 'Complete' if status.get('complete') else 'Partial'
        fig.suptitle(f'{prefix}: notebook-16 reliability interchange; not the same-reliability follow-up')
        save_figure(fig, output/'offloaded_interchange_summary.png')
    return audit, summary
