"""CPU-only analysis of immutable notebook-16 probe snapshots; no training or model load."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from safetensors import safe_open
from safetensors.torch import load_file
from scipy.special import expit
from sklearn.metrics import r2_score, roc_auc_score, balanced_accuracy_score, log_loss


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_snapshot(source, destination, output, repo, eligible_sites=None):
    manifest_path = output / 'snapshot.json'
    if manifest_path.exists():
        snapshot = json.loads(manifest_path.read_text())
        for relative, sha in snapshot['weight_and_metadata_hashes'].items():
            assert digest(destination / relative) == sha, relative
        return snapshot
    plan = json.loads((source / 'probe_plan.json').read_text())
    fingerprint = plan['probe_config_fingerprint']
    targets = plan['continuous_targets'] + plan['binary_targets']
    records, inventory, sites, hashes = [], [], [], {}
    for site in plan['sites']:
        if eligible_sites is not None and site not in eligible_sites:
            continue
        site_records = []
        counts = Counter()
        for target in targets:
            for layer in plan['layers']:
                relative = Path('reasoning_off') / site / target / f'layer_{layer:02d}'
                meta_path = source / relative.with_suffix('.json')
                weight_path = source / relative.with_suffix('.safetensors')
                if not meta_path.exists() or not weight_path.exists():
                    continue
                try:
                    record = json.loads(meta_path.read_text())
                    with safe_open(weight_path, framework='pt', device='cpu') as handle:
                        tensor_meta = handle.metadata()
                        assert tensor_meta['probe_config_fingerprint'] == fingerprint
                        assert set(handle.keys()) == {'coef', 'intercept', 'feature_mean', 'feature_scale'}
                    assert record['probe_config_fingerprint'] == fingerprint
                    assert record['reasoning'] is False
                    assert (record['site'], record['target'], record['layer']) == (site, target, layer)
                    assert record['test_rows_used'] == 0 and record['train_row_count'] == 4480
                    assert len(record['train_schedule_ids']) == 56
                except (AssertionError, OSError, ValueError, KeyError):
                    continue  # A currently incomplete atomic pair is not eligible.
                site_records.append(record)
                counts[target] += 1
        complete = all(counts[target] == 32 for target in targets)
        inventory.append({'site': site, 'complete': complete, **{t: counts[t] for t in targets}})
        if not complete:
            continue
        sites.append(site)
        records.extend(site_records)
        for record in site_records:
            relative = Path('reasoning_off') / site / record['target'] / f"layer_{record['layer']:02d}"
            for extension in ('.json', '.safetensors'):
                rel = relative.with_suffix(extension)
                (destination / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / rel, destination / rel)
                hashes[str(rel)] = digest(destination / rel)
                assert hashes[str(rel)] == digest(source / rel)
    assert sites, 'No complete sites'
    locks = {'reasoning_off': {}}
    for site in sites:
        locks['reasoning_off'][site] = {}
        for target in targets:
            ranked = sorted([r for r in records if r['site'] == site and r['target'] == target],
                            key=lambda r: (-r['best_cv_score'], r['layer']))
            locks['reasoning_off'][site][target] = [r['layer'] for r in ranked[:3]]
    snapshot = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'probe_config_fingerprint': fingerprint, 'source_probe_root': str(source),
        'source_notebook_sha256_at_snapshot': digest(repo / 'notebooks/16_noisy_channel_bayesian_activation_patching.ipynb'),
        'sites': sites, 'targets': targets, 'inventory': inventory, 'records': records,
        'locked_layers': locks, 'weight_and_metadata_hashes': hashes,
        'selection': 'All 32 layers and six targets complete at snapshot; top-3 training CV only',
    }
    write_json(destination / 'locked_layers.json', {'probe_config_fingerprint': fingerprint, 'layers': locks})
    write_jsonl(destination / 'training_cv_metrics.jsonl', records)
    write_json(manifest_path, snapshot)  # Lock persisted before any test activations are read.
    return snapshot


def completed_site_supplement(source, analysis_root, repo, results, run_root, site):
    """Freeze a newly completed site independently of the original analysis cohort."""
    root = analysis_root / f'supplement_{site}'
    snapshot = freeze_snapshot(source, root/'probes', root, repo, eligible_sites=[site])
    assert snapshot['sites'] == [site] and len(snapshot['records']) == 192
    test_rows = [row for row in results if row['split'] == 'test']
    test_schedules = {row['question_set_index'] for row in test_rows}
    assert all(set(record['train_schedule_ids']).isdisjoint(test_schedules) for record in snapshot['records'])
    cache = load_test_cache(test_rows, [site], run_root, 4096)
    locked, all_metrics, predictions = evaluate_probes(results, snapshot, root/'probes', root, cache)
    frame = pd.DataFrame(all_metrics)
    cv = pd.DataFrame(snapshot['records'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, target in zip(axes.flat, snapshot['targets']):
        training = cv[cv.target == target].sort_values('layer')
        test = frame[(frame.target == target) & (frame.subset == 'all')].sort_values('layer')
        metric = 'auroc' if target == 'reliability_sign' else 'r2'
        ax.plot(training.layer, training.best_cv_score, label='Grouped training CV')
        ax.plot(test.layer, test[metric], label='Held-out test (exploratory)')
        for layer in snapshot['locked_layers']['reasoning_off'][site][target]:
            ax.axvline(layer, color='gray', alpha=.2)
        ax.set(title=target, xlabel='Layer', ylabel='AUROC' if metric == 'auroc' else 'R²')
        # Shared scales keep a tiny Bayesian-product R² from looking comparable
        # to near-perfect reliability decoding in neighboring panels.
        ax.set_ylim((.45, 1.05) if metric == 'auroc' else (-.1, 1.05))
        ax.axhline(.5 if metric == 'auroc' else 0, color='black', lw=.6, alpha=.5)
        ax.legend(fontsize=7)
    fig.suptitle(f'Newly completed {site}: frozen separately; gray lines = CV-locked layers')
    save_figure(fig, root/'probe_curves.png')
    summary = pd.DataFrame(locked).query("lock_rank == 1 and subset == 'all'")[
        ['site','target','layer','best_cv_score','r2','auroc','row_count']]
    summary.to_csv(root/'top1_summary.csv', index=False)
    write_json(root/'status.json', {'complete': True, 'probe_count': 192, 'reasoning': False,
                                   'site': site, 'created_at': snapshot['created_at'],
                                   'original_snapshot_modified': False})
    return summary


def positions(row, site):
    if site == 'answer_line':
        return list(range(row['teacher_forced_completion_start'] + row['answer_line_generated_token_start'],
                          row['teacher_forced_completion_start'] + row['answer_line_generated_token_end']))
    value = row['prompt_token_sites'][site]
    return value if isinstance(value, list) else [value]


def load_test_cache(rows, sites, run, hidden_size):
    assert all(r['split'] == 'test' and r['reasoning'] is False for r in rows)
    matrices = {site: np.empty((32, len(rows), hidden_size), dtype=np.float32) for site in sites}
    for index, row in enumerate(rows):
        lookup = {position: j for j, position in enumerate(row['activation_token_indices'])}
        site_indices = {site: [lookup[position] for position in positions(row, site)] for site in sites}
        with safe_open(run / row['activation_path'], framework='pt', device='cpu') as handle:
            for layer in range(32):
                tensor = handle.get_tensor(f'answer.resid_post.layer_{layer}').float().numpy()
                assert tensor.shape == (len(lookup), hidden_size)
                for site in sites:
                    matrices[site][layer, index] = tensor[site_indices[site]].mean(axis=0)
        if (index + 1) % 80 == 0:
            print(f'CPU test cache: {index + 1}/{len(rows)} files, {len(sites)} sites × 32 layers', flush=True)
    return {site: ([row['row_id'] for row in rows], matrix) for site, matrix in matrices.items()}


def metrics(y, prediction, binary):
    if binary:
        return {'r2': None, 'mae': None, 'rmse': None, 'pearson': None,
                'auroc': float(roc_auc_score(y, prediction)) if len(np.unique(y)) == 2 else None,
                'balanced_accuracy': float(balanced_accuracy_score(y, prediction >= .5)) if len(np.unique(y)) == 2 else None,
                'log_loss': float(log_loss(y, prediction, labels=[0, 1]))}
    return {'r2': float(r2_score(y, prediction)) if np.std(y) > 1e-12 else None,
            'mae': float(np.mean(np.abs(y - prediction))),
            'rmse': float(np.sqrt(np.mean((y - prediction)**2))),
            'pearson': float(np.corrcoef(y, prediction)[0, 1]) if np.std(y) > 1e-12 and np.std(prediction) > 1e-12 else None,
            'auroc': None}


def evaluate_probes(results, snapshot, root, output, cache):
    row_lookup = {row['row_id']: row for row in results}
    metadata = {(r['site'], r['target'], r['layer']): r for r in snapshot['records']}
    all_metrics, locked_metrics, predictions = [], [], []
    for site in snapshot['sites']:
        row_ids, X = cache[site]
        rows = [row_lookup[row_id] for row_id in row_ids]
        reliability = np.array([row['reliabilities'][0] for row in rows])
        masks = {'all': np.ones(len(rows), dtype=bool), 'r_lt_half': reliability < .5, 'r_gt_half': reliability > .5}
        for layer in range(32):
            for target in snapshot['targets']:
                weight_path = root / 'reasoning_off' / site / target / f'layer_{layer:02d}.safetensors'
                w = {k: v.numpy() for k, v in load_file(weight_path, device='cpu').items()}
                score = ((X[layer] - w['feature_mean']) / w['feature_scale']) @ w['coef'] + float(w['intercept'][0])
                binary = target == 'reliability_sign'
                prediction = expit(score) if binary else score
                y = np.array([r[target] for r in rows])
                locks = snapshot['locked_layers']['reasoning_off'][site][target]
                rank = locks.index(layer) + 1 if layer in locks else None
                for subset, mask in masks.items():
                    record = {'site': site, 'target': target, 'layer': layer, 'reasoning': False,
                              'reasoning_mode': 'reasoning_off', 'subset': subset, 'lock_rank': rank,
                              'row_count': int(mask.sum()), 'schedule_count': len({r['question_set_index'] for r, m in zip(rows, mask) if m}),
                              'best_cv_score': metadata[site, target, layer]['best_cv_score'],
                              'probe_config_fingerprint': snapshot['probe_config_fingerprint'],
                              **metrics(y[mask], prediction[mask], binary)}
                    all_metrics.append(record)
                    if rank is not None:
                        locked_metrics.append(record)
                if rank == 1:
                    predictions.extend({'row_id': r['row_id'], 'schedule': r['question_set_index'],
                                        'site': site, 'target': target, 'layer': layer,
                                        'reliability': r['reliabilities'][0], 'truth': float(yy),
                                        'prediction': float(pp), 'emitted_correct': r['canonical_posterior_correct'],
                                        'non_tie': r['delta_a'] != 0}
                                       for r, yy, pp in zip(rows, y, prediction))
        print(f'Evaluated frozen probes on test: {site}', flush=True)
    write_jsonl(root / 'locked_test_metrics.jsonl', locked_metrics)
    write_jsonl(output / 'all_layer_test_metrics_exploratory.jsonl', all_metrics)
    pd.DataFrame(all_metrics).to_csv(output / 'all_layer_test_metrics_exploratory.csv', index=False)
    pd.DataFrame(locked_metrics).to_csv(output / 'locked_test_metrics.csv', index=False)
    frame = pd.DataFrame(predictions)
    frame.to_csv(output / 'locked_top1_test_predictions.csv', index=False)
    return locked_metrics, all_metrics, frame


def behavior_frame(results):
    frame = pd.DataFrame([{
        'row_id': r['row_id'], 'schedule': r['question_set_index'], 'split': r['split'],
        'reliability': r['reliabilities'][0], 'reliability_exact': r['reliabilities_exact'][0],
        'delta_a': r['delta_a'], 'z_bayes': r['z_bayes'], 'z_heuristic': r['z_heuristic'],
        'margin': r['answer_surface_raw_logits'][str(r['candidate_1'])] - r['answer_surface_raw_logits'][str(r['candidate_2'])],
        'emitted_choice': r['model_choice_canonical'], 'emitted_correct': r['canonical_posterior_correct'],
    } for r in results])
    frame['non_tie'] = frame.delta_a != 0
    frame['logit_choice'] = np.where(frame.margin > 0, 'C1', np.where(frame.margin < 0, 'C2', 'TIE'))
    frame['logit_correct'] = frame.margin * frame.z_bayes > 0
    frame['emitted_logit_disagree'] = frame.logit_choice != frame.emitted_choice
    return frame


def behavior_audit(results, output):
    frame = behavior_frame(results)
    non_tie = frame[frame.non_tie]
    behavior = non_tie.groupby(['split', 'reliability']).agg(
        non_tie_rows=('row_id', 'size'), schedules=('schedule', 'nunique'),
        emitted_accuracy=('emitted_correct', 'mean'), boundary_logit_accuracy=('logit_correct', 'mean'),
        emitted_logit_disagreement=('emitted_logit_disagree', 'mean')).reset_index()
    slope_rows = []
    for (split, reliability), group in frame.groupby(['split', 'reliability']):
        slope, intercept = np.polyfit(group.delta_a, group.margin, 1)
        slope_rows.append({'split': split, 'reliability': reliability, 'agreement_slope': slope,
                           'intercept': intercept, 'bayes_slope': np.log(reliability/(1-reliability))})
    slopes = pd.DataFrame(slope_rows)
    behavior.to_csv(output / 'behavior_by_reliability.csv', index=False)
    slopes.to_csv(output / 'behavior_agreement_slopes.csv', index=False)
    frame.to_csv(output / 'behavior_rows.csv', index=False)
    return behavior, slopes


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    display(fig)
    plt.close(fig)


def plot_diagnostics(results, all_metrics, prediction_frame, snapshot, output):
    root = output / 'probes' / 'figures'
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_metrics)
    sites = [s for s in ('report_1_answer', 'report_2_answer', 'report_3_answer', 'final_prompt', 'answer_line') if s in snapshot['sites']]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, target in zip(axes, ['delta_a', 'z_heuristic', 'z_bayes']):
        for site in sites:
            part = frame[(frame.site == site) & (frame.target == target) & (frame.subset == 'all')].sort_values('layer')
            ax.plot(part.layer, part.r2, label=site)
        ax.axhline(0, color='black', lw=.7)
        ax.set(title=target, xlabel='Layer', ylabel='Held-out R²', ylim=(-.15, 1))
    axes[-1].legend(fontsize=8)
    fig.suptitle('Frozen probes: all-layer test curves (exploratory; answer_line includes emitted answer)')
    save_figure(fig, root / '10_heuristic_vs_bayes_test.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    behavior = behavior_frame(results)
    for split, group in behavior[behavior.non_tie].groupby('split'):
        means = group.groupby('reliability')[['emitted_correct', 'logit_correct']].mean()
        axes[0].plot(means.index, means.emitted_correct, 'o-', label=f'{split}: emitted')
        axes[0].plot(means.index, means.logit_correct, 'x--', label=f'{split}: boundary logits')
    axes[0].set(xlabel='Reliability', ylabel='Non-tie accuracy', ylim=(0, 1))
    axes[0].legend(fontsize=8)
    for reliability, group in behavior[behavior.split == 'test'].groupby('reliability'):
        means = group.groupby('delta_a').margin.mean()
        axes[1].plot(means.index, means, marker='o', label=f'r={reliability}')
    axes[1].set(xlabel='Agreement difference', ylabel='Mean boundary margin C1 − C2', title='Held-out behavior')
    axes[1].legend(ncol=2, fontsize=7)
    save_figure(fig, root / '11_behavior.png')
    # Low-reliability per-row readouts: a within-regime R² can succeed even if the
    # cross-regime sign interaction is missing. Show both halves explicitly.
    best = frame[(frame.lock_rank == 1) & (frame.target.isin(['delta_a', 'z_bayes', 'z_heuristic']))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, target in zip(axes, ['delta_a', 'z_bayes', 'z_heuristic']):
        subset = best[best.target == target].pivot(index='site', columns='subset', values='r2').reindex(sites)
        subset.plot.bar(ax=ax)
        ax.set(title=target, ylabel='R² at CV-selected layer', ylim=(-.8, 1))
        ax.tick_params(axis='x', labelrotation=35)
    save_figure(fig, root / '12_reliability_stratified_test.png')


def match_audit(results, output):
    frame = behavior_frame(results)
    low = [r for r in results if r['reliabilities'][0] < .5 and r['delta_a'] != 0]
    fields = ['reliabilities_exact', 'membership_sets', 'observed_reports', 'presentation_order', 'candidate_1', 'candidate_2', 'reasoning']
    groups = defaultdict(list)
    for row in low:
        key = json.dumps([row[f] for f in fields], sort_keys=True)
        groups[key].append(row)
    eligible = [rows for rows in groups.values() if any(r['canonical_posterior_correct'] for r in rows) and any(not r['canonical_posterior_correct'] for r in rows)]
    audit = {
        'matching_fields': fields, 'low_reliability_non_tie_rows': len(low),
        'unique_exact_conditions': len(groups), 'repeated_exact_conditions': sum(len(g) > 1 for g in groups.values()),
        'exact_clean_corrupt_groups': len(eligible),
        'exact_directed_clean_to_corrupt_pairs': sum(sum(r['canonical_posterior_correct'] for r in g) * sum(not r['canonical_posterior_correct'] for r in g) for g in eligible),
        'high_reliability_c2_audit_count': int(((frame.reliability > .5) & (frame.delta_a > 0) & (frame.emitted_choice == 'C2')).sum()),
        'low_reliability_c1_audit_count': int(((frame.reliability < .5) & (frame.delta_a < 0) & (frame.emitted_choice == 'C1')).sum()),
        'low_c1_emitted_correct_but_nonpositive_margin': int(((frame.reliability < .5) & (frame.delta_a < 0) & (frame.emitted_choice == 'C1') & (frame.margin <= 0)).sum()),
        'conclusion': 'No exact matched clean/incorrect donors exist in this one-output-per-prompt bank. Same reliability, questions, reports, candidates and ordering fix the prompt. Repeated stochastic generations would not change pre-answer deterministic states in evaluation mode.',
    }
    write_json(output / 'exact_match_audit.json', audit)
    return audit


def write_interpretation(results, locked_metrics, all_metrics, behavior, slopes, pair_audit, snapshot, output):
    frame = pd.DataFrame(locked_metrics)
    top = frame[(frame.lock_rank == 1) & (frame.subset == 'all')]
    table = top[top.site.isin(['final_prompt', 'report_3_answer', 'answer_line'])][['site','target','layer','best_cv_score','r2','auroc']]
    table_text = '| ' + ' | '.join(table.columns) + ' |\n'
    table_text += '| ' + ' | '.join(['---'] * len(table.columns)) + ' |\n'
    for values in table.itertuples(index=False, name=None):
        table_text += '| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) and np.isfinite(v) else ('—' if pd.isna(v) else str(v)) for v in values) + ' |\n'
    text = f'''# Notebook 18: completed reasoning-off probe findings

Frozen at {snapshot['created_at']}: {len(snapshot['sites'])} complete sites,
{len(snapshot['records'])} existing probes. Training used 56 schedules / 4,480 rows;
test uses eight held-out schedules / 640 rows. Layer indices are zero-based.

{table_text}

Reliability and its sign are readily decoded. Agreement and the unsigned heuristic
are better represented than the signed Bayesian interaction near the prompt boundary.
This is consistent with experiment 2's H1: separately available ingredients are not
successfully composed into the signed product used for a direct answer. It is not
proof that the product is absent: a nonlinear or distributed code can evade a linear
probe. Nor does decoding alone show the model causally uses a variable.

The generated answer-line site includes the answer numeral itself. Its stronger
decodability cannot establish that a Bayesian state existed before the answer.
The bank holds out schedules, but not reliability values, prompt renderings or
candidate identities; all rows use candidates 2/7 in one order. It therefore does
not meet experiment 2's complete crossed generalization design. No causal rescue
claim follows from these probes.

The original report's below-chance failure can be tested against the saved behavior
plots: a positive slope of C1−C2 margin on agreement at r<0.5 has the wrong Bayesian
sign. The exact slope is log(r/(1−r)), which is negative in that regime. CSVs retain
separate emitted-answer and teacher-forced boundary-logit accuracy, excluding exact
Bayesian ties from both choice metrics.

The two user selections reproduce counts {pair_audit['high_reliability_c2_audit_count']}
and {pair_audit['low_reliability_c1_audit_count']}. Among the latter,
{pair_audit['low_c1_emitted_correct_but_nonpositive_margin']} have nonpositive saved
C1−C2 margins despite emitting C1. These must not be silently labeled logit-clean.

Exact follow-up match: {pair_audit['exact_directed_clean_to_corrupt_pairs']} donor/recipient
pairs satisfy all requested matching fields. This dataset contains one result per
exact prompt. A change in question sets, reports, reliability, prompt or ordering
would be a different matching design and must be stated explicitly. Notebook 19
preserves this constraint and records the audit; an empty design is not evidence
that patching cannot work.

All-layer test curves and transfer plots are exploratory. The top-three layer lock
was saved before reading test activations and remains unchanged. Reliability-site
agreement probes are negative controls: those sites occur before the evidence.
See `reports/research_agenda/NOTEBOOK13_UPDATE.md`, experiment 2, for the intended
causal interchange and control requirements.
'''
    (output / 'findings.md').write_text(text)
    return text
