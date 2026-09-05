"""Copy notebook 16 into a CPU-only, immutable completed-probe exploration."""
from pathlib import Path
import copy
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/16_noisy_channel_bayesian_activation_patching.ipynb'
DEST = ROOT / 'notebooks/18_noisy_channel_completed_reasoning_off_probes.ipynb'


def code(text):
    return nbformat.v4.new_code_cell(text.strip() + '\n')


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip() + '\n')


def main():
    original = nbformat.read(SOURCE, as_version=4)
    nb = copy.deepcopy(original)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    nb.cells[0] = md('''# Completed reasoning-off probes: notebook 16 exploration

This is a copy of notebook 16 restricted to completed 32-layer sweeps. The original
notebook, live kernel, training weights and activation captures are read-only inputs.
All work here uses CPU only. Results go to a separate analysis directory.

The cohort and the top three layers per target/site are frozen using training CV
before opening held-out activations. Full test curves and transfer matrices are
exploratory diagnostics, never a source of layer selection. There are eight held-out
question schedules (640 rows), so rows are not independent replicates.

`answer_line` includes the emitted answer token. Its probes can reflect the answer
already written and cannot establish a computation before the decision. Early
reliability sites precede the observations, so agreement decoding there should be null.
''')
    nb.cells[1] = md('''Run all cells in a **new kernel**. CUDA is hidden before importing torch,
and no model weights are loaded. Re-running reuses the frozen snapshot even if the
original training run advances. This bank holds out schedules only: reliability,
prompt rendering, candidate identity and ordering generalization remain untested.''')
    nb.cells[2].source = '''from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
''' + nb.cells[2].source.replace('from __future__ import annotations', '') + '''
torch.set_num_threads(2)
from scipy.special import expit
from completed_probe_exploration import (
    freeze_snapshot, load_test_cache, evaluate_probes, behavior_audit,
    plot_diagnostics, match_audit, write_interpretation,
)
'''
    nb.cells[4].source = nb.cells[4].source.replace('REASONING_VALUES = (False, True)', 'REASONING_VALUES = (False,)').replace('RUN_PROBE_TRAINING = True', 'RUN_PROBE_TRAINING = False').replace('RUN_PROBE_VISUALIZATIONS = False', 'RUN_PROBE_VISUALIZATIONS = True').replace('RUN_PROBE_TRANSFER_ANALYSIS = False', 'RUN_PROBE_TRANSFER_ANALYSIS = True')
    nb.cells[5] = code('''
MODEL_ROOT = EXPERIMENT_ROOT / 'Qwen_Qwen3.5-9B_4c87a623'
SOURCE_PROBE_ROOT = MODEL_ROOT / 'probes' / PROBE_RUN_ID
ANALYSIS_ROOT = MODEL_ROOT / 'analyses' / 'completed_reasoning_off_20260905'
PROBE_ROOT = ANALYSIS_ROOT / 'probes'
RUN_DIRS = {False: MODEL_ROOT / 'runs' / RUN_IDS[False]}
snapshot = freeze_snapshot(SOURCE_PROBE_ROOT, PROBE_ROOT, ANALYSIS_ROOT, REPO_ROOT)
PROBE_CONFIG_FINGERPRINT = snapshot['probe_config_fingerprint']
PROBE_SITES = tuple(snapshot['sites'])
display(pd.DataFrame(snapshot['inventory']))
display({'completed_sites': list(PROBE_SITES), 'probes': len(snapshot['records']),
         'snapshot_time_utc': snapshot['created_at'], 'device': 'cpu'})
''')
    nb.cells[6] = md('''## Load the existing split and captures

No datasets, splits or training plans are regenerated. The original saved labels and
schedule assignments are retained, and all reasoning-on rows are excluded.''')
    nb.cells[8] = code('''
def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\\n')
    os.replace(temporary, path)

def atomic_write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\\n' for row in rows))
    os.replace(temporary, path)

def agreement_cell(row):
    return int(row['agreement_c1']), int(row['agreement_c2'])

def row_reliability_exact(row):
    return row['reliabilities_exact'][0]
''')
    nb.cells[9] = code('''
results = [json.loads(line) for line in (RUN_DIRS[False] / 'results.jsonl').read_text().splitlines() if line.strip()]
assert len(results) == 5120 and all(row['reasoning'] is False for row in results)
assert len({row['row_id'] for row in results}) == len(results)
results_by_reasoning = {False: results}
result_by_id = {row['row_id']: row for row in results}
enriched_rows = results
dataset = results
train_rows = [row for row in results if row['split'] == 'train']
test_rows = [row for row in results if row['split'] == 'test']
test_schedule_set = {int(row['question_set_index']) for row in test_rows}
assert len(train_rows) == 4480 and len(test_rows) == 640
assert len(test_schedule_set) == 8
assert test_schedule_set.isdisjoint({row['question_set_index'] for row in train_rows})
assert all(set(record['train_schedule_ids']).isdisjoint(test_schedule_set) for record in snapshot['records'])
EXPECTED_ROWS = 5120
EXPECTED_ROWS_PER_REASONING = 5120
EXPECTED_TRAIN_ROWS = 4480
EXPECTED_TEST_ROWS = EXPECTED_TEST_ROWS_PER_REASONING = 640
TIE_CELLS = {(a, a) for a in range(4)}
display({'train_rows': len(train_rows), 'test_rows': len(test_rows),
         'test_schedules': sorted(test_schedule_set)})
''')
    nb.cells[10] = md('''### Split audit

These are the existing held-out schedules, not a new split selected after seeing performance.''')
    # This original audit cell depends only on saved labels, apart from its final diagnostic.
    nb.cells[11].source = nb.cells[11].source.split('display(pd.DataFrame(split_audit')[0]
    nb.cells[12] = md('''## Resource isolation

Only frozen probe weights and saved test activations are read. CUDA is unavailable to
this kernel. Activation matrices are cached in CPU memory and released at kernel shutdown.''')
    nb.cells[13] = code("assert not torch.cuda.is_available()\nprint('CPU-only analysis; original training process is untouched.')")
    nb.cells[14] = md('''## Existing capture inventory

The capture remains read-only. Availability is checked at every selected site while
loading the test matrices below.''')
    nb.cells[15] = code("display({'capture_run': RUN_IDS[False], 'rows': len(results), 'reasoning': False})")
    nb.cells[16] = code("assert all((RUN_DIRS[False] / row['activation_path']).is_file() for row in test_rows)\nprint('All 640 test capture files exist.')")
    nb.cells[17] = md('''## Frozen probes

The existing ridge/logistic fits and their scalers are reused. No fitting occurs.
The snapshot validates complete metadata and weight files for all 32 layers and six
targets at every included site. Other sites remain visible in the inventory as incomplete.''')
    nb.cells[18].source = nb.cells[18].source.split('def save_fitted_probe(')[0] + '''
def read_jsonl_if_present(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
'''
    nb.cells[19] = code('''
probe_cv_records = snapshot['records']
expected_probe_count = len(probe_cv_records)
locked_layers = snapshot['locked_layers']
display(pd.DataFrame([
    {'site': site, 'target': target, 'locked_layers': layers}
    for site, targets in locked_layers['reasoning_off'].items()
    for target, layers in targets.items()
]))
''')
    nb.cells[20] = md('''### Evaluate frozen probes on the held-out schedules

The primary rows below use the training-CV-selected layers. We also save all 32 test
layers as explicitly exploratory curves, without revising the layer lock. Metrics
are separated by reliability regime and include ties for regression; choice accuracy
excludes exact Bayesian ties.''')
    metric_helpers = original.cells[21].source.split('def load_probe_prediction(')[1].split('test_metric_rows:')[0]
    nb.cells[21] = code('def load_probe_prediction(' + metric_helpers + '''
test_cache = load_test_cache(test_rows, PROBE_SITES, RUN_DIRS[False], HIDDEN_SIZE)
def activation_matrix(rows, *, site, layer):
    # Transfer/scatter cells reuse the same matrices instead of rereading row files.
    ids, matrix = test_cache[site]
    lookup = {row_id: i for i, row_id in enumerate(ids)}
    indices = [lookup[row['row_id']] for row in rows]
    return matrix[int(layer), indices]

test_metric_rows, all_test_metrics, prediction_frame = evaluate_probes(
    results, snapshot, PROBE_ROOT, ANALYSIS_ROOT, test_cache)
display(pd.DataFrame(test_metric_rows).query("subset == 'all' and lock_rank == 1")[
    ['site', 'target', 'layer', 'best_cv_score', 'r2', 'auroc', 'row_count']])
''')
    # Keep notebook 16's figures 1–9, including transfer, in the copied notebook.
    nb.cells[23].source = nb.cells[23].source.replace('raise ValueError(\n            f"Degenerate probe direction: "\n            f"{MODE_NAMES[reasoning]}/{site}/{target}/layer_{layer:02d}"\n        )', 'return np.full_like(direction, np.nan)')
    nb.cells[24].source = nb.cells[24].source.replace('pca = PCA(n_components=2)', '''finite = np.isfinite(direction_matrix).all(axis=1)
            direction_rows = [row for row, keep in zip(direction_rows, finite) if keep]
            direction_matrix = direction_matrix[finite]
            pca = PCA(n_components=2, random_state=SEED)''')
    nb.cells[25].source = nb.cells[25].source.replace('if row.get("probe_config_fingerprint") == PROBE_CONFIG_FINGERPRINT', 'if row.get("probe_config_fingerprint") == PROBE_CONFIG_FINGERPRINT and row["subset"] == "all"')
    nb.cells[26] = md('''### Cross-layer and cross-location transfer

These notebook-16 diagnostics apply fixed probes to held-out activation matrices.
They are exploratory and are not used to revise layer selection. Probe direction
similarity and transfer alone do not demonstrate causal information transport.''')
    nb.cells[28] = md('''## Behavior, experiment-2 interpretation, and patch feasibility

Behavior is evaluated separately for saved logits and emitted answers. The matched
exact-question audit and completed original reliability interchange live in notebook
19. The user's clarified agreement-matched clean/incorrect follow-up is in notebook 20.''')
    nb.cells[29] = code('''
behavior, slopes = behavior_audit(results, ANALYSIS_ROOT)
display(behavior)
display(slopes)
plot_diagnostics(results, all_test_metrics, prediction_frame, snapshot, ANALYSIS_ROOT)
pair_audit = match_audit(results, ANALYSIS_ROOT)
display(pair_audit)
interpretation = write_interpretation(results, test_metric_rows, all_test_metrics,
                                    behavior, slopes, pair_audit, snapshot, ANALYSIS_ROOT)
display(Markdown(interpretation))
''')
    nb.cells[30] = code('''
final_status = {
    'reasoning_off_only': all(row['reasoning'] is False for row in results),
    'complete_sites': list(PROBE_SITES),
    'frozen_probe_count': len(probe_cv_records),
    'test_metric_count': len(all_test_metrics),
    'original_training_untouched': True,
    'cuda_available': torch.cuda.is_available(),
    'analysis_root': str(ANALYSIS_ROOT),
}
atomic_write_json(ANALYSIS_ROOT / 'notebook_status.json', final_status)
display(final_status)
''')
    nb.cells.extend([
        md('''## Newly completed site: assistant turn boundary

Training finished this additional 32-layer sweep while the original nine-site
analysis and reliability-interchange experiment were running. It is frozen in a
separate supplement, using training CV to lock layers before reading its test
activations. The original snapshot and patch-layer selections remain unchanged.
All calculations in this supplement use CPU and reasoning-off rows only.'''),
        code('''from completed_probe_exploration import completed_site_supplement
supplement_summary = completed_site_supplement(
    SOURCE_PROBE_ROOT, ANALYSIS_ROOT, REPO_ROOT, results, RUN_DIRS[False],
    'assistant_turn_boundary')
display(supplement_summary)
'''),
    ])
    nb.cells.extend([
        md('''## Newly completed site: candidate question boundary

This additional site completed its six 32-layer sweeps while the agreement-matched
patch experiment was running. Its weights and training-CV layer choices are frozen
in another separate supplement before reading its test activations. Neither the
original probe snapshot nor the running patch protocol or selection rule changes.'''),
        code('''from completed_probe_exploration import completed_site_supplement
candidate_boundary_summary = completed_site_supplement(
    SOURCE_PROBE_ROOT, ANALYSIS_ROOT, REPO_ROOT, results, RUN_DIRS[False],
    'candidate_question_boundary')
display(candidate_boundary_summary)
'''),
    ])
    # Preserve verified outputs when adding a new section. Only unchanged sources
    # qualify; changed cells must be executed again.
    if DEST.exists():
        previous = nbformat.read(DEST, as_version=4)
        prior = {cell.source: cell for cell in previous.cells if cell.cell_type == 'code'}
        for cell in nb.cells:
            if cell.cell_type == 'code' and cell.source in prior:
                cell.outputs = prior[cell.source].outputs
                cell.execution_count = prior[cell.source].execution_count
    nb.metadata['completed_probe_source'] = str(SOURCE.relative_to(ROOT))
    nb.metadata['kernelspec'] = {'display_name': 'Python 3 (MATS)', 'language': 'python', 'name': 'python3'}
    nbformat.write(nb, DEST)
    print(DEST)


if __name__ == '__main__':
    main()
