"""Build the exact-match audit and causal patching follow-up notebook."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(cells=[
        md('''# Low-reliability clean/incorrect activation patching

**Historical exact-question audit:** the user subsequently clarified that questions
and observed answers may differ when agreement totals or grids match. That active
follow-up is in [notebook 20](20_noisy_channel_agreement_matched_patching.ipynb).
The zero-pair result below applies to the earlier constraints, not the clarified task.

This notebook follows notebook 18's completed reasoning-off probes. The screen is
promising for diagnosing a missing reliability-sign interaction: the reliability
sign is available while the signed product is weak at the prompt boundary. It does
not yet identify a reliable signed-posterior direction for a behavioral repair.

The requested experiment matches **the same reliability, exact question sets,
answer pattern, candidate ordering, candidate identities, and reasoning off**.
Clean means a correct emitted answer on a non-tie Bayesian comparison; corrupted
means an incorrect emitted answer. Boundary-logit correctness is reported separately.
Noisy versus clean describes behavior here, not a deliberately corrupted prompt.

The exact match is checked before loading any model. CUDA is hidden in this notebook.
The original training kernel and artifacts are read-only inputs.'''),
        code('''import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
REPO_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'pyproject.toml').exists())
sys.path.insert(0, str(REPO_ROOT / 'notebooks'))
from completed_probe_exploration import match_audit, behavior_frame, write_json, write_jsonl
from patching_exploration import layerwise_margin_decomposition, make_reliability_design, prospective_matching_audit, summarize_offloaded_interchange
MODEL_ROOT = REPO_ROOT/'artifacts/noisy_channel_bayesian_experiment_2_activation_patching/Qwen_Qwen3.5-9B_4c87a623'
ANALYSIS_ROOT = MODEL_ROOT/'analyses/completed_reasoning_off_20260905'
OUTPUT_ROOT = ANALYSIS_ROOT/'low_reliability_patching'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT = MODEL_ROOT/'runs/qwen35_9b_n9_k3_r10_s64_selected_tokens_v2_reasoning_off'
results = [json.loads(line) for line in (RUN_ROOT/'results.jsonl').read_text().splitlines() if line.strip()]
assert len(results) == 5120 and all(r['reasoning'] is False for r in results)
snapshot = json.loads((ANALYSIS_ROOT/'snapshot.json').read_text())
'''),
        md('''## Reproduce the user selections and audit the exact matching condition

There is one deterministic generation per prompt. Repeating a prompt with sampled
answer numerals would not create different residual states at its prompt boundary:
those states occur before sampling. Patching a token that already contains the
answer would instead risk copying the answer itself.'''),
        code('''audit = match_audit(results, OUTPUT_ROOT)
display(pd.DataFrame([audit]).T)
frame = behavior_frame(results)
for label, mask in [
    ('high-r incorrect C2', (frame.reliability > .5) & (frame.delta_a > 0) & (frame.emitted_choice == 'C2')),
    ('low-r correct C1', (frame.reliability < .5) & (frame.delta_a < 0) & (frame.emitted_choice == 'C1')),
]:
    selected = frame[mask]
    display(Markdown(f'### {label}: {len(selected)} rows'))
    display(selected.groupby('reliability').agg(rows=('row_id', 'size'),
              emitted_accuracy=('emitted_correct', 'mean'),
              boundary_logit_accuracy=('logit_correct', 'mean'),
              distinct_schedules=('schedule', 'nunique')))
    selected.to_csv(OUTPUT_ROOT/(label.replace(' ', '_') + '.csv'), index=False)
'''),
        code('''MATCH_FIELDS = ('reliabilities_exact', 'membership_sets', 'observed_reports',
                'presentation_order', 'candidate_1', 'candidate_2', 'reasoning')
groups = defaultdict(list)
for row in results:
    if row['reliabilities'][0] < .5 and row['delta_a'] != 0:
        groups[json.dumps([row[field] for field in MATCH_FIELDS], sort_keys=True)].append(row)
directions = []
for group in groups.values():
    clean = [row for row in group if row['canonical_posterior_correct']]
    corrupted = [row for row in group if not row['canonical_posterior_correct']]
    for good in clean:
        for bad in corrupted:
            for donor, recipient, direction in [(good, bad, 'denoising'), (bad, good, 'noising')]:
                directions.append({'donor_row_id': donor['row_id'], 'recipient_row_id': recipient['row_id'],
                                   'direction': direction, 'matching_fields': list(MATCH_FIELDS)})
write_jsonl(OUTPUT_ROOT/'exact_patch_directions.jsonl', directions)
display({'exact_directed_patches': len(directions), 'status': 'no_eligible_pairs' if not directions else 'ready'})
if not directions:
    display(Markdown(''' + repr('''**No exact matched donor/recipient pair exists.** The saved bank cannot answer whether
that intervention fixes behavior. The empty result is a data-design limitation, not
a negative causal result. Changing exact question sets while holding both candidates’
per-question agreement patterns fixed is one possible alternative; it changes the
question-matching constraint and is not executed here without that choice.''') + '''))
'''),
        md('''## Eligibility under the pending alternative matching choice

This is a count-only audit of a possible alternative, not a change to the requested
experiment. It allows different exact question sets while holding each candidate's
entire per-question agreement pattern fixed, along with reliability, observed reports,
answer ordering, identities, and reasoning off. The Bayesian comparison stays identical.
No donor is selected and no relaxed-match intervention is run.

Counts distinguish emitted-answer correctness from the stricter requirement that the
emitted answer and a non-tied boundary-logit choice agree. They are broken out by
train/test split and Bayesian winner. Reusing rows in many pairs does not create
independent observations.'''),
        code('''prospective_counts = prospective_matching_audit(results, OUTPUT_ROOT)
display(prospective_counts)
'''),
        md('''## Experiment 2: even and odd agreement slopes across depth

For each symmetric reliability pair, decode C1−C2 with the model's own final
normalization and answer-token weights applied to the saved residual at each layer.
Plot the slope of `(m_low + m_high)/2` (even) and `(m_high − m_low)/2` (odd)
on agreement difference. These are descriptive logit-lens readouts; intermediate
layers are not calibrated output distributions and do not prove causal use.

We compare `final_prompt` (before assistant output) with `answer_prefix` (immediately
before the answer numeral). Neither includes the answer numeral. Uncertainty is a
bootstrap over the eight held-out schedules, keeping report patterns and reliability
pairs together. The head-only final-layer check is verified against saved logits.'''),
        code('''decomposition, head_parity = layerwise_margin_decomposition(
    results, RUN_ROOT, REPO_ROOT/'models/Qwen--Qwen3.5-9B', OUTPUT_ROOT)
display(head_parity)
display(decomposition[decomposition.layer.isin([0, 7, 15, 23, 31])])
'''),
        md('''## Original notebook-16 reliability interchange: a separate experiment

This keeps question sets, report pattern, candidates and ordering fixed, and swaps
reliability between 0.05/0.95 or 0.25/0.75. It changes reliability, so it does not
answer the same-reliability clean/incorrect follow-up above. The 64 directed pairs
and three `final_prompt` layers are selected from the saved design and training CV.
Exact posterior ties are retained as continuous controls and excluded from choice
accuracy. The immutable design is saved before any patched forward pass.

The CPU compatibility check uses the same local checkpoint in float32 and compares
unpatched margins against the saved BF16 capture with notebook 16's 0.05 tolerance.
A failed gate prevents claims about patching the saved GPU behavior. The original
screen includes unpatched/self-patch checks, both directions, and raw margin changes.
Stronger causal claims additionally require mean/resampled and off-target controls.
Whole-residual patches are not evidence for a specific low-dimensional sign mechanism.'''),
        code('''reliability_design = make_reliability_design(results, snapshot, OUTPUT_ROOT)
display({key: value for key, value in reliability_design.items() if key != 'directions'})
parity_path = ANALYSIS_ROOT/'cpu_patch_parity.json'
cpu_parity = json.loads(parity_path.read_text()) if parity_path.exists() else {'complete': False, 'status': 'not_run'}
display(cpu_parity)
bf16_path = ANALYSIS_ROOT/'cpu_patch_parity_bf16_linear_fp32_accumulation.json'
bf16_parity = json.loads(bf16_path.read_text()) if bf16_path.exists() else {'complete': False}
display(bf16_parity)
full_capture_path = ANALYSIS_ROOT/'cpu_patch_parity_bf16_linear_fp32_accumulation_full_capture_forward.json'
full_capture_parity = json.loads(full_capture_path.read_text()) if full_capture_path.exists() else {'complete': False}
display(full_capture_parity)
write_json(OUTPUT_ROOT/'status.json', {
    'exact_followup_status': 'no_eligible_pairs' if not directions else 'ready',
    'exact_pair_count': len(directions),
    'reliability_interchange_design_directions': len(reliability_design['directions']),
    'cpu_parity_complete': cpu_parity.get('complete', False),
    'cpu_compatible': cpu_parity.get('compatible', False),
    'cpu_bf16_compatible': bf16_parity.get('compatible', False),
    'cpu_full_capture_replay_complete': full_capture_parity.get('complete', False),
    'cpu_full_capture_compatible': full_capture_parity.get('compatible', False),
    'gpu_used': False,
    'causal_rescue_measured': False,
})
'''),
        md('''## Optional full-resident runner (unused)

The offloaded run in the next section has completed this screen; there is no need
to repeat it. This alternative checks live GPU compute processes before importing torch or loading
a model. It never contacts or interrupts an existing kernel. Set the execution flag
only to request a run once the GPU is unoccupied; an occupied GPU always prevents it.
The runner reuses notebook 16's actual patching code, fixed to reasoning off, with
the snapshot's CV-selected layers and a separate output directory. Its original
unpatched-margin compatibility gate still applies.

This is notebook 16's whole-residual screen. It does **not** yet contain all the
mean/resampled/off-target controls required for experiment 2's stronger causal claim,
and it does not implement the nonexistent exact same-reliability pair design.'''),
        code('''import subprocess
RUN_ORIGINAL_INTERCHANGE = False
command = [str(REPO_ROOT/'.venv/bin/python'), str(REPO_ROOT/'scripts/run_completed_probe_patching.py')]
if RUN_ORIGINAL_INTERCHANGE:
    command.append('--execute-gpu')
check = subprocess.run(command, capture_output=True, text=True, check=False)
print(check.stdout)
if check.returncode:
    print(check.stderr)
assert check.returncode == 0
patch_path = ANALYSIS_ROOT/'original_reliability_interchange_gpu/results.jsonl'
if patch_path.exists():
    patch_rows = [json.loads(line) for line in patch_path.read_text().splitlines() if line.strip()]
    display(pd.DataFrame(patch_rows))
else:
    print('The optional full-resident runner has not been used. See the offloaded results below.')
'''),
        md('''## Native GPU replay with weights offloaded to CPU

The live fits were verified to be scikit-learn CPU jobs while the GPU had unused
compute and over 12 GiB of memory free. A separate native-BF16 GPU process transfers
weights module by module from CPU and caps its allocator at 6 GiB. It never changes
the live kernel or its allocation. The four original compatibility strata are
checked before patching, and every recipient gets a fresh baseline and saved-state
self-patches at the three locked layers.

This section reads saved outputs; it does not launch GPU work from this CPU notebook.
All records preserve exact questions/reports/ordering and change reliability. We
report accuracy against both the recipient's original Bayesian answer and the donor's
counterfactual answer, so a changed target is never mislabeled as a same-task repair.
An unfinished run is labeled partial, and no exact same-reliability repair claim is made.

The locked layers are 2, 29, and 31. A final-prompt patch after layer 31 has no
remaining layer in which to influence the later answer-prefix position; its null
effect is structurally expected under this teacher-forced scoring design. That
control must not be interpreted as evidence against a reliability representation.
Only answer-boundary logits are measured here; patched completions are not regenerated.'''),
        code('''offloaded_audit, offloaded_summary = summarize_offloaded_interchange(ANALYSIS_ROOT, OUTPUT_ROOT)
display(offloaded_audit)
display(offloaded_summary)
if offloaded_audit['status'].get('complete'):
    display(Markdown(
        f"**Completed:** {offloaded_audit['interchange_records']} reliability-interchange patches "
        f"and {offloaded_audit['self_patch_records']} self-patches. "
        f"Non-tie margin-sign changes: {offloaded_audit['non_tie_choice_changes']}; "
        f"original-task repairs: {offloaded_audit['non_tie_original_task_repairs']}; "
        f"newly incorrect choices: {offloaded_audit['non_tie_original_task_harms']}. "
        "This is a result for the reliability-changing whole-residual screen, "
        "not the requested same-reliability clean/incorrect intervention. "
        "That follow-up remains unmeasured because no exact matched pairs exist."))
'''),
    ])
    nb.metadata.kernelspec = {'display_name': 'Python 3 (MATS)', 'language': 'python', 'name': 'python3'}
    path = ROOT/'notebooks/19_noisy_channel_low_reliability_patching.ipynb'
    nbformat.write(nb, path)
    print(path)


if __name__ == '__main__':
    main()
