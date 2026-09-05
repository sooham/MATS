"""Build notebook 20 for the user's clarified agreement-matched experiment."""
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nbformat.v4.new_markdown_cell
    code=nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook(cells=[
        md('''# Same-reliability, agreement-matched clean/corrupt patching

This implements the clarified follow-up to notebooks 18–19. Exact questions and
observed answers **may differ**. Reliability (<0.5), candidate ordering/identities,
problem size and reasoning off remain fixed. We compare two nested matching rules:

1. **Equal totals:** X and Y each have the same total agreement in donor and recipient.
2. **Exact grid:** X and Y each have identical per-question agreement bit vectors.

Both preserve the pairwise Bayesian log odds and correct candidate, despite differing
surface questions/answers. Clean/corrupt is defined by the original emitted answer.
We also report a stricter subset whose emitted label agrees with a non-tied saved
answer-boundary margin. Ties in the Bayesian target are excluded.

This notebook runs on CPU and reads outputs from a separate memory-capped native GPU
replay. It never executes in, pauses, or unloads the original training kernel.'''),
        code('''import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
REPO_ROOT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'pyproject.toml').exists())
sys.path.insert(0,str(REPO_ROOT/'notebooks'))
from agreement_matched_patching import OUT, SITES, load_rows, prepare, summarize_phase
protocol=prepare()
rows=load_rows()
by_id={r['row_id']:r for r in rows}
display({k:len(protocol[k]) for k in ['discovery_pairs','grid_followup_pairs','heldout_totals_pairs']})
display({'protocol_sha256':protocol['protocol_sha256'],'reasoning':False,'notebook_device':'cpu'})
'''),
        md('''## Pair availability and split separation

The previous notebook's zero exact-question pairs are not a blocker for this revised
design. Equal totals yield held-out pairs; exact grids do not. Grid results therefore
remain training-split exploration and are never presented as held-out confirmation.
Pair counts are dependent combinations, not independent observations. Strict held-out
pairs only have C2 as the correct candidate; this limits generalization.'''),
        code('''display(pd.read_csv(OUT/'matching_counts.csv'))
for name in ['discovery_pairs','grid_followup_pairs','heldout_totals_pairs']:
    display(Markdown('### '+name))
    frame=pd.DataFrame(protocol[name])
    display(frame.groupby(['reliability','winner']).agg(pairs=('pair_id','size'),strict=('strict_boundary_pair','sum')))
'''),
        md('''## Inspect the actual agreement grids

Examples are fixed by the frozen pair order, not by patch outcomes. The totals-only
examples can move evidence across question positions; the exact-grid examples cannot.
Different reports are permitted, so matching agreement does not imply matching raw
membership bits or question wording.'''),
        code('''for cohort in ['discovery_pairs','heldout_totals_pairs']:
    for pair in protocol[cohort][:2]:
        records=[]
        for role,row_id in [('clean',pair['clean_id']),('corrupt',pair['corrupt_id'])]:
            row=by_id[row_id]
            for q,(membership,report,gx,gy) in enumerate(zip(row['membership_sets'],row['observed_reports'],row['agreement_candidate_1_by_question'],row['agreement_candidate_2_by_question'])):
                records.append({'role':role,'question':q+1,'membership_set':membership,'report':report,'X_agrees':gx,'Y_agrees':gy})
        display(Markdown(f"**{cohort}, r={pair['reliability']}, winner={pair['winner']}, same grid={pair['same_grid']}**"))
        display(pd.DataFrame(records))
'''),
        md('''## Frozen training discovery: both directions across all 32 layers

Ten strict exact-grid training pairs (one per reliability × winner) are scanned at
`final_prompt` and `answer_prefix`, before the answer numeral. Denoising patches a
clean residual into a corrupt recipient; noising reverses that pair. Every site/layer
has a saved-state self-patch control, with fresh native unpatched baselines.

For each site, select the layer in 0–30 with the largest average bidirectional
truth-signed margin effect; ties choose the earlier layer. Selection is written before
held-out patching. Layer 31 remains a diagnostic: `final_prompt` there has no remaining
cross-token influence, while `answer_prefix` there can directly transfer the donor's
readout. Even near-final answer-prefix repairs may be decision copying, not restored
Bayesian computation. This is a whole-residual experiment, not a signed-subspace test.'''),
        code('''discovery_status,discovery_rows,discovery_summary=summarize_phase('discovery')
display(discovery_status)
if not discovery_summary.empty:
    display(discovery_summary[(discovery_summary.condition=='cross_patch') & (discovery_summary.label_scope=='strict_boundary_pairs')])
lock_path=OUT/'locked_locations.json'
display(json.loads(lock_path.read_text()) if lock_path.exists() else {'locations_locked':False})
'''),
        md('''## Frozen-location evaluation and controls

Evaluation uses all 160 emitted-label held-out totals pairs (119 strict pairs), plus
117 strict-grid training pairs with discovery rows excluded, one per matching group.
The grid follow-up is not an independent test split. All matching groups/pairs and the
layer-selection rule are frozen before patched outcomes are inspected.

Controls at each selected location include self-patching, another matched donor with
the recipient's own behavior label, a label-agnostic matched-group mean from training,
and a donor patch at the off-target domain boundary. If no other strict same-label
donor exists, its fallback is explicitly labeled self; it is not counted as a genuine
resampled control. Cross-pair rows in the grid follow-up exclude discovery rows;
its control pools still use the full training split. Mean controls can include training
recipients, but never test rows.
All selected rows have the same input prefix through the domain boundary. That
off-target control is therefore an expected null, not a deliberately different
off-target perturbation.

Metrics include raw/truth-signed margin changes, repair and damage counts, conditional
candidate accuracy, and full-vocabulary next-token correctness at the saved answer
prefix. Complete patched responses are not regenerated. Pair-weighted and
recipient-weighted effects are reported separately; reused recipients/donors and seven
held-out schedules preclude treating hundreds of patches as independent observations.
The sensitivity columns `buffered_repair_count` and `buffered_damage_count` require
the truth-signed margin to cross from ≤−0.25 to ≥0.25, or the reverse. This separates
larger changes from near-tie flips and does not affect location selection. Paired
control contrasts compare the same donor/recipient edge, site and layer.
The choice-change plot uses the full-vocabulary greedy next token, not just the
sign of the two-candidate margin. These can differ at ties or when another token
outranks both candidates. Both sets of counts remain in the table.'''),
        code('''evaluation_status,evaluation_rows,evaluation_summary=summarize_phase('evaluation')
display(evaluation_status)
if not evaluation_summary.empty:
    display(evaluation_summary[(evaluation_summary.condition=='cross_patch')])
    display(evaluation_summary[(evaluation_summary.label_scope=='strict_boundary_pairs') & (evaluation_summary.condition!='cross_patch')])
    display(pd.read_csv(OUT/'evaluation/paired_control_contrasts.csv'))
'''),
        md('''## Execution audit and interpretation limits

Native GPU replay keeps model weights on CPU between module calls and caps its own
CUDA allocator at 6 GiB. Existing GPU owners are checked before each forward pass.
Every recipient must reproduce its saved margin within the original 0.05 tolerance.
Batching is used only after unpatched and self-patch parity checks; otherwise that
recipient falls back to singleton replay. Controls are compared in the same backend.
The optimized singleton path caches the unchanged full-sequence residuals before the
patch and reruns every subsequent layer. For each recipient it must exactly match
full native unpatched replay at layers 0, 15 and 31, and an actual full-forward cross
patch at each site. The output matrix stays resident within the same 6 GiB cap.

A successful clean-state transfer would establish a causal whole-state influence at
that location under this matching design. It would not, by itself, identify reliability
sign multiplication: donor states can carry a decision, token/surface features, or
other correlated variables. No cross-order, cross-identity or cross-reliability
generalization is tested here. The completed probe plan does not include this exact
answer-prefix site: weak Bayesian-product decoding at the earlier prompt boundaries
cannot be extrapolated to it. Its descriptive head readouts are not trained product
probes. See the findings report for completed numerical results.'''),
        code('''for phase in ['discovery','evaluation']:
    root=OUT/phase
    for filename in ['execution_setup.json','status.json']:
        path=root/filename
        if path.exists():
            display({phase+'/'+filename:json.loads(path.read_text())})
    path=root/'baseline_audit.jsonl'
    if path.exists():
        checks=[json.loads(s) for s in path.read_text().splitlines()]
        display({'phase':phase,'recipient_replays':len(checks),
                 'max_singleton_saved_margin_error':max((abs(r['single']['margin']-r['saved_margin']) for r in checks),default=0),
                 'batch_compatible_recipients':sum(r['batch_and_self_compatible'] for r in checks),
                 'exact_cached_suffix_replays':sum(r.get('replay_mode')=='exact_native_cached_suffix' for r in checks)})
'''),
    ])
    nb.metadata.kernelspec={'display_name':'Python 3 (MATS)','language':'python','name':'python3'}
    path=ROOT/'notebooks/20_noisy_channel_agreement_matched_patching.ipynb'
    nbformat.write(nb,path)
    print(path)


if __name__=='__main__':main()
