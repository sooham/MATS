"""Resume notebook 16's saved whole-residual experiment only on an unoccupied GPU.

Default is a read-only readiness check. --execute-gpu is the explicit execution gate.
No existing kernel is contacted, interrupted, unloaded, or shut down.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    query = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=False)
    owners = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if query.returncode:
        print(json.dumps({'status': 'gpu_status_unavailable', 'executed': False}))
        return
    if owners:
        print(json.dumps({'status': 'waiting_for_unoccupied_gpu', 'compute_processes': owners,
                          'executed': False, 'existing_kernels_contacted': False}))
        return
    if not args.execute_gpu:
        print(json.dumps({'status': 'ready', 'executed': False,
                          'command': '.venv/bin/python scripts/run_completed_probe_patching.py --execute-gpu'}))
        return

    # All imports and CUDA initialization happen after the live-process guard.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.chdir(ROOT)
    import nbformat
    original = nbformat.read(ROOT/'notebooks/16_noisy_channel_bayesian_activation_patching.ipynb', as_version=4)
    copied = nbformat.read(ROOT/'notebooks/18_noisy_channel_completed_reasoning_off_probes.ipynb', as_version=4)
    namespace = {}
    # Original imports do not hide CUDA. The offline load cells read the frozen snapshot
    # and existing result rows; they never regenerate data or train probes.
    exec(compile(original.cells[2].source, 'nb16 imports', 'exec'), namespace)
    import sys
    sys.path.insert(0, str(ROOT/'notebooks'))
    from completed_probe_exploration import freeze_snapshot
    namespace['freeze_snapshot'] = freeze_snapshot
    for index in (4, 5, 8, 9, 18, 19):
        exec(compile(copied.cells[index].source, f'nb18 cell {index}', 'exec'), namespace)
    namespace['MODEL_ID'] = str(ROOT/'models/Qwen--Qwen3.5-9B')
    namespace['tokenizer'] = namespace['AutoProcessor'].from_pretrained(
        namespace['MODEL_ID'], local_files_only=True).tokenizer
    namespace['PATCH_ROOT'] = namespace['ANALYSIS_ROOT']/'original_reliability_interchange_gpu'
    namespace['RUN_ACTIVATION_PATCHING'] = True
    # Notebook 16 still contains branches for reasoning on. Restrict their selection
    # to the user-requested active mode without editing the original notebook.
    patch_source = original.cells[29].source.replace(
        'for mode_name in MODE_NAMES.values()',
        'for mode_name in [MODE_NAMES[value] for value in REASONING_VALUES]')
    exec(compile(patch_source, 'nb16 reasoning-off patching', 'exec'), namespace)


if __name__ == '__main__':
    main()
