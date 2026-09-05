"""Continue this exact live discovery process into evaluation and notebook export.

This does not restart an interrupted/failed discovery run. It waits for the named,
verified process to exit and requires its complete result manifest before proceeding.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/noisy_channel_bayesian_experiment_2_activation_patching/Qwen_Qwen3.5-9B_4c87a623/analyses/agreement_matched_reasoning_off_20260905'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--discovery-pid',type=int,required=True)
    args=parser.parse_args()
    proc=Path(f'/proc/{args.discovery_pid}')
    setup=json.loads((OUT/'discovery/execution_setup.json').read_text())
    assert setup['pid']==args.discovery_pid
    last_print=0
    while proc.exists():
        try:
            command=(proc/'cmdline').read_bytes()
        except FileNotFoundError:
            break
        if command:
            assert b'run_agreement_matched_patching.py' in command and b'discovery' in command, 'PID no longer belongs to this discovery job'
        if time.monotonic()-last_print>60:
            result_path=OUT/'discovery/job_results.jsonl'
            count=len(result_path.read_text().splitlines()) if result_path.exists() else 0
            print(f'Verified live discovery PID {args.discovery_pid}: {count}/2560 saved jobs',flush=True)
            last_print=time.monotonic()
        time.sleep(5)
    state=json.loads((OUT/'discovery/status.json').read_text())
    assert state.get('complete'), f'Discovery exited without completion; not restarting it: {state}'
    python=str(ROOT/'.venv/bin/python')
    subprocess.run([python,str(ROOT/'scripts/run_agreement_matched_patching.py'),
                    '--phase','evaluation','--cached-suffix'],cwd=ROOT,check=True)
    assert json.loads((OUT/'evaluation/status.json').read_text())['complete']
    subprocess.run([python,str(ROOT/'scripts/build_agreement_matched_patching_notebook.py')],cwd=ROOT,check=True)
    subprocess.run([python,str(ROOT/'scripts/execute_exploration_notebook.py'),
                    str(ROOT/'notebooks/20_noisy_channel_agreement_matched_patching.ipynb')],cwd=ROOT,check=True)
    print('COMPLETE: discovery, frozen-location evaluation, and executed notebook 20',flush=True)


if __name__=='__main__':main()
