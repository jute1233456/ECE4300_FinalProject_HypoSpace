"""Four aligned low-coverage medium cases only; no training or full-suite chaining."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

JOBS = [('causal','n1_003'), ('causal','n1_002'), ('causal','n2_006'), ('3d','g3_h3_b2_002')]


def main():
    import fcntl
    release = Path(__file__).resolve().parents[1]
    root = release.parents[1]
    out = root/'results/server-mcts-2.2-stage-medium-pilot-32k'
    logs = root/'logs/mcts-2.2-stage-medium-pilot-32k'
    out.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    lock = (logs/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, digest in json.loads((release/'release_manifest.json').read_text())['sha256'].items():
        assert sha256((release/name).read_bytes()).hexdigest() == digest, name
    baseline = root/'results/server-mcts-2.1-medium-evaluation-32k/two-step'
    selected = []
    for domain, case_id in JOBS:
        p = baseline/f'{domain}-{case_id}.json'; d = json.loads(p.read_text())
        assert d.get('completed') and len(d['cases']) == 1
        c = d['cases'][0]; assert c['stop_reason'] != 'request_errors'
        selected.append(dict(domain=domain, case_id=case_id, baseline=str(p),
            baseline_sha256=sha256(p.read_bytes()).hexdigest(), baseline_recovery=c['metrics']['recovery'],
            search_cap=c['model_call_cap'], reference_count=c['reference_count']))
    manifest = dict(version='2.2', name='Stage-MCTS', jobs=selected, workers=4,
        selection='Three lowest-coverage completed causal cases and a low-coverage completed 3D case; exploratory, not representative',
        max_tokens=32768, thinking=True, api_base='http://127.0.0.1:8003/v1',
        comparison='Equal search rounds, not equal API/token budget; empty initial trees; no training')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    state = dict(pid=os.getpid(), version='2.2', output=str(out), total=4, completed=0, status='starting')
    def save():
        p=logs/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(logs/'state.json')
    save()
    base = [sys.executable, '-u', '-m', 'experiments.test_easy', '--model','Qwen3-8B',
        '--max-tokens','32768','--enable-thinking','--base-url',manifest['api_base'],
        '--method','mcts-tree','--tree-policy','llm','--duplicate-reward','-1',
        '--exploration-list','off','--two-step-reflection','--unbounded-tree',
        '--mcts-version','2.2','--dataset-root',str(release/'datasets/medium_sampled'),
        '--pause-file',str(out/'PAUSE_two-step')]
    def evaluate(job):
        output=out/f"{job['domain']}-{job['case_id']}.json"
        args=base+['--dataset',job['domain'],'--case-id',job['case_id'],
            '--search-cap',str(job['search_cap']),'--output',str(output)]
        if output.exists():
            d=json.loads(output.read_text())
            if d.get('completed'):return job['case_id'],'completed'
            args+=['--resume',str(output)]
        with output.with_suffix('.log').open('a') as log:
            code=subprocess.run(args,cwd=release,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:return job['case_id'],'process_error'
        d=json.loads(output.read_text())
        return job['case_id'],('completed' if d.get('completed') and
            d['cases'][0]['stop_reason']!='request_errors' else 'failed')
    state['status']='running';save()
    statuses=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(evaluate,j) for j in selected]):
            try:
                case,status=f.result()
            except Exception as error:
                case,status='unknown','exception:'+type(error).__name__
            statuses.append(dict(case_id=case,status=status))
            state.update(completed=sum(x['status']=='completed' for x in statuses),statuses=statuses)
            save();print(case,status,flush=True)
    state['status']='complete' if state['completed']==4 else 'needs_inspection';save()
    # User-requested handoff: resume 2.1's original evaluation/training sequence.
    old_release = root/'releases/mcts-2.1-temp-pool6-32k'
    old_logs = root/'logs/mcts-2.1-temp-pool6'
    old_state_path = old_logs/'state.json'
    old_state = json.loads(old_state_path.read_text())
    pid = old_state.get('pid')
    try:
        existing = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:
        existing = []
    (root/'results/server-mcts-2.1-medium-evaluation-32k/PAUSE_two-step').unlink(missing_ok=True)
    if not any(a.endswith(b'/run_reflection21_only.py') for a in existing):
        with (old_logs/'worker.log').open('a') as log:
            worker = subprocess.Popen([sys.executable, '-u', str(old_release/'server/run_reflection21_only.py'),
                '--root',str(root),'--release',str(old_release)],cwd=root,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        pid=worker.pid
    state.update(handoff='resumed_reflection_MCTS_2.1', resumed_controller_pid=pid)
    save()


if __name__ == '__main__': main()
