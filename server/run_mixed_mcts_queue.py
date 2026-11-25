"""Six question slots shared by Stage-MCTS 2.2 pilot and reflection MCTS 2.1."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class Job:
    version: str
    phase: str
    domain: str
    case_id: str
    output: str


def writers(output, proc_root=Path('/proc')):
    found = []
    for item in proc_root.iterdir():
        if not item.name.isdigit():
            continue
        try:
            args = (item/'cmdline').read_bytes().split(b'\0')
            if b'experiments.test_easy' in args and b'--output' in args:
                if args[args.index(b'--output')+1].decode() == str(output):
                    found.append(int(item.name))
        except (OSError, ValueError, IndexError):
            pass
    return found


def completed(output):
    if not output.exists():
        return False
    d = json.loads(output.read_text())
    return bool(d.get('completed') and len(d.get('cases', [])) == 1
                and d['cases'][0].get('stop_reason') != 'request_errors')


def case_ids(dataset):
    jobs = []
    for domain in ('causal','3d','boolean'):
        d = json.loads((dataset/f'{domain}.json').read_text())
        rows=d.get('observation_sets')
        if rows is None:
            rows=[row for group in d['datasets_by_n_observations'].values() for row in group]
        for row in rows:
            jobs.append((domain, row.get('observation_set_id') or row['observation_id']))
    return jobs


def main():
    import argparse
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    logs = root/'logs/mixed-mcts-six-question-queue'
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    releases = {v:root/'releases'/name for v,name in (
        ('2.1','mcts-2.1-temp-pool6-32k'), ('2.2','mcts-2.2-stage-pool6-32k'))}
    for release in releases.values():
        for name,digest in json.loads((release/'release_manifest.json').read_text())['sha256'].items():
            assert sha256((release/name).read_bytes()).hexdigest() == digest, name
    pilot = root/'results/server-mcts-2.2-stage-medium-pilot-32k'
    eval21 = root/'results/server-mcts-2.1-medium-evaluation-32k/two-step'
    train21 = root/'results/server-mcts-2.1-medium-training-32k/two-step'
    eval_data = releases['2.1']/'datasets/medium_sampled'
    training_data = root/'datasets/medium_training_v1'
    pilot_manifest = json.loads((pilot/'manifest.json').read_text())
    pilot_caps = {(j['domain'],j['case_id']):j['search_cap'] for j in pilot_manifest['jobs']}
    all_pilot = [Job('2.2','pilot',j['domain'],j['case_id'],
        str(pilot/f"{j['domain']}-{j['case_id']}.json")) for j in pilot_manifest['jobs']]
    all_eval = [Job('2.1','evaluation',ds,cid,str(eval21/f'{ds}-{cid}.json'))
                for ds,cid in case_ids(eval_data)]
    def progress(job):
        p=Path(job.output)
        if not p.exists():return 0
        d=json.loads(p.read_text())
        return len(d.get('interrupted_case',{}).get('records',[]))
    pending = deque([j for j in all_pilot if not completed(Path(j.output))]+
        sorted([j for j in all_eval if not completed(Path(j.output))],key=progress,reverse=True))
    statuses = []
    state = dict(pid=os.getpid(), status='running', workers=6,
        policy='One question per slot, 2.2 pilot first then unfinished 2.1; refill immediately',
        pilot_total=len(all_pilot), evaluation_total=len(all_eval), training_scheduled=False,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    active = {}
    def save():
        state.update(active=[asdict(j) for j in active.values()],
            pending=[asdict(j) for j in pending], statuses=statuses,
            pilot_completed=sum(completed(Path(j.output)) for j in all_pilot),
            evaluation_completed=sum(completed(Path(j.output)) for j in all_eval))
        p=logs/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(logs/'state.json')
    env=dict(os.environ, OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='0')
    def run_job(job):
        output=Path(job.output);output.parent.mkdir(parents=True,exist_ok=True)
        # Adopt live 2.2 writers instead of interrupting or writing their checkpoints twice.
        while writers(output):time.sleep(3)
        if completed(output):return 'completed'
        release=releases[job.version]
        dataset=training_data if job.phase=='training' else release/'datasets/medium_sampled'
        command=[sys.executable,'-u','-m','experiments.test_easy','--model','Qwen3-8B',
            '--max-tokens','32768','--enable-thinking','--base-url','http://127.0.0.1:8003/v1',
            '--method','mcts-tree','--tree-policy','llm','--duplicate-reward','-1',
            '--exploration-list','off','--two-step-reflection','--unbounded-tree',
            '--mcts-version',job.version,'--dataset-root',str(dataset),'--dataset',job.domain,
            '--case-id',job.case_id,'--output',str(output),
            '--pause-file',str(logs/f'PAUSE_{job.version}')]
        if job.version=='2.2':
            command+=['--search-cap',str(pilot_caps[job.domain,job.case_id])]
        if output.exists():command+=['--resume',str(output)]
        with output.with_suffix('.log').open('a') as log:
            code=subprocess.run(command,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        return 'completed' if code==0 and completed(output) else f'failed_exit_{code}'
    def schedule_training():
        if state['training_scheduled'] or not all(completed(Path(j.output)) for j in all_eval):return
        if not training_data.exists():
            with (logs/'generate-training.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m','experiments.generate_medium_training',
                    '--project-root',str(releases['2.1']),'--output',str(training_data)],
                    cwd=releases['2.1'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        training_jobs=[Job('2.1','training',ds,cid,str(train21/f'{ds}-{cid}.json'))
                       for ds,cid in case_ids(training_data)]
        pending.extend(j for j in training_jobs if not completed(Path(j.output)))
        # Retain the old output layout expected by the trajectory exporter.
        parent=train21.parent;parent.mkdir(parents=True,exist_ok=True)
        (parent/'manifest.json').write_text(json.dumps(dict(suite='medium',
            jobs=[(j.domain,j.case_id) for j in training_jobs],thinking=True,workers_per_method=6,
            max_tokens=32768,dataset_root=str(training_data),mcts_version='2.1',
            tree_tools=True,methods=['two-step'],scheduler='mixed-six-question-queue'),indent=2))
        state.update(training_scheduled=True,training_total=len(training_jobs))
    schedule_training()
    save()
    with ThreadPoolExecutor(max_workers=6) as pool:
        while pending or active:
            schedule_training()
            while pending and len(active)<6:
                job=pending.popleft();active[pool.submit(run_job,job)]=job
            save()
            if not active:break
            finished,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in finished:
                job=active.pop(future)
                try:status=future.result()
                except Exception as error:status='exception_'+type(error).__name__
                statuses.append(dict(**asdict(job),status=status))
                print(job.version,job.phase,job.case_id,status,flush=True)
            # May generate training after the last evaluation futures have completed.
            schedule_training()
    if state['training_scheduled'] and not any(s['status']!='completed' for s in statuses):
        with (logs/'export-training.log').open('a') as log:
            subprocess.run([sys.executable,'server/export_training_trajectories.py',
                '--run-dir',str(train21.parent)],cwd=releases['2.1'],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        state['training_exported']=True
    state['status']='complete' if state.get('training_exported') else 'needs_inspection'
    save()


if __name__=='__main__':main()
