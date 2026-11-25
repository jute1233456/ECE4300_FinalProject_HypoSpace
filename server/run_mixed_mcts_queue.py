"""Configurable question slots shared by Stage-MCTS 2.2 pilot and reflection MCTS 2.1."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
import signal
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


_completion_cache={}


def completed(output):
    if not output.exists():
        return False
    st=output.stat();stamp=(st.st_mtime_ns,st.st_size)
    previous=_completion_cache.get(str(output))
    if previous and previous[0]==stamp:return previous[1]
    d = json.loads(output.read_text())
    value=bool(d.get('completed') and len(d.get('cases', [])) == 1
               and d['cases'][0].get('stop_reason') != 'request_errors')
    _completion_cache[str(output)]=(stamp,value)
    return value


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


def baseline_ready(jobs):
    return bool(jobs) and all(completed(Path(j.output)) for j in jobs)


def reference_counts(dataset):
    counts={}
    fields={'causal':'ground_truth_graphs','3d':'ground_truth_structures','boolean':'ground_truth_expressions'}
    for domain,field in fields.items():
        d=json.loads((dataset/f'{domain}.json').read_text())
        rows=d.get('observation_sets')
        if rows is None:rows=[row for group in d['datasets_by_n_observations'].values() for row in group]
        for row in rows:
            counts[domain,row.get('observation_set_id') or row['observation_id']]=len(row[field])
    return counts


def main():
    import argparse
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--api-base', default='http://127.0.0.1:8003/v1')
    parser.add_argument('--with-training',action='store_true',help='Explicit opt-in; training is cancelled by default')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:parser.error('workers must be between 1 and 32')
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
    followup=root/'results/server-mcts-2.2-stage-medium-followup-32k'
    initial_ids={(j.domain,j.case_id) for j in all_pilot}
    all_followup=[Job('2.2','evaluation22',j.domain,j.case_id,
        str(followup/f'{j.domain}-{j.case_id}.json')) for j in all_eval
        if (j.domain,j.case_id) not in initial_ids]
    assert len(all_followup)==23
    counts=reference_counts(eval_data)
    resume_pause_paths={logs/'PAUSE_2.1'}
    drain_jobs=[]
    (logs/'PAUSE_2.1').touch()
    for j in all_eval:
        live=writers(j.output)
        if not live:continue
        for pid in live:
            raw=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'--pause-file' in raw:
                pause=Path(raw[raw.index(b'--pause-file')+1].decode())
                pause.touch();resume_pause_paths.add(pause)
        drain_jobs.append(Job(j.version,'drain21',j.domain,j.case_id,j.output))
    def progress(job):
        p=Path(job.output)
        if not p.exists():return 0
        d=json.loads(p.read_text())
        return len(d.get('interrupted_case',{}).get('records',[]))
    candidates=[j for j in all_pilot if not completed(Path(j.output))]
    pending=deque(drain_jobs+sorted(candidates,key=lambda j:(not bool(writers(j.output)),-progress(j))))
    statuses = []
    state = dict(pid=os.getpid(), status='running', workers=args.workers, api_base=args.api_base,
        policy='Finish ALL 27 Stage-MCTS2.2 evaluation cases first, then resume MCTS2.1; refill immediately',
        pilot_total=len(all_pilot), evaluation_total=len(all_eval), training_scheduled=False,
        evaluation22_total=len(all_pilot)+len(all_followup), followup_scheduled=False,
        evaluation21_scheduled=False,
        training_enabled=args.with_training,
        priority_order=['2.2 evaluation 27','2.1 unfinished evaluation']+(['2.1 training'] if args.with_training else []),
        resume21_gate='All 27 MCTS2.2 evaluation cases completed normally',
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    active = {}
    def save():
        state.update(active=[asdict(j) for j in active.values()],
            pending=[asdict(j) for j in pending], statuses=statuses,
            pilot_completed=sum(completed(Path(j.output)) for j in all_pilot),
            evaluation22_completed=sum(completed(Path(j.output)) for j in all_pilot+all_followup),
            evaluation_completed=sum(completed(Path(j.output)) for j in all_eval))
        p=logs/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(logs/'state.json')
    env=dict(os.environ, OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='0')
    def run_job(job):
        output=Path(job.output);output.parent.mkdir(parents=True,exist_ok=True)
        if job.phase=='drain21':
            while writers(output):
                log=output.with_suffix('.log')
                lines=log.read_text(errors='replace').splitlines() if log.exists() else []
                if lines and lines[-1].startswith('Paused. Remove '):
                    for pid in writers(output):
                        try:os.kill(pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                time.sleep(1)
            return 'drained'
        # Adopt live 2.2 writers instead of interrupting or writing their checkpoints twice.
        while writers(output):time.sleep(3)
        if completed(output):return 'completed'
        release=releases[job.version]
        dataset=training_data if job.phase=='training' else release/'datasets/medium_sampled'
        command=[sys.executable,'-u','-m','experiments.test_easy','--model','Qwen3-8B',
            '--max-tokens','32768','--enable-thinking','--base-url',args.api_base,
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
    def schedule_followup():
        if state['followup_scheduled']:return
        selected=[]
        for j in all_followup:
            base=eval21/f'{j.domain}-{j.case_id}.json'
            done=completed(base)
            c=json.loads(base.read_text())['cases'][0] if done else None
            cap=c['model_call_cap'] if c else 2*(counts[j.domain,j.case_id]+5)
            pilot_caps[j.domain,j.case_id]=cap
            selected.append(dict(domain=j.domain,case_id=j.case_id,baseline=str(base),
                baseline_completed_at_scheduling=done,
                baseline_sha256=sha256(base.read_bytes()).hexdigest() if done else None,
                baseline_recovery=c['metrics']['recovery'] if c else None,search_cap=cap,
                reference_count=counts[j.domain,j.case_id]))
        followup.mkdir(parents=True,exist_ok=True)
        (followup/'manifest.json').write_text(json.dumps(dict(version='2.2',name='Stage-MCTS',
            jobs=selected,workers=args.workers,api_base=args.api_base,max_tokens=32768,thinking=True,
            initial_four_directory=str(pilot),priority='Run all 27 MCTS2.2 cases before resuming MCTS2.1',
            comparison='Same case/search round cap; fresh trees; compare actual API/token costs separately'),indent=2))
        pending.extend(sorted([j for j in all_followup if not completed(Path(j.output))],
            key=lambda j:(not bool(writers(j.output)),-progress(j))))
        state['followup_scheduled']=True

    def schedule_evaluation21():
        if state['evaluation21_scheduled'] or not baseline_ready(all_pilot+all_followup):return
        for pause in resume_pause_paths:pause.unlink(missing_ok=True)
        pending.extend(sorted([j for j in all_eval if not completed(Path(j.output))],key=progress,reverse=True))
        state['evaluation21_scheduled']=True

    def schedule_training():
        if not args.with_training or state['training_scheduled'] or not state['evaluation21_scheduled'] or not baseline_ready(all_eval):return
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
            jobs=[(j.domain,j.case_id) for j in training_jobs],thinking=True,workers_per_method=args.workers,
            max_tokens=32768,dataset_root=str(training_data),mcts_version='2.1',
            tree_tools=True,methods=['two-step'],scheduler='mixed-six-question-queue'),indent=2))
        state.update(training_scheduled=True,training_total=len(training_jobs))
    schedule_followup()
    schedule_evaluation21()
    schedule_training()
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            schedule_followup()
            schedule_evaluation21()
            schedule_training()
            while pending and len(active)<args.workers:
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
            schedule_followup()
            schedule_evaluation21()
            schedule_training()
    if state['training_scheduled'] and not any(s['status'] not in ('completed','drained') for s in statuses):
        with (logs/'export-training.log').open('a') as log:
            subprocess.run([sys.executable,'server/export_training_trajectories.py',
                '--run-dir',str(train21.parent)],cwd=releases['2.1'],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        state['training_exported']=True
    evaluation_done=baseline_ready(all_pilot+all_followup) and baseline_ready(all_eval)
    state['status']='complete' if evaluation_done and (not args.with_training or state.get('training_exported')) else 'needs_inspection'
    save()


if __name__=='__main__':main()
