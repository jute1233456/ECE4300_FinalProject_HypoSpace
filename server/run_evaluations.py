"""Run smoke tests first; then four independent full benchmark processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = Path(os.environ.get('ECE_RESULT_DIR', str(ROOT / 'results/server')))
LOG = Path(os.environ.get('ECE_LOG_DIR', str(ROOT/'logs')))
THINKING = os.environ.get('ECE_THINKING') == '1'
WORKERS = int(os.environ.get('ECE_WORKERS_PER_METHOD', '4'))
SUITE = os.environ.get('ECE_DATASET_SUITE', 'hard-balanced')
if SUITE not in ('hard-balanced', 'easy', 'medium'):
    raise ValueError(f'Unknown dataset suite: {SUITE}')
if SUITE == 'easy':
    DATASET_ROOT = ROOT / 'datasets'
    DOMAINS = ('causal', '3d', 'boolean')
elif SUITE == 'medium':
    DATASET_ROOT = ROOT / 'datasets/medium_sampled'
    DOMAINS = ('causal', '3d', 'boolean')
else:
    DATASET_ROOT = ROOT / 'datasets/balanced'
    DOMAINS = ('causal', '3d', 'boolean', 'boolean_extended')


DATASET_ROOT = Path(os.environ.get('ECE_DATASET_ROOT', str(DATASET_ROOT)))

def dataset_jobs():
    from experiments.direct_evaluation import load_domain
    jobs = []
    for dataset in DOMAINS:
        benchmark, _, _ = load_domain('boolean' if dataset == 'boolean_extended' else dataset,
                                      DATASET_ROOT / f'{dataset}.json')
        jobs.extend((dataset, row.get('observation_set_id') or row['observation_id'])
                    for row in benchmark.all_observation_sets)
    return jobs


sys.path.insert(0, str(ROOT))
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
if SUITE == 'hard-balanced':
    BASE = [sys.executable, '-u', '-m', 'experiments.test_hard_balanced',
            '--model', 'Qwen3-8B', '--max-tokens', os.environ.get('ECE_MAX_TOKENS', '4096')]
else:
    BASE = [sys.executable, '-u', '-m', 'experiments.test_easy',
            '--model', 'Qwen3-8B', '--max-tokens', os.environ.get('ECE_MAX_TOKENS', '4096')]
    if SUITE == 'medium':
        BASE += ['--dataset-root', str(DATASET_ROOT)]
if THINKING: BASE += ['--enable-thinking']
METHODS = {
    'direct': ['--method', 'direct'],
    'memory': ['--method', 'memory'],
    'mcts': ['--method', 'mcts-tree', '--tree-policy', 'llm', '--duplicate-reward', '-1',
             '--exploration-list', 'off', '--child-limit', '7'],
    'two-step': ['--method', 'mcts-tree', '--tree-policy', 'llm', '--duplicate-reward', '-1',
                 '--exploration-list', 'off', '--two-step-reflection', '--unbounded-tree'],
}
if os.environ.get('ECE_METHODS'):
    selected = os.environ['ECE_METHODS'].split(',')
    if not selected or any(name not in METHODS for name in selected):
        raise ValueError('Unknown ECE_METHODS selection')
    METHODS = {name: METHODS[name] for name in selected}


def api(name):
    return os.environ.get('ECE_API_BASE', 'http://127.0.0.1:8002/v1')


def smoke_api(port):
    payload = dict(model='Qwen3-8B', messages=[dict(role='user', content='Reply with exactly: READY')],
                   max_tokens=32, temperature=0, chat_template_kwargs={'enable_thinking': False})
    request = Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                      data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    assert 'READY' in result['choices'][0]['message']['content'], result
    (LOG / f'api_smoke_{port}.json').write_text(json.dumps(result, indent=2))


def command(name, output, smoke=False, domain=None):
    args = BASE + ['--base-url', api(name)] + METHODS[name] + ['--output', str(output),
            '--pause-file', str(OUT / f'PAUSE_{name}')]
    if name in ('mcts', 'two-step') and os.environ.get('ECE_FULL_TREE') == '1':
        args += ['--full-tree']
    if name in ('mcts', 'two-step') and os.environ.get('ECE_TREE_TOOLS') == '1':
        args += ['--tree-tools']
    if name in ('mcts', 'two-step') and os.environ.get('ECE_MCTS_VERSION'):
        args += ['--mcts-version', os.environ['ECE_MCTS_VERSION']]
    if smoke:
        # Three domain cases with a short budget; keep formal method settings unchanged.
        cases = {'causal': 'n6_hard_029', '3d': 'g3_h3_b3_006', 'boolean': 'n1_inputs_00_outputs_0'}
        if SUITE in ('easy', 'medium'):
            cases = {key: next(case_id for ds, case_id in dataset_jobs() if ds == key)
                     for key in ('causal', '3d', 'boolean')}
        args += ['--dataset', domain, '--case-id', cases[domain],
                 '--queries', '1' if THINKING else '2', '--search-cap', '1' if THINKING else '4']
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get('completed'):
            return None
        args += ['--resume', str(output)]
    return args


def run():
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'], check=True)
    ports = [int(x) for x in os.environ.get('ECE_API_PORTS', '8000,8001').split(',') if x]
    for port in ports:
        smoke_api(port)
    def check_smoke(pair):
        name,domain=pair
        output = OUT / f'smoke_{name}_{domain}.json'
        args = command(name, output, True, domain)
        if args:
            subprocess.run(args, check=True)
        data = json.loads(output.read_text())
        if not data.get('completed') or any('error' in r or r.get('finish_reason') == 'length'
            or r.get('final_answer_missing')
            or r.get('reflection_response', {}).get('finish_reason') == 'length'
            or r.get('reflection_response', {}).get('final_answer_missing')
            for c in data['cases'] for r in c['records']):
            raise RuntimeError(f'Smoke test failed: {output}')
    smoke_jobs=[(name,domain) for name in METHODS
                for domain in (('boolean',) if THINKING else ('causal','3d','boolean'))]
    if THINKING:
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(check_smoke,smoke_jobs))
    else:
        for pair in smoke_jobs:check_smoke(pair)
    (LOG / 'evaluations.started').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    jobs = dataset_jobs()
    (OUT / 'manifest.json').write_text(json.dumps(dict(suite=SUITE, jobs=jobs,
        thinking=THINKING, workers_per_method=WORKERS,
        max_tokens=int(os.environ.get('ECE_MAX_TOKENS', '4096')),
        full_tree=os.environ.get('ECE_FULL_TREE') == '1',
        dataset_root=str(DATASET_ROOT), mcts_version=os.environ.get('ECE_MCTS_VERSION'), tree_tools=os.environ.get('ECE_TREE_TOOLS') == '1', methods=list(METHODS)), indent=2))
    def evaluate_method(name):
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        def evaluate_case(job):
            dataset, case_id = job
            output = directory / f'{dataset}-{case_id}.json'
            for attempt in range(3):
                args = command(name, output)
                if args:
                    args += ['--dataset', dataset, '--case-id', case_id]
                    with output.with_suffix('.log').open('a') as log:
                        process = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode:
                        return dict(method=name, dataset=dataset, case_id=case_id, status='process_error')
                data = json.loads(output.read_text())
                if data.get('completed') and len(data['cases']) == 1 and data['cases'][0].get('stop_reason') != 'request_errors':
                    return dict(method=name, dataset=dataset, case_id=case_id, status='completed')
                # Retry failed cases from scratch, retaining earlier attempts for diagnosis.
                output.rename(output.with_suffix(f'.failed-{int(time.time())}-{attempt}.json'))
            return dict(method=name, dataset=dataset, case_id=case_id, status='request_errors')
        statuses = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(evaluate_case, job) for job in jobs]
            for future in as_completed(futures):
                status = future.result()
                statuses.append(status)
                temporary = directory / 'status.tmp'
                temporary.write_text(json.dumps(statuses, indent=2))
                temporary.replace(directory / 'status.json')
                print(f'{name} {len(statuses)}/{len(jobs)} {status}', flush=True)
        return name, statuses
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(evaluate_method, METHODS))
    (LOG / 'evaluations.status.json').write_text(json.dumps(results, indent=2))
    subprocess.run([sys.executable, 'server/report_results.py','--run-dir',str(OUT)], check=True)
    if os.environ.get('ECE_TREE_TOOLS') == '1' and os.environ.get('ECE_MCTS_VERSION') != '2.1':
        subprocess.run([sys.executable, 'server/compare_tree_tools.py', '--new', str(OUT),
                        '--output', str(OUT/'tree_tool_comparison.json')], check=True)
    if any(s['status'] != 'completed' for statuses in results.values() for s in statuses):
        raise RuntimeError('Some cases require inspection; see evaluations.status.json')
    print('ALL_FOUR_EVALUATIONS_FINISHED', flush=True)
    (LOG / 'evaluations.complete').touch()
    next_suite = os.environ.get('ECE_NEXT_SUITE')
    if next_suite:
        env = dict(os.environ)
        env['ECE_DATASET_SUITE'] = next_suite
        env['ECE_RESULT_DIR'] = os.environ.get('ECE_NEXT_RESULT_DIR', str(OUT.parent / f'server-{next_suite}'))
        env['ECE_LOG_DIR'] = os.environ.get('ECE_NEXT_LOG_DIR', str(LOG.parent / next_suite))
        env.pop('ECE_NEXT_SUITE', None)
        env.pop('ECE_NEXT_RESULT_DIR', None)
        env.pop('ECE_NEXT_LOG_DIR', None)
        print(f'Starting next suite: {next_suite}', flush=True)
        subprocess.run([sys.executable, '-u', 'server/run_evaluations.py'], env=env, check=True)


if __name__ == '__main__':
    run()
