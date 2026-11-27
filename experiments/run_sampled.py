"""Run the four methods against the sampled small benchmark on local Ark/DeepSeek.

Mirrors ``server/run_evaluations.py`` but drives ``experiments.test_hard_balanced``
against ``datasets/sampled.json`` (the ~120-reference-per-domain subset) using the
Ark Coding Plan answer model. MCTS methods use ``--tree-tools`` (native function
calling), which is the local DeepSeek equivalent of the server Qwen tree tools.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Thread
from .rate_limit import recover_quota

ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "direct": ["--method", "direct"],
    "memory": ["--method", "memory"],
    "mcts": ["--method", "mcts-tree", "--tree-policy", "llm", "--duplicate-reward", "-1",
             "--exploration-list", "off", "--child-limit", "7", "--tree-tools"],
    "two-step": ["--method", "mcts-tree", "--tree-policy", "llm", "--duplicate-reward", "-1",
                 "--exploration-list", "off", "--two-step-reflection", "--unbounded-tree", "--tree-tools"],
}
METHODS["revision"] = [*METHODS["two-step"], "--mcts-version", "2.2", "--stage22-revision"]


def sampled_jobs():
    data = json.loads((ROOT / "datasets/sampled.json").read_text(encoding="utf-8"))
    jobs = []
    for domain in ("causal", "3d", "boolean"):
        for item in data[domain]:
            dataset = "boolean_extended" if item["case_id"].startswith("extended_") else domain
            jobs.append((dataset, item["case_id"]))
    return jobs


def run_case(method, dataset, case_id, output, pause_file=None):
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("completed"):
            return dict(method=method, dataset=dataset, case_id=case_id, status="already_completed")
    args = [sys.executable, "-u", "-m", "experiments.test_hard_balanced",
            "--dataset", dataset, "--case-id", case_id, "--provider-token-limit",
            "--pause-file", str(pause_file or output.parent.parent / "PAUSE"), *METHODS[method]]
    args += ["--resume", str(output)] if output.exists() else ["--output", str(output)]
    with output.with_suffix(".log").open("a", encoding="utf-8") as log:
        process = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return dict(method=method, dataset=dataset, case_id=case_id,
                status="finished" if process.returncode == 0 else "process_error")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--methods", default="direct,memory,mcts,two-step")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/sampled-local")
    parser.add_argument("--provider", choices=("ark", "deepseek"), default="ark")
    args = parser.parse_args()
    selected = args.methods.split(",")
    if not selected or any(name not in METHODS for name in selected):
        parser.error("Unknown method selection")
    if args.provider == "deepseek":
        for method in selected:
            METHODS[method] = [*METHODS[method], "--provider", "deepseek"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    Thread(target=recover_quota, args=(args.run_dir / "PAUSE",), daemon=True).start()
    jobs = sampled_jobs()
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for method in selected:
            method_dir = args.run_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            for dataset, case_id in jobs:
                output = method_dir / f"{dataset}-{case_id}.json"
                futures.append(pool.submit(run_case, method, dataset, case_id, output))
        finished = 0
        for future in as_completed(futures):
            finished += 1
            status = future.result()
            print(f"{finished}/{len(futures)} {status}", flush=True)


if __name__ == "__main__":
    main()
