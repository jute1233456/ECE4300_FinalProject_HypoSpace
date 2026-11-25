"""Run independent formal cases concurrently, each with its own checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from time import sleep
from datetime import datetime, timezone

from .direct_evaluation import ROOT


def main(shared_pool=None, pending=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("causal", "3d", "boolean", "boolean_extended"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--global-pause", type=Path)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    args, extra = parser.parse_known_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    reserved = ("--resume", "--continue-from", "--output", "--case-id", "--pause-file", "--import-progress")
    if any(word.split("=")[0] in reserved for word in extra):
        parser.error("Checkpoint paths, case selection and pause files are managed by this runner")
    jobs = []
    for dataset in (args.dataset,) if args.dataset else ("causal", "3d", "boolean", "boolean_extended"):
        data = json.loads((ROOT / "datasets/balanced" / f"{dataset}.json").read_text())
        if dataset == "causal":
            rows = data["datasets"]
        elif dataset == "3d":
            rows = data["observation_sets"]
        else:
            rows = [row for group in data["datasets_by_n_observations"].values() for row in group]
        jobs.extend((dataset, row.get("observation_set_id") or row["observation_id"]) for row in rows)
    jobs = jobs[:args.limit]
    seeds = {}
    if args.source:
        for path in [args.source, *args.supplement]:
            run = json.loads(path.read_text(encoding="utf-8"))
            cases = list(run["cases"])
            if run.get("interrupted_case"):
                cases.append(run["interrupted_case"])
            for case in cases:
                key = (case["domain"], case["case_id"])
                if key not in seeds or len(case["records"]) > len(seeds[key][1]["records"]):
                    seeds[key] = (path, case)
        jobs = [job for job in jobs if (
            ("boolean" if job[0] == "boolean_extended" else job[0], job[1]) not in seeds
            or seeds[("boolean" if job[0] == "boolean_extended" else job[0], job[1])][1].get("stop_reason")
            in (None, "request_errors")
        )]
    print(f"{len(jobs)} cases, up to {args.workers} concurrent workers", flush=True)
    if args.plan:
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.run_dir / "manifest.json"
    settings = {"jobs": jobs, "arguments": extra, "model": "deepseek-v4.1-flash-economy"}
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json.dumps(settings)):
            parser.error("Run directory belongs to a different experiment")
    else:
        manifest.write_text(json.dumps(settings, indent=2))

    def run(job):
        while args.global_pause and args.global_pause.exists():
            sleep(1)
        dataset, case_id = job
        output = args.run_dir / f"{dataset}-{case_id}.json"
        if output.exists():
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved["completed"]:
                failed = next((case for case in saved["cases"]
                               if case.get("stop_reason") == "request_errors"), None)
                if failed is None:
                    return {"dataset": dataset, "case_id": case_id, "status": "already_completed"}
                saved.update(completed=False, cases=[], summary={},
                             interrupted_case={key: failed[key] for key in
                                               ("domain", "case_id", "prompt", "records")})
                temporary = output.with_suffix(".tmp")
                temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(output)
        command = [sys.executable, "-u", "-m", "experiments.test_hard_balanced",
                   "--dataset", dataset, "--case-id", case_id,
                   "--pause-file", str(args.global_pause or args.run_dir / "PAUSE"), *extra]
        command += ["--resume", str(output)] if output.exists() else ["--output", str(output)]
        key = ("boolean" if dataset == "boolean_extended" else dataset, case_id)
        if not output.exists() and key in seeds:
            command += ["--seed-case", str(seeds[key][0])]
        with output.with_suffix(".log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return {"dataset": dataset, "case_id": case_id,
                "status": "finished" if result.returncode == 0 else "process_error",
                "returncode": result.returncode}

    if shared_pool is not None:
        for job in jobs:
            pending.append(shared_pool.submit(run, job))
        return
    statuses = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(f"{len(statuses)}/{len(jobs)} {status}", flush=True)
            summary = {"updated_utc": datetime.now(timezone.utc).isoformat(), "processes": statuses}
            temporary = args.run_dir / "status.tmp"
            temporary.write_text(json.dumps(summary, indent=2))
            temporary.replace(args.run_dir / "status.json")


if __name__ == "__main__":
    main()
