"""Twenty shared slots for nineteen interleaved Stage-2.2 cases and one legacy case."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from threading import Thread
from pathlib import Path
import argparse

from .run_sampled import ROOT, METHODS, run_case, sampled_jobs
from .rate_limit import recover_quota


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-reflection", action="store_true")
    args = parser.parse_args()
    root = ROOT / ("results/stage22-partition-sampled" if args.partition_reflection
                   else "results/stage22-sampled")
    output_dir = root / "stage22"
    output_dir.mkdir(parents=True, exist_ok=True)
    Thread(target=recover_quota, args=(root / "PAUSE",), daemon=True).start()
    METHODS["stage22"] = [*METHODS["two-step"], "--mcts-version", "2.2"]
    if args.partition_reflection:
        METHODS["stage22"].append("--partition-reflection")
    groups = [[job for job in sampled_jobs()
               if ("boolean" if job[0] == "boolean_extended" else job[0]) == domain]
              for domain in ("causal", "3d", "boolean")]
    jobs = [job for row in zip_longest(*groups) for job in row if job]
    with ThreadPoolExecutor(max_workers=20) as pool:
        pending = [] if args.partition_reflection else [pool.submit(
            run_case, "two-step", "3d", "g3_h3_b4_001",
            ROOT / "results/sampled-local/two-step/3d-g3_h3_b4_001.json", root / "PAUSE")]
        for dataset, case_id in jobs:
            print(f"Enqueue Stage-2.2 {dataset}/{case_id}", flush=True)
            pending.append(pool.submit(run_case, "stage22", dataset, case_id,
                           output_dir / f"{dataset}-{case_id}.json"))
        for index, future in enumerate(as_completed(pending), 1):
            print(f"{index}/20 {future.result()}", flush=True)


if __name__ == "__main__":
    main()
