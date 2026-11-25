"""Run an ordered agent queue with a shared concurrency limit and pause switch."""
import argparse
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch
from time import sleep
from threading import Thread
from .rate_limit import recover_quota
from .parallel_balanced import main as enqueue

from .direct_evaluation import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--wait-for-paused", action="store_true")
    parser.add_argument("--control", choices=("pause", "resume"))
    args = parser.parse_args()
    pause = ROOT / "results" / "PAUSE_QUEUED_AGENTS"
    if args.control:
        if args.control == "pause":
            pause.touch()
        else:
            pause.unlink(missing_ok=True)
        print(f"Queue {args.control}: {pause}", flush=True)
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.wait_for_paused and not args.plan:
        logs = ["reboot-memory.log", "reboot-mcts.log", "reboot-two-step.log",
                "first-memory.log", "first-mcts-tree.log", "first-two-step.log"]
        while any("Paused." not in (ROOT / "results" / name).read_text(
                encoding="utf-8", errors="replace").splitlines()[-1] for name in logs):
            sleep(2)
    tree = ["--method", "mcts-tree", "--tree-policy", "llm",
            "--duplicate-reward", "-1", "--exploration-list", "off"]
    groups = [
        ("memory", "memory-branch-examples_hard_balanced_20261117T103505331321Z.json",
         ["--method", "memory"]),
        ("mcts-tree", "mcts-tree-list-off-branch-examples_hard_balanced_20261117T103507674253Z.json",
         [*tree, "--child-limit", "7"]),
        ("two-step", "mcts-tree-list-off-branch-examples-two-step-unbounded_hard_balanced_20261117T103508829111Z.json",
         [*tree, "--two-step-reflection", "--unbounded-tree"]),
    ]
    pending = []
    if not args.plan:
        Thread(target=recover_quota, args=(pause,), daemon=True).start()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    for name, source, flags in groups:
        command = [sys.executable, "-u", "-m", "experiments.parallel_balanced",
                   "--workers", str(args.workers), "--run-dir", f"results/queued-{name}",
                   "--source", f"results/{source}", "--global-pause", str(pause),
                   "--provider-token-limit", *flags]
        supplement = ROOT / "results" / f"first-case-resume-{name}.json"
        if supplement.exists():
            command += ["--supplement", str(supplement)]
        if args.plan:
            command += ["--plan"]
        print(f"Starting group {name}", flush=True)
        if args.plan:
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            with patch.object(sys, "argv", ["parallel_balanced", *command[4:]]):
                enqueue(shared_pool=pool, pending=pending)
    for index, future in enumerate(as_completed(pending), 1):
        print(f"{index}/{len(pending)} {future.result()}", flush=True)
    pool.shutdown()


if __name__ == "__main__":
    main()
