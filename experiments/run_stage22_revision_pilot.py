"""Three fresh-tree regression cases for the Stage-2.2 prompt revision."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

from .run_sampled import ROOT, METHODS, run_case
from .rate_limit import recover_quota


def main():
    root = ROOT / "results/stage22-revision-pilot"
    output = root / "stage22"
    output.mkdir(parents=True, exist_ok=True)
    Thread(target=recover_quota, args=(root / "PAUSE",), daemon=True).start()
    METHODS["revision"] = [*METHODS["two-step"], "--mcts-version", "2.2", "--stage22-revision"]
    jobs = [("boolean", "n2_inputs_00_10_outputs_01"),
            ("3d", "g3_h3_b3_006"),
            ("boolean", "n4_inputs_00_01_10_11_outputs_0110")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(run_case, "revision", domain, case,
                   output / f"{domain}-{case}.json") for domain, case in jobs]
        for future in as_completed(pending):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
