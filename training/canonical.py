"""Merge repeated runs of one task into a single canonical dataset file.

Rule (see AGENTS.md): several runs of the same task, prompt version and question set are
snapshots of ONE dataset, not separate datasets. The canonical file keeps, per case, the
snapshot with the most records; every other snapshot must be a record-level prefix of it.
Raw run directories stay on disk as archive only and are never re-read as a second source.

Usage:
    python -m training.canonical --out training/sources
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# One task = one canonical file. Runs are snapshots of the same sampled-19 Stage-MCTS 2.2
# revision task (same question list, same revision prompt, same completion policy).
TASKS = {
    "ds22-sampled19": {
        "task": "Stage-MCTS 2.2 revision, sampled 19 questions, DeepSeek teacher",
        "mcts_version": "2.2",
        "mcts_variant": "two-step",
        "case_list": "datasets/sampled.json",
        "runs": [
            {"label": "merged", "path": "results/ds-stage22-merged/revision", "note": "v1 handoff source"},
            {"label": "official", "path": "results/ds-stage22-official/revision", "note": "intermediate snapshot"},
            {"label": "ark-resumed", "path": "results/ds-stage22-ark-resumed/revision", "note": "latest snapshot"},
        ],
        "excluded_runs": [
            {"path": "results/stage22-sampled/stage22", "reason": "old prompt version: tree_config.stage22_revision absent"},
            {"path": "results/stage22-partition-sampled/stage22", "reason": "no completed or partial records"},
            {"path": "results/stage22-revision-pilot/stage22", "reason": "2 rounds on 3 pilot questions, superseded by full runs"},
            {"path": "results/stage22-revision19-20261126T042923Z", "reason": "cancelled wrong provider, excluded from comparisons"},
        ],
    },
}


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def record_signature(record: dict) -> str:
    return sha256(json.dumps(
        [record.get("exploration_round"), record.get("hypothesis_key"), record.get("answer")],
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()


def load_run(path: Path) -> dict:
    """case_id -> {case, source_path, sha256, n_records, done, mtime}."""
    out = {}
    for file in sorted(path.glob("*.json")):
        if file.name.endswith(".tree.json"):
            continue
        data = json.loads(file.read_text(encoding="utf-8"))
        digest = file_sha(file)
        entries = list(data.get("cases") or [])
        if data.get("interrupted_case"):
            entries.append(data["interrupted_case"])
        for case in entries:
            prev = out.get(case["case_id"])
            if prev and len(prev["case"].get("records") or []) >= len(case.get("records") or []):
                continue
            out[case["case_id"]] = {
                "case": case,
                "source_path": str(file),
                "sha256": digest,
                "n_records": len(case.get("records") or []),
                "done": bool(case.get("metrics")),
                "mtime": file.stat().st_mtime,
            }
    return out


def consolidate(task_name: str, spec: dict, out_dir: Path) -> dict:
    runs = {run["label"]: load_run(ROOT / run["path"]) for run in spec["runs"]}
    case_ids = sorted(set().union(*[set(run) for run in runs.values()]))
    lines, summary = [], []
    for case_id in case_ids:
        snapshots = [
            {"label": label, **{k: v for k, v in run[case_id].items() if k != "case"}}
            for label, run in runs.items() if case_id in run
        ]
        chosen = max(snapshots, key=lambda s: (s["n_records"], s["mtime"]))
        base = runs[chosen["label"]][case_id]
        sig = [record_signature(r) for r in base["case"].get("records") or []]
        for snap in snapshots:
            other = runs[snap["label"]][case_id]["case"].get("records") or []
            other_sig = [record_signature(r) for r in other]
            snap["is_prefix_of_chosen"] = other_sig == sig[: len(other_sig)]
        case = base["case"]
        lines.append(json.dumps({
            "case_id": case_id,
            "dataset": case.get("domain"),
            "domain": case.get("domain"),
            "status": "completed" if case.get("metrics") else "incomplete",
            "reference_count": case.get("reference_count"),
            "stop_reason": case.get("stop_reason"),
            "metrics": case.get("metrics"),
            "records": case.get("records") or [],
            "provenance": {
                "chosen": {k: chosen[k] for k in ("label", "source_path", "sha256", "n_records", "done")},
                "snapshots": [{k: s[k] for k in ("label", "source_path", "sha256", "n_records", "done", "is_prefix_of_chosen")} for s in snapshots],
            },
        }, ensure_ascii=False))
        summary.append({
            "case_id": case_id,
            "status": "completed" if case.get("metrics") else "incomplete",
            "n_records": chosen["n_records"],
            "chosen": chosen["label"],
            "chosen_sha256": chosen["sha256"],
            "all_snapshots_are_prefixes": all(s["is_prefix_of_chosen"] for s in snapshots),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{task_name}-canonical.jsonl"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "task": spec["task"],
        "task_name": task_name,
        "mcts_version": spec["mcts_version"],
        "mcts_variant": spec["mcts_variant"],
        "case_list": spec["case_list"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "one task = one canonical file; per case keep the snapshot with most records; "
                "all other snapshots must be record-level prefixes; raw run directories are archive only",
        "canonical_path": str(target),
        "canonical_sha256": file_sha(target),
        "runs": spec["runs"],
        "excluded_runs": spec["excluded_runs"],
        "cases": summary,
        "completed_cases": sum(1 for c in summary if c["status"] == "completed"),
        "total_cases": len(summary),
        "total_records": sum(c["n_records"] for c in summary),
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ds22-sampled19", choices=sorted(TASKS))
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "sources")
    args = parser.parse_args()
    manifest = consolidate(args.task, TASKS[args.task], args.out)
    print(json.dumps({k: manifest[k] for k in (
        "task_name", "canonical_path", "completed_cases", "total_cases", "total_records")},
        ensure_ascii=False, indent=2))
    bad = [c for c in manifest["cases"] if not c["all_snapshots_are_prefixes"]]
    if bad:
        print("PREFIX CONFLICTS:", json.dumps(bad, ensure_ascii=False))


if __name__ == "__main__":
    main()
