"""Summarize per-case progress and coverage across source checkpoints and queued dirs.

The formal 136-case runs are split across an old per-method checkpoint (source) and a
queue of per-case files (results/queued-*). This merges both, takes the newest result
per case, fills missing reference counts from the fixed datasets, and reports per-case
coverage plus equal-weight totals by domain and method.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .direct_evaluation import ROOT

METHODS = {
    "memory": ("results/memory-branch-examples_hard_balanced_20261117T103505331321Z.json",
               "results/queued-memory"),
    "mcts-tree": ("results/mcts-tree-list-off-branch-examples_hard_balanced_20261117T103507674253Z.json",
                  "results/queued-mcts-tree"),
    "two-step": ("results/mcts-tree-list-off-branch-examples-two-step-unbounded_hard_balanced_20261117T103508829111Z.json",
                 "results/queued-two-step"),
}

DATASET_FILES = {
    "causal": "causal.json",
    "3d": "3d.json",
    "boolean": "boolean.json",
    "boolean_extended": "boolean_extended.json",
}


def dataset_reference_counts():
    counts = {}
    for domain, fname in DATASET_FILES.items():
        data = json.loads((ROOT / "datasets" / "balanced" / fname).read_text(encoding="utf-8"))
        if domain == "causal":
            rows = data["datasets"]
        elif domain == "3d":
            rows = data["observation_sets"]
        else:
            rows = [row for group in data["datasets_by_n_observations"].values() for row in group]
        for row in rows:
            case_id = row.get("observation_set_id") or row["observation_id"]
            if domain == "causal":
                n = len(row["ground_truth_graphs"])
            elif domain == "3d":
                n = len(row["ground_truth_structures"])
            else:
                n = len(row["ground_truth_expressions"])
            counts[(domain, case_id)] = n
    return counts


def found_count(records):
    seen = set()
    for record in records:
        if record.get("reference_hit") and record.get("hypothesis_key") is not None:
            seen.add(json.dumps(record["hypothesis_key"], sort_keys=True))
    return len(seen)


def load_case(case):
    records = case.get("records", [])
    stop = case.get("stop_reason")
    return {
        "domain": case["domain"],
        "case_id": case["case_id"],
        "status": stop if stop is not None else "in_progress",
        "found": found_count(records),
        "reference_count": case.get("reference_count"),
    }


def merge_method(src_path, qdir):
    by_key = {}
    src = json.loads((ROOT / src_path).read_text(encoding="utf-8"))
    for case in src.get("cases", []):
        by_key[(case["domain"], case["case_id"])] = case
    if src.get("interrupted_case"):
        case = src["interrupted_case"]
        by_key[(case["domain"], case["case_id"])] = case
    for path in (ROOT / qdir).glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        for case in run.get("cases", []):
            by_key[(case["domain"], case["case_id"])] = case
        if run.get("interrupted_case"):
            case = run["interrupted_case"]
            by_key[(case["domain"], case["case_id"])] = case
    return by_key


def summarize_method(name, src_path, qdir, refcounts):
    rows = []
    for key, case in merge_method(src_path, qdir).items():
        row = load_case(case)
        row["reference_count"] = row["reference_count"] or refcounts.get(key)
        row["coverage"] = (row["found"] / row["reference_count"]
                           if row["reference_count"] else None)
        rows.append(row)
    return rows


def totals(rows):
    by_domain = defaultdict(list)
    status = Counter()
    for row in rows:
        by_domain[row["domain"]].append(row)
        status[row["status"]] += 1
    coverage = [row["coverage"] for row in rows if row["coverage"] is not None]
    result = {
        "cases": len(rows),
        "status": dict(status),
        "recovery": mean(coverage) if coverage else None,
    }
    for domain, domain_rows in sorted(by_domain.items()):
        domain_coverage = [row["coverage"] for row in domain_rows if row["coverage"] is not None]
        result[domain] = {
            "cases": len(domain_rows),
            "recovery": mean(domain_coverage) if domain_coverage else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", action="store_true", help="Print every case")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    refcounts = dataset_reference_counts()
    report = {}
    for name, (src_path, qdir) in METHODS.items():
        rows = summarize_method(name, src_path, qdir, refcounts)
        report[name] = totals(rows)
        if args.detail:
            print(f"\n=== {name} ===")
            for row in sorted(rows, key=lambda r: (r["domain"], r["case_id"])):
                print(f"  {row['domain']}/{row['case_id']}: {row['found']}/{row['reference_count']} "
                      f"({row['coverage']:.1%}) {row['status']}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
