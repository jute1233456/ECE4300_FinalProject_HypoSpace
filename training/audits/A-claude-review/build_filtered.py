#!/usr/bin/env python3
"""Filter the A dataset with the Claude semantic-review decisions.

Contract:
  * read the original A JSONL exactly as bytes and never modify it;
  * require decisions.jsonl to cover every sample_id exactly once, with no
    duplicate, missing or extra ids, and with matching case_id/stage;
  * emit keep / reject / uncertain rows in the original order, reusing the
    original JSON line bytes (only a trailing newline is normalised if the
    source line lacked one);
  * record source and output SHA256 plus per-stage and per-case counts in
    manifest.json.  The input ledger is not read or written.

Usage:
    python training/audits/A-claude-review/build_filtered.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_SOURCE = ROOT / "training" / "data" / "ds22-all-completed" / "train.jsonl"
DEFAULT_DECISIONS = HERE / "decisions.jsonl"
DEFAULT_OUT = HERE

REQUIRED = ("sample_id", "case_id", "stage", "decision", "reason_code",
            "evidence", "explanation", "confidence")
DECISIONS = ("keep", "reject", "uncertain")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source(path: Path):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise SystemExit(f"empty source: {path}")
    if not raw.endswith(b"\n"):
        raise SystemExit(f"source does not end with a newline: {path}")
    rows = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            raise SystemExit(f"blank source line {number}: {path}")
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise SystemExit(f"bad JSON on source line {number}: {error}")
        rows.append((row, line))
    return raw, rows


def load_decisions(path: Path):
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise SystemExit(f"blank decision line {number}: {path}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"bad JSON on decision line {number}: {error}")
        missing = [key for key in REQUIRED if key not in row]
        if missing:
            raise SystemExit(f"decision line {number} missing {missing}")
        if row["decision"] not in DECISIONS:
            raise SystemExit(f"decision line {number} has invalid decision {row['decision']!r}")
        if not str(row["sample_id"]).strip():
            raise SystemExit(f"decision line {number} has empty sample_id")
        rows.append(row)
    return raw, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source_raw, source_rows = load_source(args.source)
    decisions_raw, decisions = load_decisions(args.decisions)

    source_by_id = {}
    for row, _ in source_rows:
        sid = row.get("sample_id")
        if not sid:
            raise SystemExit("source row without sample_id")
        if sid in source_by_id:
            raise SystemExit(f"duplicate sample_id in source: {sid}")
        source_by_id[sid] = row

    decision_by_id = {}
    for row in decisions:
        sid = row["sample_id"]
        if sid in decision_by_id:
            raise SystemExit(f"duplicate sample_id in decisions: {sid}")
        decision_by_id[sid] = row

    missing = sorted(set(source_by_id) - set(decision_by_id))
    extra = sorted(set(decision_by_id) - set(source_by_id))
    if missing or extra:
        raise SystemExit(f"decision/source mismatch: missing={missing[:5]} ({len(missing)}), "
                         f"extra={extra[:5]} ({len(extra)})")
    if len(decisions) != len(source_rows):
        raise SystemExit(f"decision count {len(decisions)} != source count {len(source_rows)}")

    for sid, row in source_by_id.items():
        decision = decision_by_id[sid]
        if decision["case_id"] != row.get("case_id"):
            raise SystemExit(f"case_id mismatch for {sid}: {decision['case_id']} != {row.get('case_id')}")
        if decision["stage"] != row.get("stage"):
            raise SystemExit(f"stage mismatch for {sid}: {decision['stage']} != {row.get('stage')}")

    buckets = {name: [] for name in DECISIONS}
    for row, line in source_rows:
        buckets[decision_by_id[row["sample_id"]]["decision"]].append(line)

    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, filename in (("keep", "train.filtered.jsonl"),
                           ("reject", "rejected.jsonl"),
                           ("uncertain", "uncertain.jsonl")):
        data = b"".join(buckets[name])
        path = args.out / filename
        path.write_bytes(data)
        outputs[name] = {"path": str(path), "sha256": sha256_bytes(data),
                         "samples": len(buckets[name])}

    stage_counts = OrderedDict()
    case_counts = OrderedDict()
    for row, _ in source_rows:
        stage = row.get("stage")
        case = row.get("case_id")
        decision = decision_by_id[row["sample_id"]]["decision"]
        stage_counts.setdefault(stage, Counter())[decision] += 1
        case_counts.setdefault(case, Counter())[decision] += 1
    stage_counts = OrderedDict((k, dict(v)) for k, v in sorted(stage_counts.items()))
    case_counts = OrderedDict((k, dict(v)) for k, v in sorted(case_counts.items()))

    decision_counts = Counter(d["decision"] for d in decisions)
    reason_counts = Counter(d["reason_code"] for d in decisions)
    confidence_counts = Counter(d["confidence"] for d in decisions)
    manifest = {
        "audit": "A-claude-review",
        "generated_by": "build_filtered.py",
        "source": {"path": str(args.source), "sha256": sha256_bytes(source_raw),
                   "samples": len(source_rows)},
        "decisions": {"path": str(args.decisions), "sha256": sha256_bytes(decisions_raw),
                      "samples": len(decisions)},
        "outputs": outputs,
        "total_samples": len(source_rows),
        "decision_counts": {name: decision_counts.get(name, 0) for name in DECISIONS},
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "stage_decision_counts": stage_counts,
        "case_decision_counts": case_counts,
        "invariants": {
            "ids_exactly_covered": True,
            "no_duplicate_ids": True,
            "no_missing_ids": True,
            "no_extra_ids": True,
            "original_order_preserved": True,
            "original_json_lines_reused": True,
            "source_modified": False,
            "ledger_modified": False,
        },
        "note": ("Original A train.jsonl is read only. Filtering reuses each original "
                 "JSON line; only the original trailing newlines are retained."),
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "source": manifest["source"],
        "decision_counts": manifest["decision_counts"],
        "stage_decision_counts": manifest["stage_decision_counts"],
        "outputs": manifest["outputs"],
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
