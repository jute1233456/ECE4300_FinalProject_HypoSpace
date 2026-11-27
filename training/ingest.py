"""Ingest ledger: remember exactly which records have already become training samples.

The ledger is append-only and keyed by (case_id, record_index, stage, prompt hash, output hash)
via sample_id. A later run of the builder therefore only emits genuinely new samples, no matter
how many times the same task was re-run or how many snapshot directories exist on disk.

Usage:
    python -m training.ingest --check training/data/ds22-extended-n1/train.jsonl
    python -m training.ingest --append training/data/ds22-extended-n1/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "training" / "ingest" / "ledger.jsonl"


def load_ledger(path: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    if not Path(path).exists():
        return entries
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        entries[row["sample_id"]] = row
    return entries


def entry_for(sample: dict, *, canonical_sha256: str) -> dict:
    assistant = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
    user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    return {
        "sample_id": sample["sample_id"],
        "case_id": sample["case_id"],
        "family_id": sample.get("family_id"),
        "stage": sample["stage"],
        "round_index": sample.get("round_index"),
        "canonical_sha256": canonical_sha256,
        "prompt_sha256": sha256(user.encode("utf-8")).hexdigest(),
        "assistant_sha256": sha256(assistant.encode("utf-8")).hexdigest(),
        "builder_version": (sample.get("quality_flags") or {}).get("builder_version"),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def append_ledger(path: Path, samples: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    known = load_ledger(path)
    rows, added = [], 0
    for sample in samples:
        if sample["sample_id"] in known:
            continue
        rows.append(entry_for(sample, canonical_sha256=sample.get("canonical_sha256", "")))
        known[sample["sample_id"]] = rows[-1]
        added += 1
    if rows:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    known = load_ledger(args.ledger)
    samples = []
    for file in args.files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(json.loads(line))
    fresh = [s for s in samples if s["sample_id"] not in known]
    print(json.dumps({"samples": len(samples), "already_ingested": len(samples) - len(fresh),
                      "new": len(fresh)}, ensure_ascii=False))
    if args.append and fresh:
        print("appended:", append_ledger(args.ledger, fresh))


if __name__ == "__main__":
    main()
