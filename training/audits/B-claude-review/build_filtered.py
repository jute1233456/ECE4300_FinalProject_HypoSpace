"""Build the screened B dataset: verify decision coverage, then split by decision.

Only filters: kept rows are emitted byte-for-byte in the original order. Never edits the
source dataset or the ledger.
"""
import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "training/data/ds22-new4-20261126/train.jsonl"
DECISIONS = ROOT / "training/audits/B-claude-review/decisions.jsonl"
OUT = ROOT / "training/audits/B-claude-review"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    lines = [line for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    by_id = {row["sample_id"]: (index, line) for index, (row, line) in enumerate(zip(rows, lines))}

    decisions = [json.loads(line) for line in args.decisions.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [d["sample_id"] for d in decisions]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate sample_id in decisions")
    missing = sorted(set(by_id) - set(ids))
    extra = sorted(set(ids) - set(by_id))
    if missing or extra:
        raise SystemExit("coverage mismatch: missing=%d extra=%d (first missing=%s)"
                         % (len(missing), len(extra), missing[:3]))
    for d in decisions:
        index, _ = by_id[d["sample_id"]]
        if d["case_id"] != rows[index]["case_id"] or d["stage"] != rows[index]["stage"]:
            raise SystemExit("decision metadata mismatch: " + d["sample_id"])
        if d["decision"] not in {"keep", "reject", "uncertain"}:
            raise SystemExit("bad decision: " + d["sample_id"])

    buckets = {"keep": [], "reject": [], "uncertain": []}
    for d in sorted(decisions, key=lambda x: by_id[x["sample_id"]][0]):
        buckets[d["decision"]].append(by_id[d["sample_id"]][1])

    args.out.mkdir(parents=True, exist_ok=True)
    names = {"keep": "train.filtered.jsonl", "reject": "rejected.jsonl", "uncertain": "uncertain.jsonl"}
    for key, name in names.items():
        (args.out / name).write_text("".join(line + "\n" for line in buckets[key]), encoding="utf-8")

    by_stage = Counter((d["decision"], d["stage"]) for d in decisions)
    by_case = Counter((d["decision"], d["case_id"]) for d in decisions)
    manifest = {
        "source": str(args.source),
        "source_sha256": digest(args.source),
        "decisions": str(args.decisions),
        "decisions_sha256": digest(args.decisions),
        "counts": {key: len(value) for key, value in buckets.items()},
        "by_stage": {"%s/%s" % k: v for k, v in sorted(by_stage.items())},
        "by_case": {"%s/%s" % k: v for k, v in sorted(by_case.items())},
        "outputs": {names[key]: {"count": len(buckets[key]),
                                 "sha256": digest(args.out / names[key])} for key in buckets},
        "note": "Semantic review only; kept rows are unmodified source lines. "
                "uncertain is excluded from the recommended training set.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
