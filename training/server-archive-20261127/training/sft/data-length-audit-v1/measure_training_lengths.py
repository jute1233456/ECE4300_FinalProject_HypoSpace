"""Measure real Qwen3 chat-template token lengths of stage SFT samples (CPU only, no GPU).

Encoding follows server/train_qlora.py: system+user prefix from add_generation_prompt,
full sequence from system+user+assistant, loss only over the assistant span. Reports the
prefix length separately because that is the fixed cost paid before any answer token.

    .venv-inference/bin/python server/measure_training_lengths.py \
        --samples training/data/ds22-all-completed/train.jsonl \
        --tokenizer models/Qwen3-8B --out training/data/ds22-all-completed/lengths.json
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def summarize(values):
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--buckets", default="2048,4096,8192,10240,12288,16384,32768")
    args = parser.parse_args()
    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    thinking = args.thinking == "on"
    rows = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]

    per_stage = defaultdict(list)
    over_bucket = {b: defaultdict(int) for b in buckets}
    details, mismatches, missing = [], 0, []
    for row in rows:
        messages = row["messages"]
        if [m["role"] for m in messages] != ["system", "user", "assistant"]:
            missing.append(row["sample_id"])
            continue
        prefix = tokenizer.apply_chat_template([messages[0], messages[1]], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=thinking)
        full = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=thinking)
        if full[:len(prefix)] != prefix:
            mismatches += 1
        assistant_len = len(full) - len(prefix)
        stage = row["stage"]
        per_stage[stage].append(len(full))
        per_stage[stage + ":prefix"].append(len(prefix))
        per_stage[stage + ":assistant"].append(assistant_len)
        for bucket in buckets:
            if len(full) > bucket:
                over_bucket[bucket][stage] += 1
        details.append({"sample_id": row["sample_id"], "case_id": row["case_id"], "stage": stage,
                        "prefix_tokens": len(prefix), "full_tokens": len(full),
                        "assistant_tokens": assistant_len})

    report = {
        "samples": str(args.samples),
        "tokenizer": args.tokenizer,
        "thinking": args.thinking,
        "template_prefix_mismatches": mismatches,
        "skipped_missing_roles": missing,
        "per_stage": {key: summarize(values) for key, values in sorted(per_stage.items())},
        "over_bucket": {str(b): dict(v) for b, v in over_bucket.items()},
        "all": summarize([d["full_tokens"] for d in details]),
        "longest": sorted(details, key=lambda d: -d["full_tokens"])[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**report, "details": details}, ensure_ascii=False, indent=2),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
