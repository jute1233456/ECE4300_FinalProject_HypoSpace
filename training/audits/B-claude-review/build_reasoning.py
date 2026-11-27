"""Build train.reasoning_only.jsonl from kept non-hit rounds (selection/expansion only).

These rounds produced no verified standard solution, so they never enter the main dataset.
Their reasoning is kept only when the audit judged it sound on its own; every row carries
candidate_verified=false and round_outcome=invalid.
"""
import json
import re
from hashlib import sha256

from training.stage_sft import (DECISIONS_HEADER, INSTRUCTIONS, describe, menu_and_options,
                                split_context, stage_text, tree_payload, tree_text)
from llm.qwen_stage_protocol import isolate_stage

CASES = {"n1_inputs_01_outputs_1", "n1_inputs_10_outputs_1", "n6_hard_021", "n6_hard_029"}
CANON = "training/sources/ds22-sampled19-canonical.jsonl"
DECISIONS = "training/audits/B-claude-review/decisions-reasoning.jsonl"
OUT = "training/audits/B-claude-review/train.reasoning_only.jsonl"


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


kept = {}
for line in open(DECISIONS, encoding="utf-8"):
    row = json.loads(line)
    if row["decision"] == "keep":
        kept[(row["case_id"], row["sample_id"], row["stage"])] = row

rows = []
for line in open(CANON, encoding="utf-8"):
    case = json.loads(line)
    if case["case_id"] not in CASES:
        continue
    for index, record in enumerate(case["records"]):
        if not record.get("api_requests") or record.get("reference_hit"):
            continue
        selection_text = stage_text(record, "selection")
        expansion_text = stage_text(record, "expansion")
        if not selection_text:
            continue
        try:
            context, _ = split_context(record.get("prompt") or "")
            payload = tree_payload(record)
            menu, options, nodes = menu_and_options(payload)
            match = re.match(r"Branch:[ \t]*(\d+)", selection_text.strip(), re.I)
            if not match or int(match.group(1)) >= len(options):
                continue
            selected = describe(options[int(match.group(1))], nodes)
        except Exception:
            continue
        base = context + "\n\nSelected branch:\n" + selected
        prompts = {
            "selection": context + "\n\nComplete tree read from disk:\n" + tree_text(payload)
                         + "\n\nSelectable branch menu:\n" + menu + DECISIONS_HEADER,
            "expansion": base + DECISIONS_HEADER + "selection: " + selection_text,
        }
        targets = {"selection": selection_text, "expansion": expansion_text}
        for stage in ("selection", "expansion"):
            sid = "nh-" + digest(f"{case['case_id']}|{index}|{stage}|{targets[stage]}")[:16]
            if (case["case_id"], sid, stage) not in kept:
                continue
            prompt = prompts[stage]
            user, scope = isolate_stage(stage, prompt, INSTRUCTIONS[stage], examples=True)
            rows.append({
                "sample_id": sid,
                "case_id": case["case_id"],
                "family_id": case["case_id"],
                "stage": stage,
                "round_index": index,
                "messages": [{"role": "system", "content": scope},
                             {"role": "user", "content": user},
                             {"role": "assistant", "content": targets[stage]}],
                "source": {"canonical_path": CANON, "canonical_sha256": digest(open(CANON, encoding="utf-8").read()),
                           "case_id": case["case_id"], "record_index": index},
                "quality_flags": {"input_reconstructed": True, "plan_unverified": True,
                                  "reference_verified": False, "candidate_verified": False,
                                  "round_outcome": "invalid", "protocol": "isolated_examples_v1",
                                  "builder_version": "reasoning-only-1"},
            })

with open(OUT, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"kept_reasoning_samples": len(rows), "output": OUT, "sha256": digest(open(OUT, encoding="utf-8").read())}))
