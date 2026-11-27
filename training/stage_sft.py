"""Rebuild per-stage Stage-MCTS 2.2 training samples from canonical run records.

A run record stores the teacher's stage OUTPUTS but not the exact per-stage wire inputs
(the DeepSeek client never persisted request.prompt/system). This module rebuilds those
inputs from what the record does keep:

  record.prompt        -> original task + independent review (split exactly like
                          SplitTreeToolClient.generate does)
  record.tool_trace    -> the complete tree JSON that was read from disk this round
  node["branch"]       -> the selectable-branch menu at snapshot time
  record.plan_path     -> the expansion step that was actually used
  api_requests[*]      -> selection / expansion / candidate stage outputs

Rebuilt inputs are then wrapped with the SAME isolation template the deployed Qwen run
uses (llm.qwen_stage_protocol.isolate_stage with examples), and every sample is flagged
input_reconstructed=true. Teacher reasoning is NOT certified correct: plan_unverified=true.

Rules (see docs/TRAINING_DATA_PIPELINE.md):
  - only rounds with reference_hit=true and no truncation/protocol error are eligible;
  - candidates are deduplicated by structural hypothesis_key, first occurrence wins;
  - a rebuilt input is accepted only if the teacher output still holds inside it
    (Branch index inside the menu, Plan equals plan_path, candidate equals record.answer);
  - samples split by question family, never by record.

Usage:
    python -m training.stage_sft --canonical training/sources/ds22-sampled19-canonical.jsonl \
        --out training/data/ds22-extended-n1 --cases extended_n1_inputs_00_outputs_1
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from llm.qwen_stage_protocol import isolate_stage

ROOT = Path(__file__).resolve().parents[1]
BUILDER_VERSION = "stage-sft-1"
DECISIONS_HEADER = "\n\nDecisions from earlier stages in this round:\n"
REVIEW_MARKER = "\n\nIndependent review for this attempt: "
TASK_SPLIT = "\n\nChoose the direction most likely"

SELECTION_INSTRUCTION = (
    "Choose one listed branch using the tree and history. Output the first line "
    "Branch: <number>, followed by Reason:, Untried direction:, and Avoid: lines. "
    "Explain what legal choice remains and what repeated construction to avoid. "
    "Do not solve the problem or declare exhaustion."
)
EXPANSION_INSTRUCTION = (
    "Propose exactly one concrete next constraint for this branch. First line "
    "Plan: <step>, followed by Changed condition: and Expected difference:. "
    "Compare existing children and answers; rewording an existing step is not new. "
    "If no distinct step is available output Plan: NONE and explain why. "
    "Do not output EXHAUSTED: inability to find a step is not proof of exhaustion. "
    "Check all explicit problem constraints, including expression depth. "
    "Do not submit a complete candidate."
)
CANDIDATE_INSTRUCTION = (
    "Only produce one complete candidate in the original benchmark answer format, "
    "consistent with the selected branch and next step. Do not select branches or "
    "output Branch:, Plan:, Reflection: or EXHAUSTED. Once a complete valid candidate "
    "is available, submit it even if repeated. Do not keep reasoning only to avoid repetition."
    " Check the original structural limits and selected path before answering. "
    "Use the previous decision rationale and avoid the identified repeated construction. "
    "Submit any complete candidate you obtain, including a repeated one."
)
INSTRUCTIONS = {
    "selection": SELECTION_INSTRUCTION,
    "expansion": EXPANSION_INSTRUCTION,
    "candidate": CANDIDATE_INSTRUCTION,
}


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def record_signature(record: dict) -> str:
    return digest(json.dumps(
        [record.get("exploration_round"), record.get("hypothesis_key"), record.get("answer")],
        sort_keys=True, ensure_ascii=False,
    ))


# ---------------------------------------------------------------- reconstruction

def split_context(prompt: str) -> tuple[str, bool]:
    """Mirror SplitTreeToolClient.generate: task before the combined action instruction."""
    task = prompt.split(TASK_SPLIT, 1)[0]
    review = prompt.rsplit(REVIEW_MARKER, 1)[1] if REVIEW_MARKER in prompt else ""
    return task + ("\n\nIndependent review: " + review if review else ""), bool(review)


def tree_payload(record: dict) -> dict:
    reads = [t for t in (record.get("tool_trace") or []) if t.get("name") == "read_tree"]
    if not reads:
        raise ValueError("missing read_tree trace")
    result = reads[-1].get("result")
    if not isinstance(result, dict) or "nodes" not in result:
        raise ValueError("read_tree trace has no node list")
    return result


def tree_text(payload: dict) -> str:
    """Reproduce TreeFileTools.execute's returned string byte for byte."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def path_of(node: dict, nodes: dict) -> list[dict]:
    path, current = [], node
    while current is not None:
        path.append(current)
        parent = current.get("parent_id")
        current = nodes.get(parent) if parent else None
    return list(reversed(path))


def describe(node: dict, nodes: dict) -> str:
    """Mirror ReasoningTreeAgent.describe using the stored snapshot."""
    path = path_of(node, nodes)
    plans = " / ".join(part.get("plan", "") for part in path[1:]) or "Start from the observations"
    children = " | ".join(nodes[c].get("plan", "") for c in node.get("children") or []) or "(none)"
    return f"{plans}. Mode: {node.get('mode')}. Existing child steps: {children}."


def menu_and_options(payload: dict) -> tuple[str, list[dict], dict]:
    nodes = {n["id"]: n for n in payload["nodes"]}
    options = sorted((n for n in nodes.values() if n.get("branch") is not None),
                     key=lambda n: n["branch"])
    menu = "\n".join(f"{n['branch']}: {describe(n, nodes)}" for n in options)
    return menu, options, nodes


def stage_text(record: dict, name: str) -> str:
    for request in record.get("api_requests") or []:
        if request.get("stage") == name:
            return ((request.get("stage_response") or {}).get("text") or "").strip()
    return ""


def plan_from_expansion(text: str) -> str:
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    match = re.match(r"Plan:[ \t]*(.*)", first, re.I)
    if not match:
        return ""
    value = match.group(1).strip()
    return "" if value.casefold() == "none" else value


def verify_selection(text: str, count: int) -> int:
    match = re.match(r"Branch:[ \t]*(\d+)", text.strip(), re.I)
    if not match or int(match.group(1)) >= count:
        raise ValueError(f"selection Branch outside reconstructed menu (count={count})")
    return int(match.group(1))


# ---------------------------------------------------------------- sample building

def make_sample(*, case_id: str, family_id: str, stage: str, prompt: str, assistant: str,
                record_index: int, source: dict, extra_flags: dict) -> dict:
    user, scope = isolate_stage(stage, prompt, INSTRUCTIONS[stage], examples=True)
    sample_id = digest(f"{case_id}|{record_index}|{stage}|{digest(prompt)}|{digest(assistant)}")[:20]
    return {
        "sample_id": sample_id,
        "case_id": case_id,
        "family_id": family_id,
        "stage": stage,
        "round_index": record_index,
        "messages": [
            {"role": "system", "content": scope},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "quality_flags": {
            "input_reconstructed": True,
            "plan_unverified": True,
            "reference_verified": True,
            "has_independent_review": extra_flags["has_independent_review"],
            "protocol": "isolated_examples_v1",
            "builder_version": BUILDER_VERSION,
        },
    }


def build_case(case_row: dict, *, source_path: str, source_sha: str,
               include_unverified_reasoning: bool = True) -> tuple[list[dict], list[dict]]:
    case_id = case_row["case_id"]
    family_id = case_id  # one record per question family here; split is by case
    samples, issues = [], []
    seen_keys: dict[str, int] = {}
    for index, record in enumerate(case_row.get("records") or []):
        if not record.get("api_requests"):
            issues.append({"record": index, "reason": "empty record"})
            continue
        if record.get("protocol_error"):
            issues.append({"record": index, "reason": f"protocol_error: {record['protocol_error']}"})
            continue
        if record.get("finish_reason") == "length" or record.get("final_answer_missing"):
            issues.append({"record": index, "reason": "truncated"})
            continue
        if not record.get("reference_hit"):
            issues.append({"record": index, "reason": "not_reference_hit"})
            continue
        key = json.dumps(record.get("hypothesis_key"), sort_keys=True, ensure_ascii=False)
        if key in seen_keys:
            issues.append({"record": index, "reason": f"duplicate_of_record_{seen_keys[key]}"})
            continue
        selection = stage_text(record, "selection")
        expansion = stage_text(record, "expansion")
        candidate = stage_text(record, "candidate")
        if not selection or not candidate:
            issues.append({"record": index, "reason": "missing stage output"})
            continue
        try:
            context, has_review = split_context(record.get("prompt") or "")
            payload = tree_payload(record)
            menu, options, nodes = menu_and_options(payload)
            branch = verify_selection(selection, len(options))
        except ValueError as error:
            issues.append({"record": index, "reason": f"reconstruction failed: {error}"})
            continue
        selected = describe(options[branch], nodes)
        plan = plan_from_expansion(expansion) if expansion else ""
        plan_path = record.get("plan_path") or []
        if plan and plan_path and plan.strip() != str(plan_path[-1]).strip():
            issues.append({"record": index, "reason": "plan does not match plan_path"})
            continue
        answer = (record.get("answer") or "").strip()
        if answer and candidate.strip() != answer:
            issues.append({"record": index, "reason": "candidate does not match record.answer"})
            continue
        seen_keys[key] = index
        source = {
            "canonical_path": source_path,
            "canonical_sha256": source_sha,
            "case_id": case_id,
            "record_index": index,
            "exploration_round": record.get("exploration_round"),
        }
        base = context + "\n\nSelected branch:\n" + selected
        selection_prompt = (context + "\n\nComplete tree read from disk:\n" + tree_text(payload)
                            + "\n\nSelectable branch menu:\n" + menu + DECISIONS_HEADER)
        samples.append(make_sample(case_id=case_id, family_id=family_id, stage="selection",
            prompt=selection_prompt, assistant=selection, record_index=index, source=source,
            extra_flags={"has_independent_review": has_review}))
        if expansion and include_unverified_reasoning:
            expansion_prompt = (base + DECISIONS_HEADER + "selection: " + selection)
            samples.append(make_sample(case_id=case_id, family_id=family_id, stage="expansion",
                prompt=expansion_prompt, assistant=expansion, record_index=index, source=source,
                extra_flags={"has_independent_review": has_review}))
        candidate_prompt = (base + "\nNext step: " + (plan or "(no additional constraint)")
                            + DECISIONS_HEADER + "selection: " + selection
                            + ("\n" + "expansion: " + expansion if expansion else ""))
        samples.append(make_sample(case_id=case_id, family_id=family_id, stage="candidate",
            prompt=candidate_prompt, assistant=candidate, record_index=index, source=source,
            extra_flags={"has_independent_review": has_review}))
    return samples, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default="", help="comma separated case ids; empty = all completed")
    parser.add_argument("--valid-cases", default="", help="case ids held out for validation")
    parser.add_argument("--ledger", type=Path, default=ROOT / "training" / "ingest" / "ledger.jsonl")
    parser.add_argument("--append-ledger", action="store_true")
    parser.add_argument("--new-only", action="store_true",
                        help="skip cases whose samples are already fully in the ledger")
    args = parser.parse_args()

    from training.ingest import load_ledger

    known = load_ledger(args.ledger)
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    valid_cases = {c.strip() for c in args.valid_cases.split(",") if c.strip()}
    source_sha = sha256(args.canonical.read_bytes()).hexdigest()

    train, valid, issues, skipped_cases = [], [], [], []
    for line in args.canonical.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if wanted and row["case_id"] not in wanted:
            continue
        if row.get("status") != "completed":
            continue
        samples, found = build_case(row, source_path=str(args.canonical), source_sha=source_sha)
        issues.extend({"case_id": row["case_id"], **i} for i in found)
        if args.new_only and samples and all(s["sample_id"] in known for s in samples):
            skipped_cases.append(row["case_id"])
            continue
        bucket = valid if row["case_id"] in valid_cases else train
        bucket.extend(samples)

    if not (train or valid):
        if skipped_cases:
            print(json.dumps({"message": "all selected cases already ingested; nothing rewritten",
                              "skipped_cases_new_only": skipped_cases}, ensure_ascii=False))
            return
        parser.error("no samples built; refusing to overwrite an existing dataset directory")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        (args.out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by_stage: dict[str, int] = {}
    for row in train + valid:
        by_stage[row["stage"]] = by_stage.get(row["stage"], 0) + 1
    new_samples = [r for r in train + valid if r["sample_id"] not in known]
    manifest = {
        "builder_version": BUILDER_VERSION,
        "protocol": "isolated_examples_v1",
        "canonical_path": str(args.canonical),
        "canonical_sha256": source_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(train),
        "valid_samples": len(valid),
        "samples_by_stage": by_stage,
        "families": {"train": sorted({r["family_id"] for r in train}),
                     "valid": sorted({r["family_id"] for r in valid})},
        "already_in_ledger": len(train) + len(valid) - len(new_samples),
        "new_samples": len(new_samples),
        "skipped_cases_new_only": skipped_cases,
        "issues": issues,
        "flags": {"input_reconstructed": True, "plan_unverified": True,
                  "reference_verified": True, "label_mask_checked": False},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    if args.append_ledger:
        from training.ingest import append_ledger
        append_ledger(args.ledger, [dict(row, canonical_sha256=source_sha) for row in new_samples])
    print(json.dumps({k: manifest[k] for k in (
        "train_samples", "valid_samples", "samples_by_stage", "new_samples",
        "already_in_ledger", "skipped_cases_new_only")}, ensure_ascii=False, indent=2))
    if issues:
        print(f"issues: {len(issues)} (see manifest.json)")


if __name__ == "__main__":
    main()
