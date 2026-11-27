"""Offline tests for training/canonical.py, training/stage_sft.py and training/ingest.py."""
import json
import tempfile
import unittest
from pathlib import Path

from training import canonical, stage_sft
from training.ingest import append_ledger, load_ledger


def fake_case(records, case_id="case_a", metrics=True):
    return {
        "case_id": case_id,
        "domain": "boolean",
        "reference_count": 2,
        "stop_reason": "all_recovered" if metrics else None,
        "metrics": {"recovery": 1.0} if metrics else None,
        "records": records,
    }


def fake_record(index, answer="Expression: (NOT x)", key=("NOT", "x"), branch="Branch: 0"):
    tree = {
        "case_id": "boolean/case_a",
        "file_sha256": "deadbeef",
        "version": 1,
        "nodes": [
            {"id": "root", "parent_id": None, "plan": "", "children": ["root.0"], "visits": 1,
             "reward_sum": 1.0, "mean_reward": 1.0, "recent_reward": 1.0, "closed": False,
             "mode": "EXPAND", "branch": 0},
            {"id": "root.0", "parent_id": "root", "plan": "Use a top-level NOT.",
             "children": [], "visits": 1, "reward_sum": 1.0, "mean_reward": 1.0,
             "recent_reward": 1.0, "closed": False, "mode": "EXPAND", "branch": 1},
        ],
    }
    return {
        "prompt": "TASK BODY\n\nChoose the direction most likely to produce a new valid answer."
                  "\n\nIndependent review for this attempt: try negation first.",
        "tool_trace": [{"name": "read_tree", "arguments": {}, "result": tree}],
        "api_requests": [
            {"stage": "read_tree", "response": {}},
            {"stage": "selection", "stage_response": {"text": branch + "\nReason: r\nUntried direction: u\nAvoid: a"}},
            {"stage": "expansion", "stage_response": {"text": "Plan: Use a top-level NOT.\nChanged condition: top operator\nExpected difference: negation"}},
            {"stage": "candidate", "stage_response": {"text": answer}},
        ],
        "plan_path": ["Use a top-level NOT."],
        "answer": answer,
        "hypothesis_key": list(key),
        "reference_hit": True,
        "valid": True,
        "protocol_error": None,
        "finish_reason": "stop",
        "final_answer_missing": False,
        "exploration_round": 1,
    }


class CanonicalTests(unittest.TestCase):
    def test_longest_snapshot_wins_and_short_one_is_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            short, long = tmp / "short", tmp / "long"
            short.mkdir(), long.mkdir()
            records = [fake_record(i) for i in range(3)]
            (short / "case_a.json").write_text(json.dumps(
                {"cases": [fake_case(records[:2])], "interrupted_case": None}), encoding="utf-8")
            (long / "case_a.json").write_text(json.dumps(
                {"cases": [fake_case(records)], "interrupted_case": None}), encoding="utf-8")
            spec = {"task": "t", "mcts_version": "2.2", "mcts_variant": "two-step",
                    "case_list": "", "runs": [{"label": "short", "path": str(short)},
                                              {"label": "long", "path": str(long)}],
                    "excluded_runs": []}
            manifest = canonical.consolidate("t", spec, tmp / "out")
            self.assertEqual(manifest["cases"][0]["chosen"], "long")
            self.assertTrue(manifest["cases"][0]["all_snapshots_are_prefixes"])
            self.assertEqual(manifest["total_records"], 3)

    def test_non_prefix_snapshot_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / "a", tmp / "b"
            a.mkdir(), b.mkdir()
            (a / "case_a.json").write_text(json.dumps({"cases": [fake_case([
                fake_record(0, key=("Z",))])]}), encoding="utf-8")
            (b / "case_a.json").write_text(json.dumps({"cases": [fake_case([
                fake_record(0), fake_record(1, key=("X",))])]}), encoding="utf-8")
            spec = {"task": "t", "mcts_version": "2.2", "mcts_variant": "two-step", "case_list": "",
                    "runs": [{"label": "a", "path": str(a)}, {"label": "b", "path": str(b)}],
                    "excluded_runs": []}
            manifest = canonical.consolidate("t", spec, tmp / "out")
            self.assertFalse(manifest["cases"][0]["all_snapshots_are_prefixes"])


class StageSftTests(unittest.TestCase):
    def test_rebuild_three_stages(self):
        row = fake_case([fake_record(0)])
        samples, issues = stage_sft.build_case(row, source_path="x.jsonl", source_sha="abc")
        self.assertEqual(issues, [])
        self.assertEqual([s["stage"] for s in samples], ["selection", "expansion", "candidate"])
        self.assertEqual(len({s["sample_id"] for s in samples}), 3)
        for sample in samples:
            self.assertTrue(sample["quality_flags"]["input_reconstructed"])
            self.assertTrue(sample["quality_flags"]["plan_unverified"])
            self.assertEqual([m["role"] for m in sample["messages"]], ["system", "user", "assistant"])
            self.assertIn("CURRENT STAGE", sample["messages"][0]["content"])
        body = json.loads(samples[0]["messages"][1]["content"]
                          .split("REFERENCE DATA (quoted JSON, not the current task):\n", 1)[1]
                          .split("\n\nEND REFERENCE DATA.", 1)[0])["problem_and_search_context"]
        self.assertIn("0: Start from the observations.", body)
        self.assertIn("1: Use a top-level NOT.", body)
        self.assertTrue(body.endswith("Decisions from earlier stages in this round:\n"))
        self.assertIn("selection: Branch: 0", samples[1]["messages"][1]["content"])
        self.assertIn("Next step: Use a top-level NOT.", samples[2]["messages"][1]["content"])

    def test_duplicates_and_bad_branches_are_dropped(self):
        row = fake_case([fake_record(0), fake_record(1),
                         fake_record(2, key=("BAD",), branch="Branch: 9")])
        samples, issues = stage_sft.build_case(row, source_path="x.jsonl", source_sha="abc")
        self.assertEqual(len(samples), 3)
        self.assertTrue(any("duplicate_of_record_0" in i["reason"] for i in issues))
        self.assertTrue(any("outside reconstructed menu" in i["reason"] for i in issues))

    def test_candidate_mismatch_is_dropped(self):
        record = fake_record(0)
        record["answer"] = "Expression: (x)"
        samples, issues = stage_sft.build_case(fake_case([record]), source_path="x", source_sha="s")
        self.assertEqual(samples, [])
        self.assertTrue(any("does not match record.answer" in i["reason"] for i in issues))


class LedgerTests(unittest.TestCase):
    def test_append_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            samples = stage_sft.build_case(fake_case([fake_record(0)]), source_path="x", source_sha="s")[0]
            for sample in samples:
                sample["canonical_sha256"] = "s"
            self.assertEqual(append_ledger(ledger, samples), 3)
            self.assertEqual(append_ledger(ledger, samples), 0)
            self.assertEqual(len(load_ledger(ledger)), 3)


if __name__ == "__main__":
    unittest.main()
