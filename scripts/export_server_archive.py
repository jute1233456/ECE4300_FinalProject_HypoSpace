#!/usr/bin/env python3
"""Build a compact, git-friendly archive of training data/params/eval metadata.
Excludes model weights, tokenizers, raw responses and large tree/result records."""
import json, os, glob, shutil, hashlib, time

STAGE = "/tmp/ECE4300-archive-20261127"
T27 = "/opt/data/private/ECE4300-train-20261127"
T26 = "/opt/data/private/ECE4300-train-20261126"
DS = "/opt/data/private/ECE4300-dpo-datasets-20261127"
Q = "/opt/data/private/ECE4300-controller-20261127/quick3"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def put(src, rel):
    dst = os.path.join(STAGE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def put_if(src, rel):
    if os.path.exists(src):
        put(src, rel)

if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)

# ---------- DPO training runs ----------
for src_name, dst_name in (("dpo-reflection-v1-ep1", "dpo-v1"),
                           ("dpo-reflection-v2-ep1", "dpo-v2"),
                           ("dpo-reflection-v3-ep1", "dpo-v3")):
    src = os.path.join(T27, src_name)
    for name in ("config.json", "README.md", "status.json", "reserved-job.json"):
        put_if(os.path.join(src, name), "training/%s/%s" % (dst_name, name))
    put_if(os.path.join(src, "pipeline.log"), "training/%s/pipeline.log.txt" % dst_name)
    put_if(os.path.join(src, "data/dpo.jsonl"), "training/%s/data/dpo.jsonl" % dst_name)
    put_if(os.path.join(src, "data/manifest.json"), "training/%s/data/manifest.json" % dst_name)
    for sub in ("adapter", "smoke"):
        for name in ("status.json", "history.json", "reference_logps.json", "adapter_config.json"):
            put_if(os.path.join(src, sub, name), "training/%s/%s/%s" % (dst_name, sub, name))
    put_if(os.path.join(src, "smoke-failed-attempt1/status.json"),
           "training/%s/smoke-failed-attempt1/status.json" % dst_name)
    put_if(os.path.join(src, "smoke-failed-attempt1/history.json"),
           "training/%s/smoke-failed-attempt1/history.json" % dst_name)

# ---------- SFT/QLoRA run params (data already in local repo, same SHA) ----------
for name in ("ga16-epoch1-v1", "ga16-epoch2-v1"):
    src = os.path.join(T26, name)
    for f in glob.glob(src + "/*"):
        if os.path.isfile(f) and not f.endswith(".jsonl") and not f.endswith(".pid"):
            put(f, "training/sft/%s/%s" % (name, os.path.basename(f)))
src = os.path.join(T26, "data-length-audit-v1")
for f in ("lengths.json", "compact-simulation.json", "measure_training_lengths.py"):
    put_if(os.path.join(src, f), "training/sft/data-length-audit-v1/" + f)
put_if(os.path.join(src, "lengths.log"), "training/sft/data-length-audit-v1/lengths.log.txt")

# ---------- DPO datasets (text only; no raw responses/states) ----------
def dataset_files(src, dst):
    for name in ("dpo.jsonl", "judgments.jsonl", "pair_manifest.jsonl", "skipped.jsonl",
                 "manifest.json", "README.md", "checks.json", "SHA256SUMS"):
        put_if(os.path.join(src, name), os.path.join(dst, name))

dataset_files(DS + "/simple-reflection-v1/dataset", "datasets/v1")
dataset_files(DS + "/simple-reflection-dpo-v2", "datasets/v2")
dataset_files(DS + "/simple-reflection-dpo-v3-additive", "datasets/v3-additive")
for inc, dst in (("simple-reflection-v2-increment", "datasets/increments/v2"),
                 ("simple-reflection-v3-increment", "datasets/increments/v3")):
    src = os.path.join(DS, inc)
    put_if(os.path.join(src, "snapshot_manifest.json"), os.path.join(dst, "snapshot_manifest.json"))
    put_if(os.path.join(src, "review/labels.json"), os.path.join(dst, "review/labels.json"))
    for p in glob.glob(os.path.join(src, "review/packet-*.txt")):
        put(p, os.path.join(dst, "review", os.path.basename(p)))

# ---------- Eval compact index ----------
os.makedirs(os.path.join(STAGE, "eval"), exist_ok=True)
for name in ("summary.json", "tasks.json", "config.json", "state.json",
             "three-case-baseline-comparison.json"):
    put_if(os.path.join(Q, name), "eval/quick3-" + name)
put_if(os.path.join(Q, "controller-mixed100.log"), "eval/controller-mixed100.log.txt")

def find_result(model, dataset, case):
    ps = [p for p in glob.glob(os.path.join(Q, "results", model, "**",
                                            "%s-%s.json" % (dataset, case)), recursive=True)
          if not p.endswith(".tree.json")]
    return ps[0] if ps else None

tasks = json.load(open(Q + "/tasks.json"))
rows = []
for t in tasks:
    p = find_result(t["model"], t["dataset"], t["case_id"])
    item = {"model": t["model"], "suite": (t["id"].split("/")[1] if len(t["id"].split("/")) > 3 else "quick3"),
            "dataset": t["dataset"], "case_id": t["case_id"], "status": t.get("status"),
            "references": t.get("references"), "result_file": os.path.relpath(p, Q) if p else None}
    if p:
        d = json.load(open(p))
        recs = []
        for c in d.get("cases") or []:
            recs += c.get("records") or []
        recs += (d.get("interrupted_case") or {}).get("records") or []
        keys = {json.dumps(r["hypothesis_key"], ensure_ascii=False, sort_keys=True)
                for r in recs if r.get("reference_hit") and r.get("hypothesis_key") is not None}
        tc = d.get("tree_config") or {}
        item.update({
            "completed": d.get("completed"), "found": len(keys), "used": len(recs),
            "call_cap": d.get("call_cap"), "requested_model": d.get("requested_model"),
            "model_strategy": tc.get("model_strategy"), "stage_models": tc.get("stage_models"),
            "search_models": sorted({r.get("model") for r in recs if r.get("model")}),
            "reflection_models": sorted({(r.get("reflection_response") or {}).get("model")
                                         for r in recs if isinstance(r.get("reflection_response"), dict)} - {None}),
            "result_sha256": sha(p),
        })
    rows.append(item)
with open(os.path.join(STAGE, "eval/eval-index.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- manifest ----------
files = []
for root, _, names in os.walk(STAGE):
    for n in names:
        p = os.path.join(root, n)
        if p.endswith("manifest.json") and root == STAGE:
            continue
        files.append({"path": os.path.relpath(p, STAGE), "size": os.path.getsize(p), "sha256": sha(p)})
manifest = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_host": "EEC430004-4090",
    "note": "Compact archive for git: training data/params/logs, DPO dataset text, compact eval index. Weights/tokenizers/raw responses/large result records excluded.",
    "excluded": ["*.safetensors", "tokenizer.json", "vocab.json", "merges.txt",
                 "raw responses/states/contexts", "quick3 *.tree.json and full result records"],
    "files": files,
}
with open(os.path.join(STAGE, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\n")
print("files", len(files), "bytes", sum(x["size"] for x in files))
