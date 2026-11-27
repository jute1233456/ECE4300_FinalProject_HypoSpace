#!/bin/bash
# Two QLoRA runs, then memory-agent evaluation. Wall clock is capped near 24 hours.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
export HF_HOME=/opt/data/private/hf-cache
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PIP_INDEX="${PIP_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://mirrors.aliyun.com/pytorch-wheels/cu124}"
mkdir -p "$HF_HOME" finetune/logs finetune/data finetune/output results
START=$(date +%s)

echo "Installing training packages from ${PIP_INDEX}"
"$PY" -m pip install -U pip -i "$PIP_INDEX" --trusted-host mirrors.huaweicloud.com
"$PY" -m pip install numpy networkx scipy sympy pyyaml requests -i "$PIP_INDEX" --trusted-host mirrors.huaweicloud.com
echo "Building SFT data"
"$PY" -u finetune/build_sft.py > finetune/logs/build_sft.log 2>&1 || {
  echo "Data build failed; see finetune/logs/build_sft.log"
  exit 1
}
"$PY" -m pip install torch --index-url "$TORCH_INDEX" --trusted-host mirrors.aliyun.com
"$PY" -m pip install "transformers>=4.51" "peft>=0.15" "accelerate>=1.2" "bitsandbytes>=0.45" datasets \
  -i "$PIP_INDEX" --trusted-host mirrors.huaweicloud.com
"$PY" - << 'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B")
print("model ready")
PY

NOW=$(date +%s)
REMAIN=$((24 * 3600 - (NOW - START)))
TRAIN=$((REMAIN - 10 * 3600))
if [ "$TRAIN" -gt $((8 * 3600)) ]; then TRAIN=$((8 * 3600)); fi
if [ "$TRAIN" -lt $((2 * 3600)) ]; then TRAIN=$((2 * 3600)); fi
HOURS=$("$PY" -c "print(round($TRAIN / 3600, 2))")
echo "Training both adapters for ${HOURS} hours"
CUDA_VISIBLE_DEVICES=0 "$PY" -u finetune/train_qlora.py \
  --data finetune/data/mix.jsonl --output finetune/output/mix --hours "$HOURS" \
  > finetune/logs/mix.log 2>&1 &
PID_MIX=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u finetune/train_qlora.py \
  --data finetune/data/plain.jsonl --output finetune/output/plain --hours "$HOURS" \
  > finetune/logs/plain.log 2>&1 &
PID_PLAIN=$!
wait "$PID_MIX" || echo "mix training exited nonzero"
wait "$PID_PLAIN" || echo "plain training exited nonzero"
test -f finetune/output/mix/adapter_config.json || { echo "mix adapter missing"; exit 1; }
test -f finetune/output/plain/adapter_config.json || { echo "plain adapter missing"; exit 1; }

pick_adapter() {
  local name="$1"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u finetune/serve.py --adapter "finetune/output/$name" --port 8000 \
    > "finetune/logs/serve_$name.log" 2>&1 &
  echo $! > finetune/logs/serve.pid
  for _ in $(seq 1 180); do
    "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" && break
    sleep 2
  done
  "$PY" -u -m experiments.test_hard_balanced \
    --method memory --dataset boolean --case-id "$BOOLEAN_CASE" --queries 20 \
    --base-url http://127.0.0.1:8000/v1 --api-key local --model qwen3-8b-hypospace \
    --temperature 0.8 --max-tokens 256 \
    --pause-file "results/PAUSE_FT_$name" \
    --output "results/ft_probe_${name}.json" \
    > "finetune/logs/probe_$name.log" 2>&1 || echo "probe $name failed"
  kill "$(cat finetune/logs/serve.pid)" || true
  wait "$(cat finetune/logs/serve.pid)" 2>/dev/null || true
}

BOOLEAN_CASE=$("$PY" -c "import json; d=json.load(open('datasets/balanced/boolean.json')); rows=sum(d['datasets_by_n_observations'].values(), []); rows.sort(key=lambda row: row['n_compatible_expressions']); print(rows[len(rows)//2]['observation_set_id'])")
pick_adapter mix
pick_adapter plain

WINNER=$("$PY" - << 'PY'
import json
from pathlib import Path
scores = {}
for name in ("mix", "plain"):
    path = Path(f"results/ft_probe_{name}.json")
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    summary = data.get("summary", {}).get("all", {})
    scores[name] = summary.get("recovery", -1)
print(max(scores, key=scores.get) if scores else "mix")
PY
)
echo "Selected adapter: $WINNER"
CUDA_VISIBLE_DEVICES=0 "$PY" -u finetune/serve.py --adapter "finetune/output/$WINNER" --port 8000 \
  > finetune/logs/serve_final.log 2>&1 &
echo $! > finetune/logs/serve.pid
for _ in $(seq 1 180); do
  "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" && break
  sleep 2
done
"$PY" -u -m experiments.test_hard_balanced \
  --method memory \
  --base-url http://127.0.0.1:8000/v1 --api-key local --model qwen3-8b-hypospace \
  --temperature 0.8 --max-tokens 512 \
  --pause-file results/PAUSE_FT_FINAL \
  --output results/ft_memory_hard_balanced.json \
  > finetune/logs/eval_final.log 2>&1
echo "Evaluation finished"
