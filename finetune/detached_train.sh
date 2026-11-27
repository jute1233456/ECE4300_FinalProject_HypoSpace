#!/bin/bash
# Survives SSH/Cursor disconnect. Powering the machine off still stops it.
trap '' HUP
cd /opt/data/private/ECE4300_FinalProject_HypoSpace
PY=/opt/data/private/ECE4300_FinalProject_HypoSpace/.venv/bin/python
export MODELSCOPE_CACHE=/opt/data/private/modelscope-cache
export HF_HOME=/opt/data/private/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MODELSCOPE_CACHE" finetune/logs finetune/output
if [[ ! -s finetune/model_path.txt ]]; then
  echo "Resuming Qwen3-8B download"
  "$PY" - << 'PY'
from modelscope import snapshot_download
path = snapshot_download("Qwen/Qwen3-8B", cache_dir="/opt/data/private/modelscope-cache")
open("/opt/data/private/ECE4300_FinalProject_HypoSpace/finetune/model_path.txt", "w").write(path + "\n")
print("MODEL_PATH", path, flush=True)
PY
fi
MODEL=$(tr -d '\n' < finetune/model_path.txt)
echo "Training both adapters for 8 hours from $MODEL"
CUDA_VISIBLE_DEVICES=0 "$PY" -u finetune/train_qlora.py \
  --model "$MODEL" --data finetune/data/mix.jsonl --output finetune/output/mix --hours 8 \
  > finetune/logs/mix.log 2>&1 &
echo $! > finetune/logs/mix.pid
CUDA_VISIBLE_DEVICES=1 "$PY" -u finetune/train_qlora.py \
  --model "$MODEL" --data finetune/data/plain.jsonl --output finetune/output/plain --hours 8 \
  > finetune/logs/plain.log 2>&1 &
echo $! > finetune/logs/plain.pid
wait $(cat finetune/logs/mix.pid) || echo "mix training exited nonzero"
wait $(cat finetune/logs/plain.pid) || echo "plain training exited nonzero"
echo "training finished"
