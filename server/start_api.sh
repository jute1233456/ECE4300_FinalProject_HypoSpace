#!/usr/bin/env bash
set -euo pipefail
cd '/opt/data/private/ECE4300 Final Project'
export HF_HOME="$PWD/cache/huggingface"
export VLLM_CACHE_ROOT="$PWD/cache/vllm"
export OMP_NUM_THREADS=8
export VLLM_USAGE_STATS=0
gpu="${1:-0}"
port="${2:-8000}"
export CUDA_VISIBLE_DEVICES="$gpu"
exec .venv-inference/bin/vllm serve "$PWD/models/Qwen3-8B" \
  --served-model-name Qwen3-8B --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 --enforce-eager \
  --generation-config vllm --disable-log-requests
