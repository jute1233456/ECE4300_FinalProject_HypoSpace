#!/usr/bin/env bash
set -euo pipefail
cd '/opt/data/private/ECE4300 Final Project'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export ECE_THINKING=1 ECE_MAX_TOKENS=16384 ECE_WORKERS_PER_METHOD=2
export ECE_RESULT_DIR="$PWD/results/server-thinking-v1"
export ECE_LOG_DIR="$PWD/logs/thinking-v1"
mkdir -p "$ECE_LOG_DIR"
.venv-inference/bin/python -u server/run_evaluations.py
