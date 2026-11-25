#!/usr/bin/env bash
set -euo pipefail
cd '/opt/data/private/ECE4300 Final Project'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export ECE_THINKING=1 ECE_MAX_TOKENS=16384 ECE_WORKERS_PER_METHOD=1 ECE_DATASET_SUITE=easy
export ECE_TREE_TOOLS=1 ECE_METHODS=mcts,two-step
export ECE_RESULT_DIR="${ECE_RESULT_DIR:-$PWD/results/server-easy-tree-tools-v2}"
export ECE_LOG_DIR="${ECE_LOG_DIR:-$PWD/logs/easy-tree-tools-v2}"
mkdir -p "$ECE_LOG_DIR"
.venv-tree-tools/bin/python -u server/run_evaluations.py
