#!/usr/bin/env bash
set -euo pipefail
cd '/opt/data/private/ECE4300 Final Project'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
.venv-inference/bin/python server/manage_evaluation.py resume
.venv-inference/bin/python -u server/run_evaluations.py
touch logs/pipeline.complete
