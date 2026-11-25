#!/usr/bin/env bash
set -euo pipefail
cd '/opt/data/private/ECE4300 Final Project'
export MODELSCOPE_CACHE="$PWD/cache/modelscope"
export HF_HOME="$PWD/cache/huggingface"
python3 -m venv .venv-download
.venv-download/bin/python -m pip install modelscope requests huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv-download/bin/python server/download_model.py
