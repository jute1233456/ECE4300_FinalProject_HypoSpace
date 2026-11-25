#!/usr/bin/env bash
set -euo pipefail
cd '/opt/data/private/ECE4300 Final Project'
mkdir -p logs models results/server cache
export PIP_CACHE_DIR="$PWD/cache/pip"
export HF_HOME="$PWD/cache/huggingface"
export MODELSCOPE_CACHE="$PWD/cache/modelscope"
export OMP_NUM_THREADS=8
python3 -m venv .venv-inference
.venv-inference/bin/python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv-inference/bin/python -m pip install -r requirements-server.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv-inference/bin/python -m pip freeze > logs/environment.lock.txt
.venv-inference/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
if [ ! -f logs/model_source.json ]; then
  .venv-inference/bin/python server/download_model.py
fi
touch logs/bootstrap.complete
