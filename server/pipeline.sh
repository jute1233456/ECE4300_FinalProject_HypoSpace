#!/usr/bin/env bash
set -euo pipefail
cd '/opt/data/private/ECE4300 Final Project'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
while [ ! -f logs/bootstrap.complete ] || [ ! -f logs/model_source.json ]; do
  sleep 10
done
.venv-inference/bin/python -m pip check > logs/pip-check.log
.venv-inference/bin/python - <<'PY'
from pathlib import Path
import hashlib, json
root=Path.cwd(); model=root/'models/Qwen3-8B'
files={str(p.relative_to(model)): {'size':p.stat().st_size} for p in model.iterdir() if p.is_file()}
for name in ['config.json','model.safetensors.index.json','tokenizer_config.json']:
    p=model/name
    if p.exists(): files[name]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'logs/model_manifest.json').write_text(json.dumps({'source':json.loads((root/'logs/model_source.json').read_text()),'files':files},indent=2))
PY
CUDA_VISIBLE_DEVICES=1 .venv-inference/bin/python server/check_trainability.py > logs/trainability.log 2>&1
nohup bash server/start_api.sh 0 8000 > logs/api_8000.log 2>&1 < /dev/null &
echo "$!" > logs/api_8000.pid
nohup bash server/start_api.sh 1 8001 > logs/api_8001.log 2>&1 < /dev/null &
echo "$!" > logs/api_8001.pid
for port in 8000 8001; do
  ready=0
  for attempt in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then ready=1; break; fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then echo "API $port failed"; exit 1; fi
done
nohup .venv-inference/bin/python -m uvicorn server.api_gateway:app --host 127.0.0.1 --port 8002 > logs/gateway.log 2>&1 < /dev/null &
echo "$!" > logs/gateway.pid
for attempt in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8002/health > /dev/null; then break; fi
  sleep 2
done
curl -sf http://127.0.0.1:8002/health > /dev/null
.venv-inference/bin/python -u server/run_evaluations.py
touch logs/pipeline.complete
