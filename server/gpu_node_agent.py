"""Manage only explicitly adopted ECE4300 services; never kill unknown GPU owners."""
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def command_hash(pid):
    return hashlib.sha256((Path('/proc') / str(pid) / 'cmdline').read_bytes()).hexdigest()


def create_app(path):
    import httpx
    from fastapi import FastAPI, Request, HTTPException
    from .pool_metrics import engine_jobs
    config_path = Path(path)
    config = json.loads(config_path.read_text())
    root = config_path.parent
    token = Path(config["token_file"]).read_text().strip()
    locks = {str(g["index"]): asyncio.Lock() for g in config["gpus"]}
    app = FastAPI()

    def auth(request):
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
            raise HTTPException(403, "Management token required")

    def gpu(index):
        matches = [g for g in config["gpus"] if str(g["index"]) == index]
        if not matches:
            raise HTTPException(404, "Unknown managed GPU")
        return matches[0]

    def record(index):
        p = root / f"gpu-{index}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def write(index, value):
        p = root / f"gpu-{index}.json"
        tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.chmod(0o600); tmp.replace(p)

    def alive(rec):
        if not rec or 'pid' not in rec:
            return False
        try:
            return command_hash(rec["pid"]) == rec["cmd_hash"]
        except FileNotFoundError:
            return False

    async def stop(index):
        rec = record(index)
        if not rec or 'pid' not in rec:
            return
        try:
            actual = command_hash(rec["pid"])
        except FileNotFoundError:
            return
        if actual != rec["cmd_hash"]:
            raise HTTPException(409, "PID identity changed; refusing to signal it")
        async with httpx.AsyncClient() as client:
            try:
                headers = {"Authorization": "Bearer " + rec["api_key"]} if rec.get("api_key") else {}
                reply = await client.get(rec["url"] + '/metrics', headers=headers, timeout=10)
                reply.raise_for_status()
                if engine_jobs(reply.text) != 0:
                    raise HTTPException(409, "GPU still has running or queued requests")
            except httpx.HTTPError as error:
                raise HTTPException(409, "Cannot prove engine idle; service preserved") from error
        if rec.get("owned_group") and os.getpgid(rec["pid"]) == rec["pid"]:
            os.killpg(rec["pid"], signal.SIGTERM)
        else:
            os.kill(rec["pid"], signal.SIGTERM)
        for _ in range(120):
            if not alive(rec):
                break
            await asyncio.sleep(0.5)
        else:
            raise HTTPException(409, "Service did not exit; no forced kill of adopted process")
        write(index, dict(status="stopped", previous_pid=rec["pid"]))

    def compute_owners(index):
        identity = subprocess.check_output(['nvidia-smi', '-i', index, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        return [line.strip() for line in output.splitlines() if line.split(',')[0].strip() == identity]

    @app.get('/health')
    async def health():
        return dict(status='ok', node=config['node'], gpus=[dict(index=g['index'],
                    status=('serving' if alive(record(str(g['index']))) else 'stopped')) for g in config['gpus']])

    @app.post('/gpus/{index}/stop')
    async def stop_gpu(index: str, request: Request):
        auth(request); gpu(index)
        async with locks[index]:
            await stop(index)
            # Engine workers may take a moment after their API parent exits.
            for _ in range(120):
                if not compute_owners(index):
                    return dict(status='released', index=index)
                await asyncio.sleep(0.5)
            raise HTTPException(409, 'GPU has remaining compute owners; preserved')

    @app.post('/gpus/{index}/serve')
    async def serve(index: str, request: Request):
        auth(request); definition = gpu(index)
        async with locks[index]:
            rec = record(index)
            if alive(rec):
                async with httpx.AsyncClient() as client:
                    reply=await client.get(rec['url']+'/v1/models',headers={'Authorization':'Bearer '+rec['api_key']} if rec.get('api_key') else {},timeout=10)
                    reply.raise_for_status()
                return dict(status='already_serving',url=rec['url'],pid=rec['pid'],
                            models=[x['id'] for x in reply.json()['data']],managed_auth=bool(rec.get('owned_group')))
            if compute_owners(index):
                raise HTTPException(409, 'GPU occupied; training or unknown processes will not be stopped')
            adapters = config['adapters']
            for item in adapters:
                asset = Path(item['path']) / 'adapter_model.safetensors'
                if hashlib.sha256(asset.read_bytes()).hexdigest() != item['sha256']:
                    raise HTTPException(409, 'Adapter checksum mismatch')
            port = definition['port']
            command = [config['python'], '-m', 'vllm.entrypoints.openai.api_server',
                '--model', config['model'], '--served-model-name', 'Qwen3-8B',
                '--host', config['ip'], '--port', str(port), '--dtype', 'bfloat16',
                '--max-model-len', '32768', '--gpu-memory-utilization', '0.93',
                '--max-num-seqs', '1', '--max-num-batched-tokens', '4096', '--enforce-eager',
                '--generation-config', 'vllm', '--enable-lora',
                '--max-lora-rank', '16', '--max-loras', '1', '--max-cpu-loras', '8']
            if adapters:
                command += ['--lora-modules', *[a['name']+'='+a['path'] for a in adapters]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=index, VLLM_API_KEY=token,
                       VLLM_ALLOW_RUNTIME_LORA_UPDATING='True', VLLM_USAGE_STATS='0', OMP_NUM_THREADS='4',
                       VLLM_CACHE_ROOT=str(root / ('cache-gpu'+index)))
            with (root / ('server-gpu'+index+'.log')).open('a') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            # Popen returns after exec; verify exact process identity for future stop operations.
            rec = dict(pid=process.pid, cmd_hash=command_hash(process.pid), owned_group=True,
                       url=f"http://{config['ip']}:{port}", api_key=token, status='starting')
            write(index, rec)
            async with httpx.AsyncClient() as client:
                for _ in range(180):
                    if process.poll() is not None:
                        raise HTTPException(500, 'Model service exited; inspect node log')
                    try:
                        reply = await client.get(rec['url']+'/v1/models',headers={'Authorization':'Bearer '+token},timeout=5)
                        reply.raise_for_status()
                        names = {x['id'] for x in reply.json()['data']}
                        if names >= {'Qwen3-8B', *[a['name'] for a in adapters]}:
                            rec['status']='ready'; write(index, rec)
                            return dict(status='ready', pid=process.pid, url=rec['url'], models=sorted(names),managed_auth=True)
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(3)
            raise HTTPException(504, 'Service startup timeout; retained for inspection')

    @app.post('/gpus/{index}/load')
    async def load_adapter(index: str, request: Request):
        auth(request); gpu(index)
        body = await request.json()
        asset = Path(body['path']).resolve()
        if not asset.is_relative_to(Path(config['asset_root']).resolve()):
            raise HTTPException(400, 'Adapter path must be inside this project asset directory')
        if hashlib.sha256((asset/'adapter_model.safetensors').read_bytes()).hexdigest() != body['sha256']:
            raise HTTPException(409, 'Adapter checksum mismatch')
        async with locks[index]:
            rec = record(index)
            if not alive(rec):
                raise HTTPException(409, 'No running managed service')
            known = next((a for a in config['adapters'] if a['name'] == body['name']), None)
            if known and known['sha256'] != body['sha256']:
                raise HTTPException(409, 'Use a new immutable model name for different weights')
            async with httpx.AsyncClient() as client:
                reply = await client.post(rec['url']+'/v1/load_lora_adapter',
                    headers={'Authorization':'Bearer '+token},
                    json={'lora_name':body['name'],'lora_path':str(asset)},timeout=180)
                if reply.status_code != 200:
                    raise HTTPException(409, 'LoRA runtime load failed: '+reply.text[:200])
            if not known:
                config['adapters'].append(dict(name=body['name'],path=str(asset),sha256=body['sha256']))
                tmp=config_path.with_suffix('.tmp');tmp.write_text(json.dumps(config,indent=2));tmp.chmod(0o600);tmp.replace(config_path)
            return dict(status='loaded',name=body['name'])
    return app


def app_factory():
    return create_app(os.environ['ECE_NODE_CONFIG'])
