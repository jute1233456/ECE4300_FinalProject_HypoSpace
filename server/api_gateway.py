"""Balance model replicas (local + remote) by estimated in-flight token load.

Backends are configured via the ECE_BACKENDS environment variable, a comma-separated
list of OpenAI-compatible base URLs (without the /v1 suffix). The gateway forwards each
non-streaming chat completion to the backend with the lowest estimated in-flight token
load, so idle GPUs — including remote machines — pick up work automatically.
"""
import asyncio
import json
import os

import httpx
from fastapi import FastAPI, Request, Response

BACKENDS = [backend.strip() for backend in os.environ.get(
    'ECE_BACKENDS', 'http://127.0.0.1:8000,http://127.0.0.1:8001').split(',') if backend.strip()]

app = FastAPI()
lock = asyncio.Lock()
loads = [0] * len(BACKENDS)
requests = [0] * len(BACKENDS)
client = httpx.AsyncClient(timeout=float(os.environ.get('ECE_GATEWAY_TIMEOUT', '3600')), limits=httpx.Limits(max_connections=64))


@app.get('/health')
async def health():
    return {'status': 'ok', 'request_timeout_seconds': float(os.environ.get('ECE_GATEWAY_TIMEOUT','3600')), 'backends': BACKENDS,
            'estimated_load': loads, 'inflight_requests': requests}


@app.get('/v1/models')
async def models():
    response = await client.get(BACKENDS[0] + '/v1/models')
    return Response(response.content, status_code=response.status_code, media_type='application/json')


@app.post('/v1/chat/completions')
async def chat(request: Request):
    body = await request.json()
    if body.get('stream'):
        return Response(json.dumps({'error': {'message': 'This project gateway supports non-streaming requests.'}}),
                        status_code=400, media_type='application/json')
    weight = sum(len(str(m.get('content', ''))) for m in body.get('messages', [])) // 4 + 1024
    async with lock:
        index = min(range(len(BACKENDS)), key=lambda i: (loads[i], requests[i], i))
        loads[index] += weight
        requests[index] += 1
    try:
        response = await client.post(BACKENDS[index] + '/v1/chat/completions', json=body)
        return Response(response.content, status_code=response.status_code,
                        media_type='application/json', headers={'X-ECE4300-Replica': str(index)})
    except httpx.HTTPError:
        return Response(json.dumps({'error': {'message': 'Replica connection failed; inspect server logs.'}}),
                        status_code=502, media_type='application/json')
    finally:
        async with lock:
            loads[index] -= weight
            requests[index] -= 1
