"""Authenticated replica pool; actual engine load includes requests from older gateways."""
import asyncio
import json
import os
from pathlib import Path
import time

import httpx
from fastapi import FastAPI, Request, Response
from .pool_metrics import engine_jobs, choose_backend

BACKENDS=json.loads(os.environ['ECE_POOL_BACKENDS'])
KEYS=json.loads(Path(os.environ['ECE_POOL_KEYS_FILE']).read_text())
app=FastAPI()
lock=asyncio.Lock()
poll_lock=asyncio.Lock()
inflight=[0]*len(BACKENDS)
token_load=[0]*len(BACKENDS)
observed=[None]*len(BACKENDS)
updated=0.0
client=httpx.AsyncClient(timeout=3600,limits=httpx.Limits(max_connections=96))


def headers(url):
    return {'Authorization':'Bearer '+KEYS[url]} if url in KEYS else {}


async def refresh():
    global updated
    async with poll_lock:
        if time.monotonic()-updated<0.5:return
        async def one(i,url):
            try:
                r=await client.get(url+'/metrics',headers=headers(url),timeout=3)
                r.raise_for_status();observed[i]=engine_jobs(r.text)
            except (httpx.HTTPError,ValueError):observed[i]=None
        await asyncio.gather(*(one(i,url) for i,url in enumerate(BACKENDS)))
        updated=time.monotonic()


@app.get('/health')
async def health():
    await refresh()
    return dict(status='ok',backends=BACKENDS,inflight_requests=inflight,
                estimated_load=token_load,observed_engine_jobs=observed)


@app.get('/v1/models')
async def models():
    r=await client.get(BACKENDS[0]+'/v1/models',headers=headers(BACKENDS[0]))
    return Response(r.content,status_code=r.status_code,media_type='application/json')


@app.post('/v1/chat/completions')
async def chat(request:Request):
    body=await request.json()
    if body.get('stream'):
        return Response(json.dumps({'error':{'message':'Use non-streaming requests'}}),status_code=400,media_type='application/json')
    await refresh()
    weight=sum(len(str(m.get('content',''))) for m in body.get('messages',[]))//4+1024
    async with lock:
        i=choose_backend(observed,inflight,token_load)
        inflight[i]+=1;token_load[i]+=weight
    try:
        r=await client.post(BACKENDS[i]+'/v1/chat/completions',json=body,headers=headers(BACKENDS[i]))
        return Response(r.content,status_code=r.status_code,media_type='application/json',
                        headers={'X-ECE4300-Replica':str(i)})
    except httpx.HTTPError:
        return Response(json.dumps({'error':{'message':'Replica connection failed'}}),status_code=502,media_type='application/json')
    finally:
        async with lock:
            inflight[i]-=1;token_load[i]-=weight
