"""Compatibility listener: preserve old clients while all routing lives in one pool."""
import httpx
from fastapi import FastAPI, Request, Response

app = FastAPI()
client = httpx.AsyncClient(timeout=3600)


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def forward(path: str, request: Request):
    response = await client.request(request.method, "http://127.0.0.1:8005/" + path,
                                    content=await request.body(), headers={"Content-Type": "application/json"})
    return Response(response.content, status_code=response.status_code, media_type="application/json")
