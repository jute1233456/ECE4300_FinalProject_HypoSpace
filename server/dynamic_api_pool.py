"""Hot-configured, model-aware gateway. Management is loopback and token protected."""
import asyncio
import hmac
import hashlib
import json
import os
from pathlib import Path
import time

ROLES = {"baseline", "validation", "shared", "draining", "reserved", "training", "disabled"}


class Registry:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}
        self.config = {}
        self.mtime = None
        self.reload()

    @staticmethod
    def validate(config):
        """Check a candidate pool config without touching live entries."""
        incoming = config["backends"]
        if len({x["id"] for x in incoming}) != len(incoming):
            raise ValueError("Duplicate GPU ID")
        if len({x["url"] for x in incoming}) != len(incoming):
            raise ValueError("Duplicate service URL")
        for item in incoming:
            if item["role"] not in ROLES or not item.get("models"):
                raise ValueError("Invalid backend capabilities")
        return True

    def reload(self):
        raw = self.path.read_bytes()
        # NFS can retain identical/cached mtimes across rapid atomic configuration updates.
        stamp = hashlib.sha256(raw).digest()
        if stamp == self.mtime:
            return
        config = json.loads(raw)
        self.validate(config)
        incoming = config["backends"]
        for item in incoming:
            old = self.entries.get(item["id"])
            if old and old["inflight"] and (old["spec"]["url"] != item["url"]
                                             or old["spec"]["models"] != item["models"]):
                raise ValueError("Drain before changing service identity or models")
        for entry in self.entries.values():
            entry["retired"] = True
        for item in incoming:
            entry = self.entries.setdefault(item["id"], dict(inflight=0, load=0, observed=None,
                                                              checked=0.0, retired=False))
            entry.update(spec=item, retired=False)
        self.config, self.mtime = config, stamp

    def choose(self, model, profile):
        allowed = {profile, "shared"}
        ready = [e for e in self.entries.values()
                 if not e["retired"] and e["spec"]["role"] in allowed
                 and model in e["spec"]["models"] and e["observed"] is not None
                 and time.monotonic() - e["checked"] < 10
                 and max(e["inflight"], e["observed"]) < e["spec"].get("max_inflight", 1)]
        return min(ready, key=lambda e: (max(e["inflight"], e["observed"]), e["load"], e["spec"]["id"])) if ready else None

    def acquire(self, model, profile, weight):
        entry = self.choose(model, profile)
        if entry is not None:
            entry["inflight"] += 1
            entry["load"] += weight
        return entry

    @staticmethod
    def release(entry, weight):
        entry["inflight"] -= 1
        entry["load"] -= weight
        assert entry["inflight"] >= 0 and entry["load"] >= 0


def create_app(path):
    import httpx
    from fastapi import FastAPI, Request, Response, HTTPException
    from .pool_metrics import engine_jobs
    registry = Registry(path)
    lock, polling = asyncio.Lock(), asyncio.Lock()
    client = httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=128))
    token = Path(registry.config["token_file"]).read_text().strip()
    app = FastAPI()
    app.state.registry = registry

    def headers(entry):
        value = entry["spec"].get("api_key")
        return {"Authorization": "Bearer " + value} if value else {}

    async def refresh(force=False):
        async with polling:
            async with lock:
                registry.reload()
                entries = list(registry.entries.values())
            async def one(entry):
                if entry["retired"] or entry["spec"]["role"] in {"reserved", "training", "disabled"}:
                    return
                if not force and time.monotonic() - entry["checked"] < 0.5:
                    return
                try:
                    reply = await client.get(entry["spec"]["url"] + "/metrics", headers=headers(entry), timeout=3)
                    reply.raise_for_status()
                    entry["observed"] = engine_jobs(reply.text)
                except (httpx.HTTPError, ValueError):
                    entry["observed"] = None
                entry["checked"] = time.monotonic()
            await asyncio.gather(*(one(entry) for entry in entries))

    def authorize(request):
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
            raise HTTPException(403, "Management token required")

    @app.get("/health")
    async def health():
        await refresh()
        rows = [dict(id=e["spec"]["id"], url=e["spec"]["url"], role=e["spec"]["role"],
                     models=e["spec"]["models"], inflight=e["inflight"], observed=e["observed"],
                     retired=e["retired"], node=e["spec"].get("node"), gpu=e["spec"].get("gpu"),
                     node_api=e["spec"].get("node_api"), owner=e["spec"].get("owner"))
                for e in registry.entries.values()]
        capacity = {profile: sum(not row["retired"] and row["role"] in (profile, "shared")
                                and row["observed"] is not None for row in rows)
                    for profile in ("baseline", "validation")}
        return dict(status="ok", backends=rows, capacity=capacity,
                    validation_workers=registry.config.get("validation_workers", 6))

    @app.post("/admin/backends/{identity}")
    async def update(identity: str, request: Request):
        authorize(request)
        patch = await request.json()
        if set(patch) - {"role", "models", "url", "owner", "max_inflight", "api_key"}:
            raise HTTPException(400, "Unsupported backend change")
        if "role" in patch and patch["role"] not in ROLES:
            raise HTTPException(400, "Unknown role")
        async with lock:
            registry.reload()
            config = json.loads(json.dumps(registry.config))
            matches = [x for x in config["backends"] if x["id"] == identity]
            if not matches:
                raise HTTPException(404, "Unknown GPU")
            current = registry.entries[identity]
            if current["inflight"] and any(k in patch and patch[k] != current["spec"].get(k)
                                            for k in ("url", "models", "api_key")):
                raise HTTPException(409, "Drain first")
            # A card may become a training/reserved card only after its gateway-visible work ends.
            # Engine-level idleness is proven separately by the node agent before the service stops.
            if patch.get("role") in {"training", "reserved"} and current["inflight"]:
                raise HTTPException(409, "Drain first: card still has active requests")
            matches[0].update(patch)
            Registry.validate(config)
            temp = registry.path.with_suffix(".tmp")
            temp.write_text(json.dumps(config, indent=2)); temp.chmod(0o600); temp.replace(registry.path)
            registry.reload()
        return dict(status="updated", id=identity, role=matches[0]["role"])

    @app.post("/admin/validation-workers")
    async def workers(request: Request):
        authorize(request)
        count = (await request.json())["count"]
        if not isinstance(count, int) or not 0 <= count <= 32:
            raise HTTPException(400, "Invalid worker count")
        async with lock:
            registry.reload()
            registry.config["validation_workers"] = count
            temp = registry.path.with_suffix(".tmp")
            temp.write_text(json.dumps(registry.config, indent=2)); temp.chmod(0o600); temp.replace(registry.path)
            registry.reload()
        return dict(validation_workers=count)

    async def models():
        await refresh()
        names = sorted({name for e in registry.entries.values() if not e["retired"]
                        and e["spec"]["role"] in {"baseline", "validation", "shared"}
                        and e["observed"] is not None for name in e["spec"]["models"]})
        return dict(object="list", data=[dict(id=name, object="model", owned_by="ECE4300") for name in names])

    async def forward(request, profile):
        if request.client and request.client.host not in ('127.0.0.1', '::1', 'testclient', 'test'):
            authorize(request)
        body = await request.json()
        if body.get("stream"):
            raise HTTPException(400, "Only non-streaming requests are supported")
        model = body.get("model", "")
        weight = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4 + 1024
        entry = None
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if await request.is_disconnected():
                raise HTTPException(499, "Client disconnected before dispatch")
            await refresh()
            async with lock:
                entry = registry.acquire(model, profile, weight)
                # Only cards that could legally serve this profile count as "known". A model that
                # exists solely on validation-only cards must not make a baseline request wait 30 min.
                eligible = {profile, "shared"}
                known = any(model in e["spec"]["models"] for e in registry.entries.values()
                            if not e["retired"] and e["spec"]["role"] in eligible)
            if entry is not None:
                break
            if not known:
                raise HTTPException(404, "Requested model is not available for this profile; no fallback")
            await asyncio.sleep(0.25)
        if entry is None:
            raise HTTPException(503, "No matching GPU became available")
        try:
            reply = await client.post(entry["spec"]["url"] + "/v1/chat/completions", json=body, headers=headers(entry))
            return Response(reply.content, status_code=reply.status_code, media_type="application/json",
                            headers={"X-ECE4300-GPU": entry["spec"]["id"], "X-ECE4300-Model": model})
        except httpx.HTTPError as error:
            raise HTTPException(502, "Backend request failed; not replayed automatically") from error
        finally:
            async with lock:
                registry.release(entry, weight)

    @app.get("/v1/models")
    @app.get("/validation/v1/models")
    async def list_models():
        return await models()

    @app.post("/v1/chat/completions")
    async def baseline(request: Request):
        return await forward(request, "baseline")

    @app.post("/validation/v1/chat/completions")
    async def validation(request: Request):
        return await forward(request, "validation")

    return app


def app_factory():
    return create_app(os.environ["ECE_DYNAMIC_POOL_CONFIG"])
