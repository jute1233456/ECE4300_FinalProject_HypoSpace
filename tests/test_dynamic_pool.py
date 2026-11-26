"""Offline tests for the dynamic compute pool. No GPU, no live service, no real backend."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from server.dynamic_api_pool import Registry
import server.poolctl as poolctl

METRICS = "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
RUNNING = "vllm:num_requests_running 2\nvllm:num_requests_waiting 1\n"


class FakeResponse:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self.content = (json.dumps(payload) if payload is not None else text).encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return json.loads(self.text)


class FakeClient:
    metrics = METRICS
    posted = []

    def __init__(self, *args, **kwargs):
        pass

    async def get(self, url, headers=None, timeout=None):
        return FakeResponse(200, FakeClient.metrics)

    async def post(self, url, json=None, headers=None):
        FakeClient.posted.append({"url": url, "body": json})
        return FakeResponse(200, payload={"id": "reply", "model": json.get("model")})


def write_config(root: Path, backends, token="secret", workers=6):
    token_file = root / "token.txt"
    token_file.write_text(token)
    config = {"token_file": str(token_file), "validation_workers": workers, "backends": backends}
    path = root / "pool.json"
    path.write_text(json.dumps(config))
    return path


def card(identity, url, role, models, node="n0", gpu=0):
    return {"id": identity, "url": url, "role": role, "models": models,
            "node": node, "gpu": gpu, "node_api": "http://127.0.0.1:9"}


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = write_config(self.root, [
            card("g0", "http://g0:1", "baseline", ["Qwen3-8B"]),
            card("g1", "http://g1:1", "validation", ["Qwen3-8B", "recip-ep1"], gpu=1),
        ])
        FakeClient.posted = []
        FakeClient.metrics = METRICS
        patcher = patch("httpx.AsyncClient", FakeClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        from server.dynamic_api_pool import create_app
        self.app = create_app(self.path)
        self.client = TestClient(self.app)

    def post(self, path, model):
        return self.client.post(path, json={"model": model, "messages": [{"role": "user", "content": "hi"}]})

    def test_routing_by_model_and_profile(self):
        reply = self.post("/v1/chat/completions", "Qwen3-8B")
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.headers["X-ECE4300-GPU"], "g0")
        reply = self.post("/validation/v1/chat/completions", "recip-ep1")
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.headers["X-ECE4300-GPU"], "g1")

    def test_no_base_model_fallback(self):
        started = time.monotonic()
        reply = self.post("/v1/chat/completions", "recip-ep1")
        self.assertEqual(reply.status_code, 404)
        self.assertLess(time.monotonic() - started, 5, "must fail fast, not wait for a validation card")

    def test_unknown_model_is_404(self):
        reply = self.post("/v1/chat/completions", "does-not-exist")
        self.assertEqual(reply.status_code, 404)

    def test_draining_card_accepts_nothing_new(self):
        token = {"Authorization": "Bearer secret"}
        self.assertEqual(self.client.post("/admin/backends/g1", json={"role": "draining"},
                                          headers=token).status_code, 200)
        reply = self.post("/validation/v1/chat/completions", "recip-ep1")
        self.assertEqual(reply.status_code, 404)

    def test_admin_requires_token_and_valid_role(self):
        self.assertEqual(self.client.post("/admin/backends/g0", json={"role": "shared"}).status_code, 403)
        reply = self.client.post("/admin/backends/g0", json={"role": "root"},
                                 headers={"Authorization": "Bearer secret"})
        self.assertEqual(reply.status_code, 400)

    def test_identity_change_during_inflight_is_rejected(self):
        self.app.state.registry.entries["g0"]["inflight"] = 1
        reply = self.client.post("/admin/backends/g0", json={"url": "http://other:1"},
                                 headers={"Authorization": "Bearer secret"})
        self.assertEqual(reply.status_code, 409)


class RegistryTests(unittest.TestCase):
    def config(self, backends):
        return {"token_file": "x", "backends": backends}

    def test_validate_rejects_duplicates_and_bad_roles(self):
        with self.assertRaises(ValueError):
            Registry.validate(self.config([card("a", "http://a", "baseline", ["m"]),
                                           card("a", "http://b", "baseline", ["m"])]))
        with self.assertRaises(ValueError):
            Registry.validate(self.config([card("a", "http://a", "root", ["m"])]))
        with self.assertRaises(ValueError):
            Registry.validate(self.config([card("a", "http://a", "baseline", [])]))

    def test_invalid_reload_keeps_previous_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            path.write_text(json.dumps(self.config([card("a", "http://a", "baseline", ["m"])])))
            registry = Registry(path)
            path.write_text(json.dumps(self.config([card("a", "http://a", "root", ["m"])])))
            with self.assertRaises(ValueError):
                registry.reload()
            self.assertEqual(registry.entries["a"]["spec"]["role"], "baseline")


class DrainTests(unittest.TestCase):
    def test_drain_refuses_when_metrics_unavailable(self):
        health = {"backends": [{"id": "g0", "inflight": 0, "observed": None, "owner": None}]}
        with patch.object(poolctl, "call", return_value=health), \
             patch.object(poolctl, "change_role") as change:
            with self.assertRaises(TimeoutError):
                poolctl.drain("http://gw", "t", ["g0"], "train")
            change.assert_called_once()

    def test_drain_waits_for_zero_inflight_and_zero_engine_jobs(self):
        busy = {"backends": [{"id": "g0", "inflight": 1, "observed": 0, "owner": None}]}
        idle = {"backends": [{"id": "g0", "inflight": 0, "observed": 0, "owner": None}]}
        with patch.object(poolctl, "call", side_effect=[idle, busy, idle]), \
             patch.object(poolctl, "change_role"), \
             patch.object(poolctl.time, "sleep"):
            entries = poolctl.drain("http://gw", "t", ["g0"], "train", timeout=30)
        self.assertEqual(entries[0]["observed"], 0)

    def test_drain_rejects_other_owner(self):
        health = {"backends": [{"id": "g0", "inflight": 0, "observed": 0, "owner": "someone"}]}
        with patch.object(poolctl, "call", return_value=health):
            with self.assertRaises(ValueError):
                poolctl.drain("http://gw", "t", ["g0"], "train")


class NodeAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "token.txt").write_text("secret")
        self.config = {"node": "test", "ip": "127.0.0.1", "python": "python", "model": "/nonexistent",
                       "asset_root": str(self.root), "token_file": str(self.root / "token.txt"),
                       "gpus": [{"index": 0, "port": 18100}], "adapters": []}
        self.path = self.root / "node.json"
        self.path.write_text(json.dumps(self.config))
        self.addCleanup(self.tmp.cleanup)

    def app(self):
        from server.gpu_node_agent import create_app
        return TestClient(create_app(self.path))

    def test_serve_refuses_occupied_gpu(self):
        with patch("server.gpu_node_agent.subprocess.check_output",
                   side_effect=["uuid-0", "uuid-0, 1234\n"]):
            reply = self.app().post("/gpus/0/serve", json={}, headers={"Authorization": "Bearer secret"})
        self.assertEqual(reply.status_code, 409)
        self.assertIn("occupied", reply.text)

    def test_serve_refuses_adapter_checksum_mismatch(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"not-the-real-weights")
        self.config["adapters"] = [{"name": "recip-ep1", "path": str(adapter), "sha256": "0" * 64}]
        self.path.write_text(json.dumps(self.config))
        with patch("server.gpu_node_agent.subprocess.check_output", side_effect=["uuid-0", ""]):
            reply = self.app().post("/gpus/0/serve", json={}, headers={"Authorization": "Bearer secret"})
        self.assertEqual(reply.status_code, 409)
        self.assertIn("checksum", reply.text)


class SwapSim:
    def __init__(self):
        self.jobs = 0
        self.delay = 0.0


SIM = {}


class SwapFakeClient:
    """Backend simulator: metrics expose real in-flight count, chat replies after a delay."""

    def __init__(self, *args, **kwargs):
        pass

    async def get(self, url, headers=None, timeout=None):
        base = url[: -len("/metrics")] if url.endswith("/metrics") else url
        sim = SIM.get(base)
        jobs = sim.jobs if sim else 0
        return FakeResponse(200, f"vllm:num_requests_running {jobs}\nvllm:num_requests_waiting 0\n")

    async def post(self, url, json=None, headers=None):
        base = url.rsplit("/v1/chat/completions", 1)[0]
        sim = SIM[base]
        sim.jobs += 1
        try:
            await asyncio.sleep(sim.delay)
            return FakeResponse(200, payload={"id": "reply", "model": json.get("model")})
        finally:
            sim.jobs -= 1


class CardSwapTests(unittest.IsolatedAsyncioTestCase):
    """Training card -> inference, inference card -> training, with no interrupted request."""

    def make_app(self, backends):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.path = write_config(root, backends)
        SIM.clear()
        SIM.update({b["url"]: SwapSim() for b in backends})
        from server.dynamic_api_pool import create_app
        with patch("httpx.AsyncClient", SwapFakeClient):
            return create_app(self.path)

    async def asyncSetUp(self):
        self.app = self.make_app([
            card("g0", "http://g0:1", "baseline", ["Qwen3-8B"]),
            card("g1", "http://g1:1", "baseline", ["Qwen3-8B"]),
            card("g2", "http://g2:1", "training", ["Qwen3-8B"]),
        ])
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                        base_url="http://pool")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def ask(self, model="Qwen3-8B"):
        return await self.client.post("/v1/chat/completions",
                                      json={"model": model, "messages": [{"role": "user", "content": "hi"}]})

    async def set_role(self, identity, role, **extra):
        return await self.client.post(f"/admin/backends/{identity}", json=dict(role=role, **extra),
                                      headers={"Authorization": "Bearer secret"})

    async def health(self):
        return (await self.client.get("/health")).json()

    async def test_inference_to_training_waits_for_running_request(self):
        SIM["http://g0:1"].delay = 1.0
        long_request = asyncio.create_task(self.ask())
        await asyncio.sleep(0.25)

        cards = {row["id"]: row for row in (await self.health())["backends"]}
        self.assertEqual(cards["g0"]["inflight"], 1)
        self.assertEqual((await self.set_role("g0", "training")).status_code, 409,
                         "an active card must not be taken for training")

        self.assertEqual((await self.set_role("g0", "draining")).status_code, 200)
        other = await self.ask()
        self.assertEqual(other.status_code, 200)
        self.assertEqual(other.headers["X-ECE4300-GPU"], "g1", "queue continues on another card")

        reply = await long_request
        self.assertEqual(reply.status_code, 200, "the running request is never cut off")
        self.assertEqual(reply.headers["X-ECE4300-GPU"], "g0")
        self.assertEqual((await self.set_role("g0", "training")).status_code, 200)

    async def test_training_card_joins_inference_when_training_ends(self):
        self.assertEqual((await self.set_role("g2", "baseline")).status_code, 200)
        for identity in ("g0", "g1"):
            self.assertEqual((await self.set_role(identity, "draining")).status_code, 200)
        reply = await self.ask()
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.headers["X-ECE4300-GPU"], "g2", "released training card serves again")

    async def test_card_returns_after_training_and_serves_next_request(self):
        self.assertEqual((await self.set_role("g0", "training")).status_code, 200)
        reply = await self.ask()
        self.assertEqual(reply.headers["X-ECE4300-GPU"], "g1", "training card is out of rotation")
        for identity in ("g1", "g2"):
            self.assertEqual((await self.set_role(identity, "draining")).status_code, 200)
        self.assertEqual((await self.set_role("g0", "baseline")).status_code, 200)
        reply = await self.ask()
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.headers["X-ECE4300-GPU"], "g0", "card returns to inference after training")

    async def test_no_request_is_sent_to_a_training_card(self):
        reply = await self.ask()
        self.assertNotEqual(reply.headers.get("X-ECE4300-GPU"), "g2")
        rows = {row["id"]: row for row in (await self.health())["backends"]}
        self.assertEqual(rows["g2"]["role"], "training")


class ExampleConfigTests(unittest.TestCase):
    def test_example_pool_config_is_valid_and_keeps_adapters_off_baseline(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config/dynamic-pool.example.json").read_text())
        Registry.validate(config)
        roles = [item["role"] for item in config["backends"]]
        self.assertEqual(roles.count("baseline"), 7)
        self.assertEqual(roles.count("validation"), 2)
        baseline_models = {m for item in config["backends"] if item["role"] == "baseline"
                           for m in item["models"]}
        self.assertEqual(baseline_models, {"Qwen3-8B"},
                         "baseline cards must never advertise adapter names")

    def test_example_node_config_shape(self):
        root = Path(__file__).resolve().parents[1]
        node = json.loads((root / "config/gpu-node.example.json").read_text())
        for key in ("node", "ip", "python", "model", "asset_root", "token_file", "gpus", "adapters"):
            self.assertIn(key, node)
        self.assertTrue(node["gpus"] and "index" in node["gpus"][0] and "port" in node["gpus"][0])


if __name__ == "__main__":
    unittest.main()
