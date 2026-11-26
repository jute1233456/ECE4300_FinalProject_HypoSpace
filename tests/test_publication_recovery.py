"""Regression checks for exec identity races and resumable LoRA publication."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from server.gpu_node_agent import command_hash, started_command_hash
from server.run_sft_handoff import publish_adapter


class IdentityTests(unittest.TestCase):
    def test_empty_command_line_is_not_a_live_identity(self):
        with patch.object(Path, "read_bytes", return_value=b""):
            with self.assertRaises(FileNotFoundError):command_hash(100)

    def test_launch_waits_for_exact_command_after_empty_exec_window(self):
        raw=b"python\0-m\0vllm\0"
        child=SimpleNamespace(pid=100,poll=lambda:None)
        with patch.object(Path,"read_bytes",side_effect=[b"",b"old-parent\0",raw]):
            result=asyncio.run(started_command_hash(child,["python","-m","vllm"]))
        self.assertEqual(result,hashlib.sha256(raw).hexdigest())


class PublicationTests(unittest.TestCase):
    def test_partial_publish_skips_loaded_card_and_loads_missing_card(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/"adapter";out.mkdir();assets=root/"assets";assets.mkdir()
            for name in ["adapter_config.json","adapter_model.safetensors"]:(out/name).write_bytes(b"same")
            asset=assets/"new";asset.mkdir()
            for name in ["adapter_config.json","adapter_model.safetensors"]:(asset/name).write_bytes(b"same")
            (root/"token").write_text("test")
            cfg=dict(name="new",asset_root=str(assets),output=str(out),token_file=str(root/"token"),gpus=["g0","g1"],gateway="pool",owner="job")
            entries=[dict(id="g0",url="gpu0",node_api="agent",gpu=0,models=["base","new"]),dict(id="g1",url="gpu1",node_api="agent",gpu=1,models=["base"])]
            calls=[]
            def api(url,payload=None,*args,**kwargs):
                calls.append(url)
                if url=="gpu0/v1/models":return dict(data=[dict(id="new",root=str(asset))])
                if url=="gpu1/v1/models":return dict(data=[])
                if url.endswith("chat/completions"):return dict(choices=[{}])
                return dict(status="ok")
            with patch("server.run_sft_handoff.drain",side_effect=[[entries[0]],[entries[1]]]),patch("server.run_sft_handoff.call",side_effect=api):
                digest=publish_adapter(cfg)
            self.assertEqual(digest,hashlib.sha256(b"same").hexdigest())
            self.assertNotIn("agent/gpus/0/load",calls)
            self.assertIn("agent/gpus/1/load",calls)
            self.assertIn("pool/admin/backends/g0",calls)
            self.assertIn("pool/admin/backends/g1",calls)


if __name__=="__main__":unittest.main()
