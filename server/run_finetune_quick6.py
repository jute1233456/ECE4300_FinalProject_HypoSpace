"""Owned two-GPU pipeline: independent training groups, then fixed MCTS 2.2 quick6.

Never changes the shared inference pool. Run under a single persistent process.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import time
from urllib.request import Request, urlopen


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def case_result(data, references):
    cases = data.get("cases") or []
    case = cases[0] if cases else data.get("interrupted_case", {})
    records = case.get("records", [])
    recovered = {json.dumps(r["hypothesis_key"], sort_keys=True)
                 for r in records if r.get("reference_hit") and r.get("hypothesis_key") is not None}
    metrics = case.get("metrics") or {}
    coverage = metrics.get("recovery", len(recovered) / references)
    return dict(completed=bool(data.get("completed") and cases
                               and case.get("stop_reason") != "request_errors"),
                coverage=coverage, references=references,
                recovered=round(coverage * references), metrics=metrics,
                stop_reason=case.get("stop_reason"), rounds=len(records),
                protocol_errors=sum(bool(r.get("protocol_error")) for r in records),
                truncated_rounds=sum(r.get("finish_reason") == "length" for r in records))


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = args.config.resolve().parent
    lock = (root / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), models={}, evaluations={}, config_sha256=
                 hashlib.sha256(args.config.read_bytes()).hexdigest())
    engines = []

    def status(phase, **extra):
        state.update(phase=phase, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        atomic(root / "state.json", state)
        print(json.dumps(dict(phase=phase, **extra)), flush=True)

    def run(command, cwd, log, env=None):
        with Path(log).open("a") as output:
            result = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {log}")

    def check_adapter(path, expected_steps=None):
        import torch
        from safetensors.torch import load_file
        path = Path(path)
        weights = load_file(str(path / "adapter_model.safetensors"))
        if not weights or not all(torch.isfinite(v).all().item() for v in weights.values()):
            raise ValueError("Nonfinite or empty adapter")
        bs = [v for k, v in weights.items() if "lora_B" in k]
        if not bs or not all(torch.count_nonzero(v).item() for v in bs):
            raise ValueError("Adapter has untrained LoRA B tensors")
        if expected_steps is not None:
            saved = read(path / f"checkpoint-{expected_steps}/trainer_state.json")
            assert saved["global_step"] == expected_steps and saved["epoch"] == expected_steps / 20
        return dict(parameters=sum(v.numel() for v in weights.values()), tensors=len(weights))

    def summarize():
        groups = {}
        for name in state["models"]:
            rows = [r for r in state["evaluations"].values() if r["model"] == name]
            done = [r for r in rows if r.get("completed")]
            groups[name] = dict(completed=len(done), total=6,
                macro_coverage=sum(r["coverage"] for r in done) / 6 if len(done) == 6 else None,
                macro_coverage_completed=sum(r["coverage"] for r in done) / len(done) if done else None,
                found=sum(r["recovered"] for r in done),
                references_completed=sum(r["references"] for r in done),
                full_set_references=114, final=len(done) == 6,
                protocol_errors=sum(r["protocol_errors"] for r in rows))
        atomic(root / "summary.json", groups)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", OMP_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    os.environ["OMP_NUM_THREADS"] = "2"
    try:
        # Frozen evaluation release is used unchanged, including stage-isolation examples.
        code = Path(config["eval_code"])
        for name, digest in read(code / "release_manifest.json")["sha256"].items():
            assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest, name
        for name, digest in config["train_sha256"].items():
            assert hashlib.sha256((Path(config["train_code"]) / name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256(Path(config["train_file"]).read_bytes()).hexdigest() == config["data_sha256"]
        assert hashlib.sha256(Path(config["quick_manifest"]).read_bytes()).hexdigest() == config["quick_sha256"]
        for name, digest in read(config["quick_manifest"])["source_sha256"].items():
            assert hashlib.sha256((code / name.replace("\\", "/")).read_bytes()).hexdigest() == digest, name
        status("waiting_for_existing_training")
        while read(config["existing_two_epoch_status"])["phase"] not in ("completed", "failed"):
            time.sleep(10)
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if any(row.strip().isdigit() for row in apps.splitlines()):
            raise RuntimeError("GPUs have another compute owner; refusing to compete")

        models = [dict(name="base", adapter=None)]
        for item in config["existing_models"]:
            try:
                check_adapter(item["adapter"], item["steps"])
                models.append(dict(name=item["name"], adapter=item["adapter"]))
            except Exception as error:
                state["models"][item["name"]] = dict(status="invalid_adapter", error=str(error))

        for group in config["variants"]:
            name = group["name"]
            directory = root / name
            directory.mkdir(exist_ok=True)
            adapter = directory / "adapter"
            marker = directory / "complete.json"
            try:
                status("training_variant", active_model=name)
                if not marker.exists():
                    atomic(directory / "manifest.json", dict(group, initialization="base", train_file=config["train_file"],
                           data_sha256=config["data_sha256"], scheduler="warmup-reciprocal", decay_shape=2,
                           warmup_ratio=0.1, batch_size=1, gradient_accumulation=16,
                           total_updates=20 * group["epochs"]))
                    command = [config["train_python"], "-u", "-m", "server.train_qlora",
                        "--model", config["base_model"], "--train-file", config["train_file"],
                        "--output", str(adapter), "--epochs", str(group["epochs"]),
                        "--lr", str(group["lr"]), "--lr-schedule", "warmup-reciprocal",
                        "--warmup-ratio", "0.1", "--decay-shape", "2", "--batch-size", "1",
                        "--grad-accum", "16", "--lora-r", "16", "--max-length", "14336", "--multi-gpu"]
                    run(command, config["train_code"], directory / "train.log", env)
                    info = check_adapter(adapter, 20 * group["epochs"])
                    atomic(marker, dict(status="completed", **info))
                else:
                    check_adapter(adapter, 20 * group["epochs"])
                models.append(dict(name=name, adapter=str(adapter)))
                state["models"][name] = dict(status="trained", adapter=str(adapter))
            except Exception as error:
                state["models"][name] = dict(status="training_failed", error=str(error))
                atomic(directory / "failed.json", state["models"][name])
                status("variant_failed", active_model=name)
        for item in models:
            state["models"].setdefault(item["name"], dict(status="ready", adapter=item["adapter"]))

        status("starting_validation_servers", active_model=None)
        adapters = [f"{m['name']}={m['adapter']}" for m in models if m["adapter"]]
        for gpu, port in enumerate(config["ports"]):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            server_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_USAGE_STATS="0",
                              VLLM_CACHE_ROOT=str(root / f"vllm-cache-gpu{gpu}"))
            command = [config["train_python"], "-m", "vllm.entrypoints.openai.api_server",
                "--model", config["base_model"], "--served-model-name", "Qwen3-8B",
                "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
                "--max-model-len", "32768", "--gpu-memory-utilization", "0.93",
                "--max-num-seqs", "1", "--max-num-batched-tokens", "4096", "--enforce-eager",
                "--generation-config", "vllm", "--disable-log-requests", "--enable-lora",
                "--max-lora-rank", "16", "--max-loras", "1", "--max-cpu-loras", "8",
                "--lora-modules", *adapters]
            with (root / f"server-gpu{gpu}.log").open("a") as log:
                process = subprocess.Popen(command, env=server_env, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            engines.append(process)
            atomic(root / f"server-gpu{gpu}.json", dict(pid=process.pid, port=port, command=command))
        for process, port in zip(engines, config["ports"]):
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Validation server on {port} exited; see server log")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
                        names = {r["id"] for r in json.load(response)["data"]}
                    assert names >= {"Qwen3-8B", *(m["name"] for m in models if m["adapter"])}
                    break
                except Exception:
                    time.sleep(5)
            else:
                raise TimeoutError(f"Validation server startup timeout on {port}")
        # Registration alone does not exercise LoRA kernels. Probe every adapter before long jobs.
        for model in models:
            payload = dict(model=model["name"] if model["adapter"] else "Qwen3-8B",
                messages=[dict(role="user", content="Reply with OK.")], max_tokens=8,
                chat_template_kwargs=dict(enable_thinking=False))
            request = Request(f"http://127.0.0.1:{config['ports'][0]}/v1/chat/completions",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=180) as response:
                probe = json.load(response)
            assert probe.get("choices") and probe.get("usage", {}).get("completion_tokens", 0) > 0
            atomic(root / f"probe-{model['name']}.json", probe)

        jobs = queue.Queue()
        quick = read(config["quick_manifest"])
        for case in sorted(quick["jobs"], key=lambda j: (j["references"], j["dataset"], j["case_id"])):
            for model in models:
                jobs.put(dict(case, **model))
        status("validating", total_jobs=jobs.qsize(), workers=2)

        def worker(port):
            while True:
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    break
                name, dataset, cid = job["name"], job["dataset"], job["case_id"]
                directory = root / "results" / name
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"{dataset}-{cid}.json"
                model_name = name if job["adapter"] else "Qwen3-8B"
                command = [config["eval_python"], "-u", "-m", "experiments.test_hard_balanced",
                    "--model", model_name, "--enable-thinking", "--base-url", f"http://127.0.0.1:{port}/v1",
                    "--max-tokens", "32768", "--dataset-root", str(code / "datasets/medium_sampled"),
                    "--dataset", dataset, "--case-id", cid, "--method", "mcts-tree", "--tree-policy", "llm",
                    "--duplicate-reward", "-1", "--exploration-list", "off", "--two-step-reflection",
                    "--unbounded-tree", "--tree-tools", "--mcts-version", "2.2", "--stage22-revision",
                    "--pause-file", str(root / "PAUSE_VALIDATION")]
                row = dict(model=name, dataset=dataset, case_id=cid, references=job["references"],
                           completed=False, recovered=0, coverage=0, protocol_errors=0)
                try:
                    existing = read(target) if target.exists() else None
                    if existing is None or not case_result(existing, job["references"])["completed"]:
                        command += ["--resume" if existing else "--output", str(target)]
                        eval_env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                        TOKENIZERS_PARALLELISM="false")
                        run(command, code, target.with_suffix(".log"), eval_env)
                    row.update(case_result(read(target), job["references"]))
                except Exception as error:
                    row["error"] = str(error)
                # Separate per-job files are atomic; aggregate snapshots use one lock below.
                with result_lock:
                    state["evaluations"][f"{name}/{dataset}/{cid}"] = row
                    summarize()
                    status("validating", finished_jobs=len(state["evaluations"]))
                jobs.task_done()

        import threading
        result_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, port) for port in config["ports"]]
            for future in futures:
                future.result()
        summarize()
        complete = all(r.get("completed") for r in state["evaluations"].values())
        training_ok = all(state["models"][g["name"]]["status"] != "training_failed" for g in config["variants"])
        status("completed" if complete and training_ok and len(state["evaluations"]) == 36 else "needs_inspection")
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for process in engines:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
