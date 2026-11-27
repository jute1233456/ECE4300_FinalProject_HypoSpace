"""Serve a LoRA adapter with the OpenAI chat-completions shape."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCK = Lock()
STATE = {}
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def chat_text(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate(body):
    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    messages = body["messages"]
    text = chat_text(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    max_new = body.get("max_tokens") or 512
    temperature = body.get("temperature", 0.8)
    kwargs = {
        "max_new_tokens": max_new,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.95)
    else:
        kwargs.update(do_sample=False)
    with LOCK, torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True)
    content = THINK.sub("", content).strip()
    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(new_tokens.shape[0])
    return {
        "model": body.get("model") or "local",
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path.rstrip("/").endswith("/chat/completions"):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                payload = json.dumps(generate(body)).encode("utf-8")
                self.send_response(200)
            except Exception as error:
                payload = json.dumps({"error": type(error).__name__}).encode("utf-8")
                self.send_response(500)
                print(f"Generation failed: {type(error).__name__}: {error}", flush=True)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        print(self.address_string(), fmt % args, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    STATE["tokenizer"] = tokenizer
    STATE["model"] = model
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving {args.adapter} on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
