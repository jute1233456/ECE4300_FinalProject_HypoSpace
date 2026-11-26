"""QLoRA SFT for stage samples. Messages come from training/data/*/train.jsonl.

Rules enforced here:
  - loss only over the assistant span (system/user/padding are -100);
  - chat-template prefix must be identical for the prefix-only and full encodings;
  - no silent truncation: any sample longer than --max-length aborts before training;
  - dynamic padding: each batch is padded to its own longest sample, so batch size 1
    pays exactly the sample length (median ~2.2k, longest ~13.7k tokens).

Single card:  python -m server.train_qlora --train-file ... --output ...
Two cards:   CUDA_VISIBLE_DEVICES=0,1 python -m server.train_qlora ... --multi-gpu
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]


def message_pair(row):
    messages = row.get("messages")
    if messages:
        if [m["role"] for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{row.get('sample_id')}: expected system/user/assistant roles")
        return messages
    return [{"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-8B"))
    parser.add_argument("--max-length", type=int, default=14336)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("linear", "warmup-reciprocal"), default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--decay-shape", type=float, default=2.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--multi-gpu", action="store_true", help="device_map=auto across visible GPUs")
    args = parser.parse_args()
    if not 0 <= args.warmup_ratio < 1 or args.decay_shape < 0:
        parser.error("Require 0 <= warmup-ratio < 1 and decay-shape >= 0")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(line) for line in args.train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty training dataset")
    encoded, too_long = [], []
    for row in rows:
        messages = message_pair(row)
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)
        if full[:len(prefix)] != prefix:
            raise ValueError(f"{row.get('sample_id')}: chat template prefix mismatch")
        if len(full) > args.max_length:
            too_long.append((row.get("sample_id"), len(full)))
            continue
        encoded.append({"input_ids": full,
                        "attention_mask": [1] * len(full),
                        "labels": [-100] * len(prefix) + full[len(prefix):]})
    if too_long:
        raise ValueError(f"{len(too_long)} samples exceed --max-length {args.max_length}: "
                         f"{too_long[:5]} ... (drop them explicitly or raise the cap)")
    print(f"encoded {len(encoded)} samples; max_tokens={max(len(e['input_ids']) for e in encoded)}",
          flush=True)

    class Samples(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    def collate(batch):
        longest = max(len(item["input_ids"]) for item in batch)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([item["input_ids"] + [pad] * (longest - len(item["input_ids"])) for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] + [0] * (longest - len(item["attention_mask"])) for item in batch]),
            "labels": torch.tensor([item["labels"] + [-100] * (longest - len(item["labels"])) for item in batch]),
        }

    class AssistantOnlyTrainer(Trainer):
        """Loss over the assistant span only. The shared LM head would otherwise materialise
        [seq, vocab] logits for every prompt token (6+ GiB at 13.7k), which OOMs a 24 GiB card."""

        def create_scheduler(self, num_training_steps, optimizer=None):
            if getattr(self.args, "ece_lr_schedule", "linear") != "warmup-reciprocal":
                return super().create_scheduler(num_training_steps, optimizer)
            if self.lr_scheduler is None:
                from server.lr_schedules import warmup_reciprocal
                warmup = self.args.get_warmup_steps(num_training_steps)
                shape = self.args.ece_decay_shape
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer if optimizer is not None else self.optimizer,
                    lambda step: warmup_reciprocal(step, num_training_steps, warmup, shape))
                self._created_lr_scheduler = True
                print(f"scheduler=warmup-reciprocal; total_updates={num_training_steps}; "
                      f"warmup_updates={warmup}; shape={shape}", flush=True)
            return self.lr_scheduler

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            if labels.shape[0] != 1:
                raise ValueError("assistant-only loss requires per_device_train_batch_size=1")
            row = labels[0]
            positions = (row != -100).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                raise ValueError("sample has no assistant tokens")
            first = int(positions[0])
            keep = int(row.shape[0] - first + 1)  # include the token that predicts the first answer token
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            logits_to_keep=keep, use_cache=False)
            logits = outputs.logits[:, :-1, :].float().reshape(-1, outputs.logits.shape[-1])
            target = row[-keep:][1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100)
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if args.multi_gpu else {"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True),
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
        lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False
    model.print_trainable_parameters()

    training = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, bf16=True, gradient_checkpointing=True, logging_steps=1,
        save_strategy="epoch", report_to="none", seed=33550336, dataloader_num_workers=2,
        optim="paged_adamw_8bit", remove_unused_columns=False,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="linear",
    )
    training.ece_lr_schedule = args.lr_schedule
    training.ece_decay_shape = args.decay_shape
    trainer = AssistantOnlyTrainer(model=model, args=training, train_dataset=Samples(), data_collator=collate)
    # Loss is a per-microbatch mean; Trainer must normalize gradient accumulation.
    trainer.model_accepts_loss_kwargs = False
    print(f"model_accepts_loss_kwargs={trainer.model_accepts_loss_kwargs}; "
          f"gradient_accumulation_steps={args.grad_accum}", flush=True)
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    for index in range(torch.cuda.device_count()):
        print(f"gpu{index} max_memory_allocated_gb={torch.cuda.max_memory_allocated(index)/2**30:.2f}")


if __name__ == "__main__":
    main()
