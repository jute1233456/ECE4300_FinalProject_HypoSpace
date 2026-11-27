"""QLoRA fine-tune of one 8B model on a single GPU."""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class StopAfter(TrainerCallback):
    def __init__(self, seconds):
        self.deadline = time.time() + seconds

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.deadline:
            control.should_training_stop = True
            control.should_save = True


def render(tokenizer, messages):
    prompt_messages = messages[:-1]
    try:
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise SystemExit("Chat template is not a prefix of the full conversation")
    return prompt, full


def encode_rows(tokenizer, path, max_length):
    encoded = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            messages = json.loads(line)["messages"]
            prompt, full = render(tokenizer, messages)
            tokenized = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
            cut = len(prompt)
            input_ids = tokenized["input_ids"]
            labels = [
                token if start >= cut else -100
                for token, (start, _end) in zip(input_ids, tokenized["offset_mapping"])
            ]
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]
            if not any(label != -100 for label in labels):
                skipped += 1
                continue
            encoded.append((input_ids, labels))
    print(f"Encoded {len(encoded)} sequences, skipped {skipped}", flush=True)
    return encoded


def pack(encoded, max_length):
    packed = []
    ids, labels = [], []
    for item_ids, item_labels in encoded:
        if ids and len(ids) + len(item_ids) > max_length:
            packed.append((ids, labels))
            ids, labels = [], []
        ids.extend(item_ids)
        labels.extend(item_labels)
    if ids:
        packed.append((ids, labels))
    print(f"Packed into {len(packed)} sequences of at most {max_length} tokens", flush=True)
    return packed


class PackedSet(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids, labels = self.rows[index]
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = pack(encode_rows(tokenizer, args.data, args.max_length), args.max_length)
    if not rows:
        raise SystemExit("No training rows")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    pad_id = tokenizer.pad_token_id

    def collate(features):
        width = max(len(item["input_ids"]) for item in features)
        input_ids, labels, mask = [], [], []
        for item in features:
            gap = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * gap)
            labels.append(item["labels"] + [-100] * gap)
            mask.append(item["attention_mask"] + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=args.epochs,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,
            bf16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            optim="paged_adamw_8bit",
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            seed=20261126,
        ),
        train_dataset=PackedSet(rows),
        data_collator=collate,
        callbacks=[StopAfter(args.hours * 3600)],
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved adapter to {args.output}", flush=True)


if __name__ == "__main__":
    main()
