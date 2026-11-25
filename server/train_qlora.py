"""Future SFT entry point. JSONL rows contain prompt and response; never use test answers."""
import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--epochs', type=float, default=1)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    path = str(root/'models/Qwen3-8B')
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(line) for line in args.train_file.read_text().splitlines() if line.strip()]
    encoded = []
    for row in rows:
        messages = [{'role':'user','content':row['prompt']}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(messages+[{'role':'assistant','content':row['response']}], tokenize=True, enable_thinking=False)
        if full[:len(prefix)] != prefix:
            raise ValueError('Chat template prefix mismatch; inspect labels before training')
        if len(full) > args.max_length:
            raise ValueError('Training sample exceeds max length; do not silently truncate answers')
        padding = args.max_length - len(full)
        encoded.append({'input_ids':full+[tokenizer.pad_token_id]*padding,
                        'attention_mask':[1]*len(full)+[0]*padding,
                        'labels':[-100]*len(prefix)+full[len(prefix):]+[-100]*padding})
    if not encoded:
        raise ValueError('Empty training dataset')
    class Samples(Dataset):
        def __len__(self): return len(encoded)
        def __getitem__(self, index): return encoded[index]
    model = AutoModelForCausalLM.from_pretrained(path, device_map={'':0}, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.config.use_cache=False
    training = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=1e-4,
        bf16=True, gradient_checkpointing=True, logging_steps=1, save_strategy='epoch',
        report_to='none', seed=33550336, dataloader_num_workers=2, optim='paged_adamw_8bit')
    trainer = Trainer(model=model,args=training,train_dataset=Samples(),data_collator=default_data_collator)
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == '__main__':
    main()
