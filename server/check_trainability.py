"""One QLoRA backward pass; no optimizer step and no model weights are saved."""
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

root = Path(__file__).resolve().parents[1]
path = str(root / 'models/Qwen3-8B')
torch.manual_seed(33550336)
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map={'': 0}, torch_dtype=torch.bfloat16,
    attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True))
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    task_type='CAUSAL_LM'))
model.config.use_cache = False
text = tokenizer.apply_chat_template([
    {'role':'user','content':'Return the Boolean expression representing x AND y.'},
    {'role':'assistant','content':'Expression: (x) AND (y)'}], tokenize=False, enable_thinking=False)
batch = tokenizer(text, return_tensors='pt').to('cuda')
model.train()
loss = model(**batch, labels=batch['input_ids']).loss
assert torch.isfinite(loss), loss
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert any(g.abs().sum() > 0 for g in grads)
report = {'loss': loss.item(), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
          'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
          'status':'QLoRA forward/backward succeeded; no optimizer step; original weights unchanged'}
(root/'logs/trainability.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
