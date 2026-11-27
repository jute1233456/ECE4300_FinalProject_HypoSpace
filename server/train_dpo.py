"""Completion-only sigmoid DPO continuing a checked SFT LoRA.

Reference log probabilities are frozen before the first optimizer update using
the same quantized model AND SFT adapter. No disabled-adapter/base reference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from server.lr_schedules import warmup_reciprocal
from server.run_finetune_quick6 import atomic


def preference_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    margin = beta * ((chosen - reference_chosen) - (rejected - reference_rejected))
    return -F.logsigmoid(margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['output']) / ('smoke' if args.smoke else 'adapter')
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    source = Path(cfg['sft_adapter'])
    assert hashlib.sha256((source/'adapter_model.safetensors').read_bytes()).hexdigest() == cfg['sft_sha256']
    data = Path(cfg['data'])
    assert hashlib.sha256(data.read_bytes()).hexdigest() == cfg['data_sha256']
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True)
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    samples = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(row['prompt'], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False)
        pair = []
        for side in ('chosen', 'rejected'):
            full = tokenizer.apply_chat_template(row['prompt']+row[side], tokenize=True, enable_thinking=False)
            assert full[:len(prefix)] == prefix and len(full) > len(prefix)
            assert len(full) <= cfg['max_length'], 'No silent truncation permitted'
            pair.append((full, len(prefix)))
        samples.append(pair)
    if args.smoke:
        samples = [max(samples, key=lambda pair: max(len(x[0]) for x in pair))]
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], device_map='auto',
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, source, is_trainable=True)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    device = model.get_input_embeddings().weight.device

    def logprob(sample):
        ids, start = sample
        tokens = torch.tensor([ids], device=device)
        keep = len(ids)-start+1
        output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                       logits_to_keep=keep, use_cache=False)
        logits = output.logits[0, :-1].float()
        targets = tokens[0, start:].to(logits.device)
        # Sum, not mean: standard sequence-level sigmoid DPO.
        return -F.cross_entropy(logits, targets, reduction='sum')

    status = dict(phase='reference',pairs=len(samples),reference='frozen-sft-ep1',
                  reference_adapter_sha256=cfg['sft_sha256'],smoke=args.smoke)
    atomic(root/'status.json',status)
    reference = []
    model.eval()
    with torch.no_grad():
        for i,pair in enumerate(samples):
            reference.append([float(logprob(x)) for x in pair])
            print(json.dumps(dict(phase='reference',completed=i+1,total=len(samples))),flush=True)
    atomic(root/'reference_logps.json',reference)
    # At initialization DPO must equal log(2), since policy == SFT reference.
    model.train()
    with torch.no_grad():
        c,r = [logprob(x) for x in samples[0]]
        initial = float(preference_loss(c,r,*reference[0],cfg['beta']))
    assert abs(initial-math.log(2)) < .03, ('reference mismatch',initial)
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],
                    lr=cfg['lr'],weight_decay=0.0)
    epochs = 1 if args.smoke else cfg['epochs']
    accum = cfg['grad_accum']
    total = epochs*math.ceil(len(samples)/accum)
    warmup = min(total-1,max(1,math.ceil(total*cfg['warmup_ratio']))) if total>1 else 0
    step = 0; started = time.time();history=[]
    for epoch in range(epochs):
        order = list(range(len(samples)));random.shuffle(order)
        for offset in range(0,len(order),accum):
            group = order[offset:offset+accum]
            optimizer.zero_grad(set_to_none=True)
            losses=[];margins=[]
            lr=cfg['lr']*warmup_reciprocal(step+1,total+1,warmup,2.0)
            for param_group in optimizer.param_groups:param_group['lr']=lr
            for index in group:
                # Exact two-pass gradient: compute the scalar DPO derivative
                # first, then backprop each completion separately. Dropout is
                # disabled and weights do not change between these passes.
                # Holding both full-sequence activation graphs can OOM after
                # optimizer state and variable-length allocator blocks grow.
                with torch.no_grad():
                    c,r=[logprob(x) for x in samples[index]]
                loss=preference_loss(c,r,*reference[index],cfg['beta'])
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite DPO loss')
                margin=cfg['beta']*((c-reference[index][0])-(r-reference[index][1]))
                slope=(cfg['beta']*torch.sigmoid(-margin)).detach()
                for side,sign in [(0,-1),(1,1)]:
                    score=logprob(samples[index][side])
                    (score*sign*slope.to(score.device)/len(group)).backward()
                    del score
                losses.append(float(loss.detach()))
                margins.append(float((cfg['beta']*((c-reference[index][0])-(r-reference[index][1]))).detach()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient norm')
            optimizer.step();step+=1
            status=dict(phase='training',global_step=step,total_updates=total,epoch=epoch+(offset+len(group))/len(order),
                loss=sum(losses)/len(losses),reward_margin=sum(margins)/len(margins),lr=lr,
                effective_batch=len(group),grad_norm=float(norm),elapsed_seconds=time.time()-started)
            history.append(status);atomic(root/'status.json',status)
            print(json.dumps(status),flush=True)
    model.save_pretrained(root,safe_serialization=True);tokenizer.save_pretrained(root)
    atomic(root/'history.json',history)
    from safetensors.torch import load_file
    weights=load_file(str(root/'adapter_model.safetensors'))
    original=load_file(str(source/'adapter_model.safetensors'))
    assert set(weights)==set(original)
    changed=sum(not torch.equal(v,original[k]) for k,v in weights.items())
    assert changed>0 and all(torch.isfinite(v).all() for v in weights.values())
    status.update(phase='completed',initial_loss=initial,changed_tensors=changed,
        adapter_sha256=hashlib.sha256((root/'adapter_model.safetensors').read_bytes()).hexdigest(),
        peak_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())])
    atomic(root/'status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
