"""Independent CPU Transformers control; leave GPU services and results unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
path=str(root/'models/Qwen3-8B')
print('Loading CPU BF16 model',flush=True)
tokenizer=AutoTokenizer.from_pretrained(path)
model=AutoModelForCausalLM.from_pretrained(path,device_map='cpu',torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
case=json.loads((root/'results/server/memory/boolean-n1_inputs_00_outputs_0.json').read_text())['cases'][0]
tests=[('simple_boolean','Find one Boolean expression using x,y,AND,OR,NOT that evaluates to 0 when x=0,y=0. Reply only with Expression: followed by the expression.'),('full_prompt',case['prompt'])]
results=[]
for name,prompt in tests:
    text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=False)
    batch=tokenizer(text,return_tensors='pt')
    start=time.monotonic()
    with torch.inference_mode():output=model.generate(**batch,do_sample=False,max_new_tokens=16)
    result={'name':name,'text':tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True),
            'seconds':time.monotonic()-start,'backend':'Transformers CPU BF16 SDPA'}
    print(json.dumps(result),flush=True)
    results.append(result)
(root/'logs/cpu_backend_diagnosis.json').write_text(json.dumps(results,indent=2))
