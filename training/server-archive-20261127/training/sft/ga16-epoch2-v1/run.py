import sys,os,json,subprocess,datetime,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
def status(phase,**extra):
 data={'phase':phase,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
 temporary=root/'status.tmp';temporary.write_text(json.dumps(data,indent=2));temporary.replace(root/'status.json')
 print(json.dumps(data),flush=True)
try:
 manifest=json.loads((root/'run_manifest.json').read_text())
 output=root/'adapter-epoch2'
 command=[sys.executable,'-u',str(root/'train_qlora.py'),'--model',manifest['model'],'--train-file',str(root/'train.jsonl'),'--output',str(output),'--epochs','2','--batch-size','1','--grad-accum','16','--lr','1e-4','--lora-r','16','--max-length','14336','--multi-gpu']
 status('training',command=command,expected_optimizer_steps=40)
 with (root/'train.log').open('w') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=root)
 if result.returncode:
  status('failed',exit_code=result.returncode);raise SystemExit(result.returncode)
 state=json.loads((output/'checkpoint-40/trainer_state.json').read_text())
 assert state['global_step']==40 and state['epoch']==2.0,state
 status('validating_saved_adapter')
 import torch
 from safetensors.torch import load_file
 weights=load_file(str(output/'adapter_model.safetensors'))
 assert all(torch.isfinite(v).all().item() for v in weights.values())
 bs=[v for k,v in weights.items() if 'lora_B' in k]
 assert bs and all(torch.count_nonzero(v).item()>0 for v in bs)
 status('completed',epoch=2.0,global_step=40,adapter=str(output),finite_adapter_tensors=len(weights),nonzero_lora_B_tensors=len(bs))
except Exception:
 status('failed',traceback=traceback.format_exc());raise
