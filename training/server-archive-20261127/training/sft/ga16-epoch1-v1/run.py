import sys,subprocess,json,os,datetime,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
def status(phase,**extra):
 data={'phase':phase,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
 (root/'status.json').write_text(json.dumps(data,indent=2))
 print(json.dumps(data),flush=True)
def run(phase,cmd,env=None):
 status(phase,command=cmd)
 with (root/(phase+'.log')).open('w') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root)
 if result.returncode:
  status('failed',failed_phase=phase,exit_code=result.returncode)
  raise SystemExit(result.returncode)
try:
 cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 run('loss-validation',[sys.executable,str(root/'validate_qlora_loss.py')],cpu)
 manifest=json.loads((root/'run_manifest.json').read_text())
 base=[sys.executable,'-u',str(root/'train_qlora.py'),'--model',manifest['model'],'--train-file',str(root/'train.jsonl'),'--batch-size','1','--grad-accum','16','--lr','1e-4','--lora-r','16','--max-length','14336','--multi-gpu']
 run('smoke-ga16',base+['--output',str(root/'smoke-ga16'),'--max-steps','2','--epochs','1'])
 run('epoch1',base+['--output',str(root/'adapter-epoch1'),'--epochs','1'])
 state=json.loads((root/'adapter-epoch1/checkpoint-20/trainer_state.json').read_text())
 assert state['global_step']==20 and state['epoch']==1.0,state
 status('completed',epoch=state['epoch'],global_step=state['global_step'],adapter=str(root/'adapter-epoch1'))
except Exception:
 status('failed',traceback=traceback.format_exc())
 raise
