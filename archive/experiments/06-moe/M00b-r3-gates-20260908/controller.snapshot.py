#!/usr/bin/env python3
"""Gate Phase7 on real four-rank resume/profile checks, then run fixed full arms."""
import os,sys,time,json,hashlib,subprocess,fcntl,signal,traceback
from pathlib import Path
import torch
R=Path('/data/projects/minimind-lab')
A=Path('/data/artifacts/minimind-lab/phase7-formal-r3-20260908')
DATA=Path('/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1')
TRAIN=R/'minimind/trainer/train_phase7.py'
CFG=R/'configs/moe/phase7-formal.json'
child=None
def stop(*_):
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 raise SystemExit('controller stopped; inspect child boundary checkpoint before resume')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def idle():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
 used={int(line.split(',')[0]):int(line.split(',')[1]) for line in rows.splitlines()}
 assert all(used[i]<512 for i in [4,5,6,7]),('non-idle target GPUs',used)
def run(arm,kind,limit,split=False,signal_step=None,formal=False):
 global child
 idle()
 out=A/f'{arm}-{kind}'
 exp=f'M0{1 if arm=="dense" else 2}-{arm}-pretrain-1b28-20260908' if formal else f'M00b-{arm}-{kind}-20260908'
 out.mkdir(exist_ok=True);attempt=len(list(out.glob('command-*.json')))
 assert split or attempt==0,'refuse overwrite'
 data=str(DATA/'train-*-of-00040.jsonl') if kind in ('profile','formal') else str(A/'gate-train.jsonl')
 val=str(DATA/'validation-*-of-00001.jsonl') if kind in ('profile','formal') else str(A/'gate-validation.jsonl')
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(TRAIN),
 '--save_dir',str(out/'weights'),'--resume_dir',str(out/'resume'),'--save_weight','model','--best_weight','best',
 '--batch_size','32','--accumulation_steps','2','--max_seq_len','768','--validation_batch_size','32',
 '--num_workers','4','--dtype','bfloat16','--use_moe',str(int(arm=='moe')),'--data_path',data,'--validation_path',val,
 '--learning_rate','0.0005','--epochs','1','--max_steps',str(limit),'--log_interval','10',
 '--save_interval',str(250 if formal else 0),'--eval_interval',str(1000 if formal else 0),
 '--metrics_path',str(out/'metrics.jsonl'),'--experiment_id',exp,'--from_resume',str(int(split)),
 '--dataset_fingerprint','cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd' if kind in ('profile','formal') else sha(A/'gate-train.jsonl'),
 '--trainer_sha256',sha(TRAIN),'--protocol_sha256',sha(CFG),'--use_swanlab','--swanlab_project','MiniMind-Lab',
 '--swanlab_group','Phase7-Dense-MoE','--swanlab_run_name',f'Phase7-{exp}-B32x4-A2']
 if kind in ('profile','formal'):cmd+=['--expected_validation_rows','11525','--expected_validation_tokens','6400000']
 else:cmd+=['--expected_validation_rows','64']
 env={**os.environ,'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','CUBLAS_WORKSPACE_CONFIG':':4096:8','CUDA_VISIBLE_DEVICES':'4,5,6,7','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','HF_HOME':'/data/cache/huggingface','HF_ENDPOINT':'https://hf-mirror.com','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
 env.pop('PHASE7_SIGNAL_STEP',None)
 if signal_step is not None:env['PHASE7_SIGNAL_STEP']=str(signal_step)
 dump(out/f'command-{attempt}.json',cmd)
 dump(A/'status.json',{'phase':'formal-running' if formal else 'gate-running','arm':arm,'kind':kind,'attempt':attempt,'experiment_id':exp,'started_at':time.time()})
 with (out/f'driver-{attempt}.log').open('x') as log:
  started=time.time();child=subprocess.Popen(cmd,cwd=R/'minimind/trainer',env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  dump(out/f'process-{attempt}.json',{'pid':child.pid,'gpu_ids':[4,5,6,7]});rc=child.wait();child=None
 dump(out/f'exit-{attempt}.json',{'returncode':rc,'wall_seconds':time.time()-started})
 assert rc==0,(arm,kind,rc)
 metrics=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
 end=metrics[-1];assert end['event']=='completed'
 assert end['optimizer_step']==(signal_step or limit or 9038),(arm,kind,end)
 if kind in ('profile','formal'):
  assert end['resolved_total_optimizer_steps']==9038
  assert end['final_validation_tokens']==6400000 and end['final_validation_rows']==11525
 if formal:assert end['status']=='epochs_complete'
 dump(out/f'verified-{attempt}.json',end)
 print(json.dumps({'completed':arm+'-'+kind,'attempt':attempt,'steps':end['optimizer_step']}),flush=True)
 return out
def compare(a,b):
 pa=next((a/'resume').glob('*resume.pth'));pb=next((b/'resume').glob('*resume.pth'))
 x=torch.load(pa,map_location='cpu',weights_only=False);y=torch.load(pb,map_location='cpu',weights_only=False);worst=0.;checked=0
 def walk(a,b):
  nonlocal worst,checked
  if torch.is_tensor(a):
   assert a.shape==b.shape and a.dtype==b.dtype
   if a.is_floating_point():
    err=float((a-b).abs().max()) if a.numel() else 0.;worst=max(worst,err)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6),err
   else:assert torch.equal(a,b)
   checked+=1
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:walk(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for u,v in zip(a,b):walk(u,v)
  else:assert a==b,(a,b)
 walk(x['model'],y['model']);walk(x['optimizer'],y['optimizer'])
 assert x['optimizer_step']==y['optimizer_step']==12
 del x,y
 return {'tensor_count':checked,'max_abs_error':worst,'allclose_atol':1e-6,'allclose_rtol':1e-6,'real_sigterm_step':6,'resumed_to':12,'continuous_to':12}
def main():
 A.mkdir(exist_ok=True);lock=(A/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (A/'status.json').exists(),'inspect previous attempt before relaunch'
 idle()
 assert sha(DATA/'_SUCCESS')=='8c9c728ed063b214ebb74e905e0f6ccc4a510c7078408b87cda5d0c990639c7c'
 assert sha(DATA/'manifest.json')=='1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6'
 torch.set_num_threads(4)
 # Gate dataset uses the exact native dataset class downstream, not M00 hand encoding.
 with (A/'gate-train.jsonl').open('x') as out:
  for f in sorted(DATA.glob('train-*-of-00040.jsonl')):
   with f.open() as src:
    for i in range(128):out.write(next(src))
 with next(DATA.glob('validation-*-of-00001.jsonl')).open() as src,(A/'gate-validation.jsonl').open('x') as out:
  for i in range(64):out.write(next(src))
 dump(A/'source-manifest.json',{'trainer':sha(TRAIN),'protocol':sha(CFG),'model':sha(R/'minimind/model/model_minimind.py'),'dataset_class':sha(R/'minimind/dataset/lm_dataset.py'),'gate_data':sha(A/'gate-train.jsonl')})
 gates={}
 for arm in ['dense','moe']:
  continuous=run(arm,'continuous',12)
  interrupted=run(arm,'resumecheck',12,signal_step=6)
  run(arm,'resumecheck',12,split=True)
  gates[arm]=compare(continuous,interrupted);dump(A/'resume-gates.json',gates)
 for arm in ['dense','moe']:run(arm,'profile',100)
 dump(A/'GATES_PASSED.json',{'resume':gates,'profile_steps_each':100,'world_size':4,'full_validation_tokens':6400000})
 # Profile checkpoints are diagnostic; formal arms always start from scratch.
 for arm in ['dense','moe']:run(arm,'formal',0,formal=True)
 dump(A/'status.json',{'phase':'training-complete-evaluation-pending','seven_base_evaluation':'pending','promoted':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(A/'FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()})
  raise
