#!/usr/bin/env python3
"""Paired Phase7 architecture probe, not full pretraining or release evaluation."""
import os,sys,json,time,math,hashlib,fcntl,signal,gc,subprocess
from pathlib import Path
import torch
from transformers import AutoTokenizer
R=Path('/data/projects/minimind-lab');sys.path.insert(0,str(R/'minimind'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM,MOEFeedForward
A=Path('/data/artifacts/minimind-lab/phase7-probe-20260908')
DATA=Path('/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1')
stop=False
def stopping(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 import swanlab
 A.mkdir(exist_ok=True);lock=(A/'probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (A/'DONE.json').exists(),'refuse repeat'
 gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu=='7'
 used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip());assert used<512,('GPU not idle',used)
 torch.set_num_threads(4);torch.cuda.set_device(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert sha(DATA/'_SUCCESS')=='8c9c728ed063b214ebb74e905e0f6ccc4a510c7078408b87cda5d0c990639c7c'
 assert sha(DATA/'manifest.json')=='1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6'
 tok=AutoTokenizer.from_pretrained(str(R/'minimind/model'),local_files_only=True)
 texts=[];selection=[]
 for f in sorted(DATA.glob('train-*-of-00040.jsonl')):
  with f.open() as h:
   for i in range(10):
    row=json.loads(next(h));texts.append(row['text']);selection.append({'shard':f.name,'row':i,'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest()})
 assert len(texts)==400
 with next(DATA.glob('validation-*-of-00001.jsonl')).open() as h:vt=[json.loads(next(h))['text'] for _ in range(64)]
 def encode(ts):
  rows=[]
  for s in ts:
   ids=([tok.bos_token_id]+tok.encode(s,add_special_tokens=False)+[tok.eos_token_id])[:768]
   rows.append((ids+[tok.pad_token_id]*(768-len(ids)),ids+[-100]*(768-len(ids))))
  return torch.tensor([r[0] for r in rows]),torch.tensor([r[1] for r in rows])
 X,Y=encode(texts);VX,VY=encode(vt);assert (Y[:,1:]!=-100).sum()>0
 dataid=hashlib.sha256(X.numpy().tobytes()+Y.numpy().tobytes()).hexdigest()
 dump(A/'data-manifest.json',{'selection':selection,'train_rows':400,'validation_rows':64,'train_targets':int((Y[:,1:]!=-100).sum()),'tokenized_sha256':dataid,'validation_sha256':hashlib.sha256(VX.numpy().tobytes()+VY.numpy().tobytes()).hexdigest(),'tokenizer_sha256':sha(R/'minimind/model/tokenizer.json'),'scope':'prefix samples per shard, diagnostic only'})
 summaries=[]
 for seed in [42,43,44]:
  for arm in (['dense','moe'] if seed!=43 else ['moe','dense']):
   if stop:raise RuntimeError('SIGTERM before next arm')
   exp=f'M00-{arm}-probe-s{seed}-20260908';out=A/exp
   if (out/'DONE.json').exists():summaries.append(json.loads((out/'DONE.json').read_text()));continue
   assert not out.exists(),'partial attempt requires explicit recovery'
   out.mkdir();torch.manual_seed(seed)
   model=MiniMindForCausalLM(MiniMindConfig(use_moe=arm=='moe')).cuda()
   total=sum(p.numel() for p in model.parameters());expert=sum(p.numel() for n,p in model.named_parameters() if '.experts.' in n)
   active=total-expert+expert//4 if arm=='moe' else total
   opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
   order=torch.randperm(400,generator=torch.Generator().manual_seed(seed))
   cfg={'seed':seed,'arm':arm,'total_parameters':total,'nominal_active_parameters':active,'steps':100,'batch':4,'seq':768,'data_sha256':dataid,'scheduler_denominator':578371,'init':'random same seed; architecture RNG consumption differs','precision':'FP32 params + BF16 autocast','gpu':gpu,'source_sha256':sha(Path(__file__)),'model_source_sha256':sha(R/'minimind/model/model_minimind.py')}
   dump(out/'config.json',cfg)
   run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name=f'Phase7-M00-{arm.upper()}-Probe-S{seed}-100steps',group='Phase7-Dense-MoE',job_type='train',mode='online',log_dir=str(out/'swanlog'),config=cfg)
   (out/'swanlab-url.txt').write_text(run.url+'\n')
   counts={};handles=[];record_load=False
   def hook(layer):
    def f(mod,args,logits):
     if record_load:counts[layer]=torch.bincount(logits.detach().argmax(-1).flatten(),minlength=4).float()
    return f
   for i,block in enumerate(model.model.layers):
    if isinstance(block.mlp,MOEFeedForward):handles.append(block.mlp.gate.register_forward_hook(hook(i)))
   model.train();torch.cuda.reset_peak_memory_stats();records=[]
   try:
    for step in range(1,101):
     idx=order[(step-1)*4:step*4];x=X[idx].cuda();y=Y[idx].cuda()
     lr=5e-4*(.1+.45*(1+math.cos(math.pi*step/578371)))
     for g in opt.param_groups:g['lr']=lr
     opt.zero_grad(set_to_none=True);record_load=step%10==0
     torch.cuda.synchronize();t=time.perf_counter()
     with torch.autocast('cuda',dtype=torch.bfloat16):res=model(x,labels=y);loss=res.loss+res.aux_loss
     assert torch.isfinite(loss)
     loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(gn)
     opt.step();torch.cuda.synchronize();dt=time.perf_counter()-t
     targets=int((y[:,1:]!=-100).sum());v={'step':step,'ce_loss':float(res.loss),'aux_loss':float(res.aux_loss),'total_loss':float(loss),'lr':lr,'grad_norm':float(gn),'step_seconds':dt,'valid_targets':targets,'valid_targets_per_second':targets/dt,'padded_tokens_per_second':3072/dt,'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20}
     if record_load:
      for layer,c in counts.items():
       probs=c/c.sum()
       for e,p in enumerate(probs):v[f'router/layer{layer}/expert{e}_fraction']=float(p)
       v[f'router/layer{layer}/max_mean_load']=float(probs.max()*4)
      counts.clear()
     record_load=False
     with (out/'metrics.jsonl').open('a') as h:h.write(json.dumps(v)+'\n')
     swanlab.log(v,step=step);records.append(v)
     if step%25==0:print(json.dumps({'experiment':exp,**v}),flush=True)
     if stop:
      torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':cfg,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},out/'interrupted-resume.pt')
      raise RuntimeError('SIGTERM saved at optimizer boundary')
    model.eval();nll=0.;nt=0
    with torch.no_grad():
     for off in range(0,64,4):
      vx=VX[off:off+4].cuda();vy=VY[off:off+4].cuda();n=int((vy[:,1:]!=-100).sum())
      with torch.autocast('cuda',dtype=torch.bfloat16):res=model(vx,labels=vy)
      nll+=float(res.loss)*n;nt+=n
    checkpoint=out/'checkpoint.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':100,'config':cfg},checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);del state
    steady=[v for v in records if v['step']>10 and v['step']%10!=0]
    result={'experiment_id':exp,'status':'probe-completed','steps':100,'total_parameters':total,'nominal_active_parameters':active,'valid_targets':sum(v['valid_targets'] for v in records),'validation_nll':nll/nt,'validation_ppl':math.exp(nll/nt),'steady_targets_per_second':sum(v['valid_targets'] for v in steady)/sum(v['step_seconds'] for v in steady),'peak_memory_mib':max(v['peak_memory_mib'] for v in records),'checkpoint_sha256':sha(checkpoint),'checkpoint_strict_load':True,'optimizer_roundtrip':True,'swanlab_url':run.url,'not_formal_training':True,'routing_counts_include_padding':True}
    swanlab.log({k:v for k,v in result.items() if isinstance(v,(int,float)) and not isinstance(v,bool)},step=101);swanlab.finish();dump(out/'DONE.json',result);summaries.append(result)
   except Exception as e:
    dump(out/'FAILURE.json',{'error':str(e)});swanlab.finish(state='crashed');raise
   finally:
    for h in handles:h.remove()
    del model,opt,res,loss;gc.collect();torch.cuda.empty_cache()
 dump(A/'DONE.json',{'status':'probe-completed','runs':summaries,'formal_training_started':False})
if __name__=='__main__':main()
