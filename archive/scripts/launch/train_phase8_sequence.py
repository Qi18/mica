#!/usr/bin/env python3
"""Token-budget-matched CE continuation vs response distillation; fixed final."""
import sys,os,json,time,math,random,signal,argparse,subprocess,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase8_common import *
import torch
from transformers import AutoTokenizer
sys.path.insert(0,str(ROOT/'minimind'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
class TokenStream:
 def __init__(self,encoded,seed):
  self.encoded=encoded;self.seed=seed;self.epoch=0;self.cursor=0;self.draws=0;self.pending=None
  self.order=list(range(len(encoded)));random.Random(seed).shuffle(self.order)
 def window(self,budget):
  out=[];left=budget
  while left:
   if self.pending is None:
    if self.cursor==len(self.order):
     self.epoch+=1;self.cursor=0;random.Random(self.seed+self.epoch).shuffle(self.order)
    index=self.order[self.cursor];self.cursor+=1;self.draws+=1
    ids,labels=self.encoded[index];self.pending=(ids,list(labels))
   ids,labels=self.pending
   positions=[i for i in range(1,len(labels)) if labels[i]!=-100]
   count=min(left,len(positions));assert count>0
   head=[-100]*len(labels)
   for i in positions[:count]:head[i]=labels[i];labels[i]=-100
   out.append((ids,head));left-=count
   self.pending=(ids,labels) if count<len(positions) else None
  assert sum(sum(v!=-100 for v in lab[1:]) for _,lab in out)==budget
  return out
 def state(self):return {'epoch':self.epoch,'cursor':self.cursor,'draws':self.draws,'order':self.order,'pending':self.pending}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['K01','K02'],required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 c=cfg();d=Path(c['data']);root=Path(c['artifacts']);out=root/(('smoke-' if a.smoke else 'train-')+a.arm)
 assert not out.exists(),'refuse overwrite';out.mkdir()
 lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu=str(c['gpu_'+a.arm]);assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
 assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<512
 assert sha(c['student'])==c['student_sha256']
 tok=AutoTokenizer.from_pretrained(ROOT/'minimind/model',local_files_only=True)
 if a.smoke:
  source=rows(d/'candidates.jsonl')[:32];source=[{**r,'completion':r['original']} for r in source]
  dataset_proof={'smoke_fixture':'first32 original candidates; not actual K02 teacher data'}
 else:
  accepted=json.loads((d/'ACCEPTED.json').read_text());assert accepted['pairs']==4096
  assert all(sha(d/name)==digest for name,digest in accepted['files'].items())
  source=rows(d/(a.arm+'-train.jsonl'));dataset_proof=accepted
 enc=[encode(tok,r) for r in source]
 val=[encode(tok,r,r['original']) for r in rows(d/'validation.jsonl')[:(16 if a.smoke else 512)]]
 torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.40);torch.cuda.reset_peak_memory_stats()
 model=MiniMindForCausalLM(MiniMindConfig())
 model.load_state_dict(torch.load(c['student'],map_location='cpu',weights_only=True),strict=True);model=model.float().cuda()
 model.eval();short=min(val,key=lambda x:len(x[0]));long=max(val,key=lambda x:len(x[0]))
 with torch.inference_mode():
  one=collate(tok,[short],'cuda');pair=collate(tok,[short,long],'cuda');pair['labels'][1]=-100
  l1=model(**one).loss;l2=model(**pair).loss
  assert torch.allclose(l1,l2,atol=1e-4,rtol=1e-4),(l1,l2)
 dump(out/'alignment-check.json',{'passed':True,'single_nll':l1.item(),'right_padded_nll':l2.item(),'label_shift':'labels[:,1:]','eos_supervised':True,'truncation':0})
 del one,pair
 provenance={'config':c,'arm':a.arm,'smoke':a.smoke,'dataset':dataset_proof,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/phase8_common.py',CONFIG]},'student_sha256':c['student_sha256'],'checkpoint_selection':'fixed-final'}
 dump(out/'manifest.json',provenance)
 import swanlab
 run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name=f'Phase8-{a.arm}-'+('smoke' if a.smoke else ('CE-control-v1' if a.arm=='K01' else 'Qwen3-8B-sequence-distill-v1')),group='Phase8-Sequence-Distillation',job_type='training-smoke' if a.smoke else 'training',mode='online',log_dir=str(out/'swanlog'),config=provenance)
 (out/'swanlab-url.txt').write_text(run.url+'\n')
 optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
 stream=TokenStream(enc,c['seed']);steps=2 if a.smoke else c['target_budget']//c['targets_per_update']
 total_targets=0;forward_rows=0;start=time.monotonic();stop=[False]
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
 def validate():
  model.eval();loss_sum=0.;count=0
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for i in range(0,len(val),c['micro_batch']):
    b=collate(tok,val[i:i+c['micro_batch']],'cuda');n=int((b['labels'][:,1:]!=-100).sum())
    loss=model(**b).loss;assert torch.isfinite(loss);loss_sum+=loss.item()*n;count+=n
  return loss_sum/count
 def save(name,value):
  temp=out/(name+'.tmp');torch.save(value,temp);temp.replace(out/name)
 initial=validate();swanlab.log({'validation/original_reference_nll':initial},step=0)
 for step in range(1,steps+1):
  window=stream.window(c['targets_per_update']);warmup=max(1,int(steps*.05))
  scale=step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
  lr=c['learning_rate']*scale
  for group in optimizer.param_groups:group['lr']=lr
  model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;tick=time.monotonic()
  for i in range(0,len(window),c['micro_batch']):
   chunk=window[i:i+c['micro_batch']];b=collate(tok,chunk,'cuda');n=int((b['labels'][:,1:]!=-100).sum())
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**b).loss
   assert torch.isfinite(loss);(loss*n/c['targets_per_update']).backward()
   loss_sum+=loss.item()*n/c['targets_per_update'];forward_rows+=len(chunk)
  grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip']);assert torch.isfinite(grad)
  optimizer.step();total_targets+=c['targets_per_update']
  metrics={'step':step,'train/ce':loss_sum,'train/target_tokens':total_targets,'train/learning_rate':lr,'train/grad_norm':float(grad),'data/epoch':stream.epoch,'data/row_draws':stream.draws,'data/forward_rows':forward_rows,'system/update_seconds':time.monotonic()-tick,'system/elapsed_seconds':time.monotonic()-start,'system/peak_memory_mib':torch.cuda.max_memory_allocated()/2**20}
  if step%c['eval_every']==0 or step==steps:metrics['validation/original_reference_nll']=validate()
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
  swanlab.log(metrics,step=step)
  if step%10==0 or step==steps:print(json.dumps(metrics),flush=True)
  if step%c['eval_every']==0 or step==steps or stop[0]:
   save('resume.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'target_tokens':total_targets,'sampler':stream.state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'manifest':provenance})
  if stop[0]:
   dump(out/'STOPPED.json',{'step':step,'target_tokens':total_targets,'resume_equivalence_verified':False});swanlab.finish();return
 save('final.pth',{k:v.detach().cpu() for k,v in model.state_dict().items()})
 state=torch.load(out/'final.pth',map_location='cpu',weights_only=True)
 reloaded=MiniMindForCausalLM(MiniMindConfig());reloaded.load_state_dict(state,strict=True)
 assert all(torch.equal(v,state[k]) for k,v in reloaded.state_dict().items())
 assert all(torch.isfinite(v).all() for v in state.values())
 assert total_targets==steps*c['targets_per_update']
 assert sha(c['student'])==c['student_sha256']
 swanlab.finish()
 dump(out/'DONE.json',{'status':'smoke-passed' if a.smoke else 'trained-not-promoted','arm':a.arm,'steps':steps,'target_tokens':total_targets,'checkpoint':str(out/'final.pth'),'checkpoint_sha256':sha(out/'final.pth'),'strict_load':True,'source_checkpoint_unchanged':True,'wall_seconds':time.monotonic()-start,'row_draws':stream.draws,'forward_rows':forward_rows,'swanlab_url':run.url,'automatic_promotion':False})
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  c=cfg();path=Path(c['artifacts'])/('training-error-'+str(time.time_ns())+'.json');dump(path,{'traceback':traceback.format_exc()});raise
