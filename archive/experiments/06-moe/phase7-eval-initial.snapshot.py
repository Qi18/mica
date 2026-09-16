#!/usr/bin/env python3
"""Native MiniMind Dense/MoE fixed-final Base evaluation, same HFLM adapter."""
import argparse,os,sys,json,time,hashlib,subprocess,fcntl,traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
import lm_eval
R=Path('/data/projects/minimind-lab');sys.path.insert(0,str(R/'minimind'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
TASKS=['ceval-valid','cmmlu','arc_easy','piqa','openbookqa','hellaswag','social_iqa']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--arm',choices=['dense','moe'],required=True);args=a.parse_args()
 arm=args.arm;out=Path('/data/artifacts/minimind-lab')/f'phase7-{arm}-base-eval-20260909'
 assert not out.exists(),'refuse overwrite';out.mkdir()
 lock=(out/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu in ['4','5']
 used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<512
 torch.set_num_threads(4);torch.manual_seed(42)
 ck=Path('/data/artifacts/minimind-lab/phase7-formal-r3-20260908')/(arm+'-formal')/'weights'/('model_768.pth' if arm=='dense' else 'model_768_moe.pth')
 fingerprint=sha(ck)
 model=MiniMindForCausalLM(MiniMindConfig(use_moe=arm=='moe'))
 model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
 model=model.half().cuda().eval()
 tok=AutoTokenizer.from_pretrained(str(R/'minimind/model'),local_files_only=True)
 lm=HFLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=16,max_length=32768,add_bos_token=False)
 ids=tok('中国的首都是北京。',return_tensors='pt',add_special_tokens=False).input_ids.cuda()
 with torch.no_grad():
  native=model(ids).logits;adapted=lm._model_call(ids)
  assert torch.isfinite(native).all() and torch.equal(native,adapted)
 manifest={'arm':arm,'checkpoint':str(ck),'checkpoint_sha256':fingerprint,'selection':'fixed final epoch-end, not best validation','model_class':'MiniMindForCausalLM native','use_moe':arm=='moe','parameters':sum(p.numel() for p in model.parameters()),'dtype':str(model.dtype),'tasks':TASKS,'fewshot':0,'apply_chat_template':False,'batch_size':16,'max_length':32768,'seed':42,'limit':None,'tokenizer_sha256':sha(R/'minimind/model/tokenizer.json'),'model_source_sha256':sha(R/'minimind/model/model_minimind.py'),'eval_source_sha256':sha(Path(__file__)),'lm_eval_version':lm_eval.__version__,'gpu':gpu,'adapter_logits_exact':True,'add_bos_token':False}
 dump(out/'manifest.json',manifest)
 import swanlab
 run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name=f'Phase7-M0{1 if arm=="dense" else 2}-{arm.upper()}-Final-Base-Seven-20260909',group='Phase7-Dense-MoE',job_type='evaluation',mode='online',log_dir=str(out/'swanlog'),config=manifest)
 (out/'swanlab-url.txt').write_text(run.url+'\n');started=time.time()
 try:
  result=lm_eval.simple_evaluate(model=lm,tasks=TASKS,num_fewshot=0,batch_size=16,limit=None,apply_chat_template=False,log_samples=True,random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42,bootstrap_iters=10000)
  samples=result.pop('samples')
  for task,rows in samples.items():
   with (out/('samples-'+task+'.jsonl')).open('x') as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
  dump(out/'results.json',result)
  scores={t:100*result['results'][t].get('acc_norm,none',result['results'][t].get('acc,none')) for t in TASKS}
  counts={t:result['n-samples'][t]['effective'] for t in TASKS}
  assert sum(counts.values())==29638,counts
  assert sha(ck)==fingerprint
  summary={'arm':arm,'results_percent':scores,'macro_percent':sum(scores.values())/7,'effective_samples':counts,'total_samples':sum(counts.values()),'wall_seconds':time.time()-started,'checkpoint_unchanged':True,'swanlab_url':run.url}
  swanlab.log({**{'eval/'+k+'_percent':v for k,v in scores.items()},'eval/macro_percent':summary['macro_percent'],'eval/samples':summary['total_samples']},step=0);swanlab.finish()
  dump(out/'DONE.json',summary);print(json.dumps(summary),flush=True)
 except BaseException as e:
  dump(out/'FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()});swanlab.finish(state='crashed');raise
if __name__=='__main__':main()
