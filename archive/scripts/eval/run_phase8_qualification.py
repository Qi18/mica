#!/usr/bin/env python3
"""K00 native paired IFEval qualification. No training or automatic promotion."""
import argparse,os,sys,json,time,hashlib,subprocess,fcntl,traceback
from pathlib import Path
import torch,swanlab,lm_eval
from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
R=Path('/data/projects/minimind-lab');sys.path.insert(0,str(R/'minimind'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
ROOT=Path('/data/artifacts/minimind-lab/phase8-k00-teacher-gate-20260909')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
class NativeGreedy(HFLM):
 def _model_generate(self,context,max_length,stop,**kwargs):
  attention=kwargs.get('attention_mask')
  out=self.model.generate(input_ids=context,attention_mask=attention,max_new_tokens=max_length-context.shape[1],
       temperature=1.0,top_p=1.0,top_k=0,do_sample=False,eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
  self.done_count+=len(context)
  dump(self.outdir/'progress.json',{'generated_prompts':self.done_count,'expected':541,'elapsed_seconds':time.monotonic()-self.started})
  if self.done_count%10==0:
   swanlab.log({'progress/generated_prompts':self.done_count,'progress/elapsed_seconds':time.monotonic()-self.started},step=self.done_count)
   print('generated',self.done_count,'/ 541',flush=True)
  return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['student','teacher'],required=True);a=ap.parse_args()
 compatible=json.loads((ROOT/'compatibility.json').read_text());assert compatible['status']=='compatibility-passed-not-quality-qualified'
 arm=compatible['arms'][a.arm];ck=Path(arm['checkpoint']);assert sha(ck)==arm['sha256']
 out=ROOT/('ifeval-'+a.arm);assert not out.exists(),'refuse overwrite';out.mkdir()
 lock=(out/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu in ['4','5']
 used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<512
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(.35)
 model=MiniMindForCausalLM(MiniMindConfig(use_moe=a.arm=='teacher'))
 model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True);model=model.half().cuda().eval()
 tok=AutoTokenizer.from_pretrained(R/'minimind/model',local_files_only=True)
 lm=NativeGreedy(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,max_length=32768,add_bos_token=False)
 lm.outdir=out;lm.done_count=0;lm.started=time.monotonic()
 ids=tok('中国的首都是北京。',return_tensors='pt',add_special_tokens=False).input_ids.cuda()
 with torch.no_grad():
  logits=model(ids).logits;assert torch.isfinite(logits).all() and torch.equal(logits,lm._model_call(ids))
 manifest={'arm':a.arm,**arm,'model_class':'MiniMindForCausalLM native','adapter':'NativeGreedy HFLM; native greedy temperature1 avoids division by0; no sampling','dtype':'float16','fewshot':0,'task':'ifeval','n':541,'apply_chat_template':True,'max_new_tokens':1280,'batch_size':1,'max_length':32768,'seed':42,'limit':None,'add_bos_token':False,'tokenizer_sha256':sha(R/'minimind/model/tokenizer.json'),'tokenizer_config_sha256':sha(R/'minimind/model/tokenizer_config.json'),'eval_source_sha256':sha(Path(__file__)),'model_source_sha256':sha(R/'minimind/model/model_minimind.py'),'lm_eval_version':lm_eval.__version__,'gpu':gpu,'generation':'greedy argmax, top_p1, top_k0, EOS stop; harness handles decoded stop strings','preregister_sha256':sha(R/'docs/phases/phase8-distill.md')}
 dump(out/'manifest.json',manifest)
 run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name=f'Phase8-K00-{a.arm}-IFEval-qualification-20260909',group='Phase8-Distillation',job_type='evaluation',mode='online',log_dir=str(out/'swanlog'),config=manifest)
 (out/'swanlab-url.txt').write_text(run.url+'\n')
 try:
  result=lm_eval.simple_evaluate(model=lm,tasks=['ifeval'],num_fewshot=0,batch_size=1,limit=None,apply_chat_template=True,log_samples=True,random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42)
  samples=result.pop('samples')['ifeval']
  assert len(samples)==541 and result['n-samples']['ifeval']['effective']==541 and lm.done_count==541
  dump(out/'results.json',result)
  (out/'samples.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False,default=str)+'\n' for s in samples))
  assert sha(ck)==arm['sha256']
  scores={k:v for k,v in result['results']['ifeval'].items() if k.endswith('_acc,none')}
  swanlab.log({'eval/'+k.split(',')[0]:v for k,v in scores.items()},step=542);swanlab.finish()
  dump(out/'DONE.json',{'status':'evaluation-completed-not-teacher-qualified','arm':a.arm,'n':541,'scores':scores,'wall_seconds':time.monotonic()-lm.started,'checkpoint_unchanged':True,'swanlab_url':run.url})
  print('completed',a.arm,scores,flush=True)
 except Exception:
  dump(out/'FAILURE.json',{'traceback':traceback.format_exc()});swanlab.finish();raise
if __name__=='__main__':main()
