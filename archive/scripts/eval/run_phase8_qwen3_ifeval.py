#!/usr/bin/env python3
"""Full Qwen3-8B non-thinking IFEval teacher-candidate assessment; never trains."""
import json,os,sys,time,hashlib,subprocess,fcntl,traceback
from pathlib import Path
import torch,lm_eval,swanlab
from lm_eval.models.huggingface import HFLM
R=Path('/data/projects/minimind-lab')
MODEL=Path('/data/models/Qwen3-8B')
OUT=Path('/data/artifacts/minimind-lab/phase8-k00-qwen3-8b-ifeval-20260909')
def dump(p,x):
 temp=p.with_suffix(p.suffix+'.tmp')
 temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n');temp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
class TrackedQwen(HFLM):
 def _model_generate(self,context,max_length,stop,**kwargs):
  assert kwargs.get('do_sample') is False,'greedy-only qualification'
  assert max_length-context.shape[1]==1280,'frozen completion cap'
  output=super()._model_generate(context,max_length,stop,**kwargs)
  self.count+=context.shape[0]
  self.lengths.append(int(output.shape[1]-context.shape[1]))
  dump(OUT/'progress.json',{'status':'generating','generated_prompts':self.count,'expected':541,'elapsed_seconds':time.monotonic()-self.started,'last_generated_tokens':self.lengths[-1]})
  if self.count%5==0:
   swanlab.log({'progress/generated_prompts':self.count,'progress/elapsed_seconds':time.monotonic()-self.started},step=self.count)
   print('generated',self.count,'/541',flush=True)
  return output
def main():
 assert not OUT.exists(),'refuse overwrite';OUT.mkdir()
 lock=(OUT/'eval.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu=='6'
 assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<512
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(.85)
 started=time.monotonic();dump(OUT/'progress.json',{'status':'hashing model assets'})
 index=json.loads((MODEL/'model.safetensors.index.json').read_text())
 names=sorted(set(index['weight_map'].values()))+['model.safetensors.index.json','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']
 hashes={n:sha(MODEL/n) for n in names}
 dump(OUT/'model-assets.json',{'model_id':'Qwen/Qwen3-8B','local_path':str(MODEL),'revision':'not recovered from local download; content pinned by per-file SHA256','sha256':hashes,'weight_bytes':sum((MODEL/n).stat().st_size for n in set(index['weight_map'].values()))})
 print('model assets hashed',flush=True)
 dump(OUT/'progress.json',{'status':'loading Qwen3-8B'})
 lm=TrackedQwen(pretrained=str(MODEL),backend='causal',dtype='float16',device='cuda:0',batch_size=1,max_length=32768,
      add_bos_token=False,trust_remote_code=False,enable_thinking=False,chat_template_args={'enable_thinking':False},attn_implementation='sdpa')
 lm.count=0;lm.lengths=[];lm.started=time.monotonic()
 chat=[{'role':'user','content':'只回复数字2。'}]
 rendered=lm.apply_chat_template(chat)
 expected=lm.tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 assert rendered==expected and rendered!=lm.tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=True)
 ids=lm.tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids.to(lm.device)
 with torch.no_grad():
  logits=lm._model_call(ids)
  assert torch.isfinite(logits).all()
 lm.model.generation_config.do_sample=False
 lm.model.generation_config.temperature=1.0
 lm.model.generation_config.top_p=1.0
 lm.model.generation_config.top_k=0
 manifest={'experiment_id':'K00-qwen3-8b-ifeval-20260909','model_id':'Qwen/Qwen3-8B','model_path':str(MODEL),'model_assets_sha256':sha(OUT/'model-assets.json'),
   'gpu':gpu,'dtype':'float16','quantization':None,'task':'ifeval','samples_expected':541,'fewshot':0,'seed':42,'limit':None,
   'apply_chat_template':True,'enable_thinking':False,'rendered_non_thinking_template_verified':True,'do_sample':False,
   'max_new_tokens':1280,'batch_size':1,'max_length':32768,'add_bos_token':False,'lm_eval_version':lm_eval.__version__,
   'eval_source_sha256':sha(Path(__file__)),'lab_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
   'purpose':'instruction-following teacher qualification only; not broad knowledge/reasoning evaluation or distillation',
   'comparison_limits':'each model uses its own tokenizer/chat template; 1280 tokens are not equal character budgets; compare original task documents, not rendered prompt hashes'}
 dump(OUT/'manifest.json',manifest)
 run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name='Phase8-K00-Qwen3-8B-NonThinking-IFEval-20260909',
      group='Phase8-Distillation',job_type='evaluation',mode='online',log_dir=str(OUT/'swanlog'),config=manifest)
 (OUT/'swanlab-url.txt').write_text(run.url+'\n')
 result=lm_eval.simple_evaluate(model=lm,tasks=['ifeval'],num_fewshot=0,batch_size=1,limit=None,apply_chat_template=True,log_samples=True,
      random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42)
 samples=result.pop('samples')['ifeval']
 dump(OUT/'results.json',result)
 (OUT/'samples.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False,default=str)+'\n' for s in samples))
 assert len(samples)==541 and result['n-samples']['ifeval']['effective']==541 and lm.count==541
 assert result['configs']['ifeval']['generation_kwargs']['max_gen_toks']==1280
 assert all(sha(MODEL/n)==v for n,v in hashes.items()),'model assets changed'
 scores={k:v for k,v in result['results']['ifeval'].items() if k.endswith('_acc,none')}
 stats={'generation_length_mean_tokens':sum(lm.lengths)/len(lm.lengths),'at_length_cap':sum(n==1280 for n in lm.lengths)}
 swanlab.log({**{'eval/'+k.split(',')[0]:v for k,v in scores.items()},**{'eval/'+k:v for k,v in stats.items()}},step=542)
 swanlab.finish()
 dump(OUT/'DONE.json',{'status':'completed-instruction-eval-only','samples':541,'scores':scores,**stats,'model_unchanged':True,'total_seconds':time.monotonic()-started,'swanlab_url':run.url,'not_a_distillation_result':True})
 print('COMPLETED',json.dumps(scores),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  if OUT.exists():dump(OUT/f'FAILURE-{time.time_ns()}.json',{'traceback':traceback.format_exc()})
  raise
