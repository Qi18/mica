#!/usr/bin/env python3
"""Frozen native evaluation for S10/K01/K02; no checkpoint selection or promotion."""
import sys,os,argparse,json,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase8_common import *
import torch,swanlab,lm_eval
from transformers import AutoTokenizer
from run_phase8_qualification import NativeGreedy
from eval_sft_behavior import CHAT_CASES,TOOL_CASES,score_chat,run_tool_case
from lm_eval.tasks.ifeval.utils import InputExample,test_instruction_following_strict
sys.path.insert(0,str(ROOT/'minimind'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
TASKS=['ceval-valid','cmmlu','arc_easy','piqa','openbookqa','hellaswag','social_iqa']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['S10','K01','K02'],required=True);a=ap.parse_args()
 c=cfg();root=Path(c['artifacts']);out=root/('eval-'+a.arm);assert not out.exists();out.mkdir()
 data=Path(c['data']);accepted=json.loads((data/'ACCEPTED.json').read_text())
 assert all(sha(data/n)==v for n,v in accepted['files'].items()),'accepted dataset changed'
 gpu='7' if a.arm=='S10' else str(c['gpu_'+a.arm]);assert os.environ['CUDA_VISIBLE_DEVICES']==gpu
 assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<512
 if a.arm=='S10':ck=Path(c['student']);fingerprint=c['student_sha256']
 else:
  proof=json.loads((root/('train-'+a.arm)/'DONE.json').read_text());assert proof['target_tokens']==c['target_budget']
  ck=Path(proof['checkpoint']);fingerprint=proof['checkpoint_sha256']
 assert sha(ck)==fingerprint
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(.35)
 model=MiniMindForCausalLM(MiniMindConfig());model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True);model=model.half().cuda().eval()
 tok=AutoTokenizer.from_pretrained(ROOT/'minimind/model',local_files_only=True)
 lm=NativeGreedy(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,max_length=32768,add_bos_token=False)
 manifest={'arm':a.arm,'checkpoint':str(ck),'checkpoint_sha256':fingerprint,'dtype':'float16','apply_chat_template':True,'fewshot':0,'seed':42,'max_gen_toks_ifeval':1280,'max_gen_toks_new_constraints':512,'max_gen_toks_behavior':192,'source_sha256':sha(Path(__file__)),'native_generator_source_sha256':sha(ROOT/'scripts/eval/run_phase8_qualification.py'),'model_source_sha256':sha(ROOT/'minimind/model/model_minimind.py'),'tokenizer_sha256':sha(ROOT/'minimind/model/tokenizer.json'),'config_sha256':sha(CONFIG),'lm_eval_version':lm_eval.__version__,'selection':'fixed-final','gpu':gpu}
 dump(out/'manifest.json',manifest)
 run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name=f'Phase8-{a.arm}-sequence-v1-frozen-eval',group='Phase8-Sequence-Distillation',job_type='evaluation',mode='online',log_dir=str(out/'swanlog'),config=manifest)
 (out/'swanlab-url.txt').write_text(run.url+'\n')
 def generate(messages,cap):
  text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=False)
  batch=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda');batch.pop('token_type_ids',None)
  with torch.inference_mode():result=model.generate(**batch,max_new_tokens=cap,temperature=1,top_p=1,top_k=0,do_sample=False,eos_token_id=tok.eos_token_id)
  return tok.decode(result[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
 cache=root/'failed-attempt-1'/('eval-'+a.arm)
 required=['constraint-samples.jsonl','broad-manual-review.jsonl','behavior-samples.jsonl','behavior.json']
 if all((cache/n).exists() for n in required):
  old=json.loads((cache/'manifest.json').read_text())
  for k in ['checkpoint_sha256','dtype','apply_chat_template','fewshot','seed','max_gen_toks_new_constraints','max_gen_toks_behavior','native_generator_source_sha256','model_source_sha256','tokenizer_sha256','config_sha256','lm_eval_version']:
   assert old[k]==manifest[k],('cache protocol mismatch',k)
  constraint=rows(cache/'constraint-samples.jsonl')
  expected={r['id'] for r in rows(data/'test.jsonl') if r['verifier']}
  assert len(constraint)==256 and {r['id'] for r in constraint}==expected
  constraint_acc=sum(r['passed'] for r in constraint)/256
  behavior=json.loads((cache/'behavior.json').read_text())
  for n in required:
   (out/n).write_bytes((cache/n).read_bytes())
  dump(out/'reuse-behavior-receipt.json',{'source':str(cache),'protocol_verified':True,'files':{n:sha(cache/n) for n in required}})
 else:
  test=rows(Path(c['data'])/'test.jsonl');constraint=[];broad=[]
  for r in test:
   if r['verifier']:
    answer=generate(r['messages'],512);v=r['verifier']
    score=test_instruction_following_strict(InputExample(key=0,instruction_id_list=v['ids'],prompt=r['messages'][0]['content'],kwargs=v['kwargs']),answer)
    constraint.append({'id':r['id'],'response':answer,'passed':bool(score.follow_all_instructions),'per_instruction':score.follow_instruction_list})
    if len(constraint)%32==0:dump(out/'progress.json',{'phase':'new_constraint_test','done':len(constraint),'total':256})
   elif len(broad)<32:broad.append({'id':r['id'],'prompt':r['messages'][0]['content'],'original':r['original'],'response':generate(r['messages'],512)})
  assert len(constraint)==256
  write_rows(out/'constraint-samples.jsonl',constraint);write_rows(out/'broad-manual-review.jsonl',broad)
  constraint_acc=sum(r['passed'] for r in constraint)/256
  chat=[]
  for case in CHAT_CASES:
   answer=generate([{'role':'user','content':case['prompt']}],192)
   success,fmt,refusal,repeated,ratio,clean=score_chat(case,answer)
   chat.append({'id':case['id'],'response':answer,'success':success,'format_pass':fmt,'repeated':repeated})
  tools=[run_tool_case(model,tok,case,'cuda:0',192) for case in TOOL_CASES]
  write_rows(out/'behavior-samples.jsonl',chat+tools)
  behavior={'chat_success':sum(x['success'] for x in chat),'format_success':sum(x['format_pass'] for x,k in zip(chat,CHAT_CASES) if k.get('format')),'repetition_count':sum(x['repeated'] for x in chat),'tool_success':sum(x['end_to_end_pass'] for x in tools)}
  dump(out/'behavior.json',behavior)
 ifdir=out/'ifeval';ifdir.mkdir()
 if a.arm=='S10':
  cached=Path('/data/artifacts/minimind-lab/phase8-k00-teacher-gate-20260909/ifeval-student')
  m=json.loads((cached/'manifest.json').read_text());done=json.loads((cached/'DONE.json').read_text())
  assert m['sha256']==fingerprint and m['model_source_sha256']==manifest['model_source_sha256']
  assert m['eval_source_sha256']==manifest['native_generator_source_sha256'] and m['tokenizer_sha256']==manifest['tokenizer_sha256']
  assert m['batch_size']==1 and m['max_new_tokens']==1280 and m['apply_chat_template'] is True and done['n']==541
  result=json.loads((cached/'results.json').read_text());samples=rows(cached/'samples.jsonl')
  dump(ifdir/'reuse-receipt.json',{'source':str(cached),'protocol_verified':True,'source_manifest_sha256':sha(cached/'manifest.json')})
 else:
  lm.outdir=ifdir;lm.done_count=0;lm.started=time.monotonic()
  result=lm_eval.simple_evaluate(model=lm,tasks=['ifeval'],num_fewshot=0,batch_size=1,apply_chat_template=True,limit=None,log_samples=True,random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42)
  samples=result.pop('samples')['ifeval'];assert lm.done_count==541
 assert len(samples)==541 and result['n-samples']['ifeval']['effective']==541
 write_rows(ifdir/'samples.jsonl',samples);dump(ifdir/'results.json',result)
 ifscores={k:v for k,v in result['results']['ifeval'].items() if k.endswith('_acc,none')}
 dump(out/'progress.json',{'phase':'seven_benchmarks','status':'running'})
 result=lm_eval.simple_evaluate(model=lm,tasks=TASKS,num_fewshot=0,batch_size=16,apply_chat_template=True,limit=None,log_samples=True,random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42)
 seven=out/'seven';seven.mkdir();samples=result.pop('samples')
 for name,rs in samples.items():write_rows(seven/('samples-'+name+'.jsonl'),rs)
 groups=result.get('group_subtasks',{})
 def leaves(t):return sum((leaves(k) for k in groups[t]),[]) if t in groups else [t]
 counts={t:sum(result['n-samples'][x]['effective'] for x in leaves(t)) for t in TASKS};assert sum(counts.values())==29638
 scores={t:100*result['results'][t].get('acc_norm,none',result['results'][t].get('acc,none')) for t in TASKS}
 dump(seven/'results.json',result);macro=sum(scores.values())/7
 assert sha(ck)==fingerprint
 metrics={'eval/new_constraint_strict':constraint_acc,'eval/seven_macro_percent':macro,**{'eval/'+k.split(',')[0]:v for k,v in ifscores.items()},**{'eval/'+k:v for k,v in behavior.items()}}
 swanlab.log(metrics,step=1000);swanlab.finish()
 dump(out/'DONE.json',{'status':'evaluated-not-promoted','arm':a.arm,'new_constraint_strict':constraint_acc,'ifeval':ifscores,'seven':scores,'seven_macro_percent':macro,'behavior':behavior,'checkpoint_unchanged':True,'swanlab_url':run.url,'manual_broad_review_pending':True})
 print('EVAL_DONE',a.arm,metrics,flush=True)
if __name__=='__main__':main()
