#!/usr/bin/env python3
"""Offline vLLM Qwen3-8B answer generation. Teacher never sees reference answers."""
import sys,os,json,time,subprocess,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase8_common import *
def main():
 c=cfg();d=Path(c['data']);a=Path(c['artifacts']);a.mkdir(parents=True,exist_ok=True)
 assert (d/'prepare-manifest.json').exists()
 lock=(a/'teacher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
 assert int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<512
 assets=json.loads(Path(c['teacher_assets']).read_text())
 assert all(sha(Path(c['teacher'])/name)==value for name,value in assets['sha256'].items())
 import vllm
 from vllm import LLM,SamplingParams
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(c['teacher'],local_files_only=True)
 dataset=rows(d/'candidates.jsonl')
 output=d/'teacher-answers.jsonl';assert not output.exists()
 llm=LLM(model=c['teacher'],dtype='float16',tensor_parallel_size=1,gpu_memory_utilization=.70,max_model_len=4096,max_num_seqs=64,enforce_eager=True,enable_prefix_caching=True,trust_remote_code=False,seed=c['seed'])
 dump(a/'generation-config.json',{'config':c,'vllm_version':vllm.__version__,'generation_source_sha256':sha(Path(__file__)),'max_new_tokens':512,'do_sample':False,'enable_thinking':False,'inference_engine_differs_from_qualification':True,'swanlab':'backfill from independent minimind runtime after generation'})
 start=time.monotonic()
 with output.open('x') as f:
  for i in range(0,len(dataset),128):
   batch=dataset[i:i+128]
   prompts=[tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
   assert all(len(tok.encode(p))<3500 for p in prompts)
   generated=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=512,seed=c['seed']),use_tqdm=False)
   for row,result in zip(batch,generated):
    text=result.outputs[0]
    f.write(json.dumps({'id':row['id'],'completion':text.text,'finish_reason':text.finish_reason,'generated_tokens':len(text.token_ids),'prompt_token_count':len(result.prompt_token_ids)},ensure_ascii=False)+'\n')
   f.flush()
   metrics={'generated':min(i+128,len(dataset)),'total':len(dataset),'elapsed_seconds':time.monotonic()-start}
   dump(a/'generation-progress.json',metrics)
   with (a/'generation-metrics.jsonl').open('a') as log:log.write(json.dumps(metrics)+'\n')
   print(json.dumps(metrics),flush=True)
 dump(a/'GENERATION_DONE.json',{'rows':len(dataset),'sha256':sha(output),'swanlab_status':'pending_backfill','model_assets_sha256':sha(c['teacher_assets'])})
if __name__=='__main__':main()
