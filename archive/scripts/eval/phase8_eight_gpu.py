import os,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from run_phase8_qualification import NativeGreedy
from phase8_common import ROOT,dump
LM=None
OUT=None
GPU=None
class WorkerLM(NativeGreedy):
 def _model_generate(self,context,max_length,stop,**kwargs):
  out=self.model.generate(input_ids=context,attention_mask=kwargs.get('attention_mask'),max_new_tokens=max_length-context.shape[1],temperature=1.,top_p=1.,top_k=0,do_sample=False,eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
  self.done_count+=len(context)
  dump(OUT/('worker-'+str(GPU)+'-progress.json'),{'gpu':GPU,'generated':self.done_count,'unix_time':time.time()})
  return out

def initialize(queue,checkpoint,out):
 global LM,OUT,GPU
 GPU=queue.get();OUT=Path(out)
 import torch
 from transformers import AutoTokenizer
 import sys
 sys.path.insert(0,str(ROOT/'minimind'))
 from model.model_minimind import MiniMindForCausalLM,MiniMindConfig
 torch.set_num_threads(2);torch.cuda.set_device(GPU);torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(.35,device=GPU)
 model=MiniMindForCausalLM(MiniMindConfig())
 model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
 model=model.half().to('cuda:'+str(GPU)).eval();assert model.device.index==GPU
 tok=AutoTokenizer.from_pretrained(ROOT/'minimind/model',local_files_only=True)
 LM=WorkerLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,max_length=32768,add_bos_token=False)
 LM.done_count=0
 dump(OUT/('worker-'+str(GPU)+'-ready.json'),{'gpu':GPU,'pid':os.getpid(),'checkpoint':checkpoint,'actual_cuda_device':str(model.device)})

def execute(method,requests,batch):
 LM.batch_size_per_gpu=batch
 answers=getattr(LM,method)(requests)
 assert len(answers)==len(requests)
 return answers

class EightGPU(NativeGreedy):
 def start_pool(self,checkpoint,out):
  self.pool_out=Path(out);context=mp.get_context('spawn');queue=context.Queue()
  for gpu in range(8):queue.put(gpu)
  self.pool=ProcessPoolExecutor(max_workers=8,mp_context=context,initializer=initialize,initargs=(queue,str(checkpoint),str(out)))
 def dispatch(self,method,requests,batch):
  requests=list(requests);n=len(requests)
  shards=[list(range(i,n,8)) for i in range(8)]
  flat=[j for s in shards for j in s];assert sorted(flat)==list(range(n)) and len(set(flat))==n
  jobs=[(ix,self.pool.submit(execute,method,[requests[j] for j in ix],batch)) for ix in shards if ix]
  result=[None]*n
  for ix,f in jobs:
   values=f.result()
   for j,v in zip(ix,values):result[j]=v
  assert all(v is not None for v in result)
  if method=='generate_until':self.done_count=n
  dump(self.pool_out/(method+'-coverage.json'),{'requests':n,'shard_counts':[len(s) for s in shards],'unique_coverage':True,'order_restored':True,'per_worker_batch':batch})
  return result
 def generate_until(self,requests,**kwargs):return self.dispatch('generate_until',requests,1)
 def loglikelihood(self,requests,**kwargs):return self.dispatch('loglikelihood',requests,16)
 def loglikelihood_rolling(self,requests,**kwargs):return self.dispatch('loglikelihood_rolling',requests,16)
 def close_pool(self):self.pool.shutdown(wait=True)
