#!/usr/bin/env python3
"""Freeze paired candidates and independent held-outs before teacher generation."""
import sys,json,random,hashlib,importlib.util,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase8_common import *
from transformers import AutoTokenizer
from datasets import load_dataset
def main():
 c=cfg();d=Path(c['data']);assert not d.exists();d.mkdir(parents=True)
 tok=AutoTokenizer.from_pretrained(ROOT/'minimind/model',local_files_only=True)
 spec=importlib.util.spec_from_file_location('curriculum',ROOT/'scripts/data/sft/build_ifeval_curriculum_v4.py')
 cur=importlib.util.module_from_spec(spec);spec.loader.exec_module(cur)
 from eval.eval_sft_behavior import CHAT_CASES,TOOL_CASES
 blocked={norm(r['prompt']) for r in load_dataset('google/IFEval',split='train')}
 blocked.update(norm(r['prompt']) for r in CHAT_CASES+TOOL_CASES)
 blocked.update(norm(r['conversations'][0]['content']) for n in ['validation','test'] for r in rows(Path(c['s10_data'])/(n+'.jsonl')) if r['conversations'][0]['role']=='user')
 seen=set();capture={};original_verify=cur.verify
 def verify(prompt,answer,ids,kwargs):
  original_verify(prompt,answer,ids,kwargs);capture.update(ids=ids,kwargs=kwargs)
 cur.verify=verify
 def record(messages,answer,kind,meta=None):
  prompt=messages[0]['content'];key=norm(prompt)
  if key in blocked or key in seen:return None
  obj={'id':hashlib.sha256(prompt.encode()).hexdigest(),'kind':kind,'messages':messages,'original':answer,'verifier':meta}
  try:
   ids,lab=encode(tok,obj,answer)
   if sum(x!=-100 for x in lab[1:])>c['max_assistant_tokens']:return None
  except (AssertionError,ValueError):return None
  seen.add(key);return obj
 splits={}
 for split,want,offset in [('validation',256,3000000),('test',256,4000000),('train',4096,1000000)]:
  selected=[];i=0
  while len(selected)<want:
   assert i<30000,'candidate exhaustion'
   family=cur.SINGLE_IDS[i%len(cur.SINGLE_IDS)];capture.clear()
   try:raw,_=cur.single_example(family,split,offset+i)
   except ValueError:i+=1;continue
   conv=raw['conversations'];obj=record([{'role':'user','content':conv[0]['content']}],conv[-1]['content'],'constraint',dict(capture))
   if obj:selected.append(obj)
   i+=1
  broad=rows(Path(c['broad_data'])/(split+'.jsonl'));random.Random(c['seed']+offset).shuffle(broad)
  count=0
  for raw in broad:
   conv=raw.get('conversations',[])
   if len(conv)!=2 or [x['role'] for x in conv]!=['user','assistant']:continue
   if any(x.get('tools') or x.get('tool_calls') or x.get('reasoning_content') for x in conv):continue
   obj=record([{'role':'user','content':conv[0]['content']}],conv[-1]['content'],'broad')
   if obj:selected.append(obj);count+=1
   if count==want:break
  assert count==want,(split,count,want)
  random.Random(c['seed']+offset).shuffle(selected)
  splits[split]=selected
  write_rows(d/('candidates.jsonl' if split=='train' else split+'.jsonl'),selected)
 sets={s:{norm(x['messages'][0]['content']) for x in rs} for s,rs in splits.items()}
 assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['test']&sets['validation'])
 assert all(not (s&blocked) for s in sets.values())
 dump(d/'prepare-manifest.json',{'status':'prompts-frozen','seed':c['seed'],'counts':{s:len(r) for s,r in splits.items()},'cross_split_normalized_overlap':0,'frozen_ifeval_behavior_and_s10_heldout_normalized_overlap':0,'near_duplicate_limit':'exact and NFKC/alphanumeric normalized scan only; same verifier family intentionally reused','training_prior_exposure':'broad replay may have appeared in S10; newly numbered constraint prompts not copied from IFEval','files':{p.name:sha(p) for p in d.glob('*.jsonl')},'source_sha':{'builder':sha(Path(__file__)),'curriculum':sha(ROOT/'scripts/data/sft/build_ifeval_curriculum_v4.py'),'broad_train':sha(Path(c['broad_data'])/'train.jsonl')},'broad_answers_verified':'structural only, no claim all factual answers correct'})
 print('PREPARED',len(splits['train']),flush=True)
if __name__=='__main__':main()
