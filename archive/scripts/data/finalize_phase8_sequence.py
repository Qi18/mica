#!/usr/bin/env python3
"""Symmetric pair filtering: same accepted prompts for original/teacher CE arms."""
import sys,json,collections,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase8_common import *
from transformers import AutoTokenizer
from lm_eval.tasks.ifeval.utils import InputExample,test_instruction_following_strict
def main():
 c=cfg();d=Path(c['data']);a=Path(c['artifacts'])
 assert (a/'GENERATION_DONE.json').exists() and not (d/'ACCEPTED.json').exists()
 frozen=json.loads((d/'prepare-manifest.json').read_text())
 assert all(sha(d/n)==v for n,v in frozen['files'].items()),'frozen prompts changed'
 assert sha(d/'teacher-answers.jsonl')==json.loads((a/'GENERATION_DONE.json').read_text())['sha256']
 raw=rows(d/'teacher-answers.jsonl');candidates=rows(d/'candidates.jsonl')
 generated={r['id']:r for r in raw};assert len(generated)==len(raw)==len(candidates)
 tok=AutoTokenizer.from_pretrained(ROOT/'minimind/model',local_files_only=True)
 accepted=collections.defaultdict(list);reject=collections.Counter();audits=[]
 for row in candidates:
  g=generated[row['id']];answer=g['completion'].strip();reason=None
  if g['finish_reason']!='stop':reason='generation_truncated'
  elif not answer or '<think>' in answer or '</think>' in answer:reason='empty_or_thinking'
  else:
   try:
    ids,labels=encode(tok,row,answer)
    if sum(x!=-100 for x in labels[1:])>c['max_assistant_tokens']:reason='assistant_too_long'
   except (AssertionError,ValueError):reason='student_encoding_or_length'
  if reason is None and row['verifier']:
   v=row['verifier']
   try:
    good=test_instruction_following_strict(InputExample(key=0,instruction_id_list=v['ids'],prompt=row['messages'][0]['content'],kwargs=v['kwargs']),answer).follow_all_instructions
   except Exception:good=False
   if not good:reason='constraint_failed'
  if reason is None and row['kind']=='broad':
   words=answer.split()
   if len(words)>30 and len(set(words))/len(words)<.20:reason='severe_word_repetition'
  if reason:reject[reason]+=1
  else:accepted[row['kind']].append((row,answer))
  audits.append({'id':row['id'],'kind':row['kind'],'accepted':reason is None,'reason':reason})
 write_rows(d/'filter-audit.jsonl',audits)
 dump(d/'filter-summary.json',{'accepted_before_cap':{k:len(v) for k,v in accepted.items()},'rejected':dict(reject),'teacher_is_not_ground_truth':'broad factual quality not programmatically verified; manual review required before promotion'})
 assert all(len(accepted[k])>=2048 for k in ['constraint','broad']),'insufficient accepted paired data; no automatic threshold relaxation'
 selected=accepted['constraint'][:2048]+accepted['broad'][:2048]
 random.Random(c['seed']).shuffle(selected)
 for arm in ['K01','K02']:
  result=[{**r,'completion':r['original'] if arm=='K01' else answer} for r,answer in selected]
  write_rows(d/(arm+'-train.jsonl'),result)
 assert [r['id'] for r in rows(d/'K01-train.jsonl')]==[r['id'] for r in rows(d/'K02-train.jsonl')]
 targets={arm:sum(sum(x!=-100 for x in encode(tok,r)[1][1:]) for r in rows(d/(arm+'-train.jsonl'))) for arm in ['K01','K02']}
 assert all(x<c['target_budget'] for x in targets.values()),'budget must cover both corpora at least once'
 dump(d/'ACCEPTED.json',{'status':'accepted-for-controlled-pilot-not-quality-certified','pairs':4096,'kinds':{'constraint':2048,'broad':2048},'targets_per_pass':targets,'train_prompt_ids_equal':True,'truncation':0,'files':{p.name:sha(p) for p in d.glob('*.jsonl')},'common_config_sha256':sha(CONFIG)})
 print('ACCEPTED',targets,flush=True)
if __name__=='__main__':main()
