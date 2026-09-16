#!/usr/bin/env python3
"""Validate paired Phase7 outputs and bootstrap questions within each benchmark."""
import json,hashlib,re
from pathlib import Path
import numpy as np
ROOT=Path('/data/artifacts/minimind-lab')
def read(p):return json.loads(p.read_text())
def main():
 paths=[ROOT/f'phase7-{a}-base-eval-20260909' for a in ('dense','moe')]
 manifests=[read(p/'manifest.json') for p in paths]
 results=[read(p/'results.json') for p in paths]
 done=[read(p/'DONE.json') for p in paths]
 fields=['tasks','fewshot','apply_chat_template','batch_size','max_length','seed','limit','tokenizer_sha256','model_source_sha256','lm_eval_version','dtype','add_bos_token']
 for k in fields:assert manifests[0][k]==manifests[1][k],k
 for k in ['versions','n-shot','n-samples','group_subtasks']:assert results[0][k]==results[1][k],k
 # Ignore only process-local addresses in serialized callable reprs; same harness/source and paired prompt hashes checked.
 def canonical(x):return re.sub(r'(<function [^<>]+ at )0x[0-9a-f]+(>)',r'\1ADDRESS\2',json.dumps(x,sort_keys=True))
 assert canonical(results[0]['configs'])==canonical(results[1]['configs']),'configs'
 pairs=[]
 for p in paths:
  rows=[json.loads(x) for x in (p/'paired-scores.jsonl').open()]
  keyed={(r['task'],r['doc_id']):r for r in rows}
  assert len(keyed)==len(rows)==29638
  pairs.append(keyed)
 assert pairs[0].keys()==pairs[1].keys()
 for k in pairs[0]:
  for f in ['doc_hash','prompt_hash','group']:assert pairs[0][k][f]==pairs[1][k][f],(k,f)
 rng=np.random.default_rng(42);boots=[];tasks={}
 for t in manifests[0]['tasks']:
  metric='acc_norm' if 'acc_norm,none' in results[0]['results'][t] else 'acc'
  vals=np.array([pairs[1][k][metric]-pairs[0][k][metric] for k in pairs[0] if pairs[0][k]['group']==t])
  delta=done[1]['results_percent'][t]-done[0]['results_percent'][t]
  assert abs(vals.mean()*100-delta)<1e-8,(t,vals.mean()*100,delta)
  counts=np.array([(vals==-1).sum(),(vals==0).sum(),(vals==1).sum()])
  sample=rng.multinomial(len(vals),counts/len(vals),size=10000)
  boot=(sample[:,2]-sample[:,0])/len(vals)*100;boots.append(boot)
  tasks[t]={'dense_percent':done[0]['results_percent'][t],'moe_percent':done[1]['results_percent'][t],'delta_pp':delta,'n':len(vals),'improved':int(counts[2]),'regressed':int(counts[0]),'paired_bootstrap_95ci_pp':np.quantile(boot,[.025,.975]).tolist()}
 macro=np.mean(boots,axis=0)
 output={'protocol_equal':True,'paired_docs_and_prompts_equal':29638,'resamples':10000,'bootstrap_seed':42,'method':'paired question bootstrap stratified by seven benchmarks; equal-weight benchmark macro; categorical multinomial equivalent resampling','limitations':'exploratory single-training-seed; CI describes question sampling only, not training seed variability; no multiplicity correction','tasks':tasks,'macro_delta_pp':done[1]['macro_percent']-done[0]['macro_percent'],'macro_95ci_pp':np.quantile(macro,[.025,.975]).tolist()}
 out=ROOT/'phase7-base-comparison-20260909';out.mkdir(exist_ok=True)
 dest=out/'comparison.json'
 assert not dest.exists(),'refuse overwrite'
 dest.write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps(output))
if __name__=='__main__':main()
