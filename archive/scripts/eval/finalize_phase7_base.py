#!/usr/bin/env python3
"""Finalize existing completed inference artifacts; no model inference or retraining."""
from pathlib import Path
import argparse,json,hashlib,statistics
import swanlab
TASKS=['ceval-valid','cmmlu','arc_easy','piqa','openbookqa','hellaswag','social_iqa']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['dense','moe'],required=True);a=ap.parse_args()
 out=Path('/data/artifacts/minimind-lab')/f'phase7-{a.arm}-base-eval-20260909'
 assert not (out/'DONE.json').exists()
 result=json.loads((out/'results.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
 groups=result.get('group_subtasks',{})
 def leaves(t):return sum((leaves(k) for k in groups[t]),[]) if t in groups else [t]
 counts={t:sum(result['n-samples'][k]['effective'] for k in leaves(t)) for t in TASKS}
 assert sum(counts.values())==29638,counts
 compact=[]
 for task in TASKS:
  for leaf in leaves(task):
   f=out/('samples-'+leaf+'.jsonl');rows=[json.loads(l) for l in f.read_text().splitlines()]
   assert len(rows)==result['n-samples'][leaf]['effective'],leaf
   for row in rows:compact.append({'task':leaf,'group':task,'doc_id':row['doc_id'],'doc_hash':row['doc_hash'],'prompt_hash':row['prompt_hash'],'acc':row.get('acc'),'acc_norm':row.get('acc_norm')})
 scores={t:100*result['results'][t].get('acc_norm,none',result['results'][t].get('acc,none')) for t in TASKS}
 assert sha(Path(manifest['checkpoint']))==manifest['checkpoint_sha256']
 url=(out/'swanlab-url.txt').read_text().strip();rid=url.rsplit('/',1)[1]
 run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',id=rid,resume='must',mode='online',log_dir=str(out/'swanlog-finalize'),config={'finalization':'grouped task sample-count aggregation; no inference rerun','finalizer_sha256':sha(Path(__file__))})
 swanlab.log({**{'eval/'+t+'_percent':s for t,s in scores.items()},'eval/macro_percent':statistics.mean(scores.values()),'eval/samples':29638},step=0);swanlab.finish()
 summary={'arm':a.arm,'results_percent':scores,'macro_percent':statistics.mean(scores.values()),'effective_samples':counts,'total_samples':29638,'checkpoint_unchanged':True,'swanlab_url':url,'postprocessing_fix':'aggregate C-Eval/CMMLU leaf counts; original inference artifacts unchanged'}
 (out/'paired-scores.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in compact))
 (out/'DONE.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
