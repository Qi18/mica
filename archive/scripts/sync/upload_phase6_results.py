#!/usr/bin/env python3
"""Explicitly authorized historical metric backfill; never trains or uploads samples."""
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/data/artifacts/minimind-lab')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def flat(x,p=''):
 out={}
 for k,v in x.items():
  n=p+k
  if isinstance(v,dict):out.update(flat(v,n+'/'))
  elif isinstance(v,(int,float)) and not isinstance(v,bool):out[n]=v
  elif isinstance(v,list) and len(v)==2 and all(isinstance(i,(int,float)) for i in v):out[n+'/lower'],out[n+'/upper']=v
 return out
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--authorized-upload',action='store_true',required=True);parser.parse_args()
 import swanlab
 rd=ROOT/'phase6-agentic-rl-v2-summary-20260908';gd=ROOT/'phase6-general-comparison-20260908'
 rl=read(rd/'summary.json');g=read(gd/'summary.json');metrics={}
 for a,x in rl['arms'].items():
  for d in ('graph','tools'):metrics[a+'/test_'+d+'_e2e']=x['test'][d]['end_to_end_success']
  metrics[a+'/legacy_chat']=x['validation']['behavior']['chat']['success_rate']
  metrics[a+'/legacy_tool']=x['validation']['behavior']['tool']['end_to_end_success_rate']
  metrics[a+'/test_combined_e2e']=sum(x['test'][d]['end_to_end_success']*x['test'][d]['tasks'] for d in ('graph','tools'))/480
 metrics.update(flat(rl['paired_deltas'],'paired/'))
 jobs=[('A02-A03-rl-v2-summary-20260908','Phase6-Agentic-RL-v2-Fixed-Comparison-20260908',rd,metrics,{'test_tasks':480,'scope':rl['test_scope']})]
 combined={}
 for a,x in g['arms'].items():
  vals={'ifeval_percent':x['ifeval_percent'],'seven_percent':x['seven_percent'],'seven_macro_percent':x['seven_macro_percent'],'ifeval_strict_pass':x['ifeval_strict_pass']}
  combined.update(flat(vals,a+'/'));vals['evaluation_wall_seconds']=x['wall_seconds']
  exp=('A00-s10' if a=='S10' else a)+'-general-regression-20260908'
  role={'S10':'S10-Release','A01':'A01-Agent-SFT','A02':'A02-GRPO','A03':'A03-CISPO'}[a]
  jobs.append((exp,'Phase6-'+role+'-General-Regression-20260908',ROOT/f'phase6-general-{a}-20260908',flat(vals),{'model_sha256':x['manifest']['model_sha256'],'dtype':'float16','fewshot':0,'ifeval_prompts':541}))
 combined.update(flat(g['comparisons'],'paired/'))
 jobs.append(('A00-A03-general-regression-20260908','Phase6-General-Regression-Four-Arm-Comparison-20260908',gd,combined,{'protocol':g['protocol_verified'],'limitations':g['limitations']}))
 for exp,name,src,values,config in jobs:
  receipt=src/'swanlab-backfill.json'
  if receipt.exists() and read(receipt).get('status')=='uploaded':
   print(json.dumps({'skip_uploaded':exp,'url':read(receipt)['url']}),flush=True);continue
  rid=hashlib.sha256(('minimind-lab-backfill-'+exp).encode()).hexdigest()[:8]
  payload={'experiment_id':exp,'metrics':values,'config':config};digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
  write(src/'swanlab-backfill-payload.json',payload)
  run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name=name,id=rid,resume='allow',mode='online',group='Phase6-Agent-RL',job_type='evaluation',log_dir=str(src/'swanlab-backfill'),description='Historical metric backfill only; no new training/evaluation. Phase6 completed-not-promoted; S10 remains release.',config={**config,'seed':42,'backfill':True,'source_experiment':exp,'source_payload_sha256':digest,'phase_status':'completed-not-promoted','release':'S10','units':'general scores percent; Agent E2E fraction; deltas percentage points'})
  url=run.url
  record={'status':'uploading','experiment_id':exp,'run_id':rid,'url':url,'payload_sha256':digest,'metric_count':len(values)}
  write(receipt,record);swanlab.log(values,step=0);swanlab.finish()
  record.update(status='uploaded',uploaded_at=datetime.now(timezone.utc).isoformat(),scope='metrics-and-config-only')
  write(receipt,record);(src/'swanlab-url.txt').write_text(url+'\n')
  print(json.dumps(record),flush=True)
if __name__=='__main__':main()
