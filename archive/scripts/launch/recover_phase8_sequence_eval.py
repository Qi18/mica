#!/usr/bin/env python3
"""Run already-authorized bounded sequence-distillation workflow; never promote or push."""
import sys,os,time,json,subprocess,fcntl,traceback,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase8_common import *
PY='/data/venvs/minimind-eval/bin/python'
def main():
 c=cfg();a=Path(c['artifacts']);d=Path(c['data'])
 lock=(a/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (a/'PIPELINE_DONE.json').exists()
 def status(phase,**extra):dump(a/'pipeline-status.json',{'phase':phase,'unix_time':time.time(),**extra})
 def launch(label,script,args,gpu=None):
  env={**os.environ,'HF_HOME':'/data/cache/huggingface','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
  if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
  cmd=[PY,'-u',str(ROOT/script),*args]
  with (a/(label+'-driver.log')).open('x') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  dump(a/(label+'-launch.json'),{'pid':p.pid,'command':cmd,'gpu':gpu});return p
 def wait(jobs):
  failures=[]
  while jobs:
   for label,p in list(jobs.items()):
    rc=p.poll()
    if rc is not None:
     if rc:failures.append({'job':label,'exit_code':rc})
     del jobs[label]
   if jobs:time.sleep(5)
  assert not failures,failures
 status('frozen_evaluation_recovery')
 wait({arm:launch('recovery-eval-'+arm,'scripts/eval/eval_phase8_sequence.py',['--arm',arm],7 if arm=='S10' else c['gpu_'+arm]) for arm in ['S10','K01','K02']})
 status('comparison')
 import numpy as np
 done={arm:json.loads((a/('eval-'+arm)/'DONE.json').read_text()) for arm in ['S10','K01','K02']}
 records={arm:{r['id']:r for r in rows(a/('eval-'+arm)/'constraint-samples.jsonl')} for arm in done}
 assert records['S10'].keys()==records['K01'].keys()==records['K02'].keys()
 delta=np.array([int(records['K02'][k]['passed'])-int(records['K01'][k]['passed']) for k in records['K01']])
 counts=np.array([(delta==-1).sum(),(delta==0).sum(),(delta==1).sum()]);rng=np.random.default_rng(c['seed'])
 boots=rng.multinomial(len(delta),counts/len(delta),size=10000);ci=np.quantile((boots[:,2]-boots[:,0])/len(delta)*100,[.025,.975]).tolist()
 b,k1,k2=[done[k] for k in ['S10','K01','K02']]
 key='prompt_level_strict_acc,none'
 gates={'independent_constraint_gain_ci_positive':ci[0]>0,'ifeval_vs_S10_nonregression':k2['ifeval'][key]>=b['ifeval'][key],
  'seven_vs_S10_within_1pp':k2['seven_macro_percent']>=b['seven_macro_percent']-1,
  'chat':k2['behavior']['chat_success']>=8,'format':k2['behavior']['format_success']>=5,'repeat':k2['behavior']['repetition_count']<=1,'tool':k2['behavior']['tool_success']>=7}
 report={'status':'completed-not-promoted-awaiting-review','arms':done,'constraint_delta_K02_K01_pp':float(delta.mean()*100),'constraint_paired_bootstrap95ci_pp':ci,'numerical_gates':gates,'numerical_pass':all(gates.values()),'manual_broad_review_pending':True,'training_seed_count':1,'no_automatic_promotion':True,'no_commit_or_push':True,'scope':'sequence-level CE distillation, not KL/OPD','limitations':'same verifier family; finite heldout; IFEval used for teacher selection; token-matched but row exposure differs'}
 dump(a/'comparison.json',report);dump(a/'PIPELINE_DONE.json',report)
 status('completed-not-promoted-awaiting-review')
 print(json.dumps(report),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  c=cfg();dump(Path(c['artifacts'])/'PIPELINE_FAILURE.json',{'traceback':traceback.format_exc(),'action':'downstream stages not started; inspect owned logs; no automatic retries or threshold relaxation'});raise
