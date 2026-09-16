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
 status('waiting_teacher_generation')
 pid=json.loads((a/'generation-launch.json').read_text())['pid'];deadline=time.monotonic()+3600
 while not (a/'GENERATION_DONE.json').exists():
  assert time.monotonic()<deadline,'teacher generation deadline exceeded; inspect existing job'
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('teacher generator exited without DONE; inspect generation-driver.log')
  time.sleep(10)
 status('pair_filter_and_data_gate')
 wait({'finalize-data':launch('finalize-data','scripts/data/finalize_phase8_sequence.py',[])})
 accepted=json.loads((d/'ACCEPTED.json').read_text());assert accepted['pairs']==4096
 import swanlab
 run=swanlab.init(project='MiniMind-Lab',workspace='richliu0153',experiment_name='Phase8-KD00-Qwen3-8B-sequence-data-v1',group='Phase8-Sequence-Distillation',job_type='data-generation',mode='online',log_dir=str(a/'generation-swanlab-backfill'),config={'generation':json.loads((a/'generation-config.json').read_text()),'accepted':accepted,'logged_after_generation':True})
 for m in rows(a/'generation-metrics.jsonl'):swanlab.log(m,step=m['generated'])
 teacher=rows(d/'teacher-answers.jsonl')
 swanlab.log({'data/accepted_pairs':4096,'generation/total_tokens':sum(r['generated_tokens'] for r in teacher),'data/K01_targets_per_pass':accepted['targets_per_pass']['K01'],'data/K02_targets_per_pass':accepted['targets_per_pass']['K02']},step=len(teacher)+1)
 swanlab.finish();dump(a/'generation-swanlab-backfill.json',{'url':run.url,'finished':True});(a/'teacher-swanlab-url.txt').write_text(run.url+'\n')
 status('smoke_gates')
 jobs={}
 for arm in ['K01','K02']:
  proof=a/('smoke-'+arm)/'DONE.json'
  if proof.exists():assert json.loads(proof.read_text())['status']=='smoke-passed'
  else:jobs['smoke-'+arm]=launch('smoke-'+arm,'scripts/launch/train_phase8_sequence.py',['--arm',arm,'--smoke'],c['gpu_'+arm])
 wait(jobs)
 for arm in ['K01','K02']:assert json.loads((a/('smoke-'+arm)/'DONE.json').read_text())['strict_load']
 status('training_K01_K02')
 jobs={arm:launch('formal-'+arm,'scripts/launch/train_phase8_sequence.py',['--arm',arm],c['gpu_'+arm]) for arm in ['K01','K02']}
 wait(jobs)
 for arm in ['K01','K02']:
  proof=json.loads((a/('train-'+arm)/'DONE.json').read_text());assert proof['target_tokens']==4194304 and proof['steps']==512
 status('frozen_evaluation')
 wait({arm:launch('eval-'+arm,'scripts/eval/eval_phase8_sequence.py',['--arm',arm],7 if arm=='S10' else c['gpu_'+arm]) for arm in ['S10','K01','K02']})
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
