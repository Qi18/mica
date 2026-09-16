#!/usr/bin/env python3
"""Fixed-checkpoint matched Agent RL evaluation, with a separate test-freeze gate."""
import argparse,json,sys,subprocess,gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"launch"))
import eval_phase6_agent as evaluation
from train_phase6_rl_fp32 import strict_call,sha,write

ART=Path("/data/artifacts/minimind-lab")
DATA=Path("/data/datasets/minimind-lab/phase6/mixed-agent-v3-r1")
MODELS={"A01":ART/"A01-v3-mixed-lr3e6-20260908/checkpoint-570",
        "A02":ART/"A02-agentic-grpo-v2-20260908/model-fp32",
        "A03":ART/"A03-agentic-cispo-v2-20260908/model-fp32"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--arm",choices=list(MODELS),required=True)
 p.add_argument("--split",choices=["val","test"],required=True);a=p.parse_args()
 modelpath=MODELS[a.arm];out=ART/f"{a.arm}-agentic-rl-v2-{a.split}-20260908"
 assert not out.exists()
 if a.split=="test":
  freeze=json.loads(Path("experiments/05-agentic-rl/rl-v2-test-freeze.json").read_text())
  assert freeze["models"][a.arm]["sha256"]==sha(modelpath/"model.safetensors")
  for file in ("test.jsonl","tools-test.jsonl"):assert freeze["data"][file]==sha(DATA/file)
 out.mkdir()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.float32).cuda().eval()
 tok=AutoTokenizer.from_pretrained(modelpath)
 import swanlab
 run=swanlab.init(project="MiniMind-Lab",experiment_name=f"Phase6-{a.arm}-AgentRL-v2-{a.split}",
                 group="Phase6-Agent-RL",job_type="evaluation",config={"arm":a.arm,"model":str(modelpath),
                 "split":a.split,"greedy":True,"strict_duplicate_keys":True,"seed":42})
 (out/"swanlab-url.txt").write_text(run.url+"\n")
 summaries={}
 for domain,mod,file in (("graph","phase6_agent_v2",a.split+".jsonl"),("tools","phase6_retention_env","tools-"+a.split+".jsonl")):
  env=evaluation.configure_environment(mod)
  evaluation.parse_call=lambda text,env=env:strict_call(env,text)
  tasks=[json.loads(l) for l in (DATA/file).read_text().splitlines()]
  summary=evaluation.evaluate(model,tok,tasks,out/domain,autocast_enabled=False)
  summaries[domain]=summary
  swanlab.log({domain+"/"+k:v for k,v in summary.items() if isinstance(v,(int,float))})
 del model;gc.collect();torch.cuda.empty_cache()
 if a.split=="val":
  subprocess.run([sys.executable,"-u","scripts/eval/eval_sft_behavior.py","--model",str(modelpath),
                  "--output-dir",str(out/"behavior"),"--max-new-tokens","128"],check=True)
  behavior=json.loads((out/"behavior/task_eval.json").read_text());summaries["behavior"]=behavior
  swanlab.log({domain+"/"+k:v for domain in ("chat","tool") for k,v in behavior[domain].items() if isinstance(v,(int,float))})
 write(out/"summary.json",summaries)
 write(out/"eval-manifest.json",{"arm":a.arm,"split":a.split,"model":str(modelpath),"model_sha256":sha(modelpath/"model.safetensors"),
       "files":{f:sha(DATA/f) for f in [a.split+".jsonl","tools-"+a.split+".jsonl"]},
       "source":{str(p):{"sha256":sha(p),"text":p.read_text()} for p in [Path(__file__),Path("scripts/eval/eval_phase6_agent.py"),Path("scripts/launch/train_phase6_rl_fp32.py")]},
       "legacy_behavior_dtype":"fp16","agent_dtype":"float32-no-tf32"})
 swanlab.finish();write(out/"DONE.json",{"status":"evaluation-complete","arm":a.arm,"split":a.split})
 print(json.dumps({k:v.get("end_to_end_success") for k,v in summaries.items()},ensure_ascii=False))
if __name__=="__main__":main()
