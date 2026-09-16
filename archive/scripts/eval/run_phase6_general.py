#!/usr/bin/env python3
"""Offline, local-only full general regression. Never trains or publishes metrics."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path("/data/projects/minimind-lab");ART=Path("/data/artifacts/minimind-lab")
MODELS={"S10":ART/"D01-s10-preference-baseline-20260908/exported-fp32",
 "A01":ART/"A01-v3-mixed-lr3e6-20260908/checkpoint-570",
 "A02":ART/"A02-agentic-grpo-v2-20260908/model-fp32",
 "A03":ART/"A03-agentic-cispo-v2-20260908/model-fp32"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--arm",choices=list(MODELS),required=True);a=p.parse_args()
 model=MODELS[a.arm];out=ART/f"phase6-general-{a.arm}-20260908"
 assert not out.exists();out.mkdir()
 env={**os.environ,"HF_HOME":"/data/cache/huggingface","HF_ENDPOINT":"https://hf-mirror.com",
      "HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1","TOKENIZERS_PARALLELISM":"false",
      "WANDB_DISABLED":"true","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
 plan={"arm":a.arm,"model":str(model),"model_sha256":sha(model/"model.safetensors"),
       "tokenizer_sha256":{f:sha(model/f) for f in ["tokenizer.json","tokenizer_config.json"]},
       "source_sha256":sha(Path(__file__)),"precision":"float16","seed":42,"fewshot":0,"batch_size":16,
       "apply_chat_template":True,"full_dataset":True,"publish":False,"memory_fraction":.15}
 (out/"manifest.json").write_text(json.dumps(plan,indent=2)+"\n")
 started=time.monotonic()
 # Only a memory cap is wrapped around the existing harness CLI.
 entry="import torch;torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15);from lm_eval.__main__ import cli_evaluate;cli_evaluate()"
 for mode,tasks in [("ifeval","ifeval"),("seven","ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa")]:
  command=[sys.executable,"-u","-c",entry,"--model","hf","--model_args",f"pretrained={model},dtype=float16",
           "--tasks",tasks,"--num_fewshot","0","--batch_size","16","--apply_chat_template","--seed","42",
           "--log_samples","--output_path",str(out/mode)]
  (out/(mode+"-command.json")).write_text(json.dumps(command,indent=2)+"\n")
  with (out/(mode+".log")).open("x") as log:
   proc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  if proc.returncode:
   (out/"FAILURE.json").write_text(json.dumps({"mode":mode,"exit_code":proc.returncode,"model_sha256":plan["model_sha256"]})+"\n")
   raise SystemExit(proc.returncode)
  results=list((out/mode).rglob("results*.json"));assert len(results)==1
  x=json.loads(results[0].read_text());assert x["config"]["limit"] is None
  if mode=="ifeval":assert x["n-samples"]["ifeval"]["effective"]==541
  print(json.dumps({"arm":a.arm,"mode":mode,"result":str(results[0]),"elapsed":time.monotonic()-started}),flush=True)
 assert sha(model/"model.safetensors")==plan["model_sha256"]
 (out/"DONE.json").write_text(json.dumps({"status":"completed","arm":a.arm,"wall_seconds":time.monotonic()-started,
         "model_unchanged":True,"published":False},indent=2)+"\n")
 print("completed",a.arm,flush=True)
if __name__=="__main__":main()
