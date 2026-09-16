#!/usr/bin/env python3
"""Evaluate a Phase4 checkpoint on the frozen held-out preference test."""
import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/launch"))
from train_phase4 import load_rows,batch,logps,preference
sys.path.insert(0,str(ROOT/"minimind"))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from transformers import AutoTokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--base",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--split",choices=["validation","test"],default="test");p.add_argument("--batch-size",type=int,default=8);p.add_argument("--beta",type=float,default=.15);a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(ROOT/"minimind/model");rows=load_rows(a.data/(a.split+".jsonl"),tok)
 cfg=MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=False)
 def model(path):
  m=MiniMindForCausalLM(cfg);m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True);return m.cuda().eval()
 policy=model(a.checkpoint);ref=model(a.base).requires_grad_(False);results=[]
 with torch.inference_mode():
  for start in range(0,len(rows),a.batch_size):
   rs=rows[start:start+a.batch_size];x,y,m=batch(rs,"cuda",tok.pad_token_id)
   pp=logps(policy,x,y,m);rp=logps(ref,x,y,m);loss,margin,cr,rr=preference(pp,rp,a.beta);n=len(rs);length=m.sum(-1)
   for j in range(n):
    delta=float(margin[j])
    results.append({"index":start+j,"dpo_loss":float(loss[j]),"reward_margin":delta,"preference_credit":1 if delta>1e-7 else (0 if delta<-1e-7 else .5),"chosen_reward":float(cr[j]),"rejected_reward":float(rr[j]),"chosen_logp":float(pp[j]),"rejected_logp":float(pp[j+n]),"chosen_nll":float(-pp[j]/length[j]),"rejected_nll":float(-pp[j+n]/length[j+n]),"raw_length_normalized_credit":float(pp[j]/length[j]>pp[j+n]/length[j+n]),"chosen_tokens":int(length[j]),"rejected_tokens":int(length[j+n])})
 summary={k:sum(r[k] for r in results)/len(results) for k in results[0] if k!="index"}
 summary["pairs"]=len(results)
 for name,cond in [("chosen_longer",lambda r:r["chosen_tokens"]>r["rejected_tokens"]),("chosen_not_longer",lambda r:r["chosen_tokens"]<=r["rejected_tokens"])]:
  z=[r for r in results if cond(r)];summary[name+"_pairs"]=len(z);summary[name+"_preference_credit"]=sum(r["preference_credit"] for r in z)/len(z)
 a.output.mkdir(parents=True)
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 with (a.output/"per_pair.jsonl").open("w") as f:
  for r in results:f.write(json.dumps(r)+"\n")
 print(json.dumps(summary))
if __name__=="__main__":main()
