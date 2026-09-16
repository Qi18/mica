#!/usr/bin/env python3
"""Evaluate R03 free-form integer verifier with greedy and sampled metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase5_numeric_common import parse_integer


def read_jsonl(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def decode(model,tok,prompts,max_new_tokens,k,seed,temperature,top_p):
    rendered=[tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True,open_thinking=False) for p in prompts]
    inp=tok(rendered,return_tensors="pt",padding=True,return_token_type_ids=False).to(model.device)
    torch.manual_seed(seed)
    kwargs={"max_new_tokens":max_new_tokens,"do_sample":k>1,"num_return_sequences":k,"pad_token_id":tok.pad_token_id,"eos_token_id":tok.eos_token_id}
    if k>1: kwargs.update(temperature=temperature,top_p=top_p)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16): out=model.generate(**inp,**kwargs)
    result=[]
    for ids in out[:,inp.input_ids.shape[1]:]:
        eos=(ids==tok.eos_token_id).nonzero(); length=int(eos[0].item()+1) if len(eos) else len(ids)
        text=tok.decode(ids[:length],skip_special_tokens=True).strip(); result.append({"text":text,"tokens":length,"parsed":parse_integer(text)})
    return result


def grouped(rows,key):
    result={}
    for value in sorted({str(row[key]) for row in rows}):
        part=[row for row in rows if str(row[key])==value]
        result[value]={"samples":len(part),"pass_at_1":sum(r["greedy_correct"] for r in part)/len(part)}
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",type=Path,required=True); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--split",choices=["validation","test"],required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--candidate",required=True); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--sample-k",type=int,default=8); ap.add_argument("--max-new-tokens",type=int,default=4); ap.add_argument("--temperature",type=float,default=1.0); ap.add_argument("--top-p",type=float,default=1.0); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args()
    if a.output.exists(): raise SystemExit(f"output exists: {a.output}")
    data=read_jsonl(a.data/f"{a.split}.jsonl"); tok=AutoTokenizer.from_pretrained(a.model); tok.padding_side="left"; model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32).cuda().eval(); details=[]
    for start in range(0,len(data),a.batch_size):
        part=data[start:start+a.batch_size]; prompts=[r["prompt"] for r in part]
        greedy=decode(model,tok,prompts,a.max_new_tokens,1,a.seed+start,a.temperature,a.top_p)
        sampled=decode(model,tok,prompts,a.max_new_tokens,a.sample_k,a.seed+100000+start,a.temperature,a.top_p)
        for i,row in enumerate(part):
            g=greedy[i]; ss=sampled[i*a.sample_k:(i+1)*a.sample_k]; target=int(row["answer"])
            details.append({"task_id":row["task_id"],"operation":row["operation"],"template_id":row["template_id"],"answer":target,"greedy_text":g["text"],"greedy_value":g["parsed"],"greedy_tokens":g["tokens"],"greedy_valid":g["parsed"] is not None,"greedy_correct":g["parsed"]==target,"sample_texts":[s["text"] for s in ss],"sample_values":[s["parsed"] for s in ss],"sample_valid":[s["parsed"] is not None for s in ss],"sample_correct":[s["parsed"]==target for s in ss],"pass_at_k":any(s["parsed"]==target for s in ss)})
    sample_correct=[x for r in details for x in r["sample_correct"]]; sample_valid=[x for r in details for x in r["sample_valid"]]
    summary={"status":"completed","candidate":a.candidate,"split":a.split,"samples":len(details),"protocol":{"reward":"strict entire-completion integer exact match","sample_k":a.sample_k,"temperature":a.temperature,"top_p":a.top_p,"max_new_tokens":a.max_new_tokens,"seed":a.seed,"precision":"float32 parameters with BF16 autocast"},"pass_at_1":sum(r["greedy_correct"] for r in details)/len(details),"valid_at_1":sum(r["greedy_valid"] for r in details)/len(details),"sample_accuracy":sum(sample_correct)/len(sample_correct),"sample_valid":sum(sample_valid)/len(sample_valid),"sample_all_zero_rate":sum(not any(r["sample_correct"]) for r in details)/len(details),"sample_all_one_rate":sum(all(r["sample_correct"]) for r in details)/len(details),"sample_nondegenerate_rate":sum(any(r["sample_correct"]) and not all(r["sample_correct"]) for r in details)/len(details),"pass_at_k":sum(r["pass_at_k"] for r in details)/len(details),"mean_greedy_tokens":statistics.mean(r["greedy_tokens"] for r in details),"greedy_value_distribution":Counter(str(r["greedy_value"]) if r["greedy_value"] is not None else "INVALID" for r in details),"by_operation":grouped(details,"operation"),"by_template":grouped(details,"template_id"),"by_answer":grouped(details,"answer"),"data_sha256":sha256(a.data/f"{a.split}.jsonl"),"model_config_sha256":sha256(a.model/"config.json"),"limitations":["empirical pass@k is fixed sampled generation, not an unbiased estimator","metrics describe only the configured narrow exact-integer task"]}
    a.output.mkdir(parents=True); (a.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n"); (a.output/"samples.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in details)); print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
