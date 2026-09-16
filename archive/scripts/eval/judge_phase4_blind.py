#!/usr/bin/env python3
"""Blind pairwise Qwen3 judge for frozen Phase4 generated outputs."""
import argparse,hashlib,json,re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
def read(p):return {r["index"]:r for r in (json.loads(x) for x in p.open())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--baseline",type=Path,required=True);p.add_argument("--sft",type=Path,required=True);p.add_argument("--dpo",type=Path,required=True);p.add_argument("--judge",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--limit",type=int,default=0);a=p.parse_args()
 if a.output.exists():raise SystemExit("output exists")
 candidates={"D01":read(a.baseline),"D02":read(a.sft),"D03":read(a.dpo)}
 tok=AutoTokenizer.from_pretrained(a.judge);tok.padding_side="left"
 model=AutoModelForCausalLM.from_pretrained(a.judge,torch_dtype=torch.bfloat16).cuda().eval()
 jobs=[]
 for opponent in ("D01","D02"):
  for i in sorted(candidates["D03"])[:a.limit or None]:
   d=candidates["D03"][i];o=candidates[opponent][i];assert d["prompt_sha256"]==o["prompt_sha256"]
   swap=int(hashlib.sha256(f"42:{opponent}:{i}".encode()).hexdigest(),16)%2
   aa,bb=(d,o) if not swap else (o,d)
   content=f"""用户请求：
{d['prompt']}

回答A：
{aa['response']}

回答B：
{bb['response']}

按正确性、相关性、帮助程度和表达质量比较两个回答。不要偏好更长的回答；都不好或质量相当时选TIE。只输出 A、B 或 TIE。"""
   jobs.append({"opponent":opponent,"index":i,"swap":swap,"prompt":content,"a_candidate":aa["candidate"],"b_candidate":bb["candidate"],"dpo_tokens":d["generated_tokens"],"opponent_tokens":o["generated_tokens"]})
 results=[]
 with torch.inference_mode():
  for start in range(0,len(jobs),a.batch_size):
   part=jobs[start:start+a.batch_size]
   rendered=[tok.apply_chat_template([{"role":"system","content":"你是独立的回答质量评测员。严格执行输出格式。"},{"role":"user","content":j["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for j in part]
   inputs=tok(rendered,return_tensors="pt",padding=True,truncation=True,max_length=4096).to("cuda");inputs.pop("token_type_ids",None)
   ids=model.generate(**inputs,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)[:,inputs.input_ids.shape[1]:]
   for j,x in zip(part,ids):
    raw=tok.decode(x,skip_special_tokens=True).strip().upper()
    matches=re.findall(r"\b(?:TIE|A|B)\b",raw)
    verdict=matches[-1] if matches else "INVALID"
    winner=j["a_candidate"] if verdict=="A" else (j["b_candidate"] if verdict=="B" else verdict)
    results.append({**{k:v for k,v in j.items() if k!="prompt"},"raw":raw,"verdict":verdict,"winner":winner})
 a.output.mkdir(parents=True)
 with (a.output/"judgments.jsonl").open("w") as f:
  for r in results:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 summary={"judge_model":str(a.judge),"judge_config_sha256":hashlib.sha256((a.judge/"config.json").read_bytes()).hexdigest(),"seed":42,"decoding":"greedy; thinking disabled; max_new_tokens=8","comparisons":{}}
 for opponent in ("D01","D02"):
  rs=[r for r in results if r["opponent"]==opponent];valid=[r for r in rs if r["winner"]!="INVALID"]
  info={"samples":len(rs),"valid":len(valid),"invalid":len(rs)-len(valid),"D03_win":sum(r["winner"]=="D03" for r in valid),"opponent_win":sum(r["winner"]==opponent for r in valid),"tie":sum(r["winner"]=="TIE" for r in valid)}
  for label,cond in [("D03_longer",lambda r:r["dpo_tokens"]>r["opponent_tokens"]),("D03_not_longer",lambda r:r["dpo_tokens"]<=r["opponent_tokens"])]:
   z=[r for r in valid if cond(r)];info[label]={"samples":len(z),"D03_win":sum(r["winner"]=="D03" for r in z),"opponent_win":sum(r["winner"]==opponent for r in z),"tie":sum(r["winner"]=="TIE" for r in z)}
  summary["comparisons"]["D03_vs_"+opponent]=info
 (a.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
