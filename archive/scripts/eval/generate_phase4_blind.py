#!/usr/bin/env python3
"""Generate deterministic blinded Phase4 candidate responses on 200 held-out prompts."""
import argparse,hashlib,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--candidate",required=True);p.add_argument("--samples",type=int,default=200);p.add_argument("--batch-size",type=int,default=16);a=p.parse_args()
 if a.output.exists():raise SystemExit("output exists")
 tok=AutoTokenizer.from_pretrained(a.model);tok.padding_side="left"
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32).cuda().eval()
 rows=[json.loads(l) for l in (a.data/"test.jsonl").open()][:a.samples];out=[]
 with torch.inference_mode():
  for start in range(0,len(rows),a.batch_size):
   part=rows[start:start+a.batch_size];prompts=[r["chosen"][0]["content"] for r in part]
   rendered=[tok.apply_chat_template([{"role":"user","content":x}],tokenize=False,add_generation_prompt=True,open_thinking=False) for x in prompts]
   inputs=tok(rendered,return_tensors="pt",padding=True).to("cuda");inputs.pop("token_type_ids",None)
   generated=model.generate(**inputs,max_new_tokens=192,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   new=generated[:,inputs.input_ids.shape[1]:]
   for j,(prompt,ids) in enumerate(zip(prompts,new)):
    response=tok.decode(ids,skip_special_tokens=True).strip()
    out.append({"index":start+j,"candidate":a.candidate,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"prompt":prompt,"response":response,"generated_tokens":int((ids!=tok.pad_token_id).sum()),"ref_chosen":part[j]["chosen"][-1]["content"],"ref_rejected":part[j]["rejected"][-1]["content"]})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in out));print(a.output,len(out))
if __name__=="__main__":main()
