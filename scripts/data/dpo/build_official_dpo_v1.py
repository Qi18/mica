#!/usr/bin/env python3
"""Freeze official DPO pairs with deterministic templates and prompt-disjoint splits."""
import argparse, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

def norm(s): return re.sub(r"\W+", "", s.casefold())
def grams(s):
    s=norm(s); return {s[i:i+5] for i in range(max(1,len(s)-4))}
def render(tok,messages):
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False).replace("<think>\n\n</think>\n\n","")
def encode_pair(tok,row,max_length=1024):
    prefix=tok.apply_chat_template(row["chosen"][:-1],tokenize=False,add_generation_prompt=True,open_thinking=False)
    prefix=prefix.replace("<think>\n\n</think>\n\n","")
    pre=tok(prefix,add_special_tokens=False).input_ids
    output={}
    for side in ("chosen","rejected"):
        ids=tok(render(tok,row[side]),add_special_tokens=False).input_ids
        if ids[:len(pre)]!=pre: raise ValueError("prefix_mismatch")
        if len(ids)>max_length: raise ValueError("overlength")
        if len(ids)<=len(pre): raise ValueError("zero_targets")
        output[side]={"ids":ids,"start":len(pre)}
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise SystemExit("refusing existing output")
    source=Path("/data/datasets/minimind/312afb4f76391145c6902f765bb51691c09a12f5/dpo.jsonl")
    tok=AutoTokenizer.from_pretrained("tokenizer")
    refs=[r["prompt"] for r in load_dataset("google/IFEval",split="train")]
    sys.path.insert(0,str(Path("scripts/eval").resolve()))
    from eval_sft_behavior import CHAT_CASES,TOOL_CASES
    refs += [r["prompt"] for r in CHAT_CASES]
    refs += [r.get("prompt",r.get("user","")) for r in TOOL_CASES]
    seen=set(); allgrams=[];inv=defaultdict(list)
    for prompt in refs:
        g=grams(prompt);i=len(allgrams);allgrams.append(g)
        for x in g:inv[x].append(i)
    nref=len(allgrams); accepted=[]; reasons=Counter()
    raw=[json.loads(l) for l in source.open()]
    # Canonical order makes dedup independent of source file order.
    raw.sort(key=lambda r:hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
    for row in raw:
        try:
            if len(row["chosen"])!=2 or len(row["rejected"])!=2 or row["chosen"][:-1]!=row["rejected"][:-1]:
                raise ValueError("context_mismatch")
            prompt=row["chosen"][0]["content"]
            if len(norm(prompt))<12: raise ValueError("underspecified_short_prompt")
            if not all(r[-1]["role"]=="assistant" and r[-1]["content"].strip() for r in [row["chosen"],row["rejected"]]):raise ValueError("schema")
            if norm(row["chosen"][-1]["content"])==norm(row["rejected"][-1]["content"]):raise ValueError("identical_answers")
            tokens=encode_pair(tok,row)
            key=norm(prompt)
            if key in seen:raise ValueError("exact_duplicate_prompt")
            g=grams(prompt);counts=Counter(i for x in g for i in inv[x])
            near=[i for i,c in counts.items() if c/max(1,len(g)+len(allgrams[i])-c)>=.8]
            if near: raise ValueError("eval_overlap" if any(i<nref for i in near) else "near_duplicate_prompt")
            idx=len(allgrams);allgrams.append(g)
            for x in g:inv[x].append(idx)
            seen.add(key)
            accepted.append((hashlib.sha256(("42:"+key).encode()).hexdigest(),row,tokens))
        except ValueError as e: reasons[str(e)]+=1
    accepted.sort(key=lambda x:x[0])
    assert len(accepted)>4000
    splits={"validation":accepted[:1000],"test":accepted[1000:2000],"train":accepted[2000:]}
    a.output.mkdir(parents=True)
    manifest={"status":"awaiting-independent-audit","source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"revision":"312afb4f76391145c6902f765bb51691c09a12f5","source_rows":len(raw),"rejected":dict(reasons),"seed":42,"max_seq_len":1024,"files":{}}
    for split,rs in splits.items():
        f=a.output/(split+".jsonl")
        f.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for _,r,_ in rs))
        counts={side:sum(len(t[side]["ids"])-t[side]["start"] for _,_,t in rs) for side in ["chosen","rejected"]}
        manifest["files"][f.name]={"rows":len(rs),"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"targets":counts}
    (a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
