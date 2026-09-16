#!/usr/bin/env python3
"""Independent Phase4 split, prefix, target and near-duplicate audit."""
import argparse,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
def norm(s):return re.sub(r"\W+","",s.casefold())
def grams(s):
    s=norm(s);return {s[i:i+5] for i in range(max(1,len(s)-4))}
def main():
    p=argparse.ArgumentParser();p.add_argument("data",type=Path);a=p.parse_args()
    tok=AutoTokenizer.from_pretrained("tokenizer");m=json.loads((a.data/"manifest.json").read_text())
    results={};seen={};near=[];inv=defaultdict(list);sets=[];owners=[]
    for split in ("test","validation","train"):
        path=a.data/(split+".jsonl")
        assert hashlib.sha256(path.read_bytes()).hexdigest()==m["files"][path.name]["sha256"]
        rs=[json.loads(l) for l in path.open()];targets=Counter()
        for idx,r in enumerate(rs):
            assert r["chosen"][:-1]==r["rejected"][:-1]
            prompt=r["chosen"][0]["content"];key=norm(prompt)
            assert key not in seen,("duplicate",split,idx,seen.get(key))
            seen[key]=(split,idx)
            g=grams(prompt);matches=Counter(j for x in g for j in inv[x])
            for j,n in matches.items():
                if n/max(1,len(g)+len(sets[j])-n)>=.8:near.append([split,idx,owners[j]])
            j=len(sets);sets.append(g);owners.append([split,idx])
            for x in g:inv[x].append(j)
            prefix=tok.apply_chat_template(r["chosen"][:-1],tokenize=False,add_generation_prompt=True,open_thinking=False).replace("<think>\n\n</think>\n\n","")
            pre=tok(prefix,add_special_tokens=False).input_ids
            for side in ("chosen","rejected"):
                text=tok.apply_chat_template(r[side],tokenize=False,add_generation_prompt=False).replace("<think>\n\n</think>\n\n","")
                ids=tok(text,add_special_tokens=False).input_ids
                assert ids[:len(pre)]==pre and len(pre)<len(ids)<=1024
                targets[side]+=len(ids)-len(pre)
        assert dict(targets)==m["files"][path.name]["targets"]
        assert len(rs)==m["files"][path.name]["rows"]
        results[split]={"rows":len(rs),"targets":dict(targets),"prefix_mask_valid":True}
    assert not near,near[:5]
    result={"passed":True,"splits":results,"exact_duplicate_prompts":0,"near_duplicate_prompts_0_8":0,
            "scope":"all Phase4 splits; upstream S10 exposure and arbitrary semantic paraphrases not ruled out"}
    (a.data/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    m["status"]="accepted";(a.data/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n")
    (a.data/"_SUCCESS").write_text(hashlib.sha256((a.data/"manifest.json").read_bytes()).hexdigest()+"\n")
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
