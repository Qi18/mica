#!/usr/bin/env python3
"""Freeze independent v2 tasks; counterfactual validation shares prompts, changes hidden values."""
import argparse,hashlib,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import phase6_agent_v2 as env

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tokenizer",required=True)
    a=p.parse_args()
    if a.output.exists():
        raise SystemExit("refuse overwrite")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer)
    rng=random.Random(607)
    splits={}
    for split,count in (("train",800),("val",160),("test",240)):
        tasks=[]
        for i in range(count):
            def rid(kind):
                return hashlib.sha256(f"phase6-v2:{split}:{i}:{kind}".encode()).hexdigest()[:8]
            family=("chain","branch","retry","redirect")[i%4]
            root=rid("root")
            prompt=(f"请处理根记录{root}，按记录指示完成工具计算。" if split=="train" else
                    f"从记录{root}开始，遵循返回的指示完成计算。" if split=="val" else
                    f"根记录ID为{root}。请沿记录指示查询并完成计算。")
            tasks.append({"id":f"{split}-{i}","family":family,"root_id":root,"leaf_id":rid("leaf"),
                          "alias_id":rid("alias"),"distractor_id":rid("distractor"),"prompt":prompt,
                          "root_value":rng.randrange(100,900),"leaf_value":rng.randrange(100,900),
                          "distractor_value":rng.randrange(1000,1800),"route":rng.choice(["left","right"]),
                          "operation":"add" if (i//4)%2==0 else "subtract"})
        splits[split]=tasks
    ids=[t[k] for tasks in splits.values() for t in tasks for k in ("root_id","leaf_id","alias_id","distractor_id")]
    prompts=[t["prompt"] for tasks in splits.values() for t in tasks]
    assert len(set(ids))==len(ids) and len(set(prompts))==len(prompts)
    lengths,targets=[],0
    for split,tasks in splits.items():
        for t in tasks:
            state={}
            for call in env.expected_calls(t):
                assert env.parse_call(env.call_text(call["name"],call["arguments"]))==call
                obs,ok=env.execute(t,call,state)
            assert ok and obs=={"result":env.answer(t)}
            if split!="test":
                for row in env.trajectories(t):
                    enc=env.encode_row(tok,row)
                    env.collate(tok,[enc])
                    lengths.append(len(enc[0]))
                    if split=="train":
                        targets+=sum(v!=-100 for v in enc[1])
    assert max(lengths)+env.MAX_NEW_TOKENS<=env.MAX_CONTEXT
    a.output.mkdir(parents=True)
    manifest={"seed":607,"record_overlap":0,"prompt_overlap":0,"splits":{},"max_length":max(lengths),
              "truncated_rows":0,"train_supervised_tokens":targets,
              "scope":"synthetic graph families with split-specific prompt wording; not broad OOD",
              "test_policy":"frozen, no model evaluation until design frozen","oracle":"pass"}
    for split,tasks in splits.items():
        path=a.output/(split+".jsonl")
        path.write_text("".join(json.dumps(t,ensure_ascii=False)+"\n" for t in tasks))
        manifest["splits"][split]={"tasks":len(tasks),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        if split!="test":
            rows=[row for t in tasks for row in env.trajectories(t)]
            path=a.output/(split+"-decisions.jsonl")
            path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
            manifest["splits"][split].update({"decisions":len(rows),"decisions_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    cf=[{**t,"root_value":t["root_value"]+19,"leaf_value":t["leaf_value"]+7} for t in splits["val"]]
    assert all(env.answer(x)!=env.answer(y) for x,y in zip(splits["val"],cf))
    path=a.output/"val-counterfactual.jsonl"
    path.write_text("".join(json.dumps(t,ensure_ascii=False)+"\n" for t in cf))
    manifest["counterfactual_validation"]={"tasks":len(cf),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "purpose":"same prompts and records, hidden values changed; never train"}
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
