#!/usr/bin/env python3
"""Mixed A01 repair: fresh graph and tool tasks + documented broad replay."""
import argparse,hashlib,json,random,re,sys,unicodedata
from pathlib import Path
from collections import Counter
from jsonschema import Draft7Validator, ValidationError
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import phase6_agent_v2 as graph
import phase6_retention_env as toolenv
from eval_sft_behavior import CHAT_CASES,TOOL_CASES
def norm(s):return "".join(c.lower() for c in unicodedata.normalize("NFKC",s) if c.isalnum())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--tokenizer",required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit("refuse overwrite")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer);rng=random.Random(608)
    banned={norm(t["prompt"]) for t in CHAT_CASES+TOOL_CASES}
    oldroot=Path("/data/datasets/minimind-lab/phase6")
    for v in ("record-agent-v1","record-agent-v2"):
        for split in ("val","test"):
            banned.update(norm(t["prompt"]) for t in read(oldroot/v/(split+".jsonl")))
    outputs={};split_prompts={};report={"seed":608,"sources":{},"rejected":dict(),"scope":"same graph families; new task IDs; mixed replay is retention, not novel capability evidence"}
    for split,n in (("train",800),("val",160),("test",240)):
        tasks=[]
        for i in range(n):
            rid=lambda k:hashlib.sha256(f"phase6-v3:{split}:{i}:{k}".encode()).hexdigest()[:8]
            r=rid("root")
            variants=(f"请从根记录{r}开始，依照记录指示完成查询和计算。",f"处理记录{r}的计算流程，沿返回字段继续查找。",
                      f"根记录编号是{r}，按它的规则调用工具并返回结果。",f"读取根节点{r}，跟随后继记录完成运算。")
            tasks.append({"id":f"{split}-{i}","family":("chain","branch","retry","redirect")[i%4],
                          "root_id":r,"leaf_id":rid("leaf"),"alias_id":rid("alias"),"distractor_id":rid("distractor"),
                          "root_value":rng.randrange(100,900),"leaf_value":rng.randrange(100,900),"distractor_value":rng.randrange(1000,1800),
                          "operation":"add" if (i//4)%2==0 else "subtract","route":rng.choice(["left","right"]),
                          "prompt":variants[(i//8)%len(variants)]})
        outputs[split+".jsonl"]=tasks;split_prompts[split]={norm(t["prompt"]) for t in tasks}
        if split!="test":
            outputs[split+"-decisions.jsonl"]=[{**row,"domain":"graph","tools":graph.TOOLS} for t in tasks for row in graph.trajectories(t)]
        # Fresh tool tasks with entity/value ranges separated across splits.
        tools=[]
        for i in range(800 if split=="train" else 160 if split=="val" else 240):
            k=i%8;idx=i+{"train":0,"val":10000,"test":20000}[split];family=("math","unit","weather","exchange","time","translate","random","multistep")[k]
            x=idx+201;y=11+i%73
            def call(n,args):return {"name":n,"arguments":args}
            if k==0:
                calls=[call("calculate_math",{"expression":f"{x} * {y}"})];obs=[{"result":x*y}];ans=str(x*y)
                prompt=f"用计算工具计算{x}乘{y}，得到结果后只输出数字。"
            elif k==1:
                result=round(x*.621371,2);calls=[call("unit_converter",{"value":x,"from_unit":"km","to_unit":"miles"})]
                obs=[{"result":result}];ans=str(result);prompt=f"把{x}公里换成英里，调用工具后只回复数值。"
            elif k==2:
                city=f"海港城{idx}";temp=5+i%31;condition=("晴","阴","小雨")[i%3]
                calls=[call("get_current_weather",{"location":city})];obs=[{"temperature":temp,"condition":condition}]
                ans=condition;prompt=f"查询{city}天气，最后只回复天气状况，不要温度。"
            elif k==3:
                currencies=("USD","EUR","GBP","JPY","AUD","CAD","CHF","CNY")
                fr=currencies[(i//8)%8];to=currencies[((i//8)%8+1)%8];rate=round(.1+(idx%991)/100,2)
                calls=[call("get_exchange_rate",{"from_currency":fr,"to_currency":to})];obs=[{"rate":rate}]
                ans=str(rate);prompt=f"报价请求{idx}：查询{fr}到{to}的汇率，只返回工具给出的数值。"
            elif k==4:
                zone=("UTC","Asia/Tokyo","Europe/London","America/New_York")[i//8%4]
                tm=f"{(idx//60)%24:02d}:{idx%60:02d}"
                calls=[call("get_current_time",{"timezone":zone})];obs=[{"datetime":"2026-09-08 "+tm+":00","timezone":zone}]
                ans=tm;prompt=f"时钟请求{idx}：查询{zone}当前时间，最后只输出HH:MM。"
            elif k==5:
                text=f"档案编号{idx}已完成";translated=f"Record {idx} is complete"
                calls=[call("translate_text",{"text":text,"target_language":"English"})];obs=[{"translated_text":translated}]
                ans=translated;prompt=f"调用翻译工具把“{text}”译成英文，只回复工具给出的译文。"
            elif k==6:
                lo=idx+1;hi=lo+100
                calls=[call("random_number",{"min":lo,"max":hi})];obs=[{"number":lo+37}]
                ans=str(lo+37);prompt=f"用工具生成{lo}到{hi}之间的随机整数，只回复生成的整数。"
            else:
                lo=idx+1;hi=lo+100;v=lo+37
                calls=[call("random_number",{"min":lo,"max":hi}),call("calculate_math",{"expression":f"{v} * {v}"})]
                obs=[{"number":v},{"result":v*v}];ans=str(v*v)
                prompt=f"先用工具生成{lo}到{hi}间的随机数，再用计算工具求其平方，只回复最终数字。"
            available={c["name"] for c in calls};available.add(("text_length","get_current_weather","get_current_time")[i%3])
            schemas=[toolenv.SCHEMAS[name] for name in sorted(available)];rng.shuffle(schemas)
            tools.append({"id":f"tools-{split}-{i}","family":family,"tools":schemas,"prompt":prompt,"calls":calls,"observations":obs,"answer":ans})
        outputs["tools-"+split+".jsonl"]=tools
        split_prompts[split].update(norm(t["prompt"]) for t in tools)
        if split!="test":outputs[split+"-decisions.jsonl"].extend(row for t in tools for row in toolenv.trajectories(t))
    broad=Path("/data/datasets/minimind-lab/data-v1/sft-chat-repair-v2")
    # Do not use the later S09 targeted paraphrase curriculum.
    for split,file,count in (("train","train.jsonl",2600),("val","validation.jsonl",200)):
        source=broad/file;rows=read(source);rng.shuffle(rows);selected=[];seen=set();native_tools=0;chat=0;rejected=Counter()
        for source_row in rows:
            conv=source_row["conversations"]
            prompts=[norm(m.get("content") or "") for m in conv if m["role"]=="user"]
            if not prompts or any(p in banned or p in seen or p in split_prompts["test"] for p in prompts):
                rejected["prompt_overlap_or_duplicate"]+=1;continue
            if split=="val" and any(p in split_prompts["train"] for p in prompts):
                rejected["train_overlap"]+=1;continue
            try:
                available=next((m["tools"] for m in conv if m.get("tools")),None)
                if isinstance(available,str):available=json.loads(available)
                if available:available=[t if "function" in t else {"type":"function","function":t} for t in available]
                is_tool=any(m.get("tool_calls") or m["role"]=="tool" for m in conv)
                if is_tool and native_tools >= (600 if split=="train" else 50):continue
                if not is_tool and chat >= (2000 if split=="train" else 150):continue
                msgs=[];decisions=[]
                for m in conv:
                    role=m["role"];content=m.get("content") or ""
                    if role=="assistant" and m.get("tool_calls"):
                        texts=[]
                        native_calls=json.loads(m["tool_calls"]) if isinstance(m["tool_calls"],str) else m["tool_calls"]
                        for c in native_calls:
                            fn=c["function"];args=fn["arguments"]
                            if isinstance(args,str):args=json.loads(args)
                            schema=next((t["function"].get("parameters",{}) for t in available or [] if t["function"]["name"]==fn["name"]),None)
                            if schema is None:raise ValueError("missing tool schema")
                            Draft7Validator(schema).validate(args)
                            texts.append(graph.call_text(fn["name"],args))
                        content=(content+"\n" if content else "")+"\n".join(texts)
                    if role=="assistant" and content:
                        decisions.append({"messages":list(msgs),"completion":content,"tools":available,"domain":"native_tool" if is_tool else "chat",
                                          "task_id":"replay-"+sha_text(prompts[0]),"source":str(source)})
                    msgs.append({"role":role,"content":content})
            except (ValueError,TypeError,KeyError,ValidationError):
                rejected["invalid_native_json_or_schema"]+=1;continue
            try:
                enc=[graph.encode_row(tok,r) for r in decisions]
            except Exception:rejected["encoding_error"]+=1;continue
            if not enc or any(len(x[0])>1536 for x in enc):rejected["overlength"]+=1;continue
            seen.update(prompts);split_prompts[split].update(prompts);selected.extend(decisions)
            if is_tool:native_tools+=1
            else:chat+=1
            if native_tools+chat>=count:break
        assert chat==(2000 if split=="train" else 150),(split,chat)
        assert native_tools==(600 if split=="train" else 50),(split,native_tools)
        outputs[split+"-decisions.jsonl"].extend(selected)
        report["sources"][split]={"path":str(source),"sha256":sha(source),"chat_conversations":chat,"native_tool_conversations":native_tools,
                                  "decisions":len(selected),"note":"historical broad replay; not an unseen-S10 dataset"}
        report["rejected"][split]=dict(rejected)
    assert not (split_prompts["train"]&split_prompts["val"] or split_prompts["train"]&split_prompts["test"] or split_prompts["val"]&split_prompts["test"])
    assert not split_prompts["train"]&banned
    # Exact domain counts and token budgets, zero truncation.
    report["domains"]={}
    for split in ("train","val"):
        rng.shuffle(outputs[split+"-decisions.jsonl"]);counts=Counter();tokens=Counter();lengths=[]
        for row in outputs[split+"-decisions.jsonl"]:
            full,labels=graph.encode_row(tok,row);assert len(full)<=1536
            graph.collate(tok,[(full,labels)])
            counts[row["domain"]]+=1;tokens[row["domain"]]+=sum(v!=-100 for v in labels);lengths.append(len(full))
        report["domains"][split]={"decisions":dict(counts),"assistant_tokens":dict(tokens),"max_length":max(lengths)}
    for split in ("train","val","test"):
        for name,env in ((split+".jsonl",graph),("tools-"+split+".jsonl",toolenv)):
            for t in outputs[name]:
                state={}
                for c in env.expected_calls(t):
                    assert env.parse_call(env.call_text(c["name"],c["arguments"]))==c
                    obs,ok=env.execute(t,c,state)
                assert ok
    report.update({"prompt_overlap":0,"truncated_rows":0,"test_policy":"frozen; never use for LR/checkpoint selection",
                   "retention_caveat":"broad replay is capability-family targeted and includes old S10 sources; fixed legacy smoke is not independent proof"})
    a.output.mkdir(parents=True);report["files"]={}
    for name,rows in outputs.items():
        p=a.output/name;p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
        report["files"][name]={"rows":len(rows),"sha256":sha(p)}
    (a.output/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()[:16]
if __name__=="__main__":main()
