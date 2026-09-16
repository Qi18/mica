#!/usr/bin/env python3
"""Read-only model rollout probe for Phase6 RL eligibility; never updates weights."""
import argparse,json,random,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import phase6_agent_v2 as env

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(.15)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32).cuda().eval()
    tok=AutoTokenizer.from_pretrained(a.model);tok.padding_side="left"
    tasks=[json.loads(l) for l in (a.data/"train.jsonl").read_text().splitlines()]
    random.Random(42).shuffle(tasks);tasks=tasks[:32]
    import swanlab
    run=swanlab.init(project="MiniMind-Lab",experiment_name="A01-v2-Agent-RL-Eligibility-Probe-20260908",
                    group="Phase6-Agent",job_type="probe",
                    config={"model":a.model,"training_prompts":32,"samples_per_prompt":4,"seed":42,
                            "temperature":.8,"top_p":1.0,"top_k":0,"optimizer_updates":0,
                            "reward":"0.8 E2E + 0.2 ordered tool prefix"})
    (a.output/"swanlab-url.txt").write_text(run.url+"\n")
    results=[];mask_checked=False;started=time.monotonic()
    for start in range(0,len(tasks),4):
        episodes=[{"task":t,"messages":env.initial_messages(t),"state":{},"calls":[],"final":None,"done":False,
                   "trace":[],"actions":[]} for t in tasks[start:start+4] for _ in range(4)]
        for turn in range(env.MAX_TURNS):
            active=[e for e in episodes if not e["done"]]
            if not active:break
            prompts=[tok.apply_chat_template(e["messages"],tools=env.TOOLS,tokenize=False,
                                            add_generation_prompt=True,open_thinking=False) for e in active]
            inp=tok(prompts,return_tensors="pt",return_token_type_ids=False,padding=True,add_special_tokens=False).to("cuda")
            assert inp.input_ids.size(1)+env.MAX_NEW_TOKENS<=env.MAX_CONTEXT
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                out=model.generate(**inp,do_sample=True,temperature=.8,top_p=1.,top_k=0,max_new_tokens=env.MAX_NEW_TOKENS,
                                   pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            for i,(e,ids) in enumerate(zip(active,out[:,inp.input_ids.size(1):])):
                raw=ids.tolist()
                if tok.eos_token_id in raw:raw=raw[:raw.index(tok.eos_token_id)+1]
                prefix=inp.input_ids[i][inp.attention_mask[i].bool()].tolist()
                e["actions"].append((prefix+raw,[-100]*len(prefix)+raw))
                text=tok.decode(raw,skip_special_tokens=True).strip()
                e["trace"].append({"assistant":text})
                call=env.parse_call(text)
                if call is None:e["done"]=True;e["final"]=text
                else:
                    e["calls"].append(call);obs,_=env.execute(e["task"],call,e["state"])
                    e["trace"].append({"tool":obs})
                    e["messages"].extend([{"role":"assistant","content":text},{"role":"tool","content":env.dumps(obs)}])
        if not mask_checked:
            rows=[r for e in episodes for r in e["actions"]][:2]
            b=env.collate(tok,rows,"cuda")
            labels=b["labels"][:,1:];mask=labels!=-100
            def lp():
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    logits=model(input_ids=b["input_ids"],attention_mask=b["attention_mask"],use_cache=False).logits[:,:-1].float()
                    return F.log_softmax(logits,-1).gather(-1,labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)[mask]
            old,new=lp(),lp();delta=float((old-new).abs().max())
            assert delta<1e-5
            for full,lab in rows:
                first=next(i for i,v in enumerate(lab) if v!=-100)
                assert all(v==-100 for v in lab[:first]) and lab[first:]==full[first:]
            (a.output/"mask-ratio-check.json").write_text(json.dumps({"passed":True,"old_policy_repeat_error":delta,
                                                                    "scope":"teacher-forced raw logprobs; sampling distribution audit still needed"}))
            mask_checked=True
        for e in episodes:
            expected=env.expected_calls(e["task"]);prefix=0
            for ac,ex in zip(e["calls"],expected):
                if ac!=ex:break
                prefix+=1
            success=e["calls"]==expected and e["final"]==str(env.answer(e["task"]))
            results.append({"id":e["task"]["id"],"family":e["task"]["family"],"success":success,
                            "reward":.8*int(success)+.2*prefix/len(expected),"trace":e["trace"],
                            "action_tokens":sum(sum(x!=-100 for x in lab) for _,lab in e["actions"])})
        rewards=torch.tensor([r["reward"] for r in results]).reshape(-1,4)
        metrics={"sampled_e2e":sum(r["success"] for r in results)/len(results),
                 "zero_variance_groups":float((rewards.std(1,unbiased=False)<1e-8).float().mean()),
                 "mean_group_reward_std":float(rewards.std(1,unbiased=False).mean()),"episodes":len(results)}
        swanlab.log(metrics,step=start//4+1);print(json.dumps(metrics),flush=True)
    metrics.update({"optimizer_updates":0,"wall_seconds":time.monotonic()-started,"greedy_saturation_gate":"held: validation >=98%",
                    "action_tokens":sum(r["action_tokens"] for r in results)})
    (a.output/"summary.json").write_text(json.dumps(metrics,indent=2))
    (a.output/"rollouts.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results))
    swanlab.finish()

if __name__=="__main__":main()
