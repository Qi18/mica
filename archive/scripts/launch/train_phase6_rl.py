#!/usr/bin/env python3
"""Exploratory multi-turn GRPO/CISPO. Only generated assistant action tokens get gradients."""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase6_agent_v2 as graph
import phase6_retention_env as tools_env
ENV = {"graph": graph, "tools": tools_env}

def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def emit(p, x):
    with Path(p).open("a") as f: f.write(json.dumps(x, ensure_ascii=False)+"\n")
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+"\n")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def strict_call(env, text):
    # Reject duplicate JSON keys before the unchanged schema/execution verifier.
    import re
    m=re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*",text,re.S)
    if not m: return None
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError("duplicate key")
            out[k]=v
        return out
    try: json.loads(m.group(1),object_pairs_hook=unique)
    except (ValueError,TypeError): return None
    return env.parse_call(text)

def score(env, task, calls, final, done):
    expected=env.expected_calls(task);prefix=0
    for a,b in zip(calls,expected):
        if a!=b:break
        prefix+=1
    success=done and calls==expected and final==str(env.answer(task))
    return .8*float(success)+.2*prefix/len(expected), bool(success)

def sample(model,tok,tasks,domain,seed,group=4):
    env=ENV[domain];torch.manual_seed(seed);model.eval();tok.padding_side="left"
    episodes=[{"task":t,"domain":domain,"messages":env.initial_messages(t),"state":{},"calls":[],
               "final":None,"done":False,"trace":[],"actions":[],"context_overflow":False} for t in tasks for _ in range(group)]
    for turn in range(env.MAX_TURNS):
        active=[e for e in episodes if not e["done"]]
        if not active: break
        prefixes=[tok.apply_chat_template(e["messages"],tools=e["task"].get("tools",env.TOOLS),
                    tokenize=True,add_generation_prompt=True,open_thinking=False) for e in active]
        accepted=[]
        for e,prefix in zip(active,prefixes):
            if len(prefix)+env.MAX_NEW_TOKENS>env.MAX_CONTEXT:
                e["done"]=True;e["context_overflow"]=True
            else:accepted.append((e,prefix))
        if not accepted:break
        active=[e for e,p in accepted];prefixes=[p for e,p in accepted]
        inp=tok.pad({"input_ids":prefixes},padding=True,return_tensors="pt").to(model.device)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            out=model.generate(**inp,do_sample=True,temperature=1.,top_p=1.,top_k=0,
                repetition_penalty=1.,max_new_tokens=env.MAX_NEW_TOKENS,pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,return_dict_in_generate=True,output_scores=True)
        # Retain the actual sampling distribution for actor/generator drift audit.
        generation_lp=torch.stack([F.log_softmax(s.float(),-1).gather(1,out.sequences[:,inp.input_ids.size(1)+j,None]).squeeze(1)
                                   for j,s in enumerate(out.scores)],1).cpu()
        for i,(e,prefix,seq) in enumerate(zip(active,prefixes,out.sequences[:,inp.input_ids.size(1):])):
            raw=seq.tolist()
            if tok.eos_token_id in raw:raw=raw[:raw.index(tok.eos_token_id)+1]
            assert raw
            e["actions"].append({"ids":prefix+raw,"labels":[-100]*len(prefix)+raw,
                                 "generation_lp":generation_lp[i,:len(raw)].tolist()})
            text=tok.decode(raw,skip_special_tokens=True).strip()
            e["trace"].append({"role":"assistant","text":text})
            call=strict_call(env,text)
            if call is None:e["final"]=text;e["done"]=True
            else:
                e["calls"].append(call);obs,ok=env.execute(e["task"],call,e["state"])
                e["trace"].append({"role":"tool","observation":obs,"executed":ok})
                e["messages"].extend([{"role":"assistant","content":text},{"role":"tool","content":env.dumps(obs)}])
        del out,inp,generation_lp
    for e in episodes:
        e["reward"],e["success"]=score(env,e["task"],e["calls"],e["final"],e["done"])
        e["tokens"]=sum(len(a["generation_lp"]) for a in e["actions"])
        assert e["tokens"]>0
    return episodes

def logps(model,tok,actions):
    batch=graph.collate(tok,[(a["ids"],a["labels"]) for a in actions],model.device)
    labels=batch.pop("labels")[:,1:];mask=labels!=-100
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits=model(**batch,use_cache=False).logits[:,:-1].float()
    lp=F.log_softmax(logits,-1).gather(-1,labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return [r[m] for r,m in zip(lp,mask)]

def surrogate(new,old,adv,method):
    ratio=(new-old).exp()
    if method=="grpo":obj=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv);clip=((ratio<.8)|(ratio>1.2))
    else:obj=ratio.clamp(max=2.).detach()*adv*new;clip=ratio>2.
    return obj,clip,ratio

def unit_checks():
    result={}
    for method in ("grpo","cispo"):
        new=torch.tensor([0.0,math.log(3.),math.log(.1)],requires_grad=True)
        old=torch.zeros(3);adv=torch.tensor([1.,1.,-1.])
        obj,_,_=surrogate(new,old,adv,method);(-obj.sum()).backward()
        got=new.grad.tolist()
        assert got[0]<0
        if method=="grpo":assert got[1]==0 and got[2]==0
        else:assert got[1]<0 and got[2]>0
        result[method]=got
    r=torch.tensor([[1.,1.,1.,1.],[0.,1.,0.,1.]])
    a=(r-r.mean(1,keepdim=True))/(r.std(1,unbiased=False,keepdim=True)+1e-4)
    assert torch.equal(a[0],torch.zeros(4)) and abs(a[1].sum())<1e-6
    return {"passed":True,"surrogate_gradients":result,"zero_variance_advantage_zero":True}

def compact(e):
    return {k:e[k] for k in ("domain","reward","success","tokens","trace","context_overflow")} | {"id":e["task"]["id"],"family":e["task"]["family"]}

def prepare_old(model,reference,tok,episodes,audit=False):
    actions=[a for e in episodes for a in e["actions"]]
    max_delta=0.;total_delta=0.;n=0;repeat=0.
    with torch.no_grad():
        for i in range(0,len(actions),2):
            chunk=actions[i:i+2];old=logps(model,tok,chunk)
            refs=logps(reference,tok,chunk) if reference is not None else old
            if audit:
                again=logps(model,tok,chunk)
                repeat=max(repeat,max(float((x-y).abs().max()) for x,y in zip(old,again)))
            for a,x,y in zip(chunk,old,refs):
                generation=torch.tensor(a["generation_lp"],device=x.device)
                delta=(x-generation).abs()
                max_delta=max(max_delta,float(delta.max()));total_delta+=float(delta.sum());n+=len(x)
                a["old"]=x.detach().cpu();a["ref"]=y.detach().cpu()
    # BF16 cached generation and teacher-forced forward need not be bitwise identical.
    assert repeat<1e-5,repeat
    assert max_delta<.20 and total_delta/n<.02,(max_delta,total_delta/n)
    return {"repeat_logprob_max_error":repeat,"generation_teacher_max_error":max_delta,
            "generation_teacher_mean_error":total_delta/n,"action_tokens":n,
            "sampling":"temperature1/top_p1/top_k0; actual generation scores audited"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--probe",action="store_true")
    a=p.parse_args();c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/"unit-checks.json",unit_checks())
    torch.set_num_threads(4);torch.manual_seed(c["seed"]);torch.cuda.set_per_process_memory_fraction(.15)
    model=AutoModelForCausalLM.from_pretrained(c["base_model"],torch_dtype=torch.float32).cuda().eval()
    tok=AutoTokenizer.from_pretrained(c["base_model"])
    # Dropout remains disabled during policy updates as well as rollout.
    reference=None
    if not a.probe:
        reference=AutoModelForCausalLM.from_pretrained(c["base_model"],torch_dtype=torch.float32).cuda().eval()
        reference.requires_grad_(False)
    data=Path(c["data"]);pools={}
    for domain,file in (("graph","train.jsonl"),("tools","tools-train.jsonl")):
        pools[domain]=read(data/file);random.Random(c["seed"]).shuffle(pools[domain])
    sourcefiles=["scripts/launch/train_phase6_rl.py","scripts/phase6_agent_common.py","scripts/phase6_agent_v2.py","scripts/phase6_retention_env.py",
                 "scripts/eval/eval_phase6_agent.py","scripts/eval/eval_sft_behavior.py"]
    write(a.output/"provenance.json",{"config":c,"probe":a.probe,"data_files":{f:sha(data/f) for f in ["train.jsonl","tools-train.jsonl"]},
          "source":{f:{"sha256":sha(f),"text":Path(f).read_text()} for f in sourcefiles},"test_used":False})
    import swanlab
    run=swanlab.init(project="MiniMind-Lab",experiment_name=c["run_name"]+("-Probe" if a.probe else ""),
                    group="Phase6-Agent-RL",job_type="probe" if a.probe else "training",config=c)
    (a.output/"swanlab-url.txt").write_text(run.url+"\n")
    optimizer=None if a.probe else torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=0.)
    start=time.monotonic();updated=0;count=0;tokens=0;all_success=0;variable_groups=0;total_groups=0;trace=[]
    outer_steps=8 if a.probe else c["outer_steps"]
    for outer in range(outer_steps):
        eps=[]
        for di,domain in enumerate(("graph","tools")):
            eps+=sample(model,tok,pools[domain][2*outer:2*outer+2],domain,c["seed"]*100000+outer*2+di,c["group"])
        audit=prepare_old(model,reference,tok,eps,audit=(outer==0))
        if outer==0:write(a.output/"mask-ratio-check.json",{"passed":True,**audit,"scope":"each assistant action only; prior actions and observations masked in prefix"})
        rewards=torch.tensor([e["reward"] for e in eps]).view(-1,c["group"])
        std=rewards.std(1,unbiased=False);advantages=((rewards-rewards.mean(1,keepdim=True))/(std[:,None]+1e-4)).flatten()
        variable=int((std>1e-8).sum());variable_groups+=variable;total_groups+=len(std)
        count+=len(eps);tokens+=sum(e["tokens"] for e in eps);all_success+=sum(e["success"] for e in eps)
        for e,adv in zip(eps,advantages):
            for action in e["actions"]:action["adv"]=float(adv)
        actions=[action for e in eps for action in e["actions"]];denom=sum(len(x["old"]) for x in actions)
        last={}
        if not a.probe:
            for inner in range(c["inner_updates"]):
                optimizer.zero_grad(set_to_none=True);loss_total=0.;kl_total=0.;clipped=0.;ratio_max=0.;ratio_min=float("inf");drift_max=0.
                for i in range(0,len(actions),2):
                    chunk=actions[i:i+2];current=logps(model,tok,chunk);loss=0.
                    for action,new in zip(chunk,current):
                        old=action["old"].to(model.device);ref=action["ref"].to(model.device)
                        # Small actor/generator discrepancy corrected as a detached token IS factor.
                        gen=torch.tensor(action["generation_lp"],device=model.device)
                        correction=(old-gen).exp().detach()
                        obj,clip,ratio=surrogate(new,old,action["adv"],c["method"])
                        d=ref-new;kl=d.exp()-d-1
                        part=(-obj*correction+c["beta"]*kl).sum()/denom
                        loss=loss+part
                        loss_total+=float(part.detach());kl_total+=float(kl.detach().sum())/denom
                        clipped+=int(clip.sum());ratio_max=max(ratio_max,float(ratio.detach().max()));ratio_min=min(ratio_min,float(ratio.detach().min()))
                        drift_max=max(drift_max,float(d.detach().abs().max()))
                    assert torch.isfinite(loss)
                    loss.backward()
                if inner==0:
                    assert abs(ratio_max-1.)<1e-5 and abs(ratio_min-1.)<1e-5,(ratio_min,ratio_max)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(grad)
                optimizer.step();updated+=1
                last={"loss":loss_total,"kl_k3":kl_total,"clip_fraction":clipped/denom,"grad_norm":float(grad),
                      "ratio_min":ratio_min,"ratio_max":ratio_max,"reference_logprob_delta_max":drift_max}
            if last["kl_k3"]>.2:raise RuntimeError("KL safety stop; preserve attempt, no promotion")
        for e in eps:
            row=compact(e)|{"outer_step":outer+1};emit(a.output/"rollouts.jsonl",row)
        row={"outer_step":outer+1,"updates":updated,"episodes":count,"sampled_e2e":sum(e["success"] for e in eps)/len(eps),
             "reward":float(rewards.mean()),"group_reward_std":float(std.mean()),"variable_group_rate":variable/len(std),
             "variable_groups_total":variable_groups,"tokens_total":tokens,"wall_seconds":time.monotonic()-start,
             "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,**audit,**last}
        emit(a.output/"metrics.jsonl",row);swanlab.log({k:v for k,v in row.items() if isinstance(v,(int,float))},step=outer+1);print(json.dumps(row),flush=True)
    result={"status":"probe-complete" if a.probe else "exploratory-complete-not-promoted","optimizer_updates":updated,
            "episodes":count,"sampled_e2e":all_success/count,"variable_groups":variable_groups,"total_groups":total_groups,
            "assistant_tokens":tokens,"wall_seconds":time.monotonic()-start,"test_opened":False}
    if a.probe:result["eligible"]=variable_groups>=4 and 0<all_success<count
    else:
        target=a.output/"model-fp32";model.save_pretrained(target,safe_serialization=True);tok.save_pretrained(target)
        result["model_path"]=str(target);result["weight_sha256"]=sha(target/"model.safetensors")
    write(a.output/"DONE.json",result);swanlab.log({"final/"+k:v for k,v in result.items() if isinstance(v,(int,float))})
    swanlab.finish();print(json.dumps(result),flush=True)
if __name__=="__main__":main()
