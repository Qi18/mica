#!/usr/bin/env python3
"""Single-GPU paired Phase4 driver: chosen-only CE or reference-relative DPO."""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"minimind"))
sys.path.insert(0,str(ROOT/"scripts/data/dpo"))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from build_official_dpo_v1 import encode_pair

def load_rows(path,tok):
    return [encode_pair(tok,json.loads(line)) for line in path.open()]
def batch(rows,device,pad):
    entries=[r[s] for s in ("chosen","rejected") for r in rows]
    n=max(len(e["ids"]) for e in entries)
    x=torch.full((len(entries),n-1),pad,dtype=torch.long,device=device)
    y=x.clone();m=torch.zeros_like(x,dtype=torch.float32)
    for i,e in enumerate(entries):
        ids=e["ids"];x[i,:len(ids)-1]=torch.tensor(ids[:-1],device=device)
        y[i,:len(ids)-1]=torch.tensor(ids[1:],device=device)
        m[i,e["start"]-1:len(ids)-1]=1
    return x,y,m
def logps(model,x,y,m):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits=model(x).logits
    logp=F.log_softmax(logits.float(),-1).gather(-1,y.unsqueeze(-1)).squeeze(-1)
    return (logp*m).sum(-1)
def preference(policy,reference,beta):
    n=len(policy)//2
    reward=beta*(policy-reference)
    margin=reward[:n]-reward[n:]
    return -F.logsigmoid(margin),margin,reward[:n],reward[n:]

@torch.no_grad()
def evaluate(model,ref,rows,beta,pad,batch_size):
    model.eval();res=[]
    for start in range(0,len(rows),batch_size):
        rs=rows[start:start+batch_size];x,y,m=batch(rs,"cuda",pad)
        p=logps(model,x,y,m);r=logps(ref,x,y,m)
        loss,margin,cr,rr=preference(p,r,beta);n=len(rs);length=m.sum(-1)
        for j in range(n):
            res.append({"dpo_loss":float(loss[j]),"reward_margin":float(margin[j]),
                        "preference_credit":1. if margin[j]>1e-7 else (0. if margin[j]<-1e-7 else .5),
                        "chosen_reward":float(cr[j]),"rejected_reward":float(rr[j]),
                        "chosen_nll":float(-p[j]/length[j]),"rejected_nll":float(-p[j+n]/length[j+n]),
                        "raw_length_normalized_credit":float(p[j]/length[j]>p[j+n]/length[j+n]),
                        "chosen_tokens":int(length[j]),"rejected_tokens":int(length[j+n])})
    result={k:sum(r[k] for r in res)/len(res) for k in res[0]}
    result["pairs"]=len(res)
    for name,cond in [("chosen_longer",lambda r:r["chosen_tokens"]>r["rejected_tokens"]),
                      ("chosen_not_longer",lambda r:r["chosen_tokens"]<=r["rejected_tokens"])]:
        subset=[r for r in res if cond(r)]
        result[name+"_pairs"]=len(subset)
        result[name+"_preference_credit"]=sum(r["preference_credit"] for r in subset)/max(1,len(subset))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=["baseline","sft","dpo"],required=True)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True);p.add_argument("--run-name",required=True)
    p.add_argument("--lr",type=float,default=4e-8);p.add_argument("--beta",type=float,default=.15)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--global-batch",type=int,default=32)
    p.add_argument("--max-steps",type=int,default=0);p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--smoke",action="store_true");a=p.parse_args()
    if a.output.exists():raise SystemExit("output already exists")
    assert (a.data/"_SUCCESS").exists()
    import hashlib
    manifest=json.loads((a.data/"manifest.json").read_text())
    assert manifest["status"]=="accepted"
    assert (a.data/"_SUCCESS").read_text().strip()==hashlib.sha256((a.data/"manifest.json").read_bytes()).hexdigest()
    for split in ("train.jsonl","validation.jsonl"):
        assert hashlib.sha256((a.data/split).read_bytes()).hexdigest()==manifest["files"][split]["sha256"]
    assert a.global_batch%a.batch_size==0
    import hashlib
    assert hashlib.sha256(a.base.read_bytes()).hexdigest()=="46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131"
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(ROOT/"minimind/model")
    train=load_rows(a.data/"train.jsonl",tok);val=load_rows(a.data/"validation.jsonl",tok)
    if a.smoke:train=train[:64];val=val[:16]
    config=MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=False)
    model=MiniMindForCausalLM(config)
    state=torch.load(a.base,map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True);model=model.cuda()
    ref=MiniMindForCausalLM(config);ref.load_state_dict(state,strict=True);ref=ref.cuda().eval().requires_grad_(False)
    a.output.mkdir(parents=True)
    import swanlab
    run=swanlab.init(project="MiniMind-Lab",experiment_name=a.run_name,config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},"seed":42,"train_pairs":len(train),"validation_pairs":len(val),"reference_frozen":True,"precision":"bf16","global_pairs_per_update":a.global_batch})
    (a.output/"swanlab-url.txt").write_text(run.url+"\n")
    def log(event,step,values):
        rec={"event":event,"step":step,"time":time.time(),**values}
        with (a.output/"metrics.jsonl").open("a") as f:f.write(json.dumps(rec)+"\n")
        swanlab.log({event+"/"+k:v for k,v in values.items() if isinstance(v,(int,float))},step=step)
        print(json.dumps(rec),flush=True)
    start=time.time();torch.cuda.reset_peak_memory_stats()
    baseline=evaluate(model,ref,val,a.beta,tok.pad_token_id,a.batch_size)
    log("validation",0,baseline)
    assert abs(baseline["dpo_loss"]-math.log(2))<1e-4
    if a.method=="baseline":
        (a.output/"validation.json").write_text(json.dumps(baseline,indent=2)+"\n")
        swanlab.finish();return
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(42)).tolist()
    total=math.ceil(len(indices)/a.global_batch)
    if a.max_steps:total=min(total,a.max_steps)
    best=baseline["dpo_loss"]
    def save(name):
        torch.save({k:v.detach().cpu().float() for k,v in model.state_dict().items()},a.output/(name+".pth"))
    save("best")
    chosen_total=rejected_total=0
    for step in range(1,total+1):
        selected=[train[j] for j in indices[(step-1)*a.global_batch:step*a.global_batch]]
        nt=sum(len(r["chosen"]["ids"])-r["chosen"]["start"] for r in selected)
        nr=sum(len(r["rejected"]["ids"])-r["rejected"]["start"] for r in selected)
        lr=a.lr*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/max(1,total))))
        for group in opt.param_groups:group["lr"]=lr
        opt.zero_grad(set_to_none=True);model.train();loss_total=0
        for j in range(0,len(selected),a.batch_size):
            rs=selected[j:j+a.batch_size];x,y,m=batch(rs,"cuda",tok.pad_token_id);n=len(rs)
            if a.method=="sft":
                lp=logps(model,x[:n],y[:n],m[:n]);loss=-lp.sum()/nt
            else:
                with torch.no_grad():rp=logps(ref,x,y,m)
                lp=logps(model,x,y,m)
                loss=preference(lp,rp,a.beta)[0].sum()/len(selected)
            assert torch.isfinite(loss)
            loss.backward();loss_total+=float(loss.detach())
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        assert math.isfinite(grad)
        assert all(p.grad is None for p in ref.parameters())
        opt.step();chosen_total+=nt;rejected_total+=nr if a.method=="dpo" else 0
        if step==1 or step%10==0 or step==total:
            log("train",step,{"loss":loss_total,"lr":lr,"grad_norm":grad,"chosen_targets":chosen_total,"rejected_targets":rejected_total,"peak_memory_mib":torch.cuda.max_memory_allocated()/1024**2,"wall_seconds":time.time()-start})
        if step%a.eval_every==0 or step==total:
            v=evaluate(model,ref,val,a.beta,tok.pad_token_id,a.batch_size);log("validation",step,v)
            if v["dpo_loss"]<best:best=v["dpo_loss"];save("best")
            save("last")
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"next_step":step+1,"indices":indices,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),"args":vars(a)},a.output/"resume.pt")
    log("completed",total,{"best_validation_dpo_loss":best,"wall_seconds":time.time()-start,"chosen_targets":chosen_total,"rejected_targets":rejected_total})
    swanlab.finish()
if __name__=="__main__":main()
