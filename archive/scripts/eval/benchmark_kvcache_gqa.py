#!/usr/bin/env python3
"""Paired S10 native GQA/equivalent MHA cache ablation; remote A10 experiment."""
import argparse, csv, gc, hashlib, json, math, os, platform, random, subprocess, sys, time, traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

CKPT = Path("/data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/checkpoints/s08_best_val_768.pth")
EXPECTED = "46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131"
ARMS = {"I01": (8,False), "I02": (8,True), "I03": (4,False), "I04": (4,True)}
PROMPTS = ["请用简单的语言解释什么是机器学习。", "为什么天空看起来是蓝色的？", "请列出三个学习编程的建议。", "计算12乘以8，并说明计算过程。"]
TEXT = "机器学习通过数据学习规律。训练阶段更新模型参数，推理阶段使用固定参数生成结果。实验应该固定输入、权重和运行环境，并记录速度、显存以及输出的一致性。"
def dump(path,obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n"); tmp.replace(path)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def shell(*args): return subprocess.check_output(args,text=True).strip()
def telemetry():
    return shell("nvidia-smi","--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw","--format=csv,noheader")
def model_for(state, heads, dtype, flash=True):
    m = MiniMindForCausalLM(MiniMindConfig(num_key_value_heads=heads,flash_attn=flash))
    transformed={}
    for k,v in state.items():
        if heads==8 and (k.endswith("k_proj.weight") or k.endswith("v_proj.weight")):
            v=v.reshape(4,96,768).repeat_interleave(2,dim=0).reshape(768,768)
        transformed[k]=v
    m.load_state_dict(transformed,strict=True)
    return m.to(device="cuda:0",dtype=dtype).eval()
def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
def compare(a,b, dtype):
    a=a.float(); b=b.float(); delta=(a-b).abs()
    limits={"float32":{"max_abs":0.005,"relative_rmse":0.0001},
            "bfloat16":{"max_abs":0.5,"relative_rmse":0.005}}[dtype]
    rmse=(delta.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-6)).item()
    return {"max_abs":delta.max().item(),"mean_abs":delta.mean().item(),"relative_rmse":rmse,
            "argmax_agreement":(a.argmax(-1)==b.argmax(-1)).float().mean().item(),
            "limits":limits,"pass":bool(torch.isfinite(a).all() and torch.isfinite(b).all() and delta.max()<=limits["max_abs"] and rmse<=limits["relative_rmse"])}
@torch.inference_mode()
def generate(m,ids,n,cache):
    past=None; full=ids; generated=[]
    for _ in range(n):
        out=m(full if past is None else full[:,-1:],past_key_values=past,use_cache=cache,logits_to_keep=1)
        token=out.logits[:,-1].argmax(-1,keepdim=True)
        generated.append(token)
        full=torch.cat((full,token),1)
        past=out.past_key_values if cache else None
    return torch.cat(generated,1).cpu()
@torch.inference_mode()
def correctness(state,tok,outdir):
    result={"tests":[],"generations":[],"note":"BF16 token differences are measured, not silently treated as exact equivalence."}
    for dtype_name in ("float32","bfloat16"):
        dtype=getattr(torch,dtype_name); references={}
        for heads in (4,8):
            m=model_for(state,heads,dtype)
            for length in (16,128,512,1024):
                g=torch.Generator().manual_seed(42+length)
                ids=torch.randint(3,6400,(2,length+4),generator=g).cuda()
                full=m(ids,logits_to_keep=4).logits.cpu()
                p=m(ids[:,:length],use_cache=True,logits_to_keep=1).past_key_values
                steps=[]
                for i in range(4):
                    o=m(ids[:,length+i:length+i+1],past_key_values=p,use_cache=True,logits_to_keep=1)
                    steps.append(o.logits.cpu()); p=o.past_key_values
                incremental=torch.cat(steps,1)
                result["tests"].append({"dtype":dtype_name,"heads":heads,"length":length,"kind":"cache_vs_full",**compare(full,incremental,dtype_name)})
                if heads==4: references[length]=full
                else: result["tests"].append({"dtype":dtype_name,"length":length,"kind":"mha_vs_gqa",**compare(references[length],full,dtype_name)})
                del p,o
            for idx,prompt in enumerate(PROMPTS):
                ids=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_tensors="pt").cuda()
                for cache in (False,True):
                    gen=generate(m,ids,32,cache)
                    result["generations"].append({"dtype":dtype_name,"heads":heads,"cache":cache,"prompt_index":idx,"tokens":gen[0].tolist(),"text":tok.decode(gen[0])})
            del m; clear()
    result["generation_comparisons"]=[]
    for d in ("float32","bfloat16"):
        for idx in range(len(PROMPTS)):
            rows=[x for x in result["generations"] if x["dtype"]==d and x["prompt_index"]==idx]
            base=rows[0]["tokens"]
            for row in rows[1:]:
                match=[a==b for a,b in zip(base,row["tokens"])]
                result["generation_comparisons"].append({"dtype":d,"prompt_index":idx,"reference":"GQA cache_off",
                    "heads":row["heads"],"cache":row["cache"],"token_agreement":sum(match)/len(match),
                    "first_difference":next((i for i,x in enumerate(match) if not x),None)})
    result["pass"]=all(x["pass"] for x in result["tests"])
    dump(outdir/"correctness.json",result)
    return result
@torch.inference_mode()
def measure(m,ids,n,cache,trace):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    baseline_a=torch.cuda.memory_allocated(); baseline_r=torch.cuda.memory_reserved()
    begin=torch.cuda.Event(enable_timing=True); first=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    full=ids; past=None; t0=time.perf_counter(); begin.record()
    o=m(full,use_cache=cache,logits_to_keep=1)
    token=o.logits[:,-1].argmax(-1,keepdim=True)
    full=torch.cat((full,trace[:,0:1]),1); past=o.past_key_values if cache else None
    first.record(); first.synchronize(); t1=time.perf_counter()
    for step in range(1,n):
        o=m(full if past is None else full[:,-1:],past_key_values=past,use_cache=cache,logits_to_keep=1)
        token=o.logits[:,-1].argmax(-1,keepdim=True)
        full=torch.cat((full,trace[:,step:step+1]),1); past=o.past_key_values if cache else None
    end.record(); end.synchronize(); t2=time.perf_counter()
    kv_bytes=sum(t.numel()*t.element_size() for layer in past for t in layer) if cache else 0
    expected=2*8*ids.shape[0]*(ids.shape[1]+n-1)*m.config.num_key_value_heads*96*2 if cache else 0
    assert kv_bytes==expected,(kv_bytes,expected)
    row={"ttft_ms":(t1-t0)*1000,"decode_ms":(t2-t1)*1000,"e2e_ms":(t2-t0)*1000,
         "decode_tokens_s":ids.shape[0]*(n-1)/(t2-t1),"tpot_ms":(t2-t1)*1000/(n-1),
         "gpu_prefill_ms":begin.elapsed_time(first),"gpu_decode_ms":first.elapsed_time(end),
         "baseline_allocated_bytes":baseline_a,"baseline_reserved_bytes":baseline_r,
         "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
         "kv_bytes":kv_bytes,"kv_length":ids.shape[1]+n-1 if cache else 0,
         "output_sha256":hashlib.sha256(full[:,ids.shape[1]:].cpu().numpy().tobytes()).hexdigest()}
    return row
def stats(vals):
    a=sorted(vals); n=len(a)
    return {"median":(a[(n-1)//2]+a[n//2])/2,"p95":a[math.ceil(.95*n)-1],"min":a[0],"max":a[-1]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--mode",choices=["correctness","pilot","full"],default="correctness")
    args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert sha(CKPT)==EXPECTED,"Checkpoint SHA mismatch"
    state=torch.load(CKPT,map_location="cpu",weights_only=True)
    tok=AutoTokenizer.from_pretrained(ROOT/"minimind/model",local_files_only=True)
    config={"arms":ARMS,"batch_sizes":[1,4,8],"input_lengths":[128,512,1024],"new_tokens":128,
            "extra":{"batch":1,"input":512,"output":512},"warmups":5,"repeats":20,
            "checkpoint":str(CKPT),"checkpoint_sha256":EXPECTED,"dtype":"bfloat16","seed":42,
            "attention":"unmodified native: prefill SDPA, cached single-token manual attention",
            "timing":"model-only greedy, logits_to_keep=1, no EOS stop; TTFT includes one GPU synchronization; no per-decode-step synchronization",
            "correctness_limits":{"float32":{"max_abs":0.005,"relative_rmse":0.0001},"bfloat16":{"max_abs":0.5,"relative_rmse":0.005}},
            "input":"fixed repeated Chinese text, separately stored token IDs; workload, not quality benchmark",
            "order":"scenario shuffled seed42; arm block rotated per scenario; 5 warmups + 20 repeats per arm block",
            "max_position_embeddings":32768,"training_sequence_length":768}
    if args.mode=="correctness":
        dump(out/"config.json",config)
        result=correctness(state,tok,out)
        print(json.dumps({"correctness_pass":result["pass"],"tests":len(result["tests"]),"failed":[x for x in result["tests"] if not x["pass"]]},ensure_ascii=False),flush=True)
        return
    correctness_result=json.loads((out/"correctness.json").read_text())
    assert all(x["pass"] for x in correctness_result["tests"] if x["dtype"]=="float32"),"FP32 semantic gate failed"
    diagnosis=json.loads((out/"precision-diagnosis.json").read_text())
    assert all(x["pass"] for x in diagnosis["tests"] if x["dtype"]=="float32"),"Unified manual FP32 gate failed"
    # BF16 failure remains recorded; this is controlled trace replay, not exact-output acceptance.
    assert all(math.isfinite(x["relative_rmse"]) for x in correctness_result["tests"])
    config.update(workload="fixed FP32 GQA greedy token trace replay; BF16 argmax computed but not fed back",
                  bf16_consistency_gate="FAILED; unchanged thresholds; performance-only with controlled token trace",
                  mode=args.mode)
    dump(out/"config.json",config)
    if args.mode=="full": assert not (out/"raw.jsonl").exists(),"Refusing to overwrite previous measurements"
    meta={"status":"running","started_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"lab_commit":shell("git","-C",str(ROOT),"rev-parse","HEAD"),
          "minimind_source_commit":"393e387e9ad99f0f04c296e4c5e7353f4444629f",
          "source_sha256":sha(ROOT/"minimind/model/model_minimind.py"),"script_sha256":sha(__file__),
          "torch":torch.__version__,"transformers":__import__("transformers").__version__,"python":platform.python_version(),
          "hostname":platform.node(),"hardware":telemetry(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
          "checkpoint_sha256":EXPECTED,"swanlab_url":None,"minimind_dirty":bool(shell("git","-C",str(ROOT),"status","--porcelain","--","minimind")),
          "dirty_worktree":"pre-existing changes recorded in source.patch and status-before.txt","command":" ".join(sys.argv)}
    dump(out/("run.json" if args.mode=="full" else "pilot-run.json"),meta)
    scenarios=[(b,l,128) for b in (1,4,8) for l in (128,512,1024)]+[(1,512,512)]
    if args.mode=="pilot": scenarios=[(1,128,128),(8,1024,128)]
    random.Random(42).shuffle(scenarios)
    warmups,repeats=(5,20) if args.mode=="full" else (1,2)
    token_ids=tok.encode(TEXT,add_special_tokens=False)
    inputs={str(l):(token_ids*((l//len(token_ids))+1))[:l] for l in (128,512,1024)}
    dump(out/"input-tokens.json",inputs)
    raw_path=out/("raw.jsonl" if args.mode=="full" else "pilot-raw.jsonl")
    trace_path=out/"reference-traces.json"
    if trace_path.exists():
        traces=json.loads(trace_path.read_text())
    else:
        traces={}
    if any(f"{l}-{n}" not in traces for _,l,n in scenarios):
        reference=model_for(state,4,torch.float32)
        for l,n in sorted(set((l,n) for _,l,n in scenarios)):
            if f"{l}-{n}" in traces: continue
            ids=torch.tensor(inputs[str(l)],device="cuda").unsqueeze(0)
            traces[f"{l}-{n}"]=generate(reference,ids,n,True)[0].tolist()
        dump(trace_path,traces)
        del reference,ids; clear()
    summaries=[]
    for si,(b,l,n) in enumerate(scenarios):
        order=list(ARMS); shift=si%4; order=order[shift:]+order[:shift]
        for arm in order:
            heads,cache=ARMS[arm]; clear(); before=telemetry()
            m=model_for(state,heads,torch.bfloat16)
            ids=torch.tensor(inputs[str(l)],device="cuda").repeat(b,1)
            trace=torch.tensor(traces[f"{l}-{n}"],device="cuda").repeat(b,1)
            for _ in range(warmups): measure(m,ids,n,cache,trace)
            rows=[]
            for rep in range(repeats):
                row={"arm":arm,"batch":b,"input_tokens":l,"output_tokens":n,"repeat":rep,**measure(m,ids,n,cache,trace)}
                rows.append(row)
                with raw_path.open("a") as f: f.write(json.dumps(row)+"\n")
            summary={"arm":arm,"batch":b,"input_tokens":l,"output_tokens":n,"warmups":warmups,"repeats":repeats,
                     "metrics":{k:stats([r[k] for r in rows]) for k in rows[0] if k not in ("arm","output_sha256","repeat","batch","input_tokens","output_tokens")},
                     "telemetry_before":before,"telemetry_after":telemetry(),"output_hashes":sorted(set(r["output_sha256"] for r in rows))}
            summaries.append(summary)
            dump(out/("summary.json" if args.mode=="full" else "pilot-summary.json"),summaries)
            print(json.dumps({"done":len(summaries),"total":len(scenarios)*4,"arm":arm,"batch":b,"input":l,"output":n,
                "decode_tokens_s":summary["metrics"]["decode_tokens_s"]["median"],"peak_mib":summary["metrics"]["peak_allocated_bytes"]["median"]/2**20}),flush=True)
            del m,ids,trace; clear()
    assert sha(CKPT)==EXPECTED
    meta.update(status="completed-with-bf16-consistency-limitations",finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),scenarios=len(summaries),checkpoint_sha256_after=sha(CKPT))
    dump(out/("run.json" if args.mode=="full" else "pilot-run.json"),meta)
if __name__=="__main__":
    main()
