#!/usr/bin/env python3
"""Assemble Phase3 evidence from completed artifacts; optional SwanLab publication."""
import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
ART = Path("/data/artifacts/minimind-lab")
IDS = ["L00-code-baseline-20260907", "L01-code-full-ft-20260907", "L02-code-lora-r16-20260907"]
S10 = ART/"S10-ifeval-curriculum-v4-20260907"
URLS = ["n/a-no-training", "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/pz1x0ux9", "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s0l5ishg"]
TASKS = ["ceval-valid", "cmmlu", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa"]

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
def result(p):
    files=sorted(p.rglob("results*.json"))
    if len(files)!=1: raise ValueError(f"expected exactly one result in {p}, got {len(files)}")
    return files[0],read(files[0])
def scalar(v):
    while isinstance(v,list): v=v[0] if v else ""
    return v
def flatten(prefix,obj,out):
    if isinstance(obj,dict):
        for k,v in obj.items(): flatten(prefix+"/"+k,obj=v,out=out)
    elif isinstance(obj,(int,float)) and not isinstance(obj,bool): out[prefix]=obj

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--publish",action="store_true")
    args=parser.parse_args()
    summaries=[]
    for i,e in enumerate(IDS):
        a=ART/e
        out=ROOT/"experiments/03-lora"/e
        src=S10 if i==0 else a
        mp,m=result(a/"eval"/("mbpp-full" if i==0 else "mbpp"))
        op,o=result(src/"eval/official-seven-chat-template")
        ip,iv=result(src/"eval/ifeval-chat-template")
        official={}
        for t in TASKS:
            r=o["results"][t]
            official[t]=100*r.get("acc_norm,none",r.get("acc,none"))
        b=read(src/"eval/behavior/task_eval.json")
        raw=[json.loads(l) for l in next(mp.parent.glob("samples*.jsonl")).open()]
        empty=syntax=0
        examples=[]
        for r in raw:
            code=scalar(r.get("filtered_resps",[]))
            empty+=not bool(code.strip())
            try: ast.parse(code)
            except (SyntaxError,ValueError): syntax+=1
            if len(examples)<3:
                examples.append({"doc_id":r["doc_id"],"prompt":r["doc"]["text"],"response":scalar(r["resps"]),"extracted_code":code,"pass_at_1":r.get("pass_at_1")})
        assert len(raw)==500
        metrics=[] if i==0 else [json.loads(l) for l in (a/"metrics/metrics.jsonl").open()]
        done={} if i==0 else next(r for r in metrics if r["event"]=="completed")
        if i: assert done["global_step"]==594
        paths=[S10/"checkpoints/s08_best_val_768.pth"] if i==0 else sorted((a/"checkpoints").glob("*.pth"))
        checkpoints=[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)} for p in paths]
        score=m["results"]["mbpp_instruct"]["pass_at_1,extract_code"]
        ifeval={k.split(",")[0]:v*100 for k,v in iv["results"]["ifeval"].items() if k.endswith(",none") and "stderr" not in k}
        summary={"experiment_id":e,"status":"completed","mbpp_pass_at_1_percent":score*100,"mbpp_pass_count":round(score*500),"mbpp_test_count":500,
                 "official_seven_percent":official,"official_macro_percent":mean(official.values()),"ifeval_percent":ifeval,
                 "behavior":b,"training":done,"actual_assistant_targets":sum(r.get("window_valid_tokens",0) for r in metrics),
                 "mbpp_diagnostics":{"empty_extraction_count":empty,"extracted_syntax_error_count":syntax,
                 "note":"Extraction failures count as failures in the fixed harness. Syntax validity does not imply correct code."},
                 "sources":[{"path":str(p),"sha256":sha(p)} for p in [mp,op,ip,src/"eval/behavior/task_eval.json"]],
                 "checkpoint_manifest":checkpoints}
        if i:
            summary["training"]["gpu_hours"]=done["wall_seconds"]*6/3600
            summary["training"]["overall_assistant_targets_per_second"]=summary["actual_assistant_targets"]/done["wall_seconds"]
            summary["training"]["training_seed"]=42
            summary["training"]["dataset_split_seed"]=20260907
        write(out/"eval.json",summary)
        write(out/"failure_examples.json",examples)
        (out/"checkpoint-manifest.txt").write_text("\n".join(f"{p['sha256']}  {p['bytes']}  {p['path']}" for p in checkpoints)+"\n")
        receipt=a/"eval/phase3-swanlab-receipt.json"
        if args.publish and not receipt.exists():
            import swanlab
            run=swanlab.init(project="MiniMind-Lab",experiment_name=e.split("-")[0]+"-Phase3-Evaluation",config={"experiment_id":e,"phase":3,"base":"S10","protocol":"zero-shot-chat-template","train_seed":42,"dataset_split_seed":20260907,"method_comparison_lr_confound":True})
            values={}
            flatten("eval",summary["official_seven_percent"],values)
            flatten("ifeval",ifeval,values)
            flatten("behavior",b,values)
            flatten("training",summary["training"],values)
            values["mbpp/pass_at_1_percent"]=score*100
            values["eval/macro_percent"]=summary["official_macro_percent"]
            swanlab.log(values)
            url=run.url
            swanlab.finish()
            write(receipt,{"url":url,"experiment_id":e,"metrics":values})
        eval_url=read(receipt)["url"] if receipt.exists() else "pending"
        (out/"swanlab-url.txt").write_text(URLS[i]+"\n"+eval_url+"\n")
        write(out/"run.json",{"status":"completed","experiment_id":e,"source_head_before_changes":"d097e1f5aa70cbc3824bee62c7ed36dc6c253b3c",
              "training_swanlab_url":URLS[i],"evaluation_swanlab_url":eval_url,
              "training_seed":42,"split_seed":20260907,"base_sha256":"46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131",
              "artifact_dir":str(a),"method_acceptance":"not promoted; no demonstrated MBPP improvement",
              "limitations":["single training seed","different LR for Full FT and LoRA","DDP validation pads 1000 to 1002 samples","independent data audit performed after training","no optimizer/resume state saved","target-GPU guard substituted for global guard on shared node"]})
        with (out/"metrics.csv").open("w") as f:
            w=csv.writer(f,lineterminator="\n");w.writerow(["metric","value"])
            flat={};flatten("result",summary,flat)
            for k,v in flat.items():w.writerow([k,v])
        c=read(out/"config.json")
        c.update(status="completed",seed=42,training_seed=42,dataset_split_seed=20260907,actual_assistant_targets=summary["actual_assistant_targets"])
        write(out/"config.json",c)
        (out/"report.md").write_text(f"""# {e}

状态：completed；本实验结果不替换 S10 release。

- MBPP pass@1：{score*100:.2f}%（{round(score*500)}/500）。
- 七项 chat-template macro：{summary['official_macro_percent']:.4f}%。
- IFEval：{json.dumps(ifeval,ensure_ascii=False)}。
- Chat：{b['chat']['success_count']}/10；格式：{b['chat']['format_success_count']}/6；Tool E2E：{b['tool']['end_to_end_success_rate']*100:.1f}%。
- 训练实测：{json.dumps(done,ensure_ascii=False)}。
- 训练 run：{URLS[i]}；评测 run：{eval_url}。

数据、成本对照、限制及阶段判定见 [Phase 3 报告](../../../docs/phases/phase3-lora.md)。
逐项指标及原始结果路径/SHA 见 eval.json；训练过程见 metrics.jsonl（训练组）；checkpoint-manifest.txt 保留权重指纹和存储路径。
MBPP 的空提取 {empty}/500、提取后语法错误 {syntax}/500，仅用于诊断；不得把 loss 下降解释为代码正确率提升。
""")
        if i:
            (out/"metrics.jsonl").write_bytes((a/"metrics/metrics.jsonl").read_bytes())
        summaries.append(summary)
    write(ROOT/"experiments/03-lora/comparison.json",summaries)
    print(json.dumps([{k:s[k] for k in ["experiment_id","mbpp_pass_at_1_percent","official_macro_percent","ifeval_percent","mbpp_diagnostics"]} for s in summaries],ensure_ascii=False,indent=2))

if __name__=="__main__":main()
