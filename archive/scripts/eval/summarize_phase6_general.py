#!/usr/bin/env python3
"""Local-only four-arm general-regression comparison. No post-hoc pass threshold."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path("/data/artifacts/minimind-lab")
OUT=ROOT/"phase6-general-comparison-20260908"
ARMS=["S10","A01","A02","A03"]
TASKS=["ceval-valid","cmmlu","arc_easy","piqa","openbookqa","hellaswag","social_iqa"]
METRICS=["prompt_level_strict_acc","prompt_level_loose_acc","inst_level_strict_acc","inst_level_loose_acc"]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def result(p):
 files=list(p.rglob("results*.json"));assert len(files)==1,(str(p),len(files))
 return files[0],read(files[0])
def main():
 assert not OUT.exists()
 combined={"arms":{},"comparisons":{},"status":"evaluated-no-promotion","publish":False}
 samples={};snapshots={}
 for arm in ARMS:
  p=ROOT/f"phase6-general-{arm}-20260908";assert (p/"DONE.json").exists()
  ip,iv=result(p/"ifeval");sp,seven=result(p/"seven")
  samplefiles=list((p/"ifeval").rglob("samples_ifeval*.jsonl"));assert len(samplefiles)==1
  raw=[json.loads(l) for l in samplefiles[0].read_text().splitlines()];assert len(raw)==541
  samples[arm]={r["doc_id"]:r for r in raw};assert len(samples[arm])==541
  assert iv["n-samples"]["ifeval"]["effective"]==541
  assert iv["config"]["limit"] is None and seven["config"]["limit"] is None
  for x in (iv,seven):
   assert x["config"]["model_dtype"]=="torch.float16"
   assert x["config"]["random_seed"]==42 and x["config"]["batch_size"]=="16"
  snapshots[arm]={"ifeval":iv,"seven":seven}
  scores={t:100*seven["results"][t].get("acc_norm,none",seven["results"][t].get("acc,none")) for t in TASKS}
  metrics={m:100*iv["results"]["ifeval"][m+",none"] for m in METRICS}
  combined["arms"][arm]={"ifeval_percent":metrics,"ifeval_strict_pass":sum(r["prompt_level_strict_acc"] for r in raw),
     "seven_percent":scores,"seven_macro_percent":statistics.mean(scores.values()),"manifest":read(p/"manifest.json"),
     "wall_seconds":read(p/"DONE.json")["wall_seconds"],"sources":{str(x):sha(x) for x in [ip,sp,samplefiles[0]]},
     "ifeval_version":iv["versions"],"lm_eval_version":iv["lm_eval_version"],"chat_template_sha":iv["chat_template_sha"]}
 for arm in ARMS[1:]:
  for mode in ("ifeval","seven"):
   a=snapshots["S10"][mode];b=snapshots[arm][mode]
   for field in ("versions","n-samples","task_hashes","chat_template_sha","system_instruction_sha"):
    assert a[field]==b[field],(arm,mode,field)
  for i,r in samples["S10"].items():
   assert r["doc_hash"]==samples[arm][i]["doc_hash"] and r["prompt_hash"]==samples[arm][i]["prompt_hash"],(arm,i)
 for arm,base in [("A01","S10"),("A02","A01"),("A03","A01"),("A02","S10"),("A03","S10")]:
  ids=sorted(samples[base]);comparison={}
  for m in METRICS:
   x=[samples[arm][i][m] for i in ids];y=[samples[base][i][m] for i in ids]
   rng=np.random.default_rng(42);indices=rng.integers(0,len(ids),(10000,len(ids)))
   if m.startswith("prompt"):
    delta=np.asarray(x,dtype=float)-np.asarray(y,dtype=float);draws=delta[indices].mean(1)*100
    info={"difference_pp":float(delta.mean()*100),"improved":int((delta>0).sum()),"regressed":int((delta<0).sum())}
   else:
    totals=np.array([len(v) for v in x]);assert all(len(a)==len(b) for a,b in zip(x,y))
    delta=np.array([sum(a)-sum(b) for a,b in zip(x,y)],dtype=float)
    draws=delta[indices].sum(1)/totals[indices].sum(1)*100
    info={"difference_pp":float(delta.sum()/totals.sum()*100),"bootstrap_unit":"prompt cluster"}
   info["ci95_pp"]=np.quantile(draws,[.025,.975]).tolist();comparison[m]=info
  comparison["seven_delta_pp"]={t:combined["arms"][arm]["seven_percent"][t]-combined["arms"][base]["seven_percent"][t] for t in TASKS}
  comparison["seven_macro_delta_pp"]=combined["arms"][arm]["seven_macro_percent"]-combined["arms"][base]["seven_macro_percent"]
  combined["comparisons"][arm+"-minus-"+base]=comparison
 combined["protocol_verified"]={"same_task_hashes":True,"same_chat_template":True,"same_samples":True,"full_ifeval_prompts":541,
   "seven_tasks":TASKS,"seven_effective_samples":{t:snapshots["S10"]["seven"]["results"][t].get("sample_len") for t in TASKS}}
 combined["limitations"]=["single seed; no new tolerance selected after observing results","existing benchmark-family curriculum, not open-world capability proof",
  "seven macro weights each benchmark equally; inspect per-task regression","IFEval paired bootstrap by prompt, not a guarantee of future-model performance",
  "this comparison does not retrain or select a checkpoint; old Tool gate remains separate"]
 OUT.mkdir()
 (OUT/"summary.json").write_text(json.dumps(combined,ensure_ascii=False,indent=2)+"\n")
 compact=[]
 for arm in ARMS:
  for i,r in samples[arm].items():
   compact.append({"arm":arm,"doc_id":i,"doc_hash":r["doc_hash"],"prompt_hash":r["prompt_hash"],**{m:r[m] for m in METRICS}})
 (OUT/"ifeval-paired-scores.jsonl").write_text("".join(json.dumps(r)+"\n" for r in compact))
 lines=["# Phase6 通用能力回归：S10 / A01 / GRPO / CISPO","",
        "日期2026-09-08；四臂本轮完整重跑，只评测不训练，结果仅在L20。","",
        "## 协议",
        "现有lm-eval 0.4.12、IFEval v4完整541题、0-shot、seed42、FP16、batch16、chat template。",
        "IFEval沿用max_gen_toks1280和greedy。七项使用全部样本；优先acc_norm，否则acc，按七项等权macro。",
        "四臂task hashes、样本数、逐题doc/prompt hashes和chat template均一致；权重评测前后SHA不变。",
        "S10为实际训练起点的FP32导出，评测时同样转换FP16；不混用历史评测结果。","",
        "## 核心结果（%）","",
        "| 模型 | IFEval prompt strict | prompt loose | instruction strict | instruction loose | 七项macro |",
        "|---|---:|---:|---:|---:|---:|"]
 for arm,x in combined["arms"].items():
  v=x["ifeval_percent"];lines.append("| "+arm+" | "+" | ".join(f"{v[m]:.4f}" for m in METRICS)+f" | {x['seven_macro_percent']:.4f} |")
 lines+=["","## 七项明细（%）","","| 任务 | S10 | A01 | A02 GRPO | A03 CISPO |","|---|---:|---:|---:|---:|"]
 for t in TASKS:lines.append("| "+t+" | "+" | ".join(f"{combined['arms'][a]['seven_percent'][t]:.4f}" for a in ARMS)+" |")
 lines+=["","## 分离Agent SFT与RL影响","","| 对比 | IFEval strict差值pp [95%CI] | 改善/退化题数 | 七项macro差值pp |","|---|---:|---:|---:|"]
 for name,c in combined["comparisons"].items():
  v=c["prompt_level_strict_acc"]
  lines.append(f"| {name} | {v['difference_pp']:+.4f} [{v['ci95_pp'][0]:+.4f}, {v['ci95_pp'][1]:+.4f}] | {v['improved']}/{v['regressed']} | {c['seven_macro_delta_pp']:+.4f} |")
 lines+=["","区间为10000次配对bootstrap，seed42。instruction-level按prompt整组重采样，完整区间见summary.json。",
         "区间跨0表示当前题集不足以证明稳定变化，不等于能力严格不变；不根据结果临时设容差。",
         "A01-S10反映Agent SFT及混合回放的整体变化；A02/A03-A01才是本轮RL增量。",
         "","## 验收边界",
         "本轮补齐IFEval及七项通用回归证据，不自动晋级或替换S10。旧Tool仍5/8，未达7/8门槛。",
         "任务熟悉度和S10课程的benchmark-family定向性限制外推，不能把这些分数等同开放域聊天能力。",
         "本次不新增训练数据污染审计，也不使用回归结果继续训练或挑选checkpoint。",
         "各模型的完整逐题文本、harness配置和日志在CPFS；仓库只保留轻量指标/哈希和IFEval逐题布尔结果。",
         "没有SwanLab等外部上传，没有commit/push。",
         "","## 复现入口",
         "脚本：scripts/eval/run_phase6_general.py；汇总：scripts/eval/summarize_phase6_general.py。",
         "协议：docs/phases/phase6-general-regression-protocol.md。"]
 for arm,x in combined["arms"].items():lines.append(f"- {arm}: /data/artifacts/minimind-lab/phase6-general-{arm}-20260908（耗时{x['wall_seconds']:.1f}秒）。")
 (OUT/"report.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"arms":{a:{"ifeval_strict":x["ifeval_percent"]["prompt_level_strict_acc"],"strict_pass":x["ifeval_strict_pass"],"seven_macro":x["seven_macro_percent"]} for a,x in combined["arms"].items()},
  "comparisons":combined["comparisons"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
