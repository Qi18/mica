#!/usr/bin/env python3
"""Aggregate fixed three-arm Agentic RL results; never choose a checkpoint by test score."""
import json,argparse
from pathlib import Path
import numpy as np
ROOT=Path("/data/artifacts/minimind-lab")
OUT=ROOT/"phase6-agentic-rl-v2-summary-20260908"
def load(p):return json.loads(p.read_text())
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def pct(n,d):return f"{n}/{d} ({100*n/d:.2f}%)"
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--local-only',action='store_true');args=parser.parse_args()
 assert not OUT.exists();OUT.mkdir()
 result={"status":"exploratory-completed-not-promoted","single_seed":42,"arms":{},"paired_deltas":{}}
 for arm in ("A01","A02","A03"):
  val=ROOT/f"{arm}-agentic-rl-v2-val-20260908";test=ROOT/f"{arm}-agentic-rl-v2-test-20260908"
  assert (val/"DONE.json").exists() and (test/"DONE.json").exists()
  x={"validation":load(val/"summary.json"),"test":load(test/"summary.json"),"model":load(test/"eval-manifest.json"),
     "swanlab_val":(val/"swanlab-url.txt").read_text().strip(),"swanlab_test":(test/"swanlab-url.txt").read_text().strip()}
  if arm!="A01":
   method={"A02":"grpo","A03":"cispo"}[arm];train=ROOT/f"{arm}-agentic-{method}-v2-20260908"
   x["training"]=load(train/"DONE.json");x["last_metrics"]=read(train/"metrics.jsonl")[-1]
   x["swanlab_train"]=(train/"swanlab-url.txt").read_text().strip()
  result["arms"][arm]=x
 arrays={}
 for arm in result["arms"]:
  arrays[arm]={}
  for domain in ("graph","tools"):
   predictions=read(ROOT/f"{arm}-agentic-rl-v2-test-20260908"/domain/"predictions.jsonl")
   arrays[arm][domain]={p["id"]:int(p["end_to_end_success"]) for p in predictions}
 for arm,base in (("A02","A01"),("A03","A01"),("A03","A02")):
  comparisons={}
  both=[]
  for domain in ("graph","tools"):
   assert arrays[arm][domain].keys()==arrays[base][domain].keys()
   ids=sorted(arrays[arm][domain]);delta=np.array([arrays[arm][domain][i]-arrays[base][domain][i] for i in ids],dtype=float)
   both.extend(delta.tolist());rng=np.random.default_rng(42)
   samples=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
   comparisons[domain]={"difference_pp":float(delta.mean()*100),"ci95_pp":(np.quantile(samples,[.025,.975])*100).tolist(),
                        "improved_tasks":int((delta>0).sum()),"regressed_tasks":int((delta<0).sum())}
  delta=np.array(both);rng=np.random.default_rng(42);samples=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
  comparisons["combined"]={"difference_pp":float(delta.mean()*100),"ci95_pp":(np.quantile(samples,[.025,.975])*100).tolist()}
  result["paired_deltas"][arm+"-minus-"+base]=comparisons
 result["test_scope"]="240 graph + 240 tools, same synthetic families, now opened; no test-based model selection"
 result["limitations"]=["one seed; small compute budget","same-template-family synthetic fixtures, not BFCL/open-world agents",
 "task-level bootstrap assumes tasks independent; shared templates may make CI optimistic",
 "A01 failed old Tool retention; this exploratory exception does not promote any model","IFEval and seven general benchmarks not run in this attempt"]
 (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 run=None
 if not args.local_only:
  import swanlab
  run=swanlab.init(project="MiniMind-Lab",experiment_name="Phase6-Agentic-RL-v2-Fixed-Comparison-20260908",
                  group="Phase6-Agent-RL",job_type="evaluation",config={"arms":["A01","A02","A03"],"status":result["status"],"test_tasks":480,"seed":42})
  (OUT/"swanlab-url.txt").write_text(run.url+"\n")
 metrics={}
 for arm,x in result["arms"].items():
  for domain in ("graph","tools"):
   metrics[arm+"/test_"+domain+"_e2e"]=x["test"][domain]["end_to_end_success"]
  metrics[arm+"/legacy_chat"]=x["validation"]["behavior"]["chat"]["success_rate"]
  metrics[arm+"/legacy_tool"]=x["validation"]["behavior"]["tool"]["end_to_end_success_rate"]
 if run:
  swanlab.log(metrics);swanlab.finish()
 else:
  (OUT/'swanlab-url.txt').write_text('n/a-local-only-approval-required\n')
  (OUT/'upload-status.json').write_text(json.dumps({'status':'awaiting-user-approval','destination':'SwanLab MiniMind-Lab','would_upload':metrics},indent=2)+'\n')
 lines=["# Phase6 Agentic RL v2：A01 / GRPO / CISPO固定对照","","状态：exploratory-completed-not-promoted；日期2026-09-08。",
        "用户明确要求在A01旧Tool未通过时开始探索性RL。该例外不代表A01被接受，S10仍为通用基线。","","## 训练与实现",
        "三臂同起点A01-v3 LR3e-6 checkpoint-570；A01不更新，A02 GRPO、A03 CISPO。",
        "两组同seed42、LR1e-6、32outer×2inner=64updates、每outer4prompt×4rollout，共512episodes。",
        "训练仅用v3-r1 train/task pools，不用val/test。graph/tools各半；顺序和生成上限相同，实际token消耗分别记录。",
        "reward=0.8严格E2E+0.2正确有序工具前缀；工具观察只进上下文，只有生成assistant action/EOS参与loss。",
        "两组共同token-mean、beta0.01 KL，GRPO ratio clip[0.8,1.2]，CISPO upper IS cap2并stop-gradient。",
        "这是控制surrogate差异的实现：GRPO使用token-normalized变体，CISPO加入共同KL，不是原论文全配方复现。",
        "生成与teacher-forced old/ref/new统一FP32，关闭TF32和dropout；每批核验新旧ratio、掩码、真实采样概率差。",
        "T1/top_p1/top_k0；对微小生成/回算差做detached IS修正。固定末步，不按val/test选择checkpoint。","","## v1失败与v2修复",
        "v1 BF16两组分别在38/22更新后触发generation/teacher logprob最大差>0.20门禁，未保存最终模型、未评测test。",
        "保留中止日志、配置、源码和FAILURE.json，不能把不等预算中止点用于算法优劣比较。",
        "v2不放宽阈值，改为统一FP32并从同一A01重新训练。FP32 probe128条成功121条，32group中7组有方差。",
        "probe初始最大概率差约2.72e-5。正式训练审计值见各metrics.jsonl。","","## 训练结果",
        "| 臂 | 更新 | 采样E2E | 实际assistant tokens | 训练计时秒 |",
        "|---|---:|---:|---:|---:|"]
 for arm in ("A02","A03"):
  t=result["arms"][arm]["training"]
  lines.append(f"| {arm} | {t['optimizer_updates']} | {100*t['sampled_e2e']:.2f}% | {t['assistant_tokens']} | {t['wall_seconds']:.1f} |")
 lines+=["","训练reward/loss不是SFT loss或评测分数；两组on-policy样本会随模型更新分叉。训练计时不包含全部后续评测，GPU与驻留任务共享。",
         "","## 冻结测试与旧能力回归","",
         "| 臂 | graph test | tools test | 合计E2E | 旧Chat | 旧Tool |","|---|---:|---:|---:|---:|---:|"]
 for arm,x in result["arms"].items():
  g=x["test"]["graph"];t=x["test"]["tools"];b=x["validation"]["behavior"]
  gn=round(g["end_to_end_success"]*g["tasks"]);tn=round(t["end_to_end_success"]*t["tasks"])
  lines.append(f"| {arm} | {pct(gn,g['tasks'])} | {pct(tn,t['tasks'])} | {pct(gn+tn,g['tasks']+t['tasks'])} | {b['chat']['success_count']}/10 | {round(b['tool']['end_to_end_success_rate']*8)}/8 |")
 lines+=["","Agent三臂统一FP32、greedy、同任务预算；旧Chat/Tool统一既有FP16脚本/max_new_tokens128。",
         "测试此前未被模型评测，三臂SHA冻结后一次性打开；此后不再称为未见测试，也不能继续用它调参。",
         "全部逐题工具/参数/执行/最终答案/成本见各test目录predictions.jsonl与summary.json。","",
         "## 配对差异（percentage points，10000次task-level bootstrap95%区间）","",
         "| 对比 | graph差异 | tools差异 | 合计差异 [95%CI] |","|---|---:|---:|---:|"]
 for key,x in result["paired_deltas"].items():
  c=x["combined"]
  lines.append(f"| {key} | {x['graph']['difference_pp']:+.2f} | {x['tools']['difference_pp']:+.2f} | {c['difference_pp']:+.2f} [{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}] |")
 lines+=["","区间包含0时，本轮不足以证明稳定提升；即便未包含0，也只适用于当前合成任务与预算。",
         "单seed、共享模板会限制统计外推，不能据此宣称算法普遍优劣。",
         "","## 保持与收口边界",
         "A01起点旧Tool为5/8，本轮比较RL相对A01的变化，但正式保持门槛仍是S10的7/8，不能降低。",
         "IFEval、七项基准未在这轮运行。探索性结果归档不等于Phase6正式验收，默认模型不替换。",
         "","## 证据入口",
         "- 预注册：docs/phases/phase6-rl-v1-preregister.md、phase6-rl-v2-preregister.md。",
         "- 冻结：experiments/05-agentic-rl/rl-v2-test-freeze.json（models/data SHA）。",
         "- 训练：scripts/launch/train_phase6_rl_fp32.py；评测：scripts/eval/eval_phase6_rl_v2.py。",
         "- 权重仅在CPFS；训练/评测provenance含实际源码和数据指纹。未经请求未commit/push。",
         f"- 汇总SwanLab：{run.url if run else '未上传，等待用户确认；仅本地汇总'}"]
 for arm,x in result["arms"].items():
  for k in ("swanlab_train","swanlab_val","swanlab_test"):
   if k in x:lines.append(f"- {arm} {k}: {x[k]}")
 (OUT/"report.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"metrics":metrics,"paired_deltas":result["paired_deltas"],"swanlab":run.url if run else None},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
