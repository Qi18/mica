#!/usr/bin/env python3
"""Local-only artifact archiving on L20. No network, git commit, or push."""
import csv,difflib,hashlib,io,json,shutil,subprocess
from pathlib import Path
ROOT=Path("/data/projects/minimind-lab")
ART=Path("/data/artifacts/minimind-lab")
DEST=ROOT/"experiments/05-agentic-rl"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def patch_file(p,new):
 old=p.read_text() if p.exists() else ""
 if new==old:return
 assert new.endswith("\n")
 if old and not old.endswith("\n"):raise ValueError("refuse no-newline existing file")
 p.parent.mkdir(parents=True,exist_ok=True);rel=str(p.relative_to(ROOT))
 diff="".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile="a/"+rel if p.exists() else "/dev/null",tofile="b/"+rel))
 subprocess.run(["patch","-p1","--fuzz=0"],input=diff,text=True,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 assert p.read_text()==new
def main():
 jobs=[]
 for name in ["A01-v3-agent-rl-probe-20260908","A01-v3-agent-rl-fp32-probe-20260908"]:
  jobs.append((name,name,"completed",["unit-checks.json","mask-ratio-check.json","provenance.json","metrics.jsonl","rollouts.jsonl","DONE.json","swanlab-url.txt"]))
 for version in ["v1","v2"]:
  for arm,method in [("A02","grpo"),("A03","cispo")]:
   name=f"{arm}-agentic-{method}-{version}-20260908"
   files=["unit-checks.json","mask-ratio-check.json","provenance.json","metrics.jsonl","rollouts.jsonl","swanlab-url.txt"]
   files+=["FAILURE.json"] if version=="v1" else ["DONE.json"]
   jobs.append((name,name,"aborted-audit-failed" if version=="v1" else "completed-not-promoted",files))
 for arm in ["A01","A02","A03"]:
  for split in ["val","test"]:
   name=f"{arm}-agentic-rl-v2-{split}-20260908"
   files=["DONE.json","summary.json","eval-manifest.json","swanlab-url.txt",
          "graph/summary.json","graph/predictions.jsonl","tools/summary.json","tools/predictions.jsonl"]
   if split=="val":files+=["behavior/task_eval.json","behavior/samples.jsonl","behavior/system_metrics.json"]
   jobs.append((name,name,"completed",files))
 summary="phase6-agentic-rl-v2-summary-20260908"
 jobs.append(("A02-A03-rl-v2-summary-20260908",summary,"completed-not-promoted",["summary.json","report.md","swanlab-url.txt","upload-status.json"]))
 # Validate all sources before copying anything.
 for exp,src,status,names in jobs:
  assert not (DEST/exp).exists(),exp
  for n in names:
   p=ART/src/n;assert p.is_file() and p.stat().st_size<10000000,str(p)
 report=(ART/summary/"report.md").read_text()
 report+="\n## 附加审计与收益解释\n\n"
 report+="CISPO相对A01改善的10题包括graph chain1题/redirect1题，以及tool random5题/multistep2题/time1题，未发现原先通过题退化。\n"
 report+="随机数子集主要是数值参数抽取改善；其中一个graph样本是将观察结果358正确读出，另一个修复了错误lookup参数。不能把这些结果写成开放域规划涌现。\n"
 report+="旧Tool的E2E三臂均5/8，但tool-selection/argument-validity/execution子指标从A01的75%降到两组RL的62.5%；因此不能写成全部旧工具指标保持。\n"
 report+="GRPO记录的最大ratio超界token比例约0.355%，19/32个outer的末次inner非零；CISPO在cap2下没有触发上界裁剪。\n"
 report+="这不是CISPO大规模高off-policy收益的验证；单seed、少量更新与当前任务足以限制算法优劣外推。\n"
 report+="汇总上传曾被自动安全审核拦截；当前使用local-only模式生成本报告，未绕过拦截。训练及各臂评测已有SwanLab记录，汇总另待授权。\n"
 urls=[];newrows=[];count=0;size=0
 prior=json.loads((DEST/"A01-v3-mixed-lr3e6-20260908/checkpoint-manifest.json").read_text())["weights"]["model.safetensors"]
 freeze=json.loads((DEST/"rl-v2-test-freeze.json").read_text())
 assert prior["sha256"]==freeze["models"]["A01"]["sha256"]
 for exp,src,status,names in jobs:
  out=DEST/exp;out.mkdir()
  receipt={"artifact_root":str(ART/src),"files":{},"weights_copied":False}
  for n in names:
   source=ART/src/n;target=out/n;target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(source,target)
   assert sha(source)==sha(target)
   receipt["files"][n]={"sha256":sha(target),"bytes":target.stat().st_size}
   count+=1;size+=target.stat().st_size
  patch_file(out/"archive-manifest.json",json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
  url=(out/"swanlab-url.txt").read_text().strip();urls.append((exp,url,status))
  if "agentic-grpo-v2" in exp or "agentic-cispo-v2" in exp:
   d=json.loads((out/"DONE.json").read_text())
   patch_file(out/"checkpoint-manifest.json",json.dumps({"promoted":False,"model_path":d["model_path"],"weight_sha256":d["weight_sha256"],
     "base_verified_against_preexisting_A01_manifest":prior,"frozen_reference":freeze["models"]["A01"]},indent=2)+"\n")
  if exp!="A02-A03-rl-v2-summary-20260908":
   patch_file(out/"report.md",f"# {exp}\n\n状态：{status}。\n\n[完整RL对照报告](../phase6-rl-v2-report.md)。\n\nSwanLab：{url}\n\n数据、训练和评测证据见本目录；权重仅在CPFS，未复制到Git。\n")
  newrows.append([exp,"phase6-agentic",status,"234797d+phase6-working-tree","393e387e9ad99f0f04c296e4c5e7353f4444629f","A01-v3@29b3428831f3","mixed-agent-v3-r1","1xL20","fp32+bf16-autocast" if "v1-20260908" in exp or exp=="A01-v3-agent-rl-probe-20260908" else "fp32; behavior-fp16","42",url,f"experiments/05-agentic-rl/{exp}/report.md","","2026-09-08","2026-09-08"])
 patch_file(DEST/"phase6-rl-v2-report.md",report)
 p=ROOT/"experiments/registry.csv";old=p.read_text();oldrows=list(csv.reader(io.StringIO(old)))
 assert not {x[0] for x in newrows}&{x[0] for x in oldrows}
 assert all(len(x)==len(oldrows[0]) for x in newrows)
 buf=io.StringIO();csv.writer(buf,lineterminator="\n").writerows(newrows);new=old+buf.getvalue()
 assert list(csv.reader(io.StringIO(new)))[:len(oldrows)]==oldrows;patch_file(p,new)
 p=ROOT/"docs/phases/README.md";old=p.read_text()
 before="| 6 | Agent SFT 与 Agentic RL | [v3实验报告](../../experiments/05-agentic-rl/phase6-v3-report.md) | 进行中：A01-v3旧Tool5/8未过门槛，A02/A03未启动 |"
 after="| 6 | Agent SFT 与 Agentic RL | [RL v2对照报告](../../experiments/05-agentic-rl/phase6-rl-v2-report.md) | 探索性A02/A03完成；CISPO专项提升，旧Tool5/8未过门槛，未晋级 |"
 assert before in old;patch_file(p,old.replace(before,after))
 p=DEST/"README.md";old=p.read_text()
 old=old.replace("当前：A01-v3 completed-not-promoted，Phase6未收口。","当前：A02/A03探索性RL对照完成，Phase6未收口，S10继续保留。")
 old=old.replace("A02/A03未训练","当时A02/A03未训练").replace("A02/A03未启动","当时A02/A03未启动")
 patch_file(p,old+"\n## A02/A03探索性Agentic RL\n\n用户要求在A01保持门禁未过时进行探索性对照：固定末步、三臂冻结test。\nA01与GRPO均451/480，CISPO461/480（+2.08pp，配对95%CI +0.83至+3.54）。\n旧Chat8/10、旧Tool5/8；工具选择子指标下降，IFEval/七项尚未跑，不晋级。\n见[训练、评测、失败与统计报告](phase6-rl-v2-report.md)。汇总仅本地，上传待确认。\n")
 p=DEST/"phase6-v3-report.md";patch_file(p,p.read_text()+"\n## 后续状态（不修改上述SFT历史结论）\n\n随后用户明确要求探索性Agentic RL，固定三臂权重后打开了v3 frozen test做一次性比较。\n因此这些test当前已打开，不能再用于SFT/RL调参后声称独立测试；见[RL v2报告](phase6-rl-v2-report.md)。\n")
 p=ROOT/"docs/phases/swanlab-runs.md"
 patch_file(p,p.read_text()+"\n### Phase6 Agentic RL attempts（包含审计失败）\n\n"+"\n".join(f"- {e} ({s}): {u}" for e,u,s in urls)+"\n")
 for exp,src,status,names in jobs:
  m=json.loads((DEST/exp/"archive-manifest.json").read_text())
  for n,meta in m["files"].items():assert sha(DEST/exp/n)==meta["sha256"]
 subprocess.run(["git","diff","--check"],cwd=ROOT,check=True)
 print(json.dumps({"new_experiments":len(jobs),"verified_artifact_files":count,"artifact_bytes":size,"report":str(DEST/"phase6-rl-v2-report.md"),"committed":False,"pushed":False},ensure_ascii=False))
if __name__=="__main__":main()
