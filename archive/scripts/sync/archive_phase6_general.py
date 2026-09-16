#!/usr/bin/env python3
"""Archive completed general regression on L20 only; no external uploads."""
import csv,difflib,hashlib,io,json,shutil,subprocess
from pathlib import Path
R=Path("/data/projects/minimind-lab");A=Path("/data/artifacts/minimind-lab");D=R/"experiments/05-agentic-rl"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def patch(p,text):
 old=p.read_text() if p.exists() else ""
 assert text.endswith("\n") and (not old or old.endswith("\n"))
 if old==text:return
 p.parent.mkdir(parents=True,exist_ok=True);rel=str(p.relative_to(R))
 diff="".join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile="a/"+rel if p.exists() else "/dev/null",tofile="b/"+rel))
 subprocess.run(["patch","-p1","--fuzz=0"],input=diff,text=True,cwd=R,check=True,stdout=subprocess.DEVNULL)
 assert p.read_text()==text
def main():
 summary=A/"phase6-general-comparison-20260908";scores=json.loads((summary/"summary.json").read_text())
 rows=[];archived=[]
 for arm in ["S10","A01","A02","A03"]:
  exp=("A00-s10" if arm=="S10" else arm)+"-general-regression-20260908"
  src=A/f"phase6-general-{arm}-20260908";out=D/exp
  assert not out.exists();assert (src/"DONE.json").exists();out.mkdir()
  selected={n:src/n for n in ["manifest.json","DONE.json","ifeval-command.json","seven-command.json"]}
  for mode in ["ifeval","seven"]:
   found=list((src/mode).rglob("results*.json"));assert len(found)==1;selected[mode+"-results.json"]=found[0]
  receipt={"artifact_root":str(src),"archived":{},"full_logs_and_samples":"retained in CPFS only","published":False}
  for n,p in selected.items():
   shutil.copyfile(p,out/n);assert sha(out/n)==sha(p)
   receipt["archived"][n]={"source":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
  # Hash the full evidence without copying long texts into Git.
  receipt["cpfs_only"]={str(p.relative_to(src)):{"sha256":sha(p),"bytes":p.stat().st_size}
                       for p in src.rglob("*") if p.is_file() and (p.name.startswith("samples") or p.suffix==".log")}
  patch(out/"archive-manifest.json",json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
  patch(out/"swanlab-url.txt","n/a-local-only-no-upload\n")
  patch(out/"report.md",f"# {exp}\n\n完整评测完成；结果仅在L20，未上传。\n\n见[四臂通用回归报告](../phase6-general-regression-report.md)。\n原始完整日志和逐题结果：{src}。权重未改动，不进行晋级。\n")
  rows.append([exp,"phase6-agentic","completed","234797d+phase6-working-tree","393e387e9ad99f0f04c296e4c5e7353f4444629f",
               scores["arms"][arm]["manifest"]["model_sha256"],"ifeval541+official-seven-full","1xL20","float16","42",
               "n/a-local-only-no-upload",f"experiments/05-agentic-rl/{exp}/report.md","","2026-09-08","2026-09-08"])
  archived.append(out)
 exp="A00-A03-general-regression-20260908";out=D/exp;assert not out.exists();out.mkdir()
 for n in ["summary.json","report.md","ifeval-paired-scores.jsonl"]:
  shutil.copyfile(summary/n,out/n);assert sha(out/n)==sha(summary/n)
 patch(out/"swanlab-url.txt","n/a-local-only-no-upload\n")
 patch(out/"archive-manifest.json",json.dumps({"source":str(summary),"files":{n:sha(summary/n) for n in ["summary.json","report.md","ifeval-paired-scores.jsonl"]},"published":False},indent=2)+"\n")
 patch(D/"phase6-general-regression-report.md",(summary/"report.md").read_text())
 rows.append([exp,"phase6-agentic","completed","234797d+phase6-working-tree","393e387e9ad99f0f04c296e4c5e7353f4444629f","S10+A01+A02+A03; see manifests",
 "ifeval541+official-seven-full","1xL20-per-arm","float16","42","n/a-local-only-no-upload",f"experiments/05-agentic-rl/{exp}/report.md","","2026-09-08","2026-09-08"])
 p=R/"experiments/registry.csv";old=p.read_text();prior=list(csv.reader(io.StringIO(old)))
 assert not {x[0] for x in rows}&{x[0] for x in prior}
 assert all(len(x)==len(prior[0]) for x in rows)
 buf=io.StringIO();csv.writer(buf,lineterminator="\n").writerows(rows);new=old+buf.getvalue()
 assert list(csv.reader(io.StringIO(new)))[:len(prior)]==prior;patch(p,new)
 vals=scores["arms"]
 note="\n## 通用回归补充（2026-09-08，随后评测，保留上文历史状态）\n\n"
 note+="本轮S10/A01/GRPO/CISPO全部完成IFEval541题与七项全量评测，协议与样本哈希一致。\n"
 note+="\n| 模型 | IFEval strict | 七项macro |\n|---|---:|---:|\n"
 for arm in vals:note+=f"| {arm} | {vals[arm]['ifeval_strict_pass']}/541 | {vals[arm]['seven_macro_percent']:.4f}% |\n"
 note+="\n完整逐项变化与配对区间见[通用回归报告](phase6-general-regression-report.md)。\n"
 note+="不临时添加下降容差；旧Tool仍5/8，模型不晋级。完整原始证据保存在CPFS，仅轻量结果入仓库，未上传、未commit/push。\n"
 for name in ["README.md","phase6-rl-v2-report.md"]:
  p=D/name;patch(p,p.read_text()+note)
 p=R/"docs/phases/README.md";old=p.read_text()
 before="探索性A02/A03完成；CISPO专项提升，旧Tool5/8未过门槛，未晋级"
 after="探索性RL与通用回归完成；旧Tool5/8未过门槛，未晋级"
 assert before in old;patch(p,old.replace(before,after))
 p=R/"docs/phases/swanlab-runs.md"
 patch(p,p.read_text()+"\n### Phase6 general regression（仅L20本地）\n\n"+ "\n".join("- "+x[0]+": n/a-local-only-no-upload；完整证据见实验目录。" for x in rows)+"\n")
 for out in archived:
  receipt=json.loads((out/"archive-manifest.json").read_text())
  for n,v in receipt["archived"].items():assert sha(out/n)==v["sha256"]
 subprocess.run(["git","diff","--check"],cwd=R,check=True)
 print(json.dumps({"new_registry_rows":len(rows),"report":str(D/"phase6-general-regression-report.md"),"published":False,"committed":False},ensure_ascii=False))
if __name__=="__main__":main()
