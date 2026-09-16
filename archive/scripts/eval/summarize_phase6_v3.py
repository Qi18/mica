#!/usr/bin/env python3
"""Apply preregistered joint validation gates to the two fixed-final A01-v3 candidates."""
import json
from pathlib import Path
ROOT=Path("/data/artifacts/minimind-lab")
OUT=ROOT/"phase6-v3-validation-summary-20260908"
def load(p):return json.loads(p.read_text())
def main():
    if OUT.exists():raise SystemExit("refuse duplicate summary")
    results={}
    for lr in ("lr3e6","lr1e6"):
        p=ROOT/f"A01-v3-mixed-{lr}-20260908"
        graph=load(p/"validation-agent/summary.json");tools=load(p/"validation-tools/summary.json")
        behavior=load(p/"behavior/task_eval.json");chat=behavior["chat"];legacy=behavior["tool"]
        gates={"graph_e2e":graph["end_to_end_success"]>=.9,
               "graph_schema":graph["schema_validity"]>=.95,
               "graph_arguments":graph["argument_semantic_accuracy"]>=.9,
               "graph_execution":graph["execution_success"]>=.9,
               "graph_retry":graph["retry_e2e"]>=.6,
               "tools_e2e":tools["end_to_end_success"]>=.85,
               "legacy_chat":chat["success_count"]>=8,
               "legacy_tool":legacy["end_to_end_success_rate"]>=.875,
               "legacy_format":chat["format_success_count"]>=5,
               "legacy_repetition":chat["repetition_anomaly_count"]<=1}
        results[lr]={"graph":graph,"tools":tools,"behavior":behavior,"gates":gates,"all_gates":all(gates.values()),
                     "final_checkpoint":load(p/"best-checkpoint.json"),
                     "train_last":json.loads((p/"metrics.jsonl").read_text().splitlines()[-1]),
                     "swanlab_train":(p/"swanlab-url.txt").read_text().strip(),
                     "swanlab_tools":(p/"validation-tools/swanlab-url.txt").read_text().strip()}
    assert results["lr3e6"]["train_last"]["train/target_tokens"]==results["lr1e6"]["train_last"]["train/target_tokens"]
    passing=[lr for lr in results if results[lr]["all_gates"]]
    selected=max(passing,key=lambda lr:((results[lr]["graph"]["end_to_end_success"]+results[lr]["tools"]["end_to_end_success"])/2,lr=="lr1e6")) if passing else None
    summary={"candidates":results,"selected":selected,"status":"validation-pass-test-pending" if selected else "not-promoted",
             "test_opened":False,"rl_trained":False,"protocol":"phase6-v3-preregister.md; fixed-final checkpoint"}
    OUT.mkdir();(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    import swanlab
    run=swanlab.init(project="MiniMind-Lab",experiment_name="Phase6-A01-v3-Joint-Validation-20260908",
                    group="Phase6-Agent",job_type="evaluation",config={"status":summary["status"],"selected":selected,
                    "sources":{lr:{"train":r["swanlab_train"],"tools":r["swanlab_tools"]} for lr,r in results.items()}})
    (OUT/"swanlab-url.txt").write_text(run.url+"\n")
    metrics={}
    for lr,r in results.items():
        metrics.update({lr+"/graph_e2e":r["graph"]["end_to_end_success"],lr+"/new_tool_e2e":r["tools"]["end_to_end_success"],
                        lr+"/legacy_chat":r["behavior"]["chat"]["success_rate"],lr+"/legacy_tool":r["behavior"]["tool"]["end_to_end_success_rate"],
                        lr+"/all_gates":int(r["all_gates"])})
    swanlab.log(metrics);swanlab.finish()
    print(json.dumps({"selected":selected,**{lr:{"graph":r["graph"]["end_to_end_success"],"tools":r["tools"]["end_to_end_success"],
                       "chat":r["behavior"]["chat"]["success_count"],"legacy_tool":r["behavior"]["tool"]["end_to_end_success_rate"],
                       "gates":r["gates"],"train":r["swanlab_train"]} for lr,r in results.items()}},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
