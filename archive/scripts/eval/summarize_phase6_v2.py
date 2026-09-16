#!/usr/bin/env python3
"""Summarize frozen Phase6 v2 evidence; no model selection or training."""
import json
from pathlib import Path
import numpy as np

ROOT=Path("/data/artifacts/minimind-lab")
OUT=ROOT/"phase6-v2-evaluation-summary-20260908"
def get(p):return json.loads((ROOT/p).read_text())
def rows(p):return [json.loads(l) for l in (ROOT/p).read_text().splitlines()]

def main():
    if OUT.exists():raise SystemExit("refuse overwrite or duplicate cloud run")
    base=get("A00-v2-frozen-test-20260908/agent-test/summary.json")
    sft=get("A01-v2-frozen-test-20260908/agent-test/summary.json")
    bs=rows("A00-v2-frozen-test-20260908/agent-test/predictions.jsonl")
    ss=rows("A01-v2-frozen-test-20260908/agent-test/predictions.jsonl")
    assert [r["id"] for r in bs]==[r["id"] for r in ss] and len(ss)==240
    delta=np.array([float(s["end_to_end_success"])-float(b["end_to_end_success"]) for b,s in zip(bs,ss)])
    rng=np.random.default_rng(42)
    samples=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    val=rows("A01-agent-sft-v2-pilot-20260908/validation-agent/predictions.jsonl")
    cf=rows("A01-v2-counterfactual-20260908/predictions.jsonl")
    assert [r["id"] for r in val]==[r["id"] for r in cf]
    paired=sum(v["end_to_end_success"] and c["end_to_end_success"] for v,c in zip(val,cf))
    changed=sum(v["end_to_end_success"] and c["end_to_end_success"] and v["trace"][-1]!=c["trace"][-1] for v,c in zip(val,cf))
    probe=get("A01-v2-rl-eligibility-probe-20260908/summary.json")
    summary={"status":"A01 evaluated; Phase6 not accepted; RL held for zero-variance probe",
             "test":{"A00":base,"A01":sft,"paired_delta":float(delta.mean()),
                     "paired_bootstrap_ci95":np.quantile(samples,[.025,.975]).tolist(),"bootstrap_replicates":10000,"seed":42},
             "counterfactual":{"tasks":len(cf),"both_conditions_success":paired,
                               "both_success_and_final_changes":changed,
                               "interpretation":"hidden-value intervention within this environment, not general causal reasoning"},
             "rl_probe":probe,
             "behavior":{"A00":get("A00-v2-frozen-test-20260908/behavior/task_eval.json"),
                         "A01":get("A01-v2-frozen-test-20260908/behavior/task_eval.json")},
             "pending":["IFEval","seven general benchmarks","meaningful Agentic RL comparison"],
             "swanlab_sources":{}}
    names={"transfer":"A01-v2-transfer-baseline-20260908/validation-agent",
           "sft":"A01-agent-sft-v2-pilot-20260908","counterfactual":"A01-v2-counterfactual-20260908",
           "probe":"A01-v2-rl-eligibility-probe-20260908","A00_test":"A00-v2-frozen-test-20260908/agent-test",
           "A01_test":"A01-v2-frozen-test-20260908/agent-test"}
    summary["swanlab_sources"]={k:(ROOT/p/"swanlab-url.txt").read_text().strip() for k,p in names.items()}
    OUT.mkdir()
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    import swanlab
    run=swanlab.init(project="MiniMind-Lab",experiment_name="Phase6-A01-v2-Frozen-Evaluation-Summary-20260908",
                    group="Phase6-Agent",job_type="evaluation",config={"status":summary["status"],"sources":summary["swanlab_sources"]})
    (OUT/"swanlab-url.txt").write_text(run.url+"\n")
    values={"test/delta_e2e":float(delta.mean()),"counterfactual/paired_success":paired/len(cf),
            "probe/zero_variance_groups":probe["zero_variance_groups"]}
    for name,obj in (("A00",base),("A01",sft)):
        values.update({name+"/test/"+k:v for k,v in obj.items() if isinstance(v,(int,float))})
        values.update({name+"/chat/"+k:v for k,v in summary["behavior"][name]["chat"].items() if isinstance(v,(int,float))})
        values.update({name+"/tool/"+k:v for k,v in summary["behavior"][name]["tool"].items() if isinstance(v,(int,float))})
    swanlab.log(values);swanlab.finish()
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
