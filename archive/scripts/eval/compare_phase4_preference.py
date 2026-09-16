#!/usr/bin/env python3
"""Paired bootstrap comparison for Phase4 frozen preference test."""
import json,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ART=Path("/data/artifacts/minimind-lab")
IDS=["D01-s10-preference-baseline-20260908","D02-chosen-only-sft-20260908","D03-dpo-official-20260908"]
DIRS=["preference-test","preference-test-fp32-recovered","preference-test-fp32-recovered"]
def rows(e,d):return [json.loads(x) for x in (ART/e/"eval"/d/"per_pair.jsonl").open()]
def ci(values):
 s=sorted(values);return [s[int(.025*len(s))],s[int(.975*len(s))]]
def compare(a,b,seed=42):
 random.seed(seed);n=len(a);metric="preference_credit"
 diffs=[b[i][metric]-a[i][metric] for i in range(n)]
 boots=[]
 for _ in range(10000):
  boots.append(sum(diffs[random.randrange(n)] for _ in range(n))/n)
 return {"mean_difference":statistics.mean(diffs),"bootstrap_95_ci":ci(boots),"positive":sum(x>0 for x in diffs),"tie":sum(x==0 for x in diffs),"negative":sum(x<0 for x in diffs),"seed":seed,"replicates":10000}
def corr(xs,ys):
 mx,my=statistics.mean(xs),statistics.mean(ys)
 den=(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**.5
 return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den if den else None
def main():
 data={e:rows(e,d) for e,d in zip(IDS,DIRS)}
 out={"metric":"reference-relative preference credit; ties count 0.5","pairs":1000,
      "comparisons":{"D03_minus_D01":compare(data[IDS[0]],data[IDS[2]]),"D03_minus_D02":compare(data[IDS[1]],data[IDS[2]])},"candidates":{}}
 for e,rs in data.items():
  long=[r for r in rs if r["chosen_tokens"]>r["rejected_tokens"]];short=[r for r in rs if r["chosen_tokens"]<=r["rejected_tokens"]]
  out["candidates"][e]={"preference_credit":statistics.mean(r["preference_credit"] for r in rs),"mean_reward_margin":statistics.mean(r["reward_margin"] for r in rs),"chosen_longer_credit":statistics.mean(r["preference_credit"] for r in long),"chosen_not_longer_credit":statistics.mean(r["preference_credit"] for r in short),"margin_length_difference_pearson":corr([r["chosen_tokens"]-r["rejected_tokens"] for r in rs],[r["reward_margin"] for r in rs])}
 # Explicitly quantify deployment checkpoint quantization loss.
 for e in IDS[1:]:
  fp16=rows(e,"preference-test");fp32=data[e]
  out["candidates"][e]["fp16_saved_preference_credit"]=statistics.mean(r["preference_credit"] for r in fp16)
  out["candidates"][e]["fp32_minus_fp16_credit"]=out["candidates"][e]["preference_credit"]-out["candidates"][e]["fp16_saved_preference_credit"]
 out["limitations"]=["single training seed","bootstrap measures held-out pair sampling only","D01 relative reward is zero by definition","test was opened during checkpoint serialization diagnosis; no hyperparameter was tuned after opening it"]
 p=ROOT/"experiments/04-dpo/preference-comparison.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
