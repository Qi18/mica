#!/usr/bin/env python3
"""Independent read-only audit of the frozen Phase 3 dataset and actual loader."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def grams(s):
    s = norm(s)
    return {s[i:i+5] for i in range(max(1, len(s)-4))}

def overlaps(left, right):
    inv = defaultdict(list)
    gs = [grams(s) for s in right]
    for i, g in enumerate(gs):
        for x in g:
            inv[x].append(i)
    hits = []
    for i, s in enumerate(left):
        g = grams(s)
        counts = Counter(j for x in g for j in inv[x])
        for j, n in counts.items():
            score = n / max(1, len(g) + len(gs[j]) - n)
            if score >= .8:
                hits.append({"left": i, "right": j, "jaccard": score})
    return hits

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(".").resolve()))
    from dataset.lm_dataset import SFTDataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("tokenizer")
    manifest = json.loads((a.data_dir/"manifest.json").read_text())
    checks = {}
    for name, meta in manifest["files"].items():
        path = a.data_dir/name
        checks[name] = hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    prompts, stats = {}, {}
    for split in ("train", "validation"):
        ds = SFTDataset(str(a.data_dir/f"{split}.jsonl"), tok, max_length=768, augment=False)
        counts, mask_errors = [], 0
        prompts[split] = []
        for i, row in enumerate(ds.samples):
            prompts[split].append(row["conversations"][0]["content"])
            ids, labels = ds[i]
            counts.append(int((labels[1:] != -100).sum()))
            # All supervised indices must follow the assistant role delimiter.
            seq = ids.tolist()
            starts = [k+len(ds.bos_id) for k in range(len(seq)) if seq[k:k+len(ds.bos_id)] == ds.bos_id]
            active = (labels != -100).nonzero().flatten().tolist()
            mask_errors += int(not starts or any(k < min(starts) for k in active))
        stats[split] = {"rows": len(ds), "shifted_targets": sum(counts),
                        "zero_target_rows": counts.count(0), "nonassistant_mask_errors": mask_errors}
    test = [json.loads(l)["text"] for l in (a.data_dir/"mbpp_test.jsonl").open()]
    near = {"train_validation": overlaps(prompts["train"], prompts["validation"]),
            "train_mbpp": overlaps(prompts["train"], test),
            "validation_mbpp": overlaps(prompts["validation"], test)}
    prov = [json.loads(l) for l in (a.data_dir/"provenance.jsonl").open()]
    report = {"file_hash_checks": checks, "loader_stats": stats,
              "near_prompt_pairs_threshold_0_8": near, "provenance_rows": len(prov),
              "timing": "post-training independent audit; not a pre-training gate",
              "scope": "Phase3 train/validation/MBPP prompt comparison; not a full upstream S10 contamination audit"}
    report["passed"] = all(checks.values()) and all(not s["zero_target_rows"] and not s["nonassistant_mask_errors"] for s in stats.values()) and not near["train_mbpp"] and not near["validation_mbpp"] and not near["train_validation"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "near_prompt_pairs_threshold_0_8"}, ensure_ascii=False))
    print({k:len(v) for k,v in near.items()})

if __name__ == "__main__":
    main()
