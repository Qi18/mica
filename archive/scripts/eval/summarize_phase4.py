#!/usr/bin/env python3
"""Assemble compact, reproducible Phase4 comparison evidence from CPFS artifacts."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path("/data/artifacts/minimind-lab")
BLIND = ART / "phase4-blind"
IDS = {
    "D01": "D01-s10-preference-baseline-20260908",
    "D02": "D02-chosen-only-sft-20260908",
    "D03": "D03-dpo-official-20260908",
}
PREF_DIR = {"D01": "preference-test", "D02": "preference-test-fp32-recovered", "D03": "preference-test-fp32-recovered"}
TRAIN_URL = {
    "D01": "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/os17qekn",
    "D02": "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/51vdt97x",
    "D03": "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3w1au7dk",
}
CHECKPOINT = {
    "D01": ART / IDS["D01"] / "exported-fp32/model.safetensors",
    "D02": ART / IDS["D02"] / "last_fp32_recovered.pth",
    "D03": ART / IDS["D03"] / "last_fp32_recovered.pth",
}


def load(path: Path):
    return json.loads(path.read_text())


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def blind_bootstrap(rows, seed=42, replicates=10000):
    scores = [1.0 if r["winner"] == "D03" else 0.5 if r["winner"] == "TIE" else 0.0 for r in rows]
    rng = random.Random(seed)
    means = [sum(scores[rng.randrange(len(scores))] for _ in scores) / len(scores) for _ in range(replicates)]
    return {
        "score": statistics.mean(scores),
        "bootstrap_95_ci": [percentile(means, 0.025), percentile(means, 0.975)],
        "D03_win": sum(r["winner"] == "D03" for r in rows),
        "tie": sum(r["winner"] == "TIE" for r in rows),
        "opponent_win": sum(r["winner"] not in {"D03", "TIE"} for r in rows),
        "samples": len(rows),
        "seed": seed,
        "replicates": replicates,
    }


def seven_metrics(candidate):
    files = list((ART / IDS[candidate] / "eval/official-seven-fp32").glob("**/results*.json"))
    data = load(files[0])
    results = data["results"]
    groups = data.get("groups", {})
    values = {}
    for task in ("ceval-valid", "cmmlu", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa"):
        row = groups.get(task, results[task])
        key = "acc_norm,none" if "acc_norm,none" in row else "acc,none"
        values[task] = row[key]
    values["macro"] = statistics.mean(values.values())
    return values


def ifeval_metrics(candidate):
    files = list((ART / IDS[candidate] / "eval/ifeval-fp32").glob("**/results*.json"))
    row = load(files[0])["results"]["ifeval"]
    return {
        "prompt_strict": row["prompt_level_strict_acc,none"],
        "instruction_strict": row["inst_level_strict_acc,none"],
        "prompt_loose": row["prompt_level_loose_acc,none"],
        "instruction_loose": row["inst_level_loose_acc,none"],
    }


def generation_stats(candidate):
    rows = jsonl(BLIND / f"{candidate}.jsonl")
    return {
        "samples": len(rows),
        "mean_generated_tokens": statistics.mean(r["generated_tokens"] for r in rows),
        "median_generated_tokens": statistics.median(r["generated_tokens"] for r in rows),
        "max_token_cap_count": sum(r["generated_tokens"] >= 192 for r in rows),
        "empty_response_count": sum(not r["response"].strip() for r in rows),
        "artifact_sha256": sha(BLIND / f"{candidate}.jsonl"),
    }


def training_metrics(candidate):
    if candidate == "D01":
        return {"updates": 0, "wall_seconds": 0.0, "chosen_targets": 0, "rejected_targets": 0}
    rows = jsonl(ART / IDS[candidate] / "metrics.jsonl")
    done = next(r for r in reversed(rows) if r["event"] == "completed")
    train_rows = [r for r in rows if r["event"] == "train"]
    return {
        "updates": done["step"],
        "wall_seconds": done["wall_seconds"],
        "chosen_targets": done["chosen_targets"],
        "rejected_targets": done["rejected_targets"],
        "peak_memory_mib": max(r["peak_memory_mib"] for r in train_rows),
        "final_logged_train_loss": train_rows[-1]["loss"],
    }


def main():
    judge_rows = jsonl(BLIND / "judge-full/judgments.jsonl")
    blind = {
        "protocol": {
            "prompts": 200,
            "selection": "first 200 frozen test rows in deterministic file order",
            "generation": "greedy, max_new_tokens=192, open_thinking=false",
            "judge": "Qwen3-8B, blinded A/B, greedy, thinking disabled",
            "judge_config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
            "score": "win=1, tie=0.5, loss=0",
        },
        "comparisons": {},
        "generation": {c: generation_stats(c) for c in IDS},
        "judgments_sha256": sha(BLIND / "judge-full/judgments.jsonl"),
        "limitations": [
            "single judge model and single deterministic decoding",
            "high tie rate; judge result is not human ground truth",
            "bootstrap measures only the fixed 200-prompt sampling uncertainty",
            "the test set was opened only to diagnose checkpoint serialization; no hyperparameter was tuned afterward",
        ],
    }
    for opponent in ("D01", "D02"):
        rows = [r for r in judge_rows if r["opponent"] == opponent]
        item = blind_bootstrap(rows)
        longer = [r for r in rows if r["dpo_tokens"] > r["opponent_tokens"]]
        not_longer = [r for r in rows if r["dpo_tokens"] <= r["opponent_tokens"]]
        item["D03_longer"] = blind_bootstrap(longer) if longer else None
        item["D03_not_longer"] = blind_bootstrap(not_longer) if not_longer else None
        blind["comparisons"][f"D03_vs_{opponent}"] = item

    candidates = {}
    for candidate in IDS:
        pref = load(ART / IDS[candidate] / "eval" / PREF_DIR[candidate] / "summary.json")
        behavior = load(ART / IDS[candidate] / "eval/behavior-fp32/task_eval.json")
        candidates[candidate] = {
            "experiment_id": IDS[candidate],
            "preference": pref,
            "seven": seven_metrics(candidate),
            "ifeval": ifeval_metrics(candidate),
            "behavior": behavior,
            "generation": blind["generation"][candidate],
            "training": training_metrics(candidate),
            "checkpoint": {
                "path": str(CHECKPOINT[candidate]),
                "sha256": sha(CHECKPOINT[candidate]),
                "bytes": CHECKPOINT[candidate].stat().st_size,
                "dtype": "float32",
            },
            "swanlab_train_url": TRAIN_URL[candidate],
        }

    identical = {}
    outputs = {c: [r["response"] for r in jsonl(BLIND / f"{c}.jsonl")] for c in IDS}
    for a, b in (("D01", "D02"), ("D01", "D03"), ("D02", "D03")):
        identical[f"{a}_{b}"] = sum(x == y for x, y in zip(outputs[a], outputs[b]))

    comparison = {
        "status": "completed-not-promoted",
        "decision": "D03 passes the held-out preference objective but fails the pre-registered blind-quality gate against D02; retain S10 as the release checkpoint.",
        "dataset": load(ROOT / "experiments/04-dpo/data-manifest.json"),
        "candidates": candidates,
        "preference_comparison": load(ROOT / "experiments/04-dpo/preference-comparison.json"),
        "blind_judge": blind,
        "exact_identical_responses_out_of_200": identical,
        "serialization_incident": {
            "cause": "the original training driver called model.half() before saving inference checkpoints, erasing small DPO updates",
            "impact": "D03 preference credit fell from 0.718 in recovered float32 state to 0.536 in the invalid float16 checkpoint",
            "fix": "save inference state_dict tensors as float32; recovered final model from resume.pt",
            "invalidated_files": ["D02 best.pth/last.pth", "D03 best.pth/last.pth"],
        },
        "limitations": ["one training seed", "D02 checkpoint-selection metric was corrected after training because DPO loss is not its training objective", "judge-based wins are not human evaluation"],
    }
    out_dir = ROOT / "experiments/04-dpo"
    (out_dir / "blind-judge-summary.json").write_text(json.dumps(blind, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")

    for candidate, exp_id in IDS.items():
        exp = out_dir / exp_id
        c = candidates[candidate]
        (exp / "eval.json").write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")
        (exp / "checkpoint-manifest.txt").write_text(
            f"selected_checkpoint={c['checkpoint']['path']}\n"
            f"sha256={c['checkpoint']['sha256']}\n"
            f"bytes={c['checkpoint']['bytes']}\n"
            "serialization=float32\n"
            + ("invalidated_fp16=best.pth,last.pth\n" if candidate != "D01" else "")
        )
        source_metrics = ART / exp_id / "metrics.jsonl"
        if source_metrics.exists():
            keys = ("event","step","loss","lr","grad_norm","chosen_targets","rejected_targets","peak_memory_mib","wall_seconds","dpo_loss","reward_margin","preference_credit")
            (exp / "metrics.csv").write_text(",".join(keys) + "\n" + "".join(
                ",".join(str(r.get(k, "")) for k in keys) + "\n" for r in jsonl(source_metrics)
            ))
        run = load(exp / "run.json")
        run.update({
            "status": "completed-not-promoted" if candidate == "D03" else "completed",
            "training_state": "completed",
            "final_evaluation_state": "completed",
            "selected_checkpoint": c["checkpoint"],
            "phase_decision": comparison["decision"],
        })
        (exp / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({
        "status": comparison["status"],
        "blind": blind["comparisons"],
        "identical": identical,
        "output": str(out_dir / "comparison.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
