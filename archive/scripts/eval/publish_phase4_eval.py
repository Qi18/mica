#!/usr/bin/env python3
"""Publish the completed Phase4 comparison to the shared MiniMind-Lab project once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "experiments/04-dpo/comparison.json"
RECEIPT = Path("/data/artifacts/minimind-lab/phase4-evaluation-swanlab.json")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    comparison = read(COMPARISON)
    metrics = {}
    for candidate, row in comparison["candidates"].items():
        prefix = candidate.lower()
        metrics[f"{prefix}/preference_credit"] = row["preference"]["preference_credit"]
        metrics[f"{prefix}/reward_margin"] = row["preference"]["reward_margin"]
        metrics[f"{prefix}/official_seven_macro"] = row["seven"]["macro"]
        metrics[f"{prefix}/ifeval_prompt_strict"] = row["ifeval"]["prompt_strict"]
        metrics[f"{prefix}/ifeval_instruction_strict"] = row["ifeval"]["instruction_strict"]
        metrics[f"{prefix}/chat_success_rate"] = row["behavior"]["chat"]["success_rate"]
        metrics[f"{prefix}/tool_e2e"] = row["behavior"]["tool"]["end_to_end_success_rate"]
        metrics[f"{prefix}/mean_generated_tokens"] = row["generation"]["mean_generated_tokens"]
    for name, row in comparison["preference_comparison"]["comparisons"].items():
        metrics[f"preference/{name}/mean_difference"] = row["mean_difference"]
        metrics[f"preference/{name}/ci_low"] = row["bootstrap_95_ci"][0]
        metrics[f"preference/{name}/ci_high"] = row["bootstrap_95_ci"][1]
    for name, row in comparison["blind_judge"]["comparisons"].items():
        metrics[f"blind/{name}/score"] = row["score"]
        metrics[f"blind/{name}/ci_low"] = row["bootstrap_95_ci"][0]
        metrics[f"blind/{name}/ci_high"] = row["bootstrap_95_ci"][1]
        metrics[f"blind/{name}/wins"] = row["D03_win"]
        metrics[f"blind/{name}/ties"] = row["tie"]
        metrics[f"blind/{name}/losses"] = row["opponent_win"]

    payload = {
        "status": "dry-run",
        "project": "MiniMind-Lab",
        "experiment_name": "Phase4-Evaluation-D01-D02-D03",
        "group": "Phase4-DPO",
        "source": str(COMPARISON),
        "metrics": metrics,
    }
    if args.publish:
        if RECEIPT.exists():
            payload = read(RECEIPT)
        else:
            import swanlab
            run = swanlab.init(
                project="MiniMind-Lab",
                experiment_name="Phase4-Evaluation-D01-D02-D03",
                group="Phase4-DPO",
                job_type="evaluation",
                config={
                    "phase": 4,
                    "candidates": ["D01", "D02", "D03"],
                    "seed": 42,
                    "judge": "Qwen3-8B",
                    "status": comparison["status"],
                },
            )
            swanlab.log(metrics, step=0)
            url = run.url
            swanlab.finish()
            payload.update({"status": "logged", "url": url})
            write(RECEIPT, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
