#!/usr/bin/env python3
"""Publish the completed Phase5 comparison to the shared MiniMind-Lab project once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "experiments/04-grpo-cispo/comparison.json"
RECEIPT = Path("/data/artifacts/minimind-lab/phase5-evaluation-swanlab.json")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    comparison = read(COMPARISON)
    metrics = {}
    for candidate, row in comparison["candidates"].items():
        prefix = candidate.lower()
        metrics[f"{prefix}/math_pass_at_1"] = row["test"]["pass_at_1"]
        metrics[f"{prefix}/math_sample_accuracy"] = row["test"]["sample_accuracy"]
        metrics[f"{prefix}/math_pass_at_4"] = row["test"]["pass_at_k"]
        metrics[f"{prefix}/choice_valid"] = row["test"]["choice_valid_at_1"]
        metrics[f"{prefix}/strict_format"] = row["test"]["strict_format_at_1"]
        metrics[f"{prefix}/max_answer_position_share"] = row["choice_diagnostics"]["max_position_share_all_samples"]
        metrics[f"{prefix}/official_seven_macro"] = row["seven"]["macro"]
        metrics[f"{prefix}/ifeval_prompt_strict"] = row["ifeval"]["prompt_strict"]
        metrics[f"{prefix}/chat_success_rate"] = row["behavior"]["chat"]["success_rate"]
        metrics[f"{prefix}/tool_e2e"] = row["behavior"]["tool"]["end_to_end_success_rate"]
        if row["training"].get("mean_training_reward") is not None:
            metrics[f"{prefix}/mean_training_reward"] = row["training"]["mean_training_reward"]
        for gate, passed in row.get("gates", {}).items():
            metrics[f"{prefix}/gate/{gate}"] = int(passed)
    for name, row in comparison["paired_bootstrap"]["comparisons"].items():
        metrics[f"paired/{name}/pass1_difference"] = row["mean_difference"]
        metrics[f"paired/{name}/pass1_ci_low"] = row["bootstrap_95_ci"][0]
        metrics[f"paired/{name}/pass1_ci_high"] = row["bootstrap_95_ci"][1]
        metrics[f"paired/{name}/pass4_difference"] = row["pass_at_4"]["mean_difference"]

    payload = {
        "status": "dry-run",
        "project": "MiniMind-Lab",
        "experiment_name": "Phase5-Evaluation-R02A-R02B-R02C-R02D",
        "group": "Phase5-Verifiable-RL",
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
                experiment_name="Phase5-Evaluation-R02A-R02B-R02C-R02D",
                group="Phase5-Verifiable-RL",
                job_type="evaluation",
                config={
                    "phase": 5,
                    "candidates": ["R02A", "R02B", "R02C", "R02D"],
                    "seed": 42,
                    "bootstrap_replicates": 10000,
                    "status": comparison["status"],
                    "decision": comparison["decision"],
                },
            )
            swanlab.log(metrics, step=0)
            url = run.url
            swanlab.finish()
            payload.update({"status": "logged", "url": url})
            write(RECEIPT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
