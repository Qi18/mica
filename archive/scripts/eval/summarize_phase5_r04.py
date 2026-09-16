#!/usr/bin/env python3
"""Summarize the preregistered Phase5 R04 target-extraction experiment."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path("/data/artifacts/minimind-lab")
EXP = ROOT / "experiments/04-grpo-cispo"
IDS = {
    "R04A": "R04A-s10-target-baseline-20260908",
    "R04B": "R04B-target-sft-control-20260908",
    "R04C": "R04C-target-grpo-20260908",
    "R04D": "R04D-target-cispo-20260908",
}
MODELS = {
    "R04A": ART / "D01-s10-preference-baseline-20260908/exported-fp32",
    "R04B": ART / IDS["R04B"] / "model-fp32",
    "R04C": ART / IDS["R04C"] / "model-fp32",
    "R04D": ART / IDS["R04D"] / "model-fp32",
}


def load(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def paired_bootstrap(candidate, opponent, seed=42, replicates=10000):
    left = {row["task_id"]: row for row in candidate}
    right = {row["task_id"]: row for row in opponent}
    assert set(left) == set(right) and len(left) == 1000
    diffs = [
        float(left[key]["greedy_correct"]) - float(right[key]["greedy_correct"])
        for key in sorted(left)
    ]
    rng = random.Random(seed)
    means = [
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(replicates)
    ]
    return {
        "mean_difference": statistics.mean(diffs),
        "bootstrap_95_ci": [percentile(means, 0.025), percentile(means, 0.975)],
        "candidate_only_correct": sum(value > 0 for value in diffs),
        "both_same": sum(value == 0 for value in diffs),
        "opponent_only_correct": sum(value < 0 for value in diffs),
        "samples": len(diffs),
        "seed": seed,
        "replicates": replicates,
    }


def diagnostics(sample_rows):
    distribution = Counter(
        str(row["greedy_value"]) if row["greedy_value"] is not None else "INVALID"
        for row in sample_rows
    )
    digit_max = max(distribution[str(value)] for value in range(10)) / len(sample_rows)
    return {
        "distribution": dict(distribution),
        "max_digit_share": digit_max,
        "collapsed": digit_max > 0.35,
    }


def training(candidate):
    if candidate == "R04A":
        return {"method": "baseline", "outer_steps": 0, "optimizer_updates": 0}
    run = load(ART / IDS[candidate] / "run.json")
    return {**run["training"], "swanlab_url": run["swanlab_url"]}


def main():
    sample_rows = {
        candidate: rows(ART / exp_id / "eval/test/samples.jsonl")
        for candidate, exp_id in IDS.items()
    }
    comparisons = {}
    for candidate, opponent in (
        ("R04B", "R04A"),
        ("R04C", "R04A"),
        ("R04D", "R04A"),
        ("R04C", "R04B"),
        ("R04D", "R04B"),
        ("R04D", "R04C"),
    ):
        comparisons[f"{candidate}_vs_{opponent}"] = paired_bootstrap(
            sample_rows[candidate], sample_rows[opponent]
        )

    candidates = {}
    methods = {"R04A": "baseline", "R04B": "sft", "R04C": "grpo", "R04D": "cispo"}
    for candidate, exp_id in IDS.items():
        checkpoint = MODELS[candidate] / "model.safetensors"
        candidates[candidate] = {
            "experiment_id": exp_id,
            "method": methods[candidate],
            "test": load(ART / exp_id / "eval/test/summary.json"),
            "diagnostics": diagnostics(sample_rows[candidate]),
            "behavior": load(ART / exp_id / "eval/behavior/task_eval.json"),
            "training": training(candidate),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "serialization": "float32",
            },
        }

    for candidate in ("R04C", "R04D"):
        row = candidates[candidate]
        row["gates"] = {
            "pass1_ci_low_above_r04a": (
                comparisons[f"{candidate}_vs_R04A"]["bootstrap_95_ci"][0] > 0
            ),
            "no_digit_collapse": not row["diagnostics"]["collapsed"],
            "chat_at_least_7_of_10": row["behavior"]["chat"]["success_count"] >= 7,
            "tool_e2e_at_least_6_of_8": (
                row["behavior"]["tool"]["end_to_end_success_rate"] >= 0.75
            ),
        }
        row["recorded_numeric_gates_pass"] = all(row["gates"].values())
        row["promotion_status"] = "held-for-sft-control-revalidation"

    result = {
        "status": "completed-pending-sft-padding-revalidation",
        "decision": (
            "Recorded scores are provisional: SFT input IDs are left-padded but labels "
            "are right-padded. Promotion is held pending corrected SFT controls. "
            "R04 target-extraction scores do not establish repaired arithmetic capability."
        ),
        "dataset": load(EXP / "data-manifest-r04.json"),
        "data_audit": load(EXP / "data-audit-r04.json"),
        "candidates": candidates,
        "paired_bootstrap": {
            "protocol": {
                "samples": 1000,
                "seed": 42,
                "replicates": 10000,
                "pairing": "task_id",
                "metric": "greedy_correct",
            },
            "comparisons": comparisons,
        },
        "limitations": [
            "single training seed; no stability claim",
            "target extraction is a narrow controlled verifier task, not general reasoning",
            "equal optimizer updates do not imply equal FLOPs or wall-clock compute",
            "test was opened once after all formal candidates completed",
        ],
    }
    existing_path = EXP / "comparison-r04.json"
    if existing_path.exists():
        previous = load(existing_path)
        if "publication_audit" in previous:
            result["publication_audit"] = previous["publication_audit"]
    (EXP / "comparison-r04.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (EXP / "paired-bootstrap-r04.json").write_text(
        json.dumps(result["paired_bootstrap"], ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
