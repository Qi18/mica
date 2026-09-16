#!/usr/bin/env python3
"""Assemble compact, reproducible Phase5 verifier-RL evidence from CPFS artifacts."""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path("/data/artifacts/minimind-lab")
EXP_ROOT = ROOT / "experiments/04-grpo-cispo"
IDS = {
    "R02A": "R02A-s10-math-baseline-20260908",
    "R02B": "R02B-math-sft-control-20260908",
    "R02C": "R02C-math-grpo-20260908",
    "R02D": "R02D-math-cispo-20260908",
}
COMMON_ARTIFACT = {
    "R02A": "D01-s10-preference-baseline-20260908",
    "R02B": IDS["R02B"],
    "R02C": IDS["R02C"],
    "R02D": IDS["R02D"],
}
TRAIN_URL = {
    "R02A": "n/a-no-training",
    "R02B": "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/wqhvn9wd",
    "R02C": "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/6t4l6v1o",
    "R02D": "https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/k6ez8f9j",
}
CHECKPOINT = {
    "R02A": ART / "D01-s10-preference-baseline-20260908/exported-fp32/model.safetensors",
    "R02B": ART / IDS["R02B"] / "model-fp32/model.safetensors",
    "R02C": ART / IDS["R02C"] / "model-fp32/model.safetensors",
    "R02D": ART / IDS["R02D"] / "model-fp32/model.safetensors",
}


def load(path: Path):
    return json.loads(path.read_text())


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered))))]


def aligned_rows(left, right):
    a = {row["task_id"]: row for row in left}
    b = {row["task_id"]: row for row in right}
    assert set(a) == set(b) and len(a) == 400
    return [(a[key], b[key]) for key in sorted(a)]


def paired_bootstrap(left, right, field, seed=42, replicates=10000):
    pairs = aligned_rows(left, right)
    diffs = [float(a[field]) - float(b[field]) for a, b in pairs]
    rng = random.Random(seed)
    means = [sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(replicates)]
    return {
        "metric": field,
        "mean_difference": statistics.mean(diffs),
        "bootstrap_95_ci": [percentile(means, 0.025), percentile(means, 0.975)],
        "candidate_only_correct": sum(float(a[field]) > float(b[field]) for a, b in pairs),
        "both_same": sum(float(a[field]) == float(b[field]) for a, b in pairs),
        "opponent_only_correct": sum(float(a[field]) < float(b[field]) for a, b in pairs),
        "samples": len(pairs),
        "seed": seed,
        "replicates": replicates,
    }


def seven_metrics(candidate):
    root = ART / COMMON_ARTIFACT[candidate] / "eval/official-seven-fp32"
    result_file = next(root.glob("**/results*.json"))
    data = load(result_file)
    results, groups = data["results"], data.get("groups", {})
    values = {}
    for task in ("ceval-valid", "cmmlu", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa"):
        row = groups.get(task, results[task])
        key = "acc_norm,none" if "acc_norm,none" in row else "acc,none"
        values[task] = row[key]
    values["macro"] = statistics.mean(values.values())
    values["artifact_sha256"] = sha256(result_file)
    return values


def ifeval_metrics(candidate):
    root = ART / COMMON_ARTIFACT[candidate] / "eval/ifeval-fp32"
    result_file = next(root.glob("**/results*.json"))
    row = load(result_file)["results"]["ifeval"]
    return {
        "prompt_strict": row["prompt_level_strict_acc,none"],
        "instruction_strict": row["inst_level_strict_acc,none"],
        "prompt_loose": row["prompt_level_loose_acc,none"],
        "instruction_loose": row["inst_level_loose_acc,none"],
        "artifact_sha256": sha256(result_file),
    }


def behavior_metrics(candidate):
    path = ART / COMMON_ARTIFACT[candidate] / "eval/behavior-fp32/task_eval.json"
    value = load(path)
    value["artifact_sha256"] = sha256(path)
    return value


def training_metrics(candidate):
    if candidate == "R02A":
        return {"method": "baseline", "outer_steps": 0, "optimizer_updates": 0, "wall_seconds": 0.0}
    rows = jsonl(ART / IDS[candidate] / "metrics.jsonl")
    done = next(row for row in reversed(rows) if row["event"] == "completed")
    train = [row for row in rows if row["event"] == "train"]
    result = dict(done)
    if done["method"] == "sft":
        result["logged_loss_first"] = train[0]["loss"]
        result["logged_loss_last"] = train[-1]["loss"]
        result["logged_grad_norm_mean"] = statistics.mean(row["grad_norm"] for row in train)
    else:
        midpoint = len(train) // 2
        for key in ("reward", "degenerate_group_rate", "all_zero_group_rate", "all_one_group_rate", "entropy", "approx_kl", "clip_fraction", "mean_completion_length", "rollout_seconds"):
            result[f"logged_{key}_mean"] = statistics.mean(row[key] for row in train)
        result["logged_reward_first_half"] = statistics.mean(row["reward"] for row in train[:midpoint])
        result["logged_reward_second_half"] = statistics.mean(row["reward"] for row in train[midpoint:])
    return result


def choice_diagnostics(rows):
    choices = Counter(row["greedy_choice"] or "INVALID" for row in rows)
    valid = sum(choices[label] for label in "ABCD")
    return {
        "distribution": dict(choices),
        "max_position_share_all_samples": max(choices[label] for label in "ABCD") / len(rows),
        "max_position_share_among_valid": max(choices[label] for label in "ABCD") / valid if valid else 0.0,
        "unique_valid_choices": sum(choices[label] > 0 for label in "ABCD"),
        "collapsed": max(choices[label] for label in "ABCD") / len(rows) > 0.60,
    }


def main():
    samples = {candidate: jsonl(ART / exp_id / "eval/test/samples.jsonl") for candidate, exp_id in IDS.items()}
    bootstrap = {
        "protocol": {"samples": 400, "seed": 42, "replicates": 10000, "pairing": "task_id", "primary_metric": "greedy_correct"},
        "comparisons": {},
    }
    for candidate, opponent in (("R02B", "R02A"), ("R02C", "R02A"), ("R02D", "R02A"), ("R02C", "R02B"), ("R02D", "R02B"), ("R02D", "R02C")):
        name = f"{candidate}_vs_{opponent}"
        bootstrap["comparisons"][name] = paired_bootstrap(samples[candidate], samples[opponent], "greedy_correct")
        bootstrap["comparisons"][name]["pass_at_4"] = paired_bootstrap(samples[candidate], samples[opponent], "pass_at_k")

    candidates = {}
    for candidate, exp_id in IDS.items():
        candidates[candidate] = {
            "experiment_id": exp_id,
            "method": {"R02A": "baseline", "R02B": "sft", "R02C": "grpo", "R02D": "cispo"}[candidate],
            "validation": load(ART / exp_id / "eval/validation/summary.json"),
            "test": load(ART / exp_id / "eval/test/summary.json"),
            "choice_diagnostics": choice_diagnostics(samples[candidate]),
            "seven": seven_metrics(candidate),
            "ifeval": ifeval_metrics(candidate),
            "behavior": behavior_metrics(candidate),
            "training": training_metrics(candidate),
            "checkpoint": {
                "path": str(CHECKPOINT[candidate]),
                "sha256": sha256(CHECKPOINT[candidate]),
                "bytes": CHECKPOINT[candidate].stat().st_size,
                "serialization": "float32",
            },
            "swanlab_train_url": TRAIN_URL[candidate],
        }

    base = candidates["R02A"]
    for candidate in ("R02C", "R02D"):
        row = candidates[candidate]
        vs_a = bootstrap["comparisons"][f"{candidate}_vs_R02A"]
        vs_b = bootstrap["comparisons"][f"{candidate}_vs_R02B"]
        row["gates"] = {
            "pass1_ci_low_above_r02a": vs_a["bootstrap_95_ci"][0] > 0,
            "pass1_ci_low_above_r02b": vs_b["bootstrap_95_ci"][0] > 0,
            "no_answer_position_collapse": not row["choice_diagnostics"]["collapsed"],
            "seven_macro_regression_within_1pp": row["seven"]["macro"] - base["seven"]["macro"] >= -0.01,
            "ifeval_prompt_strict_regression_within_2pp": row["ifeval"]["prompt_strict"] - base["ifeval"]["prompt_strict"] >= -0.02,
            "chat_at_least_7_of_10": row["behavior"]["chat"]["success_count"] >= 7,
            "format_at_least_4_of_6": row["behavior"]["chat"]["format_success_count"] >= 4,
            "tool_e2e_at_least_6_of_8": row["behavior"]["tool"]["end_to_end_success_rate"] >= 0.75,
            "repetition_anomalies_at_most_2": row["behavior"]["chat"]["repetition_anomaly_count"] <= 2,
        }
        row["all_gates_pass"] = all(row["gates"].values())

    comparison = {
        "status": "completed-not-promoted",
        "decision": "Neither GRPO nor CISPO passes the pre-registered verifier-task gates. Both collapse to answer-position policies; retain S10 as the release checkpoint.",
        "dataset": load(EXP_ROOT / "data-manifest.json"),
        "data_audit": load(EXP_ROOT / "data-audit.json"),
        "candidates": candidates,
        "paired_bootstrap": bootstrap,
        "r01_incident": {
            "status": "invalidated-data-confound",
            "cause": "template_id deterministically predicted answer_position despite balanced marginals",
            "detected_on": "validation",
            "test_opened": False,
            "resolution": "preserve R01 as invalid evidence; rebuild v2 with all 16 answer-position by template cells supported and balanced",
        },
        "limitations": [
            "single training seed; no stability claim",
            "synthetic four-choice arithmetic is not free-form mathematical reasoning",
            "SFT and RL match prompt batches and optimizer updates, not FLOPs, target tokens, or wall-clock",
            "test was opened once only after all R02 candidates completed and was not used for tuning",
        ],
    }
    (EXP_ROOT / "paired-bootstrap.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n")
    (EXP_ROOT / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")

    csv_keys = ("event", "method", "outer_step", "optimizer_update", "wall_seconds", "completion_tokens", "peak_memory_mib", "loss", "grad_norm", "approx_kl", "sampled_logratio_to_ref", "entropy", "clip_fraction", "reward", "group_reward_std", "degenerate_group_rate", "all_zero_group_rate", "all_one_group_rate", "pass_at_group", "mean_completion_length", "rollout_seconds")
    for candidate, exp_id in IDS.items():
        exp = EXP_ROOT / exp_id
        row = candidates[candidate]
        (exp / "eval.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        (exp / "checkpoint-manifest.txt").write_text(
            f"selected_checkpoint={row['checkpoint']['path']}\nsha256={row['checkpoint']['sha256']}\nbytes={row['checkpoint']['bytes']}\nserialization=float32\n"
        )
        source = ART / exp_id / "metrics.jsonl"
        if source.exists():
            records = jsonl(source)
            lines = [",".join(csv_keys)] + [",".join(str(record.get(key, "")) for key in csv_keys) for record in records]
            (exp / "metrics.csv").write_text("\n".join(lines) + "\n")
        run = load(exp / "run.json")
        run.update({
            "status": "completed-not-promoted" if candidate in {"R02C", "R02D"} else "completed",
            "training_state": "n/a" if candidate == "R02A" else "completed",
            "validation_state": "completed",
            "test_state": "completed-once-after-all-training",
            "selected_checkpoint": row["checkpoint"],
            "swanlab_url": TRAIN_URL[candidate],
            "phase_decision": comparison["decision"],
        })
        (exp / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        (exp / "swanlab-url.txt").write_text(TRAIN_URL[candidate] + "\n")

    for exp in EXP_ROOT.glob("R01[BCD]-*"):
        source = ART / exp.name / "metrics.jsonl"
        if source.exists():
            records = jsonl(source)
            lines = [",".join(csv_keys)] + [",".join(str(record.get(key, "")) for key in csv_keys) for record in records]
            (exp / "metrics-invalid.csv").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "status": comparison["status"],
        "decision": comparison["decision"],
        "pass_at_1": {key: value["test"]["pass_at_1"] for key, value in candidates.items()},
        "pass_at_4": {key: value["test"]["pass_at_k"] for key, value in candidates.items()},
        "bootstrap": bootstrap["comparisons"],
        "gates": {key: candidates[key]["gates"] for key in ("R02C", "R02D")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
