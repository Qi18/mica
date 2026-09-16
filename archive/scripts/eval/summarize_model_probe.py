#!/usr/bin/env python3
"""Finalize Stage2 aggregate metrics, evaluation, and checkpoint manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lab-commit", required=True)
    parser.add_argument("--started-at", required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    results = experiment / "results"
    architecture = json.loads((results / "architecture.json").read_text(encoding="utf-8"))
    summaries = {
        seed: json.loads((results / f"summary-seed{seed}.json").read_text(encoding="utf-8"))
        for seed in (42, 43, 44)
    }
    failures = [seed for seed, item in summaries.items() if item.get("completed_step") != 100]
    resume = summaries[42]
    if failures or resume.get("resumed_from_step") != 50 or resume.get("optimizer_state_entries_after_resume", 0) <= 0:
        raise SystemExit(f"stage gate failed: seeds={failures}, resume={resume}")
    for variant in ("dense", "moe"):
        probe = architecture[variant]["probe"]
        if not probe["loss_finite"] or not probe["grad_finite"]:
            raise SystemExit(f"{variant} single-GPU probe is non-finite")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    token_rates = [summaries[seed]["mean_tokens_per_second_after_warmup"] for seed in summaries]
    sample_rates = [summaries[seed]["mean_samples_per_second_after_warmup"] for seed in summaries]
    recorded_at = datetime.now(timezone.utc)
    aggregate = {
        "status": "pass",
        "seeds": summaries,
        "architecture": {
            "dense_total_parameters": architecture["dense"]["parameters"]["total"],
            "moe_total_parameters": architecture["moe"]["parameters"]["total"],
            "moe_active_parameters": architecture["moe"]["parameters"]["active_per_token_estimate"],
            "dense_logits_shape": architecture["dense"]["probe"]["logits"],
            "kv_cache": architecture["dense"]["kv_cache"],
        },
        "throughput": {
            "mean_tokens_per_second": statistics.mean(token_rates),
            "stdev_tokens_per_second": statistics.stdev(token_rates),
            "mean_samples_per_second": statistics.mean(sample_rates),
            "seed_values": token_rates,
        },
        "resume": {
            "seed": 42,
            "from_step": 50,
            "to_step": 100,
            "optimizer_state_entries": resume["optimizer_state_entries_after_resume"],
            "swanlab_run_id": resume["swanlab_run_id"],
        },
        "recorded_at": recorded_at.isoformat(),
    }
    (results / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metric_rows = [
        ("dense_total_parameters", aggregate["architecture"]["dense_total_parameters"], "count", "pass"),
        ("moe_total_parameters", aggregate["architecture"]["moe_total_parameters"], "count", "pass"),
        ("moe_active_parameters", aggregate["architecture"]["moe_active_parameters"], "count", "observed"),
        ("mean_tokens_per_second", aggregate["throughput"]["mean_tokens_per_second"], "tokens/s", "observed"),
        ("stdev_tokens_per_second", aggregate["throughput"]["stdev_tokens_per_second"], "tokens/s", "observed"),
        ("mean_samples_per_second", aggregate["throughput"]["mean_samples_per_second"], "samples/s", "observed"),
        ("resume_from_step", 50, "step", "pass"),
        ("resume_to_step", 100, "step", "pass"),
        ("seed_count", 3, "count", "pass"),
    ]
    with (experiment / "metrics.csv").open("w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(["name", "value", "unit", "status", "recorded_at"])
        for name, value, unit, status in metric_rows:
            writer.writerow([name, value, unit, status, recorded_at.isoformat()])

    run_ids = [summaries[seed]["swanlab_run_id"] for seed in (42, 43, 44)]
    (experiment / "swanlab-url.txt").write_text(
        "# SwanLab project: https://swanlab.cn/@richliu0153/MiniMind-Lab\n"
        + "\n".join(f"seed{seed}: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/{summaries[seed]['swanlab_run_id']}" for seed in (42, 43, 44))
        + "\n",
        encoding="utf-8",
    )
    checkpoint_manifest = (
        "# sha256 size_bytes path role\n"
        f"{sha256_file(args.checkpoint)} {args.checkpoint.stat().st_size} {args.checkpoint} seed42_resume_step100\n"
    )
    (experiment / "checkpoint-manifest.txt").write_text(checkpoint_manifest, encoding="utf-8")

    eval_payload = {
        "status": "completed",
        "protocol": "docs/experiment_plan.md",
        "results": {
            "stage_gate": "pass",
            "architecture": "results/architecture.json",
            "summary": "results/summary.json",
            "seed_summaries": [f"results/summary-seed{seed}.json" for seed in (42, 43, 44)],
            "resume_verified": True,
        },
    }
    (experiment / "eval.json").write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_payload = {
        "experiment_id": experiment.name,
        "stage": "00-preparation",
        "preparation_step": "stage2-model-probe",
        "status": "completed",
        "baseline_lab_commit": "9961a5119e4249c48da923f3630c2e8ba362df15",
        "lab_commit": args.lab_commit,
        "minimind_commit": "393e387e9ad99f0f04c296e4c5e7353f4444629f",
        "swanlab_run_ids": run_ids,
        "started_at": args.started_at,
        "finished_at": recorded_at.isoformat(),
        "exit_code": 0,
    }
    (experiment / "run.json").write_text(json.dumps(run_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path = experiment / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["status"] = "completed"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "run_ids": run_ids}, ensure_ascii=False))


if __name__ == "__main__":
    main()
