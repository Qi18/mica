#!/usr/bin/env python3
"""Summarize MiniMind pretraining loss and sampled NVIDIA telemetry."""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


LOSS_RE = re.compile(r"\((\d+)/(\d+)\), loss: ([0-9.]+).*lr: ([0-9.]+)")


def parse_float(value: str) -> float:
    return float(value.strip().split()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--per-gpu-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--gpu-sample-interval-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.gpu_sample_interval_seconds <= 0:
        parser.error("--gpu-sample-interval-seconds must be positive")
    points = []
    text = args.driver_log.read_text(encoding="utf-8", errors="replace")
    for match in LOSS_RE.finditer(text):
        points.append({
            "step": int(match.group(1)),
            "total_steps": int(match.group(2)),
            "loss": float(match.group(3)),
            "learning_rate": float(match.group(4)),
        })
    if not points:
        raise RuntimeError("No training points found")

    gpu_rows = []
    with args.gpu_log.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 9:
                continue
            gpu_rows.append({
                "timestamp": row[0].strip(),
                "index": int(row[1]),
                "gpu_util": parse_float(row[3]),
                "memory_util": parse_float(row[4]),
                "memory_used_mib": parse_float(row[5]),
                "power_w": parse_float(row[7]),
                "temperature_c": parse_float(row[8]),
            })
    active = [row for row in gpu_rows if row["gpu_util"] > 0]
    start = datetime.fromisoformat(args.started_at.replace("Z", "+00:00"))
    finish = datetime.fromtimestamp(args.driver_log.stat().st_mtime, tz=timezone.utc)
    wall_seconds = (finish - start).total_seconds()
    total_token_slots = (
        points[-1]["step"] * args.world_size * args.per_gpu_batch_size * args.max_seq_len
    )
    windows = {}
    for window in (100, 1000):
        selected = points[-window:]
        windows[str(window)] = mean(point["loss"] for point in selected)

    result = {
        "status": "completed",
        "started_at": start.isoformat().replace("+00:00", "Z"),
        "finished_at": finish.isoformat().replace("+00:00", "Z"),
        "wall_seconds": wall_seconds,
        "wall_minutes": wall_seconds / 60,
        "train_gpu_hours": wall_seconds * args.world_size / 3600,
        "logged_points": len(points),
        "final_step": points[-1]["step"],
        "total_steps": points[-1]["total_steps"],
        "final_loss": points[-1]["loss"],
        "final_learning_rate": points[-1]["learning_rate"],
        "loss_mean_last_logged_points": windows,
        "loss_mean_first_100_logged_points": mean(point["loss"] for point in points[:100]),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "token_slot_throughput_per_second_upper_bound": total_token_slots / wall_seconds,
        "token_slot_throughput_note": "Upper bound based on padded sequence slots; it is not effective non-padding token throughput.",
        "gpu_telemetry": {
            "sample_interval_seconds": args.gpu_sample_interval_seconds,
            "rows": len(gpu_rows),
            "active_rows": len(active),
            "mean_gpu_util_percent_active": mean(row["gpu_util"] for row in active),
            "median_gpu_util_percent_active": median(row["gpu_util"] for row in active),
            "mean_memory_used_mib_active": mean(row["memory_used_mib"] for row in active),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in gpu_rows),
            "mean_power_w_per_gpu_active": mean(row["power_w"] for row in active),
            "peak_power_w": max(row["power_w"] for row in gpu_rows),
            "mean_temperature_c_active": mean(row["temperature_c"] for row in active),
            "peak_temperature_c": max(row["temperature_c"] for row in gpu_rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
