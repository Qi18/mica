#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


REQUIRED_FILES = (
    "config.json",
    "command.sh",
    "run.json",
    "metrics.csv",
    "eval.json",
    "report.md",
    "checkpoint-manifest.txt",
    "swanlab-url.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one experiment directory.")
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()

    if not experiment_dir.is_dir():
        raise SystemExit(f"not a directory: {experiment_dir}")

    missing = [
        name for name in REQUIRED_FILES if not (experiment_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(f"missing files: {', '.join(missing)}")

    parsed: dict[str, object] = {}
    for json_name in ("config.json", "run.json", "eval.json"):
        with (experiment_dir / json_name).open(encoding="utf-8") as source:
            parsed[json_name] = json.load(source)

    with (experiment_dir / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        reader = csv.reader(source)
        header = next(reader, None)
    if not header:
        raise SystemExit("metrics.csv has no header")

    run = parsed["run.json"]
    if not isinstance(run, dict) or "status" not in run:
        raise SystemExit("run.json has no status")

    print(f"experiment_dir={experiment_dir}")
    print(f"status={run['status']}")
    print("validation=pass")


if __name__ == "__main__":
    main()
