#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import shutil


ROOT_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT_DIR / "experiments/_template"
STAGE_PATTERN = re.compile(r"^[0-9]{2}-[a-z0-9-]+$")
EXPERIMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]+$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a MiniMind experiment.")
    parser.add_argument("stage", help="for example 01-pretrain")
    parser.add_argument("experiment_id")
    args = parser.parse_args()

    if not STAGE_PATTERN.fullmatch(args.stage):
        parser.error(f"invalid stage directory: {args.stage}")
    if not EXPERIMENT_PATTERN.fullmatch(args.experiment_id):
        parser.error(f"invalid experiment id: {args.experiment_id}")
    if not TEMPLATE_DIR.is_dir():
        raise SystemExit(f"template directory not found: {TEMPLATE_DIR}")

    stage_dir = ROOT_DIR / "experiments" / args.stage
    target_dir = stage_dir / args.experiment_id
    if target_dir.exists():
        raise SystemExit(f"experiment already exists: {target_dir}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_DIR, target_dir)
    for path in target_dir.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        content = content.replace("__EXPERIMENT_ID__", args.experiment_id)
        content = content.replace("__STAGE__", args.stage)
        path.write_text(content, encoding="utf-8")

    command_file = target_dir / "command.sh"
    command_file.chmod(command_file.stat().st_mode | 0o111)
    print(f"created={target_dir}")


if __name__ == "__main__":
    main()
