#!/usr/bin/env python3
"""Materialize one shared Arrow cache before spawning P02 DDP workers."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    cache_dir = args.cache_dir.resolve()
    if not str(cache_dir).startswith("/data/"):
        raise SystemExit(f"cache directory must be under /data: {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
        cache_dir=str(cache_dir),
        keep_in_memory=False,
    )
    if len(dataset) != args.expected_rows:
        raise SystemExit(
            f"dataset row mismatch: {len(dataset)} != {args.expected_rows}"
        )
    print(
        f"dataset_cache=ready rows={len(dataset)} "
        f"cache_dir={cache_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
