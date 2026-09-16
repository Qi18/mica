#!/usr/bin/env python3
"""Build a small deterministic pre-tokenized corpus for the Stage2 model probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"source not found: {args.source}")
    if args.samples <= 0 or args.max_length < 4:
        raise SystemExit("samples must be positive and max-length must be >= 4")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    rows: list[list[int]] = []
    invalid_json = 0
    with args.source.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if len(rows) >= args.samples:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            text = str(record.get("text") or "")
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length - 2,
            ).input_ids
            tokens = [tokenizer.bos_token_id, *token_ids, tokenizer.eos_token_id]
            tokens += [tokenizer.pad_token_id] * (args.max_length - len(tokens))
            rows.append(tokens)

    if len(rows) != args.samples:
        raise SystemExit(f"expected {args.samples} rows, built {len(rows)}")

    input_ids = torch.tensor(rows, dtype=torch.int32)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    payload = {
        "input_ids": input_ids,
        "labels": labels,
        "metadata": {
            "samples": args.samples,
            "max_length": args.max_length,
            "source": str(args.source),
            "source_sha256": args.source_sha256,
            "tokenizer": str(args.tokenizer),
            "tokenizer_vocab_size": len(tokenizer),
            "invalid_json_before_sample_limit": invalid_json,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, args.output)

    manifest = {
        **payload["metadata"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "output_size": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "valid_target_tokens": int((labels[:, 1:] != -100).sum().item()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
