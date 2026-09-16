#!/usr/bin/env python3
"""Measure MiniMind pretrain and SFT token budgets with the training-time tokenizer rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--pretrain-root", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0,
        "mean": round(sum(values) / len(values), 3) if values else 0.0,
    }


def tokenizer_fingerprint(path: Path) -> dict[str, Any]:
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        candidate = path / name
        if candidate.exists():
            files[name] = {
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
    return files


def pretrain_stats(root: Path, tokenizer: Any, sequence_length: int) -> dict[str, Any]:
    sources = []
    totals: defaultdict[str, int] = defaultdict(int)
    for path in sorted(root.glob("*.jsonl")):
        counts: defaultdict[str, int] = defaultdict(int)
        raw_lengths: list[int] = []
        processed_lengths: list[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                counts["rows_read"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts["invalid_json"] += 1
                    continue
                text = str(row.get("text", "")).strip()
                if not text:
                    counts["empty_text"] += 1
                    continue
                token_ids = tokenizer(text, add_special_tokens=False).input_ids
                raw_length = len(token_ids)
                processed_length = min(raw_length, sequence_length - 2) + 2
                raw_lengths.append(raw_length)
                processed_lengths.append(processed_length)
                counts["valid_rows"] += 1
                counts["raw_tokens"] += raw_length
                counts["processed_tokens"] += processed_length
                if raw_length > sequence_length - 2:
                    counts["truncated_rows"] += 1
        result = {
            "source": path.stem,
            "path": str(path),
            "sha256": sha256_file(path),
            **dict(counts),
            "raw_length": summarize_lengths(raw_lengths),
            "processed_length": summarize_lengths(processed_lengths),
        }
        sources.append(result)
        for key in (
            "rows_read",
            "valid_rows",
            "invalid_json",
            "empty_text",
            "raw_tokens",
            "processed_tokens",
            "truncated_rows",
        ):
            totals[key] += counts[key]
    totals["sequence_length"] = sequence_length
    totals["truncation_rate"] = round(
        totals["truncated_rows"] / totals["valid_rows"], 6
    ) if totals["valid_rows"] else 0.0
    return {"totals": dict(totals), "sources": sources}


def create_chat_prompt(tokenizer: Any, conversations: list[dict[str, Any]]) -> str:
    messages = []
    tools = None
    for original in conversations:
        message = dict(original)
        if message.get("role") == "system" and message.get("tools"):
            tools = (
                json.loads(message["tools"])
                if isinstance(message["tools"], str)
                else message["tools"]
            )
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )


def assistant_target_count(
    input_ids: list[int],
    bos_id: list[int],
    eos_id: list[int],
    sequence_length: int,
) -> int:
    count = 0
    index = 0
    while index < len(input_ids):
        if input_ids[index : index + len(bos_id)] == bos_id:
            start = index + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(eos_id)] == eos_id:
                    break
                end += 1
            count += max(0, min(end + len(eos_id), sequence_length) - start)
            index = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            index += 1
    return count


def sft_stats(path: Path, tokenizer: Any, sequence_length: int) -> dict[str, Any]:
    bos_id = tokenizer(
        f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
    ).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    totals: defaultdict[str, int] = defaultdict(int)
    per_source: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    input_lengths: list[int] = []
    target_lengths: list[int] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            totals["rows_read"] += 1
            try:
                row = json.loads(line)
                source = str(row.get("source", "unknown"))
                prompt = create_chat_prompt(tokenizer, row["conversations"])
                raw_input_ids = tokenizer(prompt).input_ids
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                totals["invalid_rows"] += 1
                if totals["invalid_rows"] <= 5:
                    print(f"invalid_sft_row={totals['rows_read']} error={type(exc).__name__}", flush=True)
                continue

            input_ids = raw_input_ids[:sequence_length]
            target_tokens = assistant_target_count(
                input_ids, bos_id, eos_id, sequence_length
            )
            input_lengths.append(len(input_ids))
            target_lengths.append(target_tokens)
            totals["valid_rows"] += 1
            totals["input_tokens"] += len(input_ids)
            totals["assistant_target_tokens"] += target_tokens
            per_source[source]["rows"] += 1
            per_source[source]["input_tokens"] += len(input_ids)
            per_source[source]["assistant_target_tokens"] += target_tokens
            if len(raw_input_ids) > sequence_length:
                totals["truncated_rows"] += 1
                per_source[source]["truncated_rows"] += 1
            if target_tokens == 0:
                totals["zero_target_rows"] += 1
                per_source[source]["zero_target_rows"] += 1

    totals["sequence_length"] = sequence_length
    totals["target_retention"] = round(
        1 - totals["zero_target_rows"] / totals["valid_rows"], 6
    ) if totals["valid_rows"] else 0.0
    totals["truncation_rate"] = round(
        totals["truncated_rows"] / totals["valid_rows"], 6
    ) if totals["valid_rows"] else 0.0
    sources = []
    for source, counts in sorted(per_source.items()):
        values = dict(counts)
        values["source"] = source
        values["target_tokens_per_row"] = round(
            counts["assistant_target_tokens"] / counts["rows"], 3
        )
        sources.append(values)
    return {
        "totals": dict(totals),
        "input_length": summarize_lengths(input_lengths),
        "assistant_target_length": summarize_lengths(target_lengths),
        "sources": sources,
        "training_semantics": {
            "augment": False,
            "chat_template": True,
            "assistant_mask": "MiniMind SFTDataset.generate_labels equivalent",
        },
    }


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), trust_remote_code=True
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": str(args.tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_files": tokenizer_fingerprint(args.tokenizer),
        "sequence_length": args.sequence_length,
        "pretrain": pretrain_stats(
            args.pretrain_root, tokenizer, args.sequence_length
        ),
        "sft": sft_stats(args.sft, tokenizer, args.sequence_length),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pretrain": result["pretrain"]["totals"],
                "sft": result["sft"]["totals"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if (
        result["pretrain"]["totals"].get("invalid_json", 0)
        or result["pretrain"]["totals"].get("empty_text", 0)
        or result["sft"]["totals"].get("invalid_rows", 0)
        or result["sft"]["totals"].get("zero_target_rows", 0)
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
