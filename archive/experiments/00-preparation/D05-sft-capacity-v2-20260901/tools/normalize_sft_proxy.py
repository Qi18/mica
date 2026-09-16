#!/usr/bin/env python3
"""Normalize sampled SFT sources into MiniMind conversations with stable splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned = []
    role_map = {"human": "user", "prompter": "user", "gpt": "assistant"}
    for message in messages:
        role = role_map.get(str(message.get("role", "")).lower(), str(message.get("role", "")).lower())
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant", "tool"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def adapt_ultrachat(row: dict[str, Any]) -> list[dict[str, str]]:
    return clean_messages(row.get("messages") or [])


def adapt_gsm8k(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(row["question"]).strip()},
        {"role": "assistant", "content": str(row["answer"]).strip()},
    ]


def adapt_mbpp(row: dict[str, Any]) -> list[dict[str, str]]:
    tests = "\n".join(str(item) for item in row.get("test_list") or [])
    prompt = str(row["text"]).strip()
    if tests:
        prompt += "\n\nYour solution must pass these tests:\n" + tests
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(row["code"]).strip()},
    ]


def adapt_code_alpaca(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = str(row["instruction"]).strip()
    extra = str(row.get("input", "")).strip()
    if extra:
        prompt += "\n\nInput:\n" + extra
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(row["output"]).strip()},
    ]


def adapt_alpaca_gpt4_zh(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = str(row["instruction_zh"]).strip()
    extra = str(row.get("input_zh", "")).strip()
    if extra:
        prompt += "\n\n输入：\n" + extra
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(row["output_zh"]).strip()},
    ]


def adapt_conversations(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("conversations") or [])


ADAPTERS: dict[str, Callable[[dict[str, Any]], list[dict[str, str]]]] = {
    "ultrachat_200k": adapt_ultrachat,
    "oasst1_conversations": adapt_conversations,
    "glaive_function_calling_v2_conversations": adapt_conversations,
    "strict_format_v1": adapt_conversations,
    "translation_bilingual_v1": adapt_conversations,
    "gsm8k": adapt_gsm8k,
    "alpaca_gpt4_zh": adapt_alpaca_gpt4_zh,
    "mbpp": adapt_mbpp,
    "code_alpaca_20k": adapt_code_alpaca,
}


def canonical(messages: list[dict[str, Any]]) -> bytes:
    value = []
    for item in messages:
        normalized = {
            "role": item["role"],
            "content": " ".join(str(item.get("content", "")).split()).lower(),
        }
        for key in ("reasoning_content", "tools", "tool_calls"):
            if item.get(key):
                normalized[key] = item[key]
        value.append(normalized)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return payload.encode("utf-8")


def split_group_digest(messages: list[dict[str, Any]]) -> bytes:
    user_text = "\n".join(
        str(item.get("content", ""))
        for item in messages
        if item.get("role") == "user"
    )
    normalized = unicodedata.normalize("NFKC", user_text).lower()
    group_key = "".join(character for character in normalized if character.isalnum())
    return hashlib.blake2b(group_key.encode("utf-8"), digest_size=16).digest()


def stable_split(digest: bytes) -> str:
    bucket = int.from_bytes(digest[:8], "big") % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    handles = {
        split: (args.output_root / f"sft_proxy_v1.{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    seen: set[bytes] = set()

    try:
        for source_id, adapter in ADAPTERS.items():
            path = args.raw_root / f"{source_id}.jsonl"
            if not path.exists():
                invalid_counts[f"{source_id}:missing_file"] += 1
                continue
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                        messages = adapter(row)
                    except (KeyError, TypeError, json.JSONDecodeError):
                        invalid_counts[f"{source_id}:adapter_error"] += 1
                        continue
                    if (
                        not messages
                        or not any(item["role"] == "user" for item in messages)
                        or not any(item["role"] == "assistant" for item in messages)
                    ):
                        invalid_counts[f"{source_id}:invalid_conversation"] += 1
                        continue
                    digest = hashlib.blake2b(canonical(messages), digest_size=16).digest()
                    if digest in seen:
                        duplicate_counts[source_id] += 1
                        continue
                    seen.add(digest)
                    split = stable_split(split_group_digest(messages))
                    item = {
                        "conversations": messages,
                        "source": source_id,
                        "source_record_hash": digest.hex(),
                    }
                    handles[split].write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    source_counts[source_id] += 1
                    split_counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    outputs = {}
    for split in ("train", "validation", "test"):
        path = args.output_root / f"sft_proxy_v1.{split}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs[split] = {
            "path": str(path),
            "rows": split_counts[split],
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_counts_after_dedup": dict(source_counts),
        "split_counts": dict(split_counts),
        "invalid_counts": dict(invalid_counts),
        "exact_duplicate_counts": dict(duplicate_counts),
        "deferred_sources": {},
        "split_policy": (
            "blake2b(NFKC lowercase alphanumeric concatenated user messages) "
            "modulo 1000 -> 900/50/50"
        ),
        "outputs": outputs,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 1 if invalid_counts else 0


if __name__ == "__main__":
    sys.exit(main())
