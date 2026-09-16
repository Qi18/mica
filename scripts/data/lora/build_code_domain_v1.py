#!/usr/bin/env python3
"""Build and audit the Phase 3 CodeAlpaca train/validation plus held-out MBPP test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

CODEALPACA_SHA256 = "7a213e61a8a23633a1b9a9aab424b7f0c717865eb159be7c5e3f2061e991973b"
CODEALPACA_REVISION = "152bb5e9a29651266b018106053980070a0521a1"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
SCHEMA_KEYS = ("role", "content", "reasoning_content", "tools", "tool_calls")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.casefold())


def grams(value: str, size: int = 5) -> set[str]:
    value = normalize(value)
    if len(value) < size:
        return {value} if value else set()
    return {value[i:i + size] for i in range(len(value) - size + 1)}


def format_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("instruction", "")).strip()
    extra = str(row.get("input", "")).strip()
    return prompt + (("\n\nInput:\n" + extra) if extra else "")


def msg(role: str, content: str) -> dict[str, str]:
    return {
        "role": role,
        "content": content.strip(),
        "reasoning_content": "",
        "tools": "",
        "tool_calls": "",
    }


def target_stats(tokenizer: Any, conversations: list[dict[str, str]], max_length: int) -> tuple[int, bool]:
    rendered = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(rendered, add_special_tokens=True).input_ids
    truncated = len(ids) > max_length
    ids = ids[:max_length]
    bos = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    targets = 0
    i = 0
    while i < len(ids):
        if ids[i:i + len(bos)] != bos:
            i += 1
            continue
        start = i + len(bos)
        end = start
        while end < len(ids) and ids[end:end + len(eos)] != eos:
            end += 1
        targets += max(0, min(end + len(eos), max_length) - start)
        i = end + len(eos)
    return targets, truncated


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            payload = (stable_json(row) + "\n").encode()
            handle.write(payload)
            digest.update(payload)
    return {"rows": len(rows), "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer"))
    parser.add_argument("--validation-rows", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=20260907)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--near-threshold", type=float, default=0.80)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    mbpp = [dict(row) for row in load_dataset(
        "google-research-datasets/mbpp", "full", split="test", revision=MBPP_REVISION
    )]
    test_norm = [normalize(str(row["text"])) for row in mbpp]
    exact_test: dict[str, list[int]] = defaultdict(list)
    test_grams = [grams(str(row["text"])) for row in mbpp]
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(test_norm):
        exact_test[value].append(idx)
        for gram in test_grams[idx]:
            inverted[gram].append(idx)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    with args.source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            prompt = format_prompt(row)
            output = str(row.get("output", "")).strip()
            normalized = normalize(prompt)
            reason = ""
            matched: list[int] = []
            if not prompt or not output:
                reason = "empty_prompt_or_output"
            elif normalized in seen:
                reason = "duplicate_train_prompt"
            elif normalized in exact_test:
                reason, matched = "exact_mbpp_test_prompt", exact_test[normalized]
            else:
                contained = [
                    idx for idx, test_value in enumerate(test_norm)
                    if len(normalized) >= 20 and len(test_value) >= 20
                    and (normalized in test_value or test_value in normalized)
                ]
                if contained:
                    reason, matched = "containment_mbpp_test_prompt", contained
                else:
                    prompt_grams = grams(prompt)
                    intersections: Counter[int] = Counter()
                    for gram in prompt_grams:
                        intersections.update(inverted.get(gram, ()))
                    near = [
                        idx for idx, intersection in intersections.items()
                        if intersection / max(1, len(prompt_grams) + len(test_grams[idx]) - intersection)
                        >= args.near_threshold
                    ]
                    if near:
                        reason, matched = "near_duplicate_mbpp_test_prompt", near
            if reason:
                rejected.append({
                    "source_line": line_number,
                    "reason": reason,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "mbpp_task_ids": [mbpp[idx]["task_id"] for idx in matched],
                })
                continue
            seen.add(normalized)
            origin = hashlib.sha256(stable_json(row).encode()).hexdigest()
            accepted.append({
                "origin_id": f"code_alpaca_20k:{origin}",
                "conversations": [msg("user", prompt), msg("assistant", output)],
            })

    accepted.sort(key=lambda row: hashlib.sha256(
        f"{args.split_seed}:{row['origin_id']}".encode()
    ).hexdigest())
    if len(accepted) <= args.validation_rows:
        raise SystemExit("not enough accepted rows")
    validation, train = accepted[:args.validation_rows], accepted[args.validation_rows:]
    train_payload = [{"conversations": row["conversations"]} for row in train]
    validation_payload = [{"conversations": row["conversations"]} for row in validation]
    provenance = [
        {"split": split_name, "row_index": index, "origin_id": row["origin_id"]}
        for split_name, rows in (("train", train), ("validation", validation))
        for index, row in enumerate(rows)
    ]

    split_stats: dict[str, Any] = {}
    for name, rows in (("train", train), ("validation", validation)):
        target_tokens = zero_targets = truncated_rows = schema_errors = 0
        for row in rows:
            conversations = row["conversations"]
            schema_errors += sum(tuple(item) != SCHEMA_KEYS for item in conversations)
            targets, truncated = target_stats(tokenizer, conversations, args.max_length)
            target_tokens += targets
            zero_targets += int(targets == 0)
            truncated_rows += int(truncated)
        split_stats[name] = {
            "rows": len(rows),
            "assistant_target_tokens": target_tokens,
            "zero_target_rows": zero_targets,
            "truncated_rows": truncated_rows,
            "schema_errors": schema_errors,
        }

    files = {
        "train.jsonl": write_jsonl(args.output_dir / "train.jsonl", train_payload),
        "validation.jsonl": write_jsonl(args.output_dir / "validation.jsonl", validation_payload),
        "mbpp_test.jsonl": write_jsonl(args.output_dir / "mbpp_test.jsonl", mbpp),
        "rejected_contamination.jsonl": write_jsonl(
            args.output_dir / "rejected_contamination.jsonl", rejected
        ),
        "provenance.jsonl": write_jsonl(args.output_dir / "provenance.jsonl", provenance),
    }
    sets = {
        "train": {normalize(row["conversations"][0]["content"]) for row in train},
        "validation": {normalize(row["conversations"][0]["content"]) for row in validation},
        "test": set(test_norm),
    }
    overlaps = {
        "train_validation_exact": len(sets["train"] & sets["validation"]),
        "train_test_exact": len(sets["train"] & sets["test"]),
        "validation_test_exact": len(sets["validation"] & sets["test"]),
    }
    source_sha = sha256_file(args.source)
    gates = {
        "source_rows_match": len(train) + len(validation) + len(rejected) == 20022,
        "source_sha_pinned": source_sha == CODEALPACA_SHA256,
        "mbpp_test_rows_match": len(mbpp) == 500,
        "no_zero_targets": all(x["zero_target_rows"] == 0 for x in split_stats.values()),
        "no_schema_errors": all(x["schema_errors"] == 0 for x in split_stats.values()),
        "no_exact_cross_split_overlap": all(value == 0 for value in overlaps.values()),
    }
    status = "accepted" if all(gates.values()) else "rejected"
    manifest = {
        "schema_version": 1,
        "protocol_version": "phase3-code-domain-v1r1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "domain": "python-code-instruction-following",
        "train_source": {
            "repo_id": "sahil2801/CodeAlpaca-20k",
            "revision": CODEALPACA_REVISION,
            "license": "cc-by-4.0",
            "path": str(args.source.resolve()),
            "sha256": source_sha,
        },
        "heldout_test": {
            "repo_id": "google-research-datasets/mbpp",
            "revision": MBPP_REVISION,
            "config": "full",
            "split": "test",
            "license": "cc-by-4.0",
            "used_for_training_or_selection": False,
        },
        "split": {
            "algorithm": "sha256(seed:origin_id)",
            "seed": args.split_seed,
            **split_stats,
        },
        "contamination": {
            "near_duplicate_threshold": args.near_threshold,
            "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
            **overlaps,
        },
        "files": files,
        "max_seq_len": args.max_length,
        "gates": gates,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if status != "accepted":
        raise SystemExit("dataset audit rejected; inspect manifest.json")
    (args.output_dir / "_SUCCESS").write_text(stable_json({
        "status": "accepted",
        "manifest_sha256": sha256_file(manifest_path),
    }) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
