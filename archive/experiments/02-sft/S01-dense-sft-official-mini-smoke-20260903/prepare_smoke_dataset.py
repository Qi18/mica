#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(tokenizer, conversations):
    messages = []
    tools = None
    for original in conversations:
        message = dict(original)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)


def target_count(tokenizer, conversations, max_length, bos_id, eos_id):
    ids = tokenizer(render(tokenizer, conversations)).input_ids[:max_length]
    labels = [-100] * len(ids)
    index = 0
    while index < len(ids):
        if ids[index:index + len(bos_id)] == bos_id:
            start = index + len(bos_id)
            end = start
            while end < len(ids) and ids[end:end + len(eos_id)] != eos_id:
                end += 1
            for position in range(start, min(end + len(eos_id), max_length)):
                labels[position] = ids[position]
            index = end + len(eos_id) if end < len(ids) else len(ids)
        else:
            index += 1
    return sum(value != -100 for value in labels[1:]), len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()

    total_rows = sum(1 for _ in args.input.open(encoding="utf-8"))
    generator = random.Random(args.seed)
    candidate_indices = set(generator.sample(range(total_rows), min(total_rows, args.rows + 128)))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    selected = []
    targets = []
    input_lengths = []
    rejected_zero_target = 0
    invalid = 0
    with args.input.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index not in candidate_indices:
                continue
            try:
                row = json.loads(line)
                count, input_length = target_count(tokenizer, row["conversations"], args.max_length, bos_id, eos_id)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid += 1
                continue
            if count == 0:
                rejected_zero_target += 1
                continue
            selected.append(row)
            targets.append(count)
            input_lengths.append(input_length)
            if len(selected) == args.rows:
                break
    if len(selected) != args.rows:
        raise SystemExit(f"insufficient valid rows: {len(selected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    permutation = torch.randperm(len(selected), generator=torch.Generator().manual_seed(args.seed)).tolist()
    validation_indices = set(permutation[:args.validation_size])
    validation_targets = sum(value for index, value in enumerate(targets) if index in validation_indices)
    train_targets = sum(targets) - validation_targets
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "source_rows": total_rows,
        "source_sha256": sha256(args.input),
        "selection": "python_random_sample_without_replacement_sorted_by_source_order",
        "seed": args.seed,
        "rows": len(selected),
        "validation_size": args.validation_size,
        "train_rows": len(selected) - args.validation_size,
        "assistant_targets_total": sum(targets),
        "assistant_targets_train": train_targets,
        "assistant_targets_validation": validation_targets,
        "invalid_candidates": invalid,
        "zero_target_candidates_rejected": rejected_zero_target,
        "max_length": args.max_length,
        "train_augment": False,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "tokenizer_json_sha256": sha256(args.tokenizer / "tokenizer.json")
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
