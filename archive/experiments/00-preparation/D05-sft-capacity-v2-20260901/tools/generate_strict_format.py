#!/usr/bin/env python3
"""Generate deterministic, programmatically verifiable strict-format SFT examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_json_task(index: int, rng: random.Random) -> tuple[str, str]:
    name = f"item-{index:04d}"
    score = rng.randint(10, 999)
    active = bool(rng.randint(0, 1))
    prompt = (
        "严格只输出一行 JSON，不要 Markdown，不要解释。"
        f"字段顺序必须是 name、score、active。name={name}，score={score}，active={str(active).lower()}。"
    )
    answer = json.dumps(
        {"name": name, "score": score, "active": active},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return prompt, answer


def make_xml_task(index: int, rng: random.Random) -> tuple[str, str]:
    values = [rng.randint(1, 99) for _ in range(3)]
    prompt = (
        "只输出 XML。根节点必须是 <record>，包含一个 <id> 和三个按原顺序排列的 <value>。"
        f"id={index}，values={values}。"
    )
    answer = (
        f"<record><id>{index}</id>"
        + "".join(f"<value>{value}</value>" for value in values)
        + "</record>"
    )
    return prompt, answer


def make_csv_task(index: int, rng: random.Random) -> tuple[str, str]:
    records = [(f"k{index}-{offset}", rng.randint(1, 999)) for offset in range(3)]
    shuffled = list(records)
    rng.shuffle(shuffled)
    prompt = (
        "将以下记录按 value 升序排序，只输出 CSV，第一行必须是 key,value："
        + json.dumps(shuffled, ensure_ascii=False)
    )
    ordered = sorted(records, key=lambda item: (item[1], item[0]))
    answer = "key,value\n" + "\n".join(f"{key},{value}" for key, value in ordered)
    return prompt, answer


def make_bullets_task(index: int, rng: random.Random) -> tuple[str, str]:
    count = rng.randint(2, 5)
    words = [f"词{index}-{offset}" for offset in range(count)]
    prompt = (
        f"严格输出 {count} 行；每行只能以 '- ' 开头；按给定顺序输出这些词；不要增加标题："
        + "、".join(words)
    )
    answer = "\n".join(f"- {word}" for word in words)
    return prompt, answer


def make_arithmetic_task(index: int, rng: random.Random) -> tuple[str, str]:
    left = rng.randint(10, 999)
    right = rng.randint(10, 999)
    prompt = (
        "计算下面表达式，只输出 JSON，字段只能是 expression 和 result："
        f"{left}+{right}"
    )
    answer = json.dumps(
        {"expression": f"{left}+{right}", "result": left + right},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return prompt, answer


GENERATORS = {
    "json": make_json_task,
    "xml": make_xml_task,
    "csv": make_csv_task,
    "bullets": make_bullets_task,
    "arithmetic_json": make_arithmetic_task,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    category_counts: Counter[str] = Counter()
    seen: set[bytes] = set()

    names = list(GENERATORS)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(args.rows):
            category = names[index % len(names)]
            prompt, answer = GENERATORS[category](index, rng)
            conversations: list[dict[str, Any]] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            canonical = json.dumps(conversations, ensure_ascii=False, sort_keys=True).encode("utf-8")
            digest = hashlib.blake2b(canonical, digest_size=16).digest()
            if digest in seen:
                raise RuntimeError(f"duplicate generated conversation at index {index}")
            seen.add(digest)
            row = {
                "conversations": conversations,
                "source": "strict_format_v1",
                "category": category,
                "generator_index": index,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            category_counts[category] += 1

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/data/generate_strict_format.py",
        "seed": args.seed,
        "rows": args.rows,
        "category_counts": dict(category_counts),
        "exact_duplicates": 0,
        "license": "self-generated-for-minimind-lab",
        "output": str(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
