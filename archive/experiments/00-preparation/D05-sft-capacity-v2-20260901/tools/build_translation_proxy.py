#!/usr/bin/env python3
"""Build deterministic EN-to-ZH translation conversations from aligned Alpaca fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def normalized(text: Any) -> str:
    return " ".join(str(text or "").split())


def pair(row: dict[str, Any], english_key: str, chinese_key: str, kind: str) -> dict[str, Any] | None:
    english = normalized(row.get(english_key))
    chinese = normalized(row.get(chinese_key))
    if not english or not chinese or english == chinese:
        return None
    conversations = [
        {
            "role": "user",
            "content": (
                "Translate the following English text into Simplified Chinese. "
                "Return only the translation.\n\n" + english
            ),
        },
        {"role": "assistant", "content": chinese},
    ]
    return {
        "conversations": conversations,
        "translation_direction": "en_to_zh",
        "aligned_field": kind,
        "provenance": "silk-road/alpaca-data-gpt4-chinese",
    }


def canonical(item: dict[str, Any]) -> bytes:
    payload = json.dumps(
        item["conversations"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload.encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    seen: set[bytes] = set()
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            counts["rows_read"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            candidates = (
                pair(row, "instruction", "instruction_zh", "instruction"),
                pair(row, "output", "output_zh", "output"),
            )
            for item in candidates:
                if item is None:
                    counts["missing_pair"] += 1
                    continue
                digest = hashlib.blake2b(canonical(item), digest_size=16).digest()
                if digest in seen:
                    counts["exact_duplicate"] += 1
                    continue
                seen.add(digest)
                item["source_record_hash"] = digest.hex()
                target.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"written_{item['aligned_field']}"] += 1
                counts["rows_written"] += 1

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/data/build_translation_proxy.py",
        "source": str(args.input),
        "source_revision": "81a6dfd72f416aff605e7d189bfbbc46a2511fee",
        "source_license": "apache-2.0",
        "scope": "translation proxy only; no summarization examples",
        "counts": dict(counts),
        "output": str(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 1 if counts["invalid_json"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
