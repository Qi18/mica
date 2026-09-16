#!/usr/bin/env python3
"""Download pinned OASST1 rows and build one deterministic best path per tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="OpenAssistant/oasst1")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--languages", default="en,zh")
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--conversation-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def canonical(messages: list[dict[str, str]]) -> bytes:
    normalized = [
        {"role": item["role"], "content": " ".join(item["content"].split()).lower()}
        for item in messages
    ]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def child_key(node: dict[str, Any]) -> tuple[Any, ...]:
    rank = node.get("rank")
    return (
        0 if node.get("review_result") is True else 1,
        rank if isinstance(rank, int) else 10**9,
        1 if node.get("synthetic") else 0,
        str(node["message_id"]),
    )


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from datasets import load_dataset

    allowed_languages = {value.strip() for value in args.languages.split(",") if value.strip()}
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.conversation_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.repo_id,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    raw_rows = 0
    nodes: dict[str, dict[str, Any]] = {}
    language_counts: Counter[str] = Counter()
    with args.raw_output.open("w", encoding="utf-8") as raw_handle:
        for row in dataset:
            item = dict(row)
            raw_handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            raw_rows += 1
            language_counts[str(item.get("lang", ""))] += 1
            message_id = str(item.get("message_id", ""))
            role = str(item.get("role", "")).lower()
            text = str(item.get("text", "")).strip()
            if (
                message_id
                and role in {"prompter", "assistant"}
                and text
                and not item.get("deleted")
                and str(item.get("lang", "")) in allowed_languages
            ):
                nodes[message_id] = item

    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots = []
    for node in nodes.values():
        parent_id = node.get("parent_id")
        if parent_id and str(parent_id) in nodes:
            children[str(parent_id)].append(node)
        elif node.get("role") == "prompter":
            roots.append(node)
    for values in children.values():
        values.sort(key=child_key)
    roots.sort(key=lambda item: (str(item.get("message_tree_id", "")), str(item["message_id"])))

    seen_conversations: set[bytes] = set()
    built = duplicate = incomplete = 0
    with args.conversation_output.open("w", encoding="utf-8") as output:
        for root in roots:
            path = [root]
            visited = {str(root["message_id"])}
            current = root
            for _ in range(31):
                expected = "assistant" if current["role"] == "prompter" else "prompter"
                candidates = [
                    item
                    for item in children.get(str(current["message_id"]), [])
                    if item["role"] == expected and str(item["message_id"]) not in visited
                ]
                if not candidates:
                    break
                current = candidates[0]
                path.append(current)
                visited.add(str(current["message_id"]))
            if path[-1]["role"] == "prompter":
                path.pop()
            messages = [
                {
                    "role": "user" if item["role"] == "prompter" else "assistant",
                    "content": str(item["text"]).strip(),
                }
                for item in path
            ]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                incomplete += 1
                continue
            digest = hashlib.blake2b(canonical(messages), digest_size=16).digest()
            if digest in seen_conversations:
                duplicate += 1
                continue
            seen_conversations.add(digest)
            result = {
                "conversations": messages,
                "source": "oasst1",
                "message_tree_id": root.get("message_tree_id"),
                "root_message_id": root.get("message_id"),
                "path_message_ids": [item.get("message_id") for item in path],
                "language": root.get("lang"),
            }
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            built += 1

    def file_info(path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": os.environ["HF_ENDPOINT"],
        "repo_id": args.repo_id,
        "revision": args.revision,
        "split": args.split,
        "allowed_languages": sorted(allowed_languages),
        "raw_rows": raw_rows,
        "eligible_nodes": len(nodes),
        "roots": len(roots),
        "conversations": built,
        "exact_duplicate_conversations_removed": duplicate,
        "incomplete_roots": incomplete,
        "language_counts_all_rows": dict(language_counts),
        "selection_policy": "one root-to-leaf path per tree; prefer reviewed, lower-rank, non-synthetic child",
        "raw_file": file_info(args.raw_output),
        "conversation_file": file_info(args.conversation_output),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
