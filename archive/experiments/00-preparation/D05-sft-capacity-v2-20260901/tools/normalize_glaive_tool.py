#!/usr/bin/env python3
"""Convert the Glaive function-calling sample to MiniMind tool-call messages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MARKER = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
END_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def extract_tools(system: str) -> list[dict[str, Any]]:
    tools = []
    decoder = json.JSONDecoder()
    position = 0
    while True:
        start = system.find("{", position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(system[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict) and value.get("name") and value.get("parameters"):
            tools.append(value)
        position = start + end
    return tools


def parse_function_call(content: str) -> dict[str, Any] | None:
    name_match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', content)
    arguments_match = re.search(r'["\']arguments["\']\s*:\s*\'(.*?)\'\s*}', content, re.DOTALL)
    if arguments_match is None:
        arguments_match = re.search(r'["\']arguments["\']\s*:\s*(\{.*?})\s*}', content, re.DOTALL)
    if name_match is None or arguments_match is None:
        return None
    raw_arguments = arguments_match.group(1)
    try:
        arguments: Any = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = raw_arguments
    return {
        "type": "function",
        "function": {
            "name": name_match.group(1),
            "arguments": arguments,
        },
    }


def parse_chat(chat: str, system: str) -> tuple[list[dict[str, Any]], int]:
    messages: list[dict[str, Any]] = []
    tools = extract_tools(system)
    system_content = system.removeprefix("SYSTEM:").strip()
    system_message: dict[str, Any] = {"role": "system", "content": system_content}
    if tools:
        system_message["tools"] = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    messages.append(system_message)

    matches = list(MARKER.finditer(chat))
    function_call_errors = 0
    for index, match in enumerate(matches):
        label = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
        content = chat[match.end():end].replace(END_TOKEN, "").strip()
        if not content:
            continue
        if label == "USER":
            messages.append({"role": "user", "content": content})
        elif label == "FUNCTION RESPONSE":
            messages.append({"role": "tool", "content": content})
        elif content.startswith("<functioncall>"):
            call = parse_function_call(content)
            if call is None:
                function_call_errors += 1
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": json.dumps([call], ensure_ascii=False, sort_keys=True),
                }
            )
        else:
            messages.append({"role": "assistant", "content": content})
    return messages, function_call_errors


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = written = invalid = function_call_errors = tool_rows = 0

    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line in source:
            rows += 1
            try:
                row = json.loads(line)
                messages, errors = parse_chat(str(row["chat"]), str(row["system"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                invalid += 1
                continue
            function_call_errors += errors
            if (
                not any(item["role"] == "user" for item in messages)
                or not any(item["role"] == "assistant" for item in messages)
            ):
                invalid += 1
                continue
            has_tool = any(item.get("tool_calls") for item in messages)
            tool_rows += int(has_tool)
            output.write(
                json.dumps(
                    {"conversations": messages, "source": "glaive_function_calling_v2"},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_rows": rows,
        "written_rows": written,
        "invalid_rows": invalid,
        "function_call_parse_errors": function_call_errors,
        "tool_rows": tool_rows,
        "output": str(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "output_sha256": file_sha256(args.output),
    }
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 1 if invalid or function_call_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
