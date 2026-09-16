#!/usr/bin/env python3
"""Build deterministic, complete-turn MiniMind SFT-v1 payloads.

The builder deliberately writes a pending manifest only. The independent
auditor is the sole component allowed to create '_SUCCESS'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from transformers import AutoTokenizer


MESSAGE_KEYS = ("role", "content", "reasoning_content", "tools", "tool_calls")
VALID_ROLES = {"system", "user", "assistant", "tool"}
ROLE_MAP = {"human": "user", "prompter": "user", "gpt": "assistant"}
SPLITS = ("train", "validation", "test")
PRIME = np.uint64(4_294_967_311)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    source_id: str
    repo_id: str
    revision: str
    license: str
    purpose: str
    path: Path
    raw_sha256: str
    rows: int
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/sft/build_v1.yaml"))
    parser.add_argument("--stage", choices=("smoke", "pilot", "formal_v1"), default="smoke")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def normalize_prompt(messages: list[dict[str, Any]]) -> str:
    return normalize_text(
        "\n".join(message["content"] for message in messages if message["role"] == "user")
    )


def canonical_json_list(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise BuildError(f"{field} is not valid JSON") from exc
    if not isinstance(value, list) or not value:
        raise BuildError(f"{field} must be a non-empty list")
    return stable_json(value)


def canonical_message(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise BuildError("message is not an object")
    raw_role = stable_text(message.get("role")).lower()
    role = ROLE_MAP.get(raw_role, raw_role)
    if role not in VALID_ROLES:
        raise BuildError(f"unsupported role: {raw_role!r}")
    raw_content = message.get("content")
    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content.strip()
    else:
        raise BuildError("message content must be a string")
    reasoning = message.get("reasoning_content")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise BuildError("reasoning_content must be a string or null")
        reasoning = reasoning.strip() or None
    tools = canonical_json_list(message.get("tools"), "tools")
    tool_calls = canonical_json_list(message.get("tool_calls"), "tool_calls")
    if tools is not None and role != "system":
        raise BuildError("tools are only valid on system messages")
    if tool_calls is not None and role != "assistant":
        raise BuildError("tool_calls are only valid on assistant messages")
    if reasoning is not None and role != "assistant":
        raise BuildError("reasoning_content is only valid on assistant messages")
    if role != "assistant" and not content:
        raise BuildError(f"{role} message is empty")
    if role == "assistant" and not (content or reasoning or tool_calls):
        raise BuildError("assistant message has no supervised content")
    return {
        "role": role,
        "content": content,
        "reasoning_content": reasoning,
        "tools": tools,
        "tool_calls": tool_calls,
    }


def canonical_conversation(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [canonical_message(message) for message in messages]
    if not result:
        raise BuildError("empty conversation")
    return result


def create_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    rendered_messages: list[dict[str, Any]] = []
    tools = None
    for source_message in messages:
        message = dict(source_message)
        if message["role"] == "system" and message.get("tools"):
            tools = json.loads(message["tools"])
        if message.get("tool_calls"):
            message["tool_calls"] = json.loads(message["tool_calls"])
        rendered_messages.append(message)
    return tokenizer.apply_chat_template(
        rendered_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )


def find_sequence(values: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    for index in range(start, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def render_metrics(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    bos_id: list[int],
    eos_id: list[int],
) -> dict[str, int]:
    input_ids = tokenizer(create_chat_prompt(tokenizer, messages)).input_ids
    assistant_messages = sum(message["role"] == "assistant" for message in messages)
    positions: list[int] = []
    cursor = 0
    while True:
        position = find_sequence(input_ids, bos_id, cursor)
        if position < 0:
            break
        positions.append(position)
        cursor = position + max(1, len(bos_id))
    if len(positions) != assistant_messages:
        raise BuildError(
            f"assistant marker mismatch messages={assistant_messages} markers={len(positions)}"
        )
    valid_targets = 0
    closed = 0
    for marker_index, marker_position in enumerate(positions):
        start = marker_position + len(bos_id)
        end = find_sequence(input_ids, eos_id, start)
        if end < 0:
            raise BuildError("assistant span has no closing EOS")
        if marker_index + 1 < len(positions) and end >= positions[marker_index + 1]:
            raise BuildError("assistant span crosses the next assistant marker")
        valid_targets += max(0, end + len(eos_id) - max(1, start))
        closed += 1
    if valid_targets <= 0:
        raise BuildError("zero shifted assistant targets")
    return {
        "rendered_tokens": len(input_ids),
        "assistant_messages": assistant_messages,
        "closed_assistant_spans": closed,
        "valid_targets": valid_targets,
    }


def split_into_turns(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    system: list[dict[str, Any]] = []
    body = messages
    if messages and messages[0]["role"] == "system":
        system = [messages[0]]
        body = messages[1:]
    if any(message["role"] == "system" for message in body):
        raise BuildError("system message is only allowed at position zero")
    if not body or body[0]["role"] != "user":
        raise BuildError("conversation body must start with user")
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in body:
        if message["role"] == "user":
            if current:
                turns.append(current)
            current = [message]
        elif not current:
            raise BuildError("message appears before first user")
        else:
            current.append(message)
    if current:
        turns.append(current)
    has_tools = bool(system and system[0]["tools"])
    for turn in turns:
        assistants = 0
        pending_calls = 0
        for index, message in enumerate(turn):
            role = message["role"]
            if index == 0 and role != "user":
                raise BuildError("turn does not start with user")
            if role == "assistant":
                assistants += 1
                if pending_calls:
                    raise BuildError("assistant arrived before tool responses closed")
                calls = json.loads(message["tool_calls"]) if message["tool_calls"] else []
                if calls:
                    if not has_tools:
                        raise BuildError("tool call has no system tools")
                    pending_calls = len(calls)
            elif role == "tool":
                if pending_calls <= 0:
                    raise BuildError("orphan tool response")
                pending_calls -= 1
        if assistants == 0:
            raise BuildError("turn has no assistant")
        if pending_calls:
            raise BuildError("unclosed tool call")
    return system, turns


def complete_chunks(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_length: int,
    bos_id: list[int],
    eos_id: list[int],
) -> tuple[list[tuple[list[dict[str, Any]], dict[str, int]]], int]:
    system, turns = split_into_turns(messages)
    chunks: list[tuple[list[dict[str, Any]], dict[str, int]]] = []
    current: list[list[dict[str, Any]]] = []
    dropped = 0

    def flatten(selected: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return system + [message for turn in selected for message in turn]

    for turn in turns:
        candidate = flatten(current + [turn])
        metrics = render_metrics(tokenizer, candidate, bos_id, eos_id)
        if metrics["rendered_tokens"] <= max_length:
            current.append(turn)
            continue
        if current:
            accepted = flatten(current)
            chunks.append((accepted, render_metrics(tokenizer, accepted, bos_id, eos_id)))
            current = []
        single = flatten([turn])
        single_metrics = render_metrics(tokenizer, single, bos_id, eos_id)
        if single_metrics["rendered_tokens"] <= max_length:
            current = [turn]
        else:
            dropped += 1
    if current:
        accepted = flatten(current)
        chunks.append((accepted, render_metrics(tokenizer, accepted, bos_id, eos_id)))
    return chunks, dropped


def extract_tools(system: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
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


GLAIVE_MARKER = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE):\s*")


def parse_function_call(content: str) -> dict[str, Any] | None:
    name = re.search(r"""["']name["']\s*:\s*["']([^"']+)["']""", content)
    arguments = re.search(r"""["']arguments["']\s*:\s*'(.*?)'\s*}""", content, re.S)
    if arguments is None:
        arguments = re.search(r"""["']arguments["']\s*:\s*(\{.*?\})\s*}""", content, re.S)
    if name is None or arguments is None:
        return None
    raw = arguments.group(1)
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return {
        "type": "function",
        "function": {"name": name.group(1), "arguments": parsed},
    }


def adapt_glaive(row: dict[str, Any]) -> list[dict[str, Any]]:
    chat = stable_text(row["chat"])
    system = stable_text(row["system"])
    tools = extract_tools(system)
    system_message: dict[str, Any] = {
        "role": "system",
        "content": system.removeprefix("SYSTEM:").strip(),
    }
    if tools:
        system_message["tools"] = tools
    messages: list[dict[str, Any]] = [system_message]
    matches = list(GLAIVE_MARKER.finditer(chat))
    parse_errors = 0
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
        content = chat[match.end() : end].replace("<|endoftext|>", "").strip()
        if not content:
            continue
        label = match.group(1)
        if label == "USER":
            messages.append({"role": "user", "content": content})
        elif label == "FUNCTION RESPONSE":
            messages.append({"role": "tool", "content": content})
        elif content.startswith("<functioncall>"):
            call = parse_function_call(content)
            if call is None:
                parse_errors += 1
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        else:
            messages.append({"role": "assistant", "content": content})
    if parse_errors:
        raise BuildError(f"Glaive function-call parse errors={parse_errors}")
    return messages


def boxed_split(solution: str) -> tuple[str, str] | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", solution)]
    for start in reversed(starts):
        opening = solution.find("{", start)
        depth = 0
        escaped = False
        for index in range(opening, len(solution)):
            char = solution[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and re.fullmatch(r"[\s.!。]*", solution[index + 1 :]):
                    reasoning = solution[:start].rstrip()
                    final = solution[start : index + 1].strip()
                    return (reasoning, final) if reasoning and final else None
                if depth < 0:
                    break
    terminal = re.search(
        r"(?im)^(?:final\s+answer|answer)\s*[:：]\s*[^\r\n]+[.!。]?\s*$",
        solution,
    )
    if terminal and terminal.end() == len(solution):
        reasoning = solution[: terminal.start()].rstrip()
        final = solution[terminal.start() :].strip()
        return (reasoning, final) if reasoning and final else None
    return None


NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
}


def first_number(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    lowered = text.casefold()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def quoted_values(prompt: str) -> list[str]:
    values = re.findall(r'["“]([^"”]{1,120})["”]', prompt)
    return [value.strip() for value in values if value.strip()]


NUMBER_PATTERN = (
    r"(?:\d+|"
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r")"
)


def number_value(value: str) -> int | None:
    lowered = value.casefold()
    return int(lowered) if lowered.isdigit() else NUMBER_WORDS.get(lowered)


def count_requirement(prompt: str, noun: str) -> tuple[str, int, int | None] | None:
    cleaned = re.sub(r"[*_`]", "", prompt)
    range_match = re.search(
        rf"(?:between|from)\s+({NUMBER_PATTERN})\s+(?:and|to)\s+"
        rf"({NUMBER_PATTERN})\s+{noun}",
        cleaned,
        re.I,
    )
    if range_match:
        lower = number_value(range_match.group(1))
        upper = number_value(range_match.group(2))
        if lower is not None and upper is not None:
            return "range", lower, upper
    hyphen_range = re.search(
        rf"\b({NUMBER_PATTERN})\s*[-–]\s*({NUMBER_PATTERN})\s+{noun}",
        cleaned,
        re.I,
    )
    if hyphen_range:
        lower = number_value(hyphen_range.group(1))
        upper = number_value(hyphen_range.group(2))
        if lower is not None and upper is not None:
            return "range", lower, upper
    patterns = (
        (
            "max",
            rf"(?:no more than|not more than|at most|not exceeding|"
            rf"a maximum of|up to)\s+({NUMBER_PATTERN})\s+{noun}",
        ),
        (
            "min",
            rf"(?:at least|a minimum of)\s+({NUMBER_PATTERN})\s+{noun}",
        ),
        (
            "exact",
            rf"(?:exactly|precisely|consisting of|consist of|contains?|"
            rf"using|use|in|into|of)\s+({NUMBER_PATTERN})\s+{noun}",
        ),
        ("exact", rf"\b({NUMBER_PATTERN})[- ]{noun}"),
    )
    for kind, pattern in patterns:
        match = re.search(pattern, cleaned, re.I)
        if match:
            expected = number_value(match.group(1))
            if expected is not None:
                return kind, expected, None
    return None


def count_satisfies(
    requirement: tuple[str, int, int | None] | None,
    actual: int,
) -> bool:
    if requirement is None:
        return False
    kind, lower, upper = requirement
    if kind == "exact":
        return actual == lower
    if kind == "max":
        return actual <= lower
    if kind == "min":
        return actual >= lower
    return upper is not None and lower <= actual <= upper


def paragraph_blocks(answer: str) -> list[str]:
    result: list[str] = []
    heading = re.compile(
        r"(?:#{1,6}\s+.+|\*\*[^*]+\*\*|__[^_]+__|"
        r"\{\{[^{}]+\}\}|<<[^<>]+>>|\[[^]\n]+\])"
    )
    for raw_block in re.split(r"\n\s*\n+", answer.strip()):
        block = raw_block.strip()
        if not block or heading.fullmatch(block):
            continue
        lines = block.splitlines()
        if all(
            not line.strip()
            or re.match(r"^\s*(?:[-*+] |\d+[.)] )", line)
            for line in lines
        ):
            continue
        result.append(block)
    return result


def semantic_word_count(answer: str) -> int:
    candidate = answer.strip()
    fence = chr(96) * 3
    if candidate.startswith(fence) and candidate.endswith(fence):
        candidate = candidate[len(fence) : -len(fence)].strip()
        if candidate.casefold().startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        strings: list[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, list):
                for child in item:
                    collect(child)
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)

        collect(value)
        if strings:
            candidate = " ".join(strings)
    return len(re.findall(r"(?u)\b[\w]+(?:[-'’][\w]+)*\b", candidate))


def sentence_count(answer: str) -> int:
    return len(
        [
            part
            for part in re.split(r"(?<=[.!?。！？])(?:\s+|$)", answer.strip())
            if part.strip()
        ]
    )


def bullet_groups(answer: str) -> list[int]:
    groups: list[int] = []
    active = False
    for line in answer.splitlines():
        if re.match(r"^\s*[-*+]\s+\S", line):
            if not active:
                groups.append(0)
            groups[-1] += 1
            active = True
        elif not line.strip() or (active and line[:1].isspace()):
            continue
        else:
            active = False
    return groups


def structured_list_items(answer: str) -> list[str]:
    markers = list(
        re.finditer(r"(?m)^\s*(?:\d+[.)]|[-*+])\s+\S", answer)
    )
    if not markers:
        return []
    result: list[str] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(answer)
        result.append(answer[marker.start() : end].strip())
    return result


LANGUAGE_STOPWORDS = {
    "english": {"the", "and", "of", "to", "in", "is", "that", "for", "with", "as", "on", "are", "this", "be", "by", "an", "it"},
    "spanish": {"el", "la", "los", "las", "de", "del", "y", "en", "que", "para", "con", "una", "un", "es", "por", "como"},
    "french": {"le", "la", "les", "de", "des", "et", "en", "que", "pour", "avec", "une", "un", "est", "dans", "du"},
    "german": {"der", "die", "das", "den", "dem", "des", "und", "ist", "in", "zu", "mit", "für", "von", "auf", "ein", "eine"},
    "italian": {"il", "lo", "la", "gli", "le", "di", "del", "e", "in", "che", "per", "con", "una", "un", "è"},
    "portuguese": {"o", "a", "os", "as", "de", "do", "da", "e", "em", "que", "para", "com", "uma", "um", "é"},
    "dutch": {"de", "het", "een", "en", "van", "in", "dat", "voor", "met", "op", "is"},
}


LANGUAGE_SCRIPTS = {
    "chinese": ("\u4e00", "\u9fff"),
    "japanese": ("\u3040", "\u30ff"),
    "korean": ("\uac00", "\ud7af"),
    "arabic": ("\u0600", "\u06ff"),
    "russian": ("\u0400", "\u04ff"),
    "ukrainian": ("\u0400", "\u04ff"),
    "hindi": ("\u0900", "\u097f"),
    "greek": ("\u0370", "\u03ff"),
    "hebrew": ("\u0590", "\u05ff"),
}


def response_language(prompt: str) -> str | None:
    names = sorted(set(LANGUAGE_STOPWORDS) | set(LANGUAGE_SCRIPTS))
    pattern = "|".join(names)
    matches = {
        match.casefold()
        for match in re.findall(
            rf"\b(?:in|into|using)\s+({pattern})(?:\s+language)?\b",
            prompt,
            re.I,
        )
    }
    return next(iter(matches)) if len(matches) == 1 else None


def language_matches(answer: str, language: str | None) -> bool:
    if language is None:
        return False
    if language in LANGUAGE_SCRIPTS:
        lower, upper = LANGUAGE_SCRIPTS[language]
        return sum(lower <= char <= upper for char in answer) >= 5
    tokens = re.findall(r"[A-Za-zÀ-ÿ]+", answer.casefold())
    if len(tokens) < 8:
        return False
    scores = {
        name: sum(token in stopwords for token in tokens)
        for name, stopwords in LANGUAGE_STOPWORDS.items()
    }
    target = scores[language]
    other = max(value for name, value in scores.items() if name != language)
    return target >= 3 and target >= other + 2


def highlighted_count(answer: str) -> int:
    bold = re.findall(r"\*\*.+?\*\*|__.+?__", answer, re.S)
    italic = re.findall(r"(?<!\*)\*(?!\*)([^*\n]+)(?<!\*)\*(?!\*)", answer)
    return len(bold) + len(italic)


def extended_strict_check(
    constraint: str,
    prompt: str,
    answer: str,
) -> tuple[bool, dict[str, Any] | None]:
    cleaned_prompt = re.sub(r"[*_`]", "", prompt)
    lowered_prompt = cleaned_prompt.casefold()
    if constraint == "format:title":
        first = next((line.strip() for line in answer.splitlines() if line.strip()), "")
        passed = bool(
            re.fullmatch(r'["“][^"”\n]+["”]', first)
            or re.fullmatch(r"\{\{[^{}\n]+\}\}", first)
        )
        return passed, {"first_line": first[:160], "extended_wrapper": True}
    if constraint == "content:include a postscript":
        passed = re.search(
            r"(?im)^\s*(?:p\.?\s*s\.?|postscript)(?::|\s|$)",
            answer,
        ) is not None
        return passed, {"postscript": passed}
    if constraint == "length constraints:number of words":
        requirement = count_requirement(cleaned_prompt, r"words?")
        items = structured_list_items(answer)
        per_item = re.search(
            r"\beach\b[^.\n]{0,120}\b(?:no more than|not more than|"
            r"at most|not exceeding|at least|exactly)\b",
            cleaned_prompt,
            re.I,
        )
        if per_item and len(items) >= 2:
            actuals = [semantic_word_count(item) for item in items]
            passed = all(count_satisfies(requirement, actual) for actual in actuals)
            return passed, {
                "requirement": requirement,
                "scope": "each_list_item",
                "actual_per_item": actuals,
            }
        actual = semantic_word_count(answer)
        return count_satisfies(requirement, actual), {
            "requirement": requirement,
            "actual": actual,
        }
    if constraint == "length constraints:number of sentences":
        requirement = count_requirement(cleaned_prompt, r"sentences?")
        actual = sentence_count(answer)
        return count_satisfies(requirement, actual), {
            "requirement": requirement,
            "actual": actual,
        }
    if constraint == "length constraints:number of paragraphs":
        requirement = count_requirement(cleaned_prompt, r"paragraphs?")
        actual = len(paragraph_blocks(answer))
        return count_satisfies(requirement, actual), {
            "requirement": requirement,
            "actual": actual,
        }
    if constraint == "format:number of bullet lists":
        groups = bullet_groups(answer)
        list_requirement = count_requirement(cleaned_prompt, r"bullet lists?")
        if list_requirement is not None:
            passed = count_satisfies(list_requirement, len(groups))
            return passed, {"requirement": list_requirement, "groups": groups}
        point_requirement = count_requirement(cleaned_prompt, r"bullet points?")
        if point_requirement is None and re.search(
            r"(?:format|as follows)[^\n]{0,80}[*+-]\s+\w+",
            prompt,
            re.I,
        ):
            match = re.search(
                rf"list of\s+(?:exactly\s+)?({NUMBER_PATTERN})\s+"
                r"(?:\w+\s+){0,2}(?:features?|items?|sources?|examples?)",
                cleaned_prompt,
                re.I,
            )
            if match:
                expected = number_value(match.group(1))
                if expected is not None:
                    point_requirement = "exact", expected, None
        passed = bool(groups) and count_satisfies(point_requirement, sum(groups))
        return passed, {"requirement": point_requirement, "groups": groups}
    if constraint == "format:number of highlighted sections":
        requirement = count_requirement(
            cleaned_prompt,
            r"(?:bold(?: text)?|highlighted) sections?",
        )
        if requirement is None:
            match = re.search(
                rf"highlight\s+({NUMBER_PATTERN})\s+\w+",
                cleaned_prompt,
                re.I,
            )
            if match:
                expected = number_value(match.group(1))
                if expected is not None:
                    requirement = "min", expected, None
        actual = highlighted_count(answer)
        return count_satisfies(requirement, actual), {
            "requirement": requirement,
            "actual": actual,
        }
    if constraint == "case: frequency of capital words":
        requirement = count_requirement(
            cleaned_prompt,
            r"(?:instances? of\s+)?(?:words?\s+)?(?:in\s+)?all capital(?: letters?)?",
        )
        actual = len(re.findall(r"(?<!\w)[A-Z]{2,}(?!\w)", answer))
        return count_satisfies(requirement, actual), {
            "requirement": requirement,
            "actual": actual,
        }
    if constraint == "in english and capital":
        explicit = re.search(
            r"(?:in|written in) english(?:\s+and|,)\s+"
            r"(?:in\s+)?(?:all\s+)?capital letters",
            cleaned_prompt,
            re.I,
        )
        letters = [char for char in answer if char.isalpha()]
        ascii_letters = [char for char in letters if char.isascii()]
        passed = bool(explicit and letters)
        passed = passed and len(ascii_letters) / len(letters) >= 0.95
        passed = passed and all(not char.islower() for char in letters)
        return passed, {
            "ascii_letter_fraction": (
                len(ascii_letters) / len(letters) if letters else 0.0
            ),
            "lowercase_letters": sum(char.islower() for char in letters),
        }
    if constraint == "response language":
        language = response_language(cleaned_prompt)
        passed = language_matches(answer, language)
        return passed, {"language": language, "heuristic_match": passed}
    if constraint == "keywords:frequency":
        requirements: list[tuple[str, str, int]] = []
        pattern = (
            rf"(?:word|phrase)\s+['\"“‘]([^'\"”’]+)['\"”’]"
            rf"(?:\s*\([^)]*\))?\s+"
            rf"(at least|exactly|at most|no more than)\s+"
            rf"({NUMBER_PATTERN})\s+times"
        )
        for match in re.finditer(pattern, cleaned_prompt, re.I):
            expected = number_value(match.group(3))
            if expected is not None:
                requirements.append(
                    (match.group(1), match.group(2).casefold(), expected)
                )
        evidence: list[dict[str, Any]] = []
        for value, kind, expected in requirements:
            actual = len(re.findall(re.escape(value), answer, re.I))
            normalized_kind = {
                "at least": "min",
                "exactly": "exact",
                "at most": "max",
                "no more than": "max",
            }[kind]
            evidence.append(
                {"value": value, "kind": normalized_kind, "expected": expected, "actual": actual}
            )
        passed = bool(evidence) and all(
            count_satisfies((item["kind"], item["expected"], None), item["actual"])
            for item in evidence
        )
        return passed, {"requirements": evidence}
    if constraint == "keywords:letter frequency":
        match = re.search(
            rf"letter\s+['\"“‘]?([A-Za-z])['\"”’]?"
            rf"[^.\n]{{0,80}}?"
            rf"(at least|exactly|at most|no more than)\s+"
            rf"({NUMBER_PATTERN})\s+times",
            cleaned_prompt,
            re.I,
        )
        if match is None:
            return False, None
        expected = number_value(match.group(3))
        if expected is None:
            return False, None
        kind = {
            "at least": "min",
            "exactly": "exact",
            "at most": "max",
            "no more than": "max",
        }[match.group(2).casefold()]
        if "each sentence" in lowered_prompt:
            units = [
                part
                for part in re.split(r"(?<=[.!?。！？])(?:\s+|$)", answer.strip())
                if part.strip()
            ]
        elif "each paragraph" in lowered_prompt:
            units = paragraph_blocks(answer)
        else:
            units = [answer]
        counts = [unit.casefold().count(match.group(1).casefold()) for unit in units]
        passed = bool(counts) and all(
            count_satisfies((kind, expected, None), actual) for actual in counts
        )
        return passed, {
            "character": match.group(1),
            "kind": kind,
            "expected": expected,
            "actual_per_unit": counts,
        }
    if constraint == "include keywords":
        match = re.search(
            r"(?:keywords?|words?|phrases?)\s*(?:such as|including|:)\s*"
            r"(.+?)(?:\.(?:\s|$)|$)",
            prompt,
            re.I | re.S,
        )
        values = (
            re.findall(r"""['"“‘]([^'"”’]{1,80})['"”’]""", match.group(1))
            if match
            else []
        )
        passed = bool(values) and all(
            value.casefold() in answer.casefold() for value in values
        )
        return passed, {"included_values": values}
    if constraint == "format:choose one from options":
        tuples = re.findall(
            r"""\(([^()]*(?:['"][^'"]+['"][^()]*){2,})\)""",
            prompt,
        )
        values = (
            re.findall(r"""['"]([^'"]+)['"]""", tuples[-1])
            if tuples
            else []
        )
        if not values:
            match = re.search(
                r"(?:following options|following exact phrases|choose one)"
                r".*?:\s*(.+?)(?:\.|$)",
                prompt,
                re.I | re.S,
            )
            if match:
                values = re.findall(r'["“]([^"”]+)["”]', match.group(1))
        hits = [
            value
            for value in values
            if re.search(
                rf"(?<!\w){re.escape(value)}(?!\w)",
                answer,
                re.I,
            )
        ]
        return len(values) >= 2 and len(hits) == 1, {
            "options": values,
            "matched_options": hits,
        }
    if constraint == "repeat the prompt":
        normalized_answer = " ".join(answer.casefold().split())
        colon = re.search(
            r"repeat (?:the )?(?:prompt|question|request)"
            r"(?: above)?(?: verbatim)?\s*:\s*(.+?[?!.])(?:\s|$)",
            prompt,
            re.I | re.S,
        )
        if colon:
            target = " ".join(colon.group(1).casefold().split())
            return target in normalized_answer, {"target": target[:240]}
        marker = re.search(
            r"(?:but first,\s*)?repeat (?:the )?request above verbatim",
            prompt,
            re.I,
        )
        if marker:
            target = " ".join(prompt[: marker.start()].casefold().split()).rstrip(" ,.;")
            return len(target) >= 20 and target in normalized_answer, {
                "target": target[:240]
            }
        if re.search(r"repeat the prompt verbatim", prompt, re.I):
            target = " ".join(prompt.casefold().split())
            return target in normalized_answer, {"target": target[:240]}
        end_marker = re.search(r"repeat the prompt at the end", prompt, re.I)
        if end_marker:
            question = re.search(r"^(.+?[?])", prompt, re.S)
            target = " ".join(question.group(1).casefold().split()) if question else ""
            passed = bool(target) and normalized_answer.rstrip(" .").endswith(
                target.rstrip(" .")
            )
            return passed, {"target": target[:240]}
        return False, None
    if constraint == "length constraints:first word of the nth paragraph":
        ordinals = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
        }
        requirements: list[tuple[int, str]] = []
        pattern = (
            r"(?:first word of|begin|start)\s+(?:the\s+)?"
            r"(first|second|third|fourth|fifth)\s+paragraph"
            r"(?:\s+with|\s+(?:must|should)\s+be)\s+"
            r"(?:the\s+word\s+)?['\"“‘]?([A-Za-z]+)"
        )
        for match in re.finditer(pattern, cleaned_prompt, re.I):
            requirements.append(
                (ordinals[match.group(1).casefold()], match.group(2))
            )
        blocks = paragraph_blocks(answer)
        actual: list[dict[str, Any]] = []
        for index, expected in requirements:
            if index > len(blocks):
                return False, {"requirements": requirements, "actual": actual}
            first = re.search(r"[A-Za-z]+", blocks[index - 1])
            value = first.group(0) if first else ""
            actual.append({"paragraph": index, "expected": expected, "actual": value})
        passed = bool(requirements) and all(
            item["expected"].casefold() == item["actual"].casefold()
            for item in actual
        )
        return passed, {"requirements": requirements, "actual": actual}
    if constraint == "give two responses":
        items = structured_list_items(answer)
        if re.search(r"\b(?:provide|write) two responses\b", cleaned_prompt, re.I):
            numbered = [item for item in items if re.match(r"^[12][.)]\s+", item)]
            if len(numbered) == 2 and not re.match(r"^3[.)]\s+", items[-1]):
                return True, {"numbered_responses": 2}
        stems = ("response", "summary", "example", "strategy", "event", "story")
        for stem in stems:
            left = re.search(rf"(?im)^\s*(?:[*#]+\s*)?{stem}\s*1\b", answer)
            right = re.search(rf"(?im)^\s*(?:[*#]+\s*)?{stem}\s*2\b", answer)
            if left and right and left.start() < right.start():
                return True, {"numbered_pair": stem}
        parts = [
            part.strip()
            for part in re.split(r"(?m)^\s*(?:\*{4,}|-{3,})\s*$", answer)
            if part.strip()
        ]
        return len(parts) == 2, {"separator_parts": len(parts)}
    if constraint == "format:number of sections":
        requirement = count_requirement(cleaned_prompt, r"(?:distinct\s+)?sections?")
        explicit = len(set(re.findall(r"(?im)^\s*section\s+(\d+)\b", answer)))
        if explicit:
            actual = explicit
        else:
            headings = re.findall(
                r"(?im)^\s*(?:#{1,6}\s+\S.+|\{\{[^{}]+\}\}|"
                r"<<[^<>]+>>|[A-Z]\)\s+\S.+|\*\*[^*]+\*\*)\s*$",
                answer,
            )
            actual = len(headings)
        return count_satisfies(requirement, actual), {
            "requirement": requirement,
            "actual": actual,
        }
    return False, None


def validate_strict(prompt: str, answer: str, constraints: Any) -> dict[str, Any]:
    if isinstance(constraints, str):
        try:
            constraints = json.loads(constraints)
        except json.JSONDecodeError:
            constraints = [constraints]
    if not isinstance(constraints, list) or not constraints:
        raise BuildError("strict constraints are empty")
    families: list[str] = []
    evidence: dict[str, Any] = {}
    lowered_prompt = prompt.casefold()
    for raw_constraint in constraints:
        constraint = stable_text(raw_constraint).casefold()
        passed = False
        detail: Any = None
        if constraint == "punctuation:use no comma":
            passed = "," not in answer and "，" not in answer
            detail = {"commas": answer.count(",") + answer.count("，")}
        elif constraint == "format:use json format":
            candidate = answer.strip()
            fence = chr(96) * 3
            if candidate.startswith(fence) and candidate.endswith(fence):
                candidate = candidate[len(fence) : -len(fence)].strip()
                if candidate.casefold().startswith("json"):
                    candidate = candidate[4:].lstrip()
            try:
                json.loads(candidate)
                passed = True
            except json.JSONDecodeError:
                passed = False
            detail = {"json_parse": passed}
        elif constraint == "case:in english and lowercase":
            letters = [char for char in answer if char.isalpha()]
            passed = bool(letters) and all(not char.isupper() for char in letters)
            detail = {"uppercase_letters": sum(char.isupper() for char in letters)}
        elif constraint == "format:title":
            first = next((line.strip() for line in answer.splitlines() if line.strip()), "")
            passed = bool(
                re.fullmatch(r"<<[^<>]+>>", first)
                or re.fullmatch(r"\*\*[^*]+\*\*", first)
                or re.fullmatch(r"#{1,6}\s+\S.*", first)
                or re.fullmatch(r"\[[^\[\]]+\]", first)
            )
            detail = {"first_line": first[:160]}
        elif constraint == "include keywords":
            values = quoted_values(prompt)
            if "keyword" in lowered_prompt and values:
                passed = all(value.casefold() in answer.casefold() for value in values)
            detail = {"quoted_values": values}
        elif constraint == "keywords:exclude words":
            values = quoted_values(prompt)
            passed = bool(values) and all(
                re.search(rf"(?<!\w){re.escape(value)}(?!\w)", answer, re.I) is None
                for value in values
            )
            detail = {"excluded_values": values}
        elif constraint == "specific ending":
            matches = re.findall(
                r"(?:end|ending|conclude)[^.\n]{0,100}?['\"“]([^'\"”]+)['\"”]",
                prompt,
                re.I,
            )
            ending = matches[-1].strip() if matches else ""
            passed = bool(ending) and answer.rstrip().endswith(ending)
            detail = {"ending": ending}
        elif constraint == "content:include a postscript":
            passed = re.search(r"(?im)^\s*(?:p\.?\s*s\.?|postscript)\s*:", answer) is not None
            detail = {"postscript": passed}
        elif constraint == "use quotation":
            passed = bool(re.search(r'["“][^"”]+["”]', answer))
            detail = {"quoted_span": passed}
        elif constraint == "content:number of placeholders":
            position = lowered_prompt.find("placeholder")
            expected = first_number(prompt[max(0, position - 80) : position + 80])
            actual = len(re.findall(r"\[[^\[\]]+\]|\{[^{}]+\}", answer))
            passed = expected is not None and actual >= expected
            detail = {"expected_min": expected, "actual": actual}
        elif constraint == "format:number of sections":
            position = lowered_prompt.find("section")
            expected = first_number(prompt[max(0, position - 80) : position + 80])
            actual = len(set(re.findall(r"(?im)\bsection\s+(\d+)\b", answer)))
            passed = expected is not None and actual == expected
            detail = {"expected": expected, "actual": actual}
        elif constraint == "length constraints:number of sentences":
            exact = re.search(r"exactly\s+(\d+)\s+sentences?", prompt, re.I)
            if exact:
                expected = int(exact.group(1))
                actual = len(re.findall(r"[^.!?\n][.!?](?:\s|$)", answer))
                passed = actual == expected
                detail = {"expected_exact": expected, "actual": actual}
        elif constraint == "case: frequency of capital words":
            exact = re.search(r"exactly\s+(\d+)\s+(?:words?\s+)?(?:in\s+)?all capital", prompt, re.I)
            if exact:
                expected = int(exact.group(1))
                actual = len(re.findall(r"(?<!\w)[A-Z]{2,}(?!\w)", answer))
                passed = actual == expected
                detail = {"expected_exact": expected, "actual": actual}
        if not passed:
            passed, extended_detail = extended_strict_check(
                constraint, prompt, answer
            )
            if passed:
                detail = extended_detail
        if not passed:
            raise BuildError(f"unsupported or failed strict constraint: {raw_constraint}")
        families.append(constraint)
        evidence[constraint] = detail
    return {
        "status": "passed",
        "families": sorted(set(families)),
        "evidence": evidence,
        "verifier_version": "strict-v1.2",
    }


def adapt_simple(source_id: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    if source_id == "tigerbot_alpaca_zh_0_5m":
        prompt = stable_text(row["instruction"])
        extra = stable_text(row.get("input"))
        if extra:
            prompt += "\n\n输入：\n" + extra
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": stable_text(row["output"])}]
    if source_id == "alpaca_gpt4_zh":
        prompt = stable_text(row["instruction_zh"])
        extra = stable_text(row.get("input_zh"))
        if extra:
            prompt += "\n\n输入：\n" + extra
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": stable_text(row["output_zh"])}]
    if source_id == "gsm8k":
        return [{"role": "user", "content": stable_text(row["question"])}, {"role": "assistant", "content": stable_text(row["answer"])}]
    if source_id == "magicoder_evol_instruct_110k":
        return [{"role": "user", "content": stable_text(row["instruction"])}, {"role": "assistant", "content": stable_text(row["response"])}]
    if source_id == "code_alpaca_20k":
        prompt = stable_text(row["instruction"])
        extra = stable_text(row.get("input"))
        if extra:
            prompt += "\n\nInput:\n" + extra
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": stable_text(row["output"])}]
    if source_id == "mbpp":
        tests = "\n".join(map(str, row.get("test_list") or []))
        prompt = stable_text(row["text"])
        if tests:
            prompt += "\n\nYour solution must pass these tests:\n" + tests
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": stable_text(row["code"])}]
    if source_id == "databricks_dolly_15k":
        prompt = stable_text(row["instruction"])
        context = stable_text(row.get("context"))
        if context:
            prompt += "\n\nContext:\n" + context
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": stable_text(row["response"])}]
    if source_id == "cnn_dailymail_summary":
        prompt = "Summarize the following news article faithfully and concisely.\n\nArticle:\n" + stable_text(row["article"])
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": stable_text(row["highlights"])}]
    raise BuildError(f"no simple adapter for {source_id}")


def load_sources(config: dict[str, Any], repo_root: Path) -> list[Source]:
    source_config_path = (repo_root / config["contracts"]["sources"]).resolve()
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    raw_root = Path(config["paths"]["raw_root"])
    result: list[Source] = []
    for value in source_config["sources"]:
        materialization = value["materialization"]
        path = raw_root / materialization["output"]
        done_path = Path(str(path) + ".done.json")
        if not path.exists() or not done_path.exists():
            raise BuildError(f"raw source or done evidence missing: {value['id']}")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("sha256") is None or done.get("rows") is None:
            raise BuildError(f"incomplete done evidence: {done_path}")
        if path.stat().st_size != int(done["bytes"]):
            raise BuildError(f"raw size drift: {path}")
        result.append(
            Source(
                source_id=str(value["id"]),
                repo_id=str(value["repo_id"]),
                revision=str(value["revision"]),
                license=str(value["license"]),
                purpose=str(value["purpose"]),
                path=path,
                raw_sha256=str(done["sha256"]),
                rows=int(done["rows"]),
                size_bytes=int(done["bytes"]),
            )
        )
    return result


def split_for_prompt(prompt: str) -> str:
    value = int(sha256_bytes(prompt.encode("utf-8"))[:8], 16) % 10_000
    if value < 200:
        return "validation"
    if value < 400:
        return "test"
    return "train"


BASE_SAMPLE_RATES = {
    "chinese_general": 0.05,
    "english_general": 0.12,
    "multiturn": 0.24,
    "strict_format": 1.0,
    "math": 0.04,
    "tool_call": 0.20,
    "summarization_translation": 0.08,
    "code": 0.20,
}


def sampled(origin: str, bucket: str, scale: float) -> bool:
    rate = min(1.0, BASE_SAMPLE_RATES[bucket] * scale)
    value = int(sha256_bytes(f"sample|{bucket}|{origin}".encode())[:16], 16)
    return value / float(2**64) < rate


def source_bucket_and_messages(
    source: Source,
    row: dict[str, Any],
    line_number: int,
) -> tuple[str, str, list[dict[str, Any]], str, list[dict[str, Any]]]:
    source_id = source.source_id
    transform_chain: list[dict[str, Any]] = []
    upstream_id = str(line_number)
    variant = "original"
    if source_id == "ultrachat_200k":
        upstream_id = stable_text(row["prompt_id"])
        residue = int(sha256_bytes(upstream_id.encode()), 16) % 5
        bucket = "english_general" if residue in {0, 1, 2} else "multiturn"
        messages = list(row["messages"])
    elif source_id in {"tigerbot_alpaca_zh_0_5m", "alpaca_gpt4_zh"}:
        bucket = "chinese_general"
        messages = adapt_simple(source_id, row)
    elif source_id == "numinamath_cot":
        tag = stable_text(row.get("source")).casefold()
        if tag in {"gsm8k", "math"}:
            raise BuildError(f"excluded Numina source tag: {tag}")
        split = boxed_split(stable_text(row["solution"]))
        if split is None:
            raise BuildError("Numina has no high-confidence terminal answer")
        reasoning, final = split
        bucket = "math"
        messages = [
            {"role": "user", "content": stable_text(row["problem"])},
            {"role": "assistant", "content": final, "reasoning_content": reasoning},
        ]
        variant = "reasoning_final"
        transform_chain.append({"id": "numina_reasoning_final", "version": "1.0", "status": "passed"})
    elif source_id == "gsm8k":
        bucket = "math"
        messages = adapt_simple(source_id, row)
    elif source_id == "glaive_function_calling_v2":
        bucket = "tool_call"
        messages = adapt_glaive(row)
        variant = "parsed_tool_exchange"
        transform_chain.append({"id": "glaive_tool_parser", "version": "1.0", "status": "passed"})
    elif source_id in {"cnn_dailymail_summary", "databricks_dolly_15k"}:
        if source_id == "databricks_dolly_15k" and stable_text(row.get("category")) != "summarization":
            raise BuildError("Dolly row is not summarization")
        bucket = "summarization_translation"
        messages = adapt_simple(source_id, row)
        upstream_id = stable_text(row.get("id")) or upstream_id
    elif source_id in {"magicoder_evol_instruct_110k", "code_alpaca_20k", "mbpp"}:
        bucket = "code"
        messages = adapt_simple(source_id, row)
        upstream_id = stable_text(row.get("task_id")) or upstream_id
    elif source_id == "tulu_3_sft_personas_instruction_following":
        bucket = "strict_format"
        upstream_id = stable_text(row["id"])
        messages = list(row["messages"])
        if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise BuildError("Tulu strict row is not one user/assistant pair")
        prompt = stable_text(row["prompt"])
        if normalize_text(prompt) != normalize_text(stable_text(messages[0].get("content"))):
            raise BuildError("Tulu prompt mismatch")
        report = validate_strict(prompt, stable_text(messages[1].get("content")), row.get("constraints"))
        variant = "native_verified"
        transform_chain.append({"id": "tulu_constraint_verifier", **report})
    else:
        raise BuildError(f"source not selected by formal v1 builder: {source_id}")
    return bucket, upstream_id, messages, variant, transform_chain


def choose_chunk(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    origin: str,
    bucket: str,
    max_length: int,
    bos_id: list[int],
    eos_id: list[int],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    canonical = canonical_conversation(messages)
    chunks, dropped = complete_chunks(tokenizer, canonical, max_length, bos_id, eos_id)
    eligible: list[tuple[str, list[dict[str, Any]], dict[str, int]]] = []
    for chunk, metrics in chunks:
        if bucket == "multiturn" and metrics["assistant_messages"] < 2:
            continue
        if bucket == "tool_call" and not any(
            message["role"] == "tool" or message["tool_calls"] for message in chunk
        ):
            continue
        if bucket == "strict_format" and metrics["valid_targets"] > 384:
            continue
        payload = {"conversations": chunk}
        rank = sha256_bytes(f"{origin}|{stable_json(payload)}".encode())
        eligible.append((rank, chunk, metrics))
    if not eligible:
        raise BuildError("no eligible complete chunk")
    _, chunk, metrics = min(eligible, key=lambda item: item[0])
    return chunk, metrics, dropped


def initialize_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise BuildError(f"candidate database already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE candidates (
          candidate_id TEXT PRIMARY KEY,
          origin_group_id TEXT NOT NULL UNIQUE,
          bucket TEXT NOT NULL,
          split TEXT NOT NULL,
          selection_rank TEXT NOT NULL,
          valid_targets INTEGER NOT NULL,
          rendered_tokens INTEGER NOT NULL,
          prompt_norm TEXT NOT NULL,
          prompt_digest TEXT NOT NULL,
          exact_digest TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          provenance_json TEXT NOT NULL
        );
        CREATE INDEX candidate_order ON candidates(selection_rank);
        CREATE INDEX candidate_bucket_split ON candidates(bucket, split, selection_rank);
        """
    )
    return connection


def candidate_scale(stage: str) -> float:
    return {"smoke": 1.0, "pilot": 8.0, "formal_v1": 32.0}[stage]


def populate_candidates(
    connection: sqlite3.Connection,
    sources: list[Source],
    tokenizer: Any,
    stage: str,
    max_length: int,
    max_source_rows: int | None,
) -> dict[str, Any]:
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    reports: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_rows: Counter[str] = Counter()
    bucket_tokens: Counter[str] = Counter()
    usable = {
        "ultrachat_200k", "tigerbot_alpaca_zh_0_5m", "alpaca_gpt4_zh",
        "numinamath_cot", "gsm8k", "glaive_function_calling_v2",
        "cnn_dailymail_summary", "databricks_dolly_15k",
        "magicoder_evol_instruct_110k", "code_alpaca_20k", "mbpp",
        "tulu_3_sft_personas_instruction_following",
    }
    for source in sources:
        report = reports[source.source_id]
        if source.source_id not in usable:
            report["not_selected_for_v1"] += source.rows
            continue
        with source.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if max_source_rows and line_number > max_source_rows:
                    break
                report["raw_rows_scanned"] += 1
                try:
                    row = json.loads(line)
                    if source.source_id == "ultrachat_200k":
                        origin_hint = stable_text(row["prompt_id"])
                        residue = int(sha256_bytes(origin_hint.encode()), 16) % 5
                        bucket_hint = "english_general" if residue in {0, 1, 2} else "multiturn"
                    elif source.source_id == "tulu_3_sft_personas_instruction_following":
                        origin_hint = stable_text(row["id"])
                        bucket_hint = "strict_format"
                    else:
                        origin_hint = str(line_number)
                        bucket_hint = source.purpose
                    origin = f"{source.source_id}:{origin_hint}"
                    if not sampled(origin, bucket_hint, candidate_scale(stage)):
                        report["sampling_filtered"] += 1
                        continue
                    report["sampled_rows"] += 1
                    bucket, upstream_id, messages, variant, transforms = source_bucket_and_messages(
                        source, row, line_number
                    )
                    origin = f"{source.source_id}:{upstream_id}"
                    chunk, metrics, dropped = choose_chunk(
                        tokenizer, messages, origin, bucket, max_length, bos_id, eos_id
                    )
                    prompt = normalize_prompt(chunk)
                    if not prompt:
                        raise BuildError("empty normalized user prompt")
                    payload_json = stable_json({"conversations": chunk})
                    exact_digest = sha256_bytes(payload_json.encode())
                    prompt_digest = sha256_bytes(prompt.encode())
                    candidate_id = sha256_bytes(f"{origin}|{variant}|{exact_digest}".encode())
                    split = split_for_prompt(prompt)
                    rank = sha256_bytes(f"42|select|{bucket}|{origin}|{candidate_id}".encode())
                    provenance = {
                        "candidate_id": candidate_id,
                        "source_id": source.source_id,
                        "repo_id": source.repo_id,
                        "revision": source.revision,
                        "license": source.license,
                        "source_record_locator": {"line_number": line_number},
                        "source_record_sha256": sha256_bytes(line.rstrip("\n").encode()),
                        "upstream_id": upstream_id,
                        "origin_group_id": origin,
                        "raw_sha256": source.raw_sha256,
                        "transform_chain": transforms,
                        "variant_family": variant,
                        "bucket": bucket,
                        "split": split,
                        "rendered_tokens": metrics["rendered_tokens"],
                        "shifted_assistant_target_tokens": metrics["valid_targets"],
                    }
                    connection.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            candidate_id, origin, bucket, split, rank,
                            metrics["valid_targets"], metrics["rendered_tokens"],
                            prompt, prompt_digest, exact_digest, payload_json,
                            stable_json(provenance),
                        ),
                    )
                    report["candidate_rows"] += 1
                    report["candidate_assistant_tokens"] += metrics["valid_targets"]
                    report["dropped_overlength_turns"] += dropped
                    bucket_rows[bucket] += 1
                    bucket_tokens[bucket] += metrics["valid_targets"]
                except (BuildError, KeyError, TypeError, ValueError, json.JSONDecodeError, sqlite3.IntegrityError) as exc:
                    report[f"rejected:{type(exc).__name__}:{str(exc)[:100]}"] += 1
                if line_number % 5000 == 0:
                    connection.commit()
        connection.commit()
    return {
        "by_source": {key: dict(value) for key, value in sorted(reports.items())},
        "candidate_rows_by_bucket": dict(sorted(bucket_rows.items())),
        "candidate_assistant_tokens_by_bucket": dict(sorted(bucket_tokens.items())),
    }


def hashed_shingles(text: str, size: int = 5) -> set[int]:
    return {
        int.from_bytes(
            hashlib.blake2b(text[index : index + size].encode(), digest_size=8).digest(),
            "big",
        )
        for index in range(max(0, len(text) - size + 1))
    }


class NearIndex:
    def __init__(self) -> None:
        self.by_anchor: dict[int, list[str]] = defaultdict(list)
        self.values: dict[str, set[int]] = {}

    @staticmethod
    def anchors(values: set[int]) -> list[int]:
        return sorted(values)[:32]

    def contains_near(self, text: str, threshold: float = 0.85) -> bool:
        if len(text) < 50:
            return False
        values = hashed_shingles(text)
        candidates: set[str] = set()
        for anchor in self.anchors(values):
            candidates.update(self.by_anchor.get(anchor, ()))
        for candidate_id in sorted(candidates):
            other = self.values[candidate_id]
            if len(values & other) / len(values | other) >= threshold:
                return True
        return False

    def add(self, candidate_id: str, text: str) -> None:
        if len(text) < 50:
            return
        values = hashed_shingles(text)
        self.values[candidate_id] = values
        for anchor in self.anchors(values):
            self.by_anchor[anchor].append(candidate_id)


def exact_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    buckets = list(weights)
    result = {bucket: int(total * weights[bucket]) for bucket in buckets}
    result[buckets[-1]] += total - sum(result.values())
    return result


def stage_budgets(config: dict[str, Any], stage: str) -> dict[str, int]:
    value = config["stage_budgets"][stage]
    defaults = {"smoke": (20_000, 20_000), "pilot": (40_000, 40_000)}
    if stage in defaults:
        validation, test = defaults[stage]
        return {
            "train": int(value["train"]),
            "validation": int(value.get("validation", validation)),
            "test": int(value.get("test", test)),
        }
    return {name: int(value[name]) for name in SPLITS}


def select_candidates(
    connection: sqlite3.Connection,
    budgets: dict[str, int],
    weights: dict[str, float],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    quotas = {split: exact_quotas(budgets[split], weights) for split in SPLITS}
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    tokens: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    rows: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    exact_seen: set[str] = set()
    prompt_seen: set[str] = set()
    near = NearIndex()
    rejected = Counter()
    query = """
      SELECT candidate_id, bucket, split, selection_rank, valid_targets,
             prompt_norm, prompt_digest, exact_digest, payload_json, provenance_json
      FROM candidates ORDER BY selection_rank, candidate_id
    """
    for value in connection.execute(query):
        (
            candidate_id, bucket, split, rank, valid_targets, prompt,
            prompt_digest, exact_digest, payload_json, provenance_json,
        ) = value
        if tokens[split][bucket] >= quotas[split][bucket]:
            continue
        if exact_digest in exact_seen:
            rejected["exact_conversation"] += 1
            continue
        if prompt_digest in prompt_seen:
            rejected["exact_prompt"] += 1
            continue
        if near.contains_near(prompt):
            rejected["near_prompt"] += 1
            continue
        selected[split].append(
            {
                "candidate_id": candidate_id,
                "bucket": bucket,
                "selection_rank": rank,
                "payload_json": payload_json,
                "provenance_json": provenance_json,
            }
        )
        tokens[split][bucket] += int(valid_targets)
        rows[split][bucket] += 1
        exact_seen.add(exact_digest)
        prompt_seen.add(prompt_digest)
        near.add(candidate_id, prompt)
        if all(
            tokens[name][bucket_name] >= quotas[name][bucket_name]
            for name in SPLITS for bucket_name in weights
        ):
            break
    shortages = {
        split: {
            bucket: quotas[split][bucket] - tokens[split][bucket]
            for bucket in weights if tokens[split][bucket] < quotas[split][bucket]
        }
        for split in SPLITS
    }
    shortages = {key: value for key, value in shortages.items() if value}
    if shortages:
        raise BuildError(f"sampled capacity does not meet quotas: {shortages}")
    return selected, {
        "quotas": quotas,
        "selected_rows": {split: dict(rows[split]) for split in SPLITS},
        "selected_assistant_tokens": {split: dict(tokens[split]) for split in SPLITS},
        "dedup_rejections": dict(rejected),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def tokenizer_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def write_outputs(
    output_root: Path,
    selected: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> Path:
    if output_root.exists():
        raise BuildError(f"output already exists: {output_root}")
    temporary = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise BuildError(f"temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    provenance_path = temporary / "provenance.jsonl"
    with provenance_path.open("w", encoding="utf-8") as provenance_handle:
        for split in SPLITS:
            with (temporary / f"{split}.jsonl").open("w", encoding="utf-8") as payload_handle:
                for line_number, item in enumerate(selected[split], 1):
                    payload_handle.write(item["payload_json"] + "\n")
                    provenance = json.loads(item["provenance_json"])
                    provenance["split_line_number"] = line_number
                    provenance_handle.write(stable_json(provenance) + "\n")
    manifest["files"] = {split: file_info(temporary / f"{split}.jsonl") for split in SPLITS}
    manifest["files"]["provenance"] = file_info(provenance_path)
    write_json(temporary / "manifest.json", manifest)
    write_json(temporary / "capacity_profile.json", manifest["capacity_profile"])
    write_json(
        temporary / "contamination_report.json",
        {"status": "pending_independent_audit", "fixed_eval_overlap": None},
    )
    os.replace(temporary, output_root)
    return output_root


def run_self_test() -> int:
    passing = [
        ("punctuation:use no comma", "Use no comma.", "No commas here"),
        ("case:in english and lowercase", "write in english and lowercase", "all lowercase text"),
        ("format:use json format", "Return JSON.", '{"ok":true}'),
        ("format:title", "Include a title.", "<<A Small Title>>\nBody"),
        ("specific ending", 'End with "DONE".', "Result\nDONE"),
        ("content:include a postscript", "Include a postscript.", "Body\nP.S.: note"),
        (
            "length constraints:number of words",
            "Describe each in no more than five words.",
            "1. alpha beta\n2. gamma delta",
        ),
        (
            "case: frequency of capital words",
            "Use at least two instances of words in all capital letters.",
            "Use ALPHA and BETA here.",
        ),
        (
            "format:choose one from options",
            'Your answer must contain one of the following exact phrases: "yes", "no".',
            "yes",
        ),
        (
            "give two responses",
            "Provide two responses.",
            "1. first response\n2. second response",
        ),
        (
            "response language",
            "Write your response in Spanish.",
            "La respuesta es para una persona que vive en la ciudad y con su familia.",
        ),
    ]
    for constraint, prompt, answer in passing:
        validate_strict(prompt, answer, [constraint])
    try:
        validate_strict("Use no comma.", "has, comma", ["punctuation:use no comma"])
    except BuildError:
        return 0
    raise AssertionError("strict verifier negative fixture unexpectedly passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    repo_root = Path.cwd().resolve()
    config_path = (repo_root / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 2:
        raise BuildError("build config schema_version must be 2")
    weights = {key: float(value) for key, value in config["planned_mix"].items()}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise BuildError("planned mix must sum to one")
    output_root = args.output_root or Path(config["paths"]["final_root"]) / args.stage
    work_root = args.work_root or Path(config["paths"]["work_root"])
    database_path = work_root / f"{args.stage}-candidates.sqlite"
    sources = load_sources(config, repo_root)
    tokenizer_path = repo_root / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    connection = initialize_database(database_path)
    try:
        capacity = populate_candidates(
            connection,
            sources,
            tokenizer,
            args.stage,
            int(config["sequence_length"]),
            args.max_source_rows,
        )
        budgets = stage_budgets(config, args.stage)
        selected, selection = select_candidates(connection, budgets, weights)
    finally:
        connection.close()
    manifest = {
        "schema_version": 1,
        "name": f"minimind-sft-v1-{args.stage}",
        "created_at": utc_now(),
        "status": "pending_external_audit",
        "trainable": False,
        "stage": args.stage,
        "seed": int(config["seed"]),
        "sequence_length": int(config["sequence_length"]),
        "budget_unit": "shifted_assistant_loss_target_tokens",
        "budgets": budgets,
        "planned_mix": weights,
        "selection": selection,
        "capacity_profile": capacity,
        "bindings": {
            "build_config": str(args.config),
            "build_config_sha256": file_sha256(config_path),
            "sources_config": config["contracts"]["sources"],
            "sources_config_sha256": file_sha256(repo_root / config["contracts"]["sources"]),
            "acceptance_config": config["contracts"]["acceptance"],
            "acceptance_config_sha256": file_sha256(repo_root / config["contracts"]["acceptance"]),
            "builder": "scripts/data/sft/build_sft_v1.py",
            "builder_sha256": file_sha256(Path(__file__)),
            "tokenizer": "tokenizer",
            "tokenizer_sha256": tokenizer_digest(tokenizer_path),
        },
        "source_objects": [
            {
                "source_id": source.source_id,
                "repo_id": source.repo_id,
                "revision": source.revision,
                "license": source.license,
                "path": str(source.path),
                "rows": source.rows,
                "bytes": source.size_bytes,
                "sha256": source.raw_sha256,
            }
            for source in sources
        ],
        "audit": {
            "required": True,
            "status": "pending",
            "success_marker_may_only_be_written_by": "scripts/data/sft/audit_sft_v1.py",
        },
    }
    final_path = write_outputs(output_root, selected, manifest)
    print(json.dumps({"status": "pending_external_audit", "output": str(final_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
