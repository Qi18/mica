#!/usr/bin/env python3
"""Profile full SFT assistant-target capacity without writing training candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from transformers import AutoTokenizer

from build_oasst_tree import child_key
from build_translation_proxy import pair as translation_pair
from generate_strict_format import GENERATORS as STRICT_GENERATORS
from normalize_glaive_tool import parse_chat as parse_glaive_chat
from normalize_sft_proxy import (
    adapt_alpaca_gpt4_zh,
    adapt_code_alpaca,
    adapt_gsm8k,
    adapt_mbpp,
    adapt_ultrachat,
)
from tokenizer_stats import create_chat_prompt, sha256_file, tokenizer_fingerprint


PROTOCOL_VERSION = "2026-09-01-sft-capacity-v1"
MESSAGE_KEYS = ("role", "content", "reasoning_content", "tools", "tool_calls")
VALID_ROLES = {"system", "user", "assistant", "tool"}
ROLE_MAP = {"human": "user", "prompter": "user", "gpt": "assistant"}
REPO_ROOT = Path(__file__).resolve().parents[4]


class ProfileError(RuntimeError):
    """Fail-closed profiler error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixture-output", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise ProfileError(f"temporary output already exists: {temporary}")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalized_license(value: Any) -> str:
    if isinstance(value, list):
        values = [str(item).strip().lower() for item in value if str(item).strip()]
        return ",".join(sorted(values))
    return str(value or "").strip().lower()


def normalized_json_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"{field} is not valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, list) or not parsed:
        raise ProfileError(f"{field} must be a non-empty JSON list")
    return json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_message(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProfileError("message is not an object")
    raw_role = str(message.get("role", "")).strip().lower()
    role = ROLE_MAP.get(raw_role, raw_role)
    if role not in VALID_ROLES:
        raise ProfileError(f"unsupported role: {raw_role!r}")

    raw_content = message.get("content")
    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content.strip()
    else:
        raise ProfileError("message content must be a string")

    raw_reasoning = message.get("reasoning_content")
    if raw_reasoning is None:
        reasoning_content = None
    elif isinstance(raw_reasoning, str):
        reasoning_content = raw_reasoning.strip() or None
    else:
        raise ProfileError("reasoning_content must be a string or null")

    tools = normalized_json_string(message.get("tools"), "tools")
    tool_calls = normalized_json_string(message.get("tool_calls"), "tool_calls")

    if tools is not None and role != "system":
        raise ProfileError("tools is only valid on a system message")
    if tool_calls is not None and role != "assistant":
        raise ProfileError("tool_calls is only valid on an assistant message")
    if reasoning_content is not None and role != "assistant":
        raise ProfileError("reasoning_content is only valid on an assistant message")
    if role in {"system", "user", "tool"} and not content:
        raise ProfileError(f"{role} message has empty content")
    if role == "assistant" and not (content or reasoning_content or tool_calls):
        raise ProfileError("assistant message has no supervised content")

    result = {
        "role": role,
        "content": content,
        "reasoning_content": reasoning_content,
        "tools": tools,
        "tool_calls": tool_calls,
    }
    if tuple(result) != MESSAGE_KEYS:
        raise AssertionError("canonical message key order drift")
    return result


def canonical_conversation(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [canonical_message(message) for message in messages]
    if not result:
        raise ProfileError("empty conversation")
    for message in result:
        if set(message) != set(MESSAGE_KEYS) or len(message) != len(MESSAGE_KEYS):
            raise AssertionError("canonical message schema drift")
    return result


def training_row(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"conversations": messages}


def exact_digest(messages: list[dict[str, Any]]) -> bytes:
    payload = json.dumps(
        training_row(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def parsed_list(value: str | None, field: str) -> list[Any]:
    if value is None:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{field} is not valid canonical JSON") from exc
    if not isinstance(parsed, list):
        raise ProfileError(f"{field} is not a list")
    return parsed


def split_into_turns(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    system: list[dict[str, Any]] = []
    body = messages
    if messages and messages[0]["role"] == "system":
        system = [messages[0]]
        body = messages[1:]
    if any(message["role"] == "system" for message in body):
        raise ProfileError("system message is only allowed at position zero")
    if not body or body[0]["role"] != "user":
        raise ProfileError("conversation body must start with user")

    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in body:
        if message["role"] == "user":
            if current:
                turns.append(current)
            current = [message]
        elif not current:
            raise ProfileError("message appears before the first user")
        else:
            current.append(message)
    if current:
        turns.append(current)

    has_tool_definitions = bool(system and system[0]["tools"])
    for turn in turns:
        assistants = 0
        pending_calls = 0
        for index, message in enumerate(turn):
            role = message["role"]
            if index == 0 and role != "user":
                raise ProfileError("turn does not start with user")
            if index > 0 and role == "user":
                raise AssertionError("turn boundary construction drift")
            if role == "assistant":
                assistants += 1
                if pending_calls:
                    raise ProfileError("assistant arrived before all tool responses")
                calls = parsed_list(message["tool_calls"], "tool_calls")
                if calls:
                    if not has_tool_definitions:
                        raise ProfileError("tool call has no system tool definitions")
                    pending_calls = len(calls)
            elif role == "tool":
                if pending_calls <= 0:
                    raise ProfileError("orphan tool response")
                pending_calls -= 1
            elif role == "system":
                raise ProfileError("system message inside a turn")
        if assistants == 0:
            raise ProfileError("turn has no assistant message")
        if pending_calls:
            raise ProfileError("tool call is not closed by tool responses")
    return system, turns


def find_sequence(values: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    limit = len(values) - len(needle)
    for index in range(start, limit + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def all_sequence_positions(values: list[int], needle: list[int]) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        position = find_sequence(values, needle, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + max(1, len(needle))


def render_metrics(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    bos_id: list[int],
    eos_id: list[int],
) -> dict[str, int]:
    prompt = create_chat_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt).input_ids
    assistant_messages = sum(
        1 for message in messages if message["role"] == "assistant"
    )
    marker_positions = all_sequence_positions(input_ids, bos_id)
    if len(marker_positions) != assistant_messages:
        raise ProfileError(
            "assistant marker count mismatch: "
            f"messages={assistant_messages} markers={len(marker_positions)}"
        )

    valid_targets = 0
    closed_spans = 0
    for marker_index, marker_position in enumerate(marker_positions):
        span_start = marker_position + len(bos_id)
        span_end = find_sequence(input_ids, eos_id, span_start)
        if span_end < 0:
            raise ProfileError("assistant span is missing closing EOS")
        if (
            marker_index + 1 < len(marker_positions)
            and span_end >= marker_positions[marker_index + 1]
        ):
            raise ProfileError("assistant span crosses the next assistant marker")
        supervised_end = span_end + len(eos_id)
        valid_targets += max(0, supervised_end - max(1, span_start))
        closed_spans += 1

    if valid_targets <= 0:
        raise ProfileError("shifted training labels contain no valid target")
    return {
        "rendered_tokens": len(input_ids),
        "assistant_messages": assistant_messages,
        "assistant_markers": len(marker_positions),
        "closed_assistant_spans": closed_spans,
        "valid_targets": valid_targets,
    }


def split_complete_turns(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_length: int,
    bos_id: list[int],
    eos_id: list[int],
) -> tuple[list[tuple[list[dict[str, Any]], dict[str, int]]], int]:
    system, turns = split_into_turns(messages)
    chunks: list[tuple[list[dict[str, Any]], dict[str, int]]] = []
    current_turns: list[list[dict[str, Any]]] = []
    dropped_overlength_turns = 0

    def flatten(selected: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return system + [message for turn in selected for message in turn]

    for turn in turns:
        candidate_turns = current_turns + [turn]
        candidate_messages = flatten(candidate_turns)
        candidate_metrics = render_metrics(
            tokenizer, candidate_messages, bos_id, eos_id
        )
        if candidate_metrics["rendered_tokens"] <= max_length:
            current_turns = candidate_turns
            continue

        if current_turns:
            current_messages = flatten(current_turns)
            current_metrics = render_metrics(
                tokenizer, current_messages, bos_id, eos_id
            )
            if current_metrics["rendered_tokens"] > max_length:
                raise AssertionError("accepted chunk exceeds sequence length")
            chunks.append((current_messages, current_metrics))
            current_turns = []

        single_messages = flatten([turn])
        single_metrics = render_metrics(tokenizer, single_messages, bos_id, eos_id)
        if single_metrics["rendered_tokens"] <= max_length:
            current_turns = [turn]
        else:
            dropped_overlength_turns += 1

    if current_turns:
        current_messages = flatten(current_turns)
        current_metrics = render_metrics(tokenizer, current_messages, bos_id, eos_id)
        if current_metrics["rendered_tokens"] > max_length:
            raise AssertionError("accepted final chunk exceeds sequence length")
        chunks.append((current_messages, current_metrics))
    return chunks, dropped_overlength_turns


class DigestIndex:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE seen (digest BLOB PRIMARY KEY, first_source TEXT NOT NULL)"
        )

    def add(self, digest: bytes, source_id: str) -> tuple[bool, str | None]:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO seen(digest, first_source) VALUES (?, ?)",
            (sqlite3.Binary(digest), source_id),
        )
        if cursor.rowcount == 1:
            return True, None
        row = self.connection.execute(
            "SELECT first_source FROM seen WHERE digest = ?",
            (sqlite3.Binary(digest),),
        ).fetchone()
        return False, str(row[0]) if row else "unknown"

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM seen").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class SourceProfile:
    def __init__(self, source_id: str, purpose: str, bucket: str):
        self.source_id = source_id
        self.purpose = purpose
        self.bucket = bucket
        self.counts: Counter[str] = Counter()
        self.duplicate_first_source: Counter[str] = Counter()
        self.invalid_examples: list[dict[str, Any]] = []
        self.max_rendered_tokens = 0
        self.raw_evidence: dict[str, Any] | None = None
        self.extra: dict[str, Any] = {}

    def invalid(self, kind: str, reference: Any, error: Exception | str) -> None:
        self.counts[kind] += 1
        if len(self.invalid_examples) < 5:
            self.invalid_examples.append(
                {
                    "reference": reference,
                    "kind": kind,
                    "error": str(error)[:300],
                }
            )

    def as_dict(self) -> dict[str, Any]:
        invalid_total = sum(
            count
            for key, count in self.counts.items()
            if key.startswith("invalid_") or key.endswith("_errors")
        )
        result = {
            "source_id": self.source_id,
            "purpose": self.purpose,
            "planned_mix_bucket": self.bucket,
            **dict(sorted(self.counts.items())),
            "invalid_total": invalid_total,
            "max_rendered_tokens": self.max_rendered_tokens,
            "duplicate_first_source": dict(sorted(self.duplicate_first_source.items())),
            "invalid_examples": self.invalid_examples,
        }
        if self.raw_evidence is not None:
            result["raw_evidence"] = self.raw_evidence
        result.update(self.extra)
        return result


class CapacityEngine:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        digest_index: DigestIndex,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.digest_index = digest_index
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n",
            add_special_tokens=False,
        ).input_ids
        if not self.bos_id or not self.eos_id:
            raise ProfileError("assistant marker token sequences are empty")

    def add_conversation(
        self,
        profile: SourceProfile,
        messages: Iterable[dict[str, Any]],
        reference: Any,
    ) -> None:
        profile.counts["adapted_conversations"] += 1
        try:
            canonical = canonical_conversation(messages)
            chunks, dropped = split_complete_turns(
                self.tokenizer,
                canonical,
                self.max_length,
                self.bos_id,
                self.eos_id,
            )
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            profile.invalid("invalid_conversations", reference, exc)
            return

        profile.counts["dropped_overlength_turns"] += dropped
        if dropped:
            profile.counts["conversations_with_overlength_turns"] += 1
        if not chunks:
            profile.counts["conversations_dropped_all_turns"] += 1
            return
        if len(chunks) > 1:
            profile.counts["conversations_split"] += 1

        for chunk, metrics in chunks:
            profile.counts["chunks_before_exact_dedup"] += 1
            profile.counts["assistant_target_tokens_before_exact_dedup"] += metrics[
                "valid_targets"
            ]
            digest = exact_digest(chunk)
            inserted, first_source = self.digest_index.add(
                digest, profile.source_id
            )
            if not inserted:
                profile.counts["exact_duplicate_chunks"] += 1
                profile.counts["exact_duplicate_assistant_target_tokens"] += metrics[
                    "valid_targets"
                ]
                profile.duplicate_first_source[str(first_source)] += 1
                continue

            if metrics["rendered_tokens"] > self.max_length:
                raise AssertionError("profiler accepted an overlength chunk")
            if metrics["assistant_markers"] != metrics["assistant_messages"]:
                raise AssertionError("assistant marker invariant drift")
            if metrics["closed_assistant_spans"] != metrics["assistant_messages"]:
                raise AssertionError("assistant EOS closure invariant drift")
            if metrics["valid_targets"] <= 0:
                raise AssertionError("shifted target invariant drift")

            profile.counts["unique_chunks"] += 1
            profile.counts["assistant_target_tokens"] += metrics["valid_targets"]
            profile.counts["rendered_tokens"] += metrics["rendered_tokens"]
            profile.counts["assistant_messages"] += metrics["assistant_messages"]
            profile.counts["assistant_markers"] += metrics["assistant_markers"]
            profile.counts["closed_assistant_spans"] += metrics[
                "closed_assistant_spans"
            ]
            if sum(message["role"] == "user" for message in chunk) > 1:
                profile.counts["multiturn_chunks"] += 1
            if any(
                message["role"] == "tool" or message["tool_calls"] is not None
                for message in chunk
            ):
                profile.counts["tool_chunks"] += 1
            profile.max_rendered_tokens = max(
                profile.max_rendered_tokens, metrics["rendered_tokens"]
            )


SIMPLE_ADAPTERS = {
    "ultrachat_200k": adapt_ultrachat,
    "gsm8k": adapt_gsm8k,
    "mbpp": adapt_mbpp,
    "code_alpaca_20k": adapt_code_alpaca,
    "alpaca_gpt4_zh": adapt_alpaca_gpt4_zh,
}


def adapt_dolly(row: dict[str, Any]) -> list[dict[str, Any]]:
    instruction = str(row["instruction"]).strip()
    context = str(row.get("context") or "").strip()
    response = str(row["response"]).strip()
    if not instruction or not response:
        raise ProfileError("Dolly instruction or response is empty")
    prompt = instruction
    if context:
        prompt += "\n\nContext:\n" + context
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def load_profile_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ProfileError("profile protocol_version mismatch")
    if int(value.get("sequence_length", 0)) <= 0:
        raise ProfileError("sequence_length must be positive")
    if int(value.get("target_assistant_tokens", 0)) <= 0:
        raise ProfileError("target_assistant_tokens must be positive")
    if not value.get("source_order"):
        raise ProfileError("source_order is empty")
    return value


def build_source_specs(profile_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_config_path = resolve_repo_path(profile_config["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    by_id = {str(source["id"]): dict(source) for source in source_config["sources"]}

    for supplemental in profile_config.get("supplemental_sources", []):
        source_id = str(supplemental["id"])
        if source_id in by_id:
            existing = by_id[source_id]
            for key in ("repo_id", "revision", "license"):
                if str(existing.get(key)) != str(supplemental.get(key)):
                    raise ProfileError(
                        f"supplemental source {source_id} conflicts on {key}"
                    )
        else:
            by_id[source_id] = dict(supplemental)

    data_root = Path(profile_config["data_root"])
    specs = []
    for source_id in profile_config["source_order"]:
        if source_id not in by_id:
            raise ProfileError(f"source_order references unknown source: {source_id}")
        source = dict(by_id[source_id])
        materialization = dict(source.get("materialization") or {})
        output = materialization.get("output")
        if not output:
            raise ProfileError(f"source {source_id} has no materialization output")
        source["materialization"] = materialization
        source["raw_path"] = data_root / output
        source["done_path"] = Path(str(source["raw_path"]) + ".done.json")
        source["bucket"] = (
            "summarization_translation"
            if source.get("purpose") == "summarization"
            else source.get("purpose")
        )
        specs.append(source)

    acceptance_path = resolve_repo_path(profile_config["acceptance_config"])
    acceptance = yaml.safe_load(acceptance_path.read_text(encoding="utf-8"))
    budgets = acceptance["budgets"]
    split_budgets = {
        split: int(budgets[split])
        for split in ("train", "validation", "test")
    }
    total_eligible_tokens = sum(split_budgets.values())
    bucket_weights = {
        str(key): float(value)
        for key, value in budgets["bucket_weights"].items()
    }
    planned_mix = {
        str(key): float(value)
        for key, value in source_config["planned_mix"].items()
    }
    if int(acceptance["sequence_length"]) != int(profile_config["sequence_length"]):
        raise ProfileError("acceptance sequence_length mismatch")
    if split_budgets["train"] != int(profile_config["target_assistant_tokens"]):
        raise ProfileError("profile train target differs from acceptance train budget")
    if planned_mix != bucket_weights:
        raise ProfileError("source planned_mix differs from acceptance bucket_weights")
    if not math.isclose(sum(bucket_weights.values()), 1.0, abs_tol=1e-12):
        raise ProfileError("acceptance bucket weights do not sum to one")
    if budgets.get("record_reuse_across_buckets_or_splits") is not False:
        raise ProfileError("acceptance must forbid record reuse")
    quota_rule = dict(budgets.get("quota_rule") or {})
    if quota_rule.get("maximum_overshoot") != "less_than_one_selected_complete_record":
        raise ProfileError("acceptance complete-record overshoot rule mismatch")

    return specs, {
        "path": str(source_config_path),
        "sha256": sha256_file(source_config_path),
        "planned_mix": source_config["planned_mix"],
        "target_assistant_tokens": source_config["target_assistant_tokens"],
        "sequence_length": source_config["sequence_length"],
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
            "split_budgets": split_budgets,
            "total_eligible_assistant_tokens": total_eligible_tokens,
            "bucket_weights": bucket_weights,
            "quota_rule": quota_rule,
            "record_reuse_across_buckets_or_splits": False,
        },
    }


def validate_done_evidence(source: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(source["raw_path"])
    done_path = Path(source["done_path"])
    if not raw_path.exists() and not done_path.exists():
        return {
            "source_id": source["id"],
            "status": "missing",
            "raw_path": str(raw_path),
            "done_path": str(done_path),
        }
    if raw_path.exists() != done_path.exists():
        return {
            "source_id": source["id"],
            "status": "one_sided_evidence",
            "raw_exists": raw_path.exists(),
            "done_exists": done_path.exists(),
        }
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "source_id": source["id"],
            "status": "invalid_done_json",
            "error": f"{type(exc).__name__}: {exc}",
        }

    mismatches = []
    expected = {
        "source_id": str(source["id"]),
        "repo_id": str(source["repo_id"]),
        "revision": str(source["revision"]),
    }
    for key, value in expected.items():
        if str(done.get(key)) != value:
            mismatches.append(
                {"field": key, "expected": value, "actual": done.get(key)}
            )
    if normalized_license(done.get("license")) != normalized_license(
        source.get("license")
    ):
        mismatches.append(
            {
                "field": "license",
                "expected": normalized_license(source.get("license")),
                "actual": normalized_license(done.get("license")),
            }
        )
    stat_bytes = raw_path.stat().st_size
    if int(done.get("bytes", -1)) != stat_bytes:
        mismatches.append(
            {
                "field": "bytes",
                "expected": done.get("bytes"),
                "actual": stat_bytes,
            }
        )
    required_fields = set(source["materialization"].get("required_fields") or [])
    done_fields = set(done.get("fields") or [])
    if not required_fields.issubset(done_fields):
        mismatches.append(
            {
                "field": "required_fields",
                "expected_subset": sorted(required_fields),
                "actual": sorted(done_fields),
            }
        )
    return {
        "source_id": source["id"],
        "status": "ready" if not mismatches else "evidence_mismatch",
        "raw_path": str(raw_path),
        "done_path": str(done_path),
        "rows": done.get("rows"),
        "bytes": done.get("bytes"),
        "sha256": done.get("sha256"),
        "storage": done.get("storage"),
        "mismatches": mismatches,
    }


def iter_verified_jsonl(
    source: dict[str, Any],
    profile: SourceProfile,
) -> Iterator[tuple[int, dict[str, Any]]]:
    raw_path = Path(source["raw_path"])
    done = json.loads(Path(source["done_path"]).read_text(encoding="utf-8"))
    required_fields = set(source["materialization"].get("required_fields") or [])
    digest = hashlib.sha256()
    rows = 0
    byte_count = 0

    with raw_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            rows += 1
            profile.counts["raw_rows_read"] += 1
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                profile.invalid("invalid_raw_json", line_number, exc)
                continue
            if not isinstance(row, dict):
                profile.invalid(
                    "invalid_raw_schema", line_number, "raw row is not an object"
                )
                continue
            missing = sorted(required_fields - set(row))
            if missing:
                profile.invalid(
                    "invalid_required_fields",
                    line_number,
                    "missing fields: " + ",".join(missing),
                )
                continue
            yield line_number, row

    actual_sha = digest.hexdigest()
    expected_rows = int(done["rows"])
    expected_bytes = int(done["bytes"])
    expected_sha = str(done["sha256"])
    if rows != expected_rows or byte_count != expected_bytes or actual_sha != expected_sha:
        raise ProfileError(
            f"raw evidence drift for {source['id']}: "
            f"rows={rows}/{expected_rows} bytes={byte_count}/{expected_bytes} "
            f"sha256={actual_sha}/{expected_sha}"
        )
    profile.raw_evidence = {
        "path": str(raw_path),
        "done_path": str(source["done_path"]),
        "rows": rows,
        "bytes": byte_count,
        "sha256": actual_sha,
        "revision": source["revision"],
        "license": source["license"],
        "required_fields": sorted(required_fields),
    }


def profile_simple_source(
    source: dict[str, Any],
    profile: SourceProfile,
    engine: CapacityEngine,
) -> None:
    source_id = str(source["id"])
    for line_number, row in iter_verified_jsonl(source, profile):
        try:
            if source_id == "glaive_function_calling_v2":
                messages, parse_errors = parse_glaive_chat(
                    str(row["chat"]), str(row["system"])
                )
                if parse_errors:
                    profile.counts["function_call_parse_errors"] += parse_errors
                    raise ProfileError(
                        f"Glaive function-call parse errors: {parse_errors}"
                    )
            elif source_id == "databricks_dolly_15k":
                category = str(row.get("category") or "unknown")
                profile.counts[f"dolly_category:{category}"] += 1
                if category != "summarization":
                    profile.counts["filtered_non_summarization_rows"] += 1
                    continue
                messages = adapt_dolly(row)
            else:
                adapter = SIMPLE_ADAPTERS[source_id]
                messages = adapter(row)
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            profile.invalid("invalid_adapter_rows", line_number, exc)
            continue
        engine.add_conversation(profile, messages, line_number)
    engine.digest_index.commit()


def profile_oasst(
    source: dict[str, Any],
    profile: SourceProfile,
    engine: CapacityEngine,
    languages: set[str],
) -> None:
    nodes: dict[str, dict[str, Any]] = {}
    language_counts: Counter[str] = Counter()
    for line_number, row in iter_verified_jsonl(source, profile):
        language = str(row.get("lang", ""))
        language_counts[language] += 1
        message_id = str(row.get("message_id", ""))
        role = str(row.get("role", "")).lower()
        text = str(row.get("text", "")).strip()
        if (
            message_id
            and role in {"prompter", "assistant"}
            and text
            and not row.get("deleted")
            and language in languages
        ):
            nodes[message_id] = {
                "message_id": message_id,
                "message_tree_id": row.get("message_tree_id"),
                "parent_id": row.get("parent_id"),
                "role": role,
                "text": text,
                "review_result": row.get("review_result"),
                "rank": row.get("rank"),
                "synthetic": row.get("synthetic"),
                "lang": language,
            }

    children: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    roots: list[dict[str, Any]] = []
    for node in nodes.values():
        parent_id = node.get("parent_id")
        if parent_id and str(parent_id) in nodes:
            children[str(parent_id)].append(node)
        elif node["role"] == "prompter":
            roots.append(node)
    for values in children.values():
        values.sort(key=child_key)
    roots.sort(
        key=lambda item: (
            str(item.get("message_tree_id", "")),
            str(item["message_id"]),
        )
    )

    incomplete = 0
    for root in roots:
        path = [root]
        visited = {str(root["message_id"])}
        current = root
        for _ in range(31):
            expected = "assistant" if current["role"] == "prompter" else "prompter"
            candidates = [
                item
                for item in children.get(str(current["message_id"]), [])
                if item["role"] == expected
                and str(item["message_id"]) not in visited
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
                "content": item["text"],
            }
            for item in path
        ]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            incomplete += 1
            continue
        engine.add_conversation(
            profile,
            messages,
            {
                "message_tree_id": root.get("message_tree_id"),
                "root_message_id": root["message_id"],
            },
        )

    profile.extra.update(
        {
            "oasst_rebuild": {
                "languages": sorted(languages),
                "eligible_nodes": len(nodes),
                "roots": len(roots),
                "incomplete_roots": incomplete,
                "language_counts_all_rows": dict(language_counts),
                "selection_policy": (
                    "one deterministic root-to-leaf path per tree; prefer "
                    "reviewed, lower-rank, non-synthetic child"
                ),
                "network_used": False,
            }
        }
    )
    engine.digest_index.commit()


def profile_translation(
    source: dict[str, Any],
    profile: SourceProfile,
    engine: CapacityEngine,
) -> None:
    for line_number, row in iter_verified_jsonl(source, profile):
        candidates = (
            translation_pair(
                row, "instruction", "instruction_zh", "instruction"
            ),
            translation_pair(row, "output", "output_zh", "output"),
        )
        for candidate in candidates:
            if candidate is None:
                profile.counts["missing_aligned_pairs"] += 1
                continue
            profile.counts[
                "translation_field:" + str(candidate["aligned_field"])
            ] += 1
            engine.add_conversation(
                profile,
                candidate["conversations"],
                {
                    "line_number": line_number,
                    "aligned_field": candidate["aligned_field"],
                },
            )
    engine.digest_index.commit()


def profile_strict_format(
    profile: SourceProfile,
    engine: CapacityEngine,
    rows: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    generator_names = list(STRICT_GENERATORS)
    profile.counts["generator_rows_requested"] = rows
    for index in range(rows):
        category = generator_names[index % len(generator_names)]
        prompt, answer = STRICT_GENERATORS[category](index, rng)
        profile.counts[f"strict_category:{category}"] += 1
        engine.add_conversation(
            profile,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            {"generator_index": index, "category": category},
        )
    engine.digest_index.commit()


def validate_profile_inputs(
    config_path: Path,
    profile_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specs, source_config = build_source_specs(profile_config)
    evidence = [validate_done_evidence(source) for source in specs]
    tokenizer_path = resolve_repo_path(profile_config["tokenizer"])
    tokenizer_files = tokenizer_fingerprint(tokenizer_path)
    issues = [
        item
        for item in evidence
        if item["status"] not in {"ready", "missing"}
    ]
    missing = [item for item in evidence if item["status"] == "missing"]
    result = {
        "validation": "passed" if not issues else "failed",
        "ready_for_full_profile": not issues and not missing,
        "profile_config": str(config_path),
        "profile_config_sha256": sha256_file(config_path),
        "source_config": source_config,
        "tokenizer": str(tokenizer_path),
        "tokenizer_files": tokenizer_files,
        "source_evidence": evidence,
        "missing_sources": [item["source_id"] for item in missing],
        "issues": issues,
        "network_used": False,
    }
    return result, specs, evidence


def bucket_report(
    planned_mix: dict[str, Any],
    target_tokens: int,
    profiles: dict[str, SourceProfile],
) -> dict[str, Any]:
    result = {}
    for bucket, raw_weight in planned_mix.items():
        weight = float(raw_weight)
        target = int(round(target_tokens * weight))
        members = [
            profile
            for profile in profiles.values()
            if profile.bucket == bucket
        ]
        available = sum(
            profile.counts["assistant_target_tokens"] for profile in members
        )
        result[bucket] = {
            "weight": weight,
            "target_assistant_tokens": target,
            "available_assistant_tokens": available,
            "capacity_ratio": round(available / target, 6) if target else None,
            "shortfall_assistant_tokens": max(0, target - available),
            "source_ids": [profile.source_id for profile in members],
            "subitems": {
                profile.source_id: profile.counts["assistant_target_tokens"]
                for profile in members
            },
        }
    return result


def run_profile(config_path: Path, output: Path) -> int:
    if output.exists():
        raise ProfileError(f"refusing to overwrite capacity report: {output}")
    profile_config = load_profile_config(config_path)
    validation, specs, _ = validate_profile_inputs(config_path, profile_config)
    if validation["validation"] != "passed":
        raise ProfileError("input evidence mismatch; run validate mode")
    if validation["missing_sources"]:
        raise ProfileError(
            "missing raw sources: " + ",".join(validation["missing_sources"])
        )

    tokenizer_path = resolve_repo_path(profile_config["tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    sqlite_dir = Path(profile_config["sqlite_work_dir"])
    sqlite_dir.mkdir(parents=True, exist_ok=True)
    descriptor, sqlite_name = tempfile.mkstemp(
        prefix="sft-capacity-digests-",
        suffix=".sqlite",
        dir=sqlite_dir,
    )
    os.close(descriptor)
    sqlite_path = Path(sqlite_name)
    sqlite_path.unlink()
    digest_index = DigestIndex(sqlite_path)
    engine = CapacityEngine(
        tokenizer=tokenizer,
        max_length=int(profile_config["sequence_length"]),
        digest_index=digest_index,
    )
    profiles: dict[str, SourceProfile] = {}

    try:
        for source in specs:
            source_id = str(source["id"])
            profile = SourceProfile(
                source_id=source_id,
                purpose=str(source["purpose"]),
                bucket=str(source["bucket"]),
            )
            profiles[source_id] = profile
            if source_id == "oasst1":
                profile_oasst(
                    source,
                    profile,
                    engine,
                    set(profile_config["oasst_languages"]),
                )
            else:
                profile_simple_source(source, profile, engine)

        translation_config = profile_config["derived"]["translation"]
        translation_source_id = str(translation_config["input_source"])
        source_by_id = {str(source["id"]): source for source in specs}
        translation_profile = SourceProfile(
            source_id=str(translation_config["id"]),
            purpose="translation",
            bucket="summarization_translation",
        )
        profiles[translation_profile.source_id] = translation_profile
        profile_translation(
            source_by_id[translation_source_id],
            translation_profile,
            engine,
        )

        strict_config = profile_config["derived"]["strict_format"]
        strict_profile = SourceProfile(
            source_id=str(strict_config["id"]),
            purpose="strict_format",
            bucket="strict_format",
        )
        profiles[strict_profile.source_id] = strict_profile
        profile_strict_format(
            strict_profile,
            engine,
            rows=int(strict_config["current_generator_rows"]),
            seed=int(strict_config["seed"]),
        )

        acceptance = validation["source_config"]["acceptance"]
        total_eligible_target = int(
            acceptance["total_eligible_assistant_tokens"]
        )
        buckets = bucket_report(
            acceptance["bucket_weights"],
            total_eligible_target,
            profiles,
        )
        strict_tokens = strict_profile.counts["assistant_target_tokens"]
        strict_rows = strict_profile.counts["unique_chunks"]
        strict_target = int(strict_config["target_assistant_tokens"])
        strict_total_eligible_target = int(
            buckets["strict_format"]["target_assistant_tokens"]
        )
        projected_rows = (
            math.ceil(strict_target * strict_rows / strict_tokens)
            if strict_tokens and strict_rows
            else None
        )
        projected_rows_total_eligible = (
            math.ceil(
                strict_total_eligible_target * strict_rows / strict_tokens
            )
            if strict_tokens and strict_rows
            else None
        )
        total_available = sum(
            profile.counts["assistant_target_tokens"]
            for profile in profiles.values()
        )

        blockers = []
        for bucket, item in buckets.items():
            if item["shortfall_assistant_tokens"]:
                blockers.append(
                    {
                        "id": f"capacity_shortfall:{bucket}",
                        "severity": "fatal",
                        "shortfall_assistant_tokens": item[
                            "shortfall_assistant_tokens"
                        ],
                    }
                )
        if strict_tokens < strict_target:
            blockers.append(
                {
                    "id": "strict_format_full_generation_not_materialized",
                    "severity": "fatal",
                    "projected_rows": projected_rows,
                }
            )
        report = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "created_at": utc_now(),
            "status": "capacity_profiled_not_trainable",
            "trainable": False,
            "profile_config": str(config_path),
            "profile_config_sha256": sha256_file(config_path),
            "profiler": str(Path(__file__).resolve()),
            "profiler_sha256": sha256_file(Path(__file__).resolve()),
            "source_config": validation["source_config"],
            "tokenizer": {
                "path": str(tokenizer_path),
                "class": type(tokenizer).__name__,
                "vocab_size": tokenizer.vocab_size,
                "files": tokenizer_fingerprint(tokenizer_path),
            },
            "sequence_length": int(profile_config["sequence_length"]),
            "target_assistant_tokens": int(
                profile_config["target_assistant_tokens"]
            ),
            "train_target_assistant_tokens": int(
                acceptance["split_budgets"]["train"]
            ),
            "heldout_target_assistant_tokens": (
                int(acceptance["split_budgets"]["validation"])
                + int(acceptance["split_budgets"]["test"])
            ),
            "total_eligible_target_assistant_tokens": total_eligible_target,
            "source_precedence_for_exact_dedup": (
                [str(source["id"]) for source in specs]
                + [
                    translation_profile.source_id,
                    strict_profile.source_id,
                ]
            ),
            "exact_digest_policy": (
                "SHA256 of canonical training payload; SQLite PRIMARY KEY; "
                "first source in deterministic precedence owns collisions"
            ),
            "training_schema_contract": {
                "top_level_keys": ["conversations"],
                "message_keys": list(MESSAGE_KEYS),
                "optional_values": "explicit null",
                "provenance": "sidecar only in the later formal builder",
            },
            "rendering_gates": {
                "complete_turn_splitting": True,
                "rendered_tokens_max": int(profile_config["sequence_length"]),
                "assistant_marker_equals_assistant_messages": True,
                "every_assistant_span_closed_by_eos": True,
                "shifted_valid_targets_min": 1,
            },
            "global": {
                "unique_chunks": digest_index.count(),
                "assistant_target_tokens": total_available,
                "eligible_capacity": {
                    "required_assistant_target_tokens": total_eligible_target,
                    "available_assistant_target_tokens": total_available,
                    "shortfall_assistant_target_tokens": max(
                        0, total_eligible_target - total_available
                    ),
                    "surplus_assistant_target_tokens": max(
                        0, total_available - total_eligible_target
                    ),
                    "meets_total_requirement": (
                        total_available >= total_eligible_target
                    ),
                    "note": (
                        "The total gate does not replace every split/bucket "
                        "quota and no-record-reuse gates."
                    ),
                },
                "exact_duplicate_chunks": sum(
                    profile.counts["exact_duplicate_chunks"]
                    for profile in profiles.values()
                ),
            },
            "sources": {
                source_id: profile.as_dict()
                for source_id, profile in profiles.items()
            },
            "planned_mix_buckets": buckets,
            "summarization_translation_breakdown": {
                "summarization": profiles[
                    "databricks_dolly_15k"
                ].counts["assistant_target_tokens"],
                "translation": translation_profile.counts[
                    "assistant_target_tokens"
                ],
                "combined": buckets["summarization_translation"][
                    "available_assistant_tokens"
                ],
            },
            "strict_format_projection": {
                "current_generator_rows": int(
                    strict_config["current_generator_rows"]
                ),
                "measured_unique_chunks": strict_rows,
                "measured_assistant_target_tokens": strict_tokens,
                "train_target_assistant_tokens": strict_target,
                "projected_rows_for_train_target": projected_rows,
                "total_eligible_target_assistant_tokens": (
                    strict_total_eligible_target
                ),
                "projected_rows_for_total_eligible_target": (
                    projected_rows_total_eligible
                ),
                "projection_assumption": (
                    "linear scaling at the measured deterministic generator "
                    "assistant-target-token average"
                ),
            },
            "final_build_blockers": blockers
            + [
                {
                    "id": "source_capacity_mix_not_frozen",
                    "severity": "fatal",
                    "reason": (
                        "A1 only measures capacity; A2/A3 must resolve derived "
                        "data and freeze quotas before formal build."
                    ),
                }
            ],
            "final_build_allowed": False,
            "network_used": False,
        }
        atomic_write_json(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": report["status"],
                    "sources": len(profiles),
                    "unique_chunks": report["global"]["unique_chunks"],
                    "assistant_target_tokens": report["global"][
                        "assistant_target_tokens"
                    ],
                    "blockers": len(report["final_build_blockers"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        digest_index.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            candidate = Path(str(sqlite_path) + suffix)
            if candidate.exists():
                candidate.unlink()
    return 0


def run_self_test(tokenizer_path: Path, fixture_output: Path | None) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    with tempfile.TemporaryDirectory(prefix="sft-capacity-self-test-") as directory:
        index = DigestIndex(Path(directory) / "digests.sqlite")
        engine = CapacityEngine(tokenizer, 768, index)
        profile = SourceProfile("self_test", "test", "test")

        tools = json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup one value.",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"],
                        },
                    },
                }
            ],
            ensure_ascii=False,
        )
        tool_calls = json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"key": "alpha"},
                    },
                }
            ],
            ensure_ascii=False,
        )
        long_messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use tools when needed.", "tools": tools},
            {"role": "user", "content": "Look up alpha."},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "content": "{\"value\": 7}"},
            {"role": "assistant", "content": "The value is 7."},
        ]
        for turn_index in range(6):
            long_messages.extend(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Turn {turn_index}: "
                            + ("explain capacity profiling carefully " * 28)
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Answer {turn_index}: "
                            + ("complete turns remain intact " * 28)
                        ),
                    },
                ]
            )

        canonical = canonical_conversation(long_messages)
        chunks, dropped = split_complete_turns(
            tokenizer,
            canonical,
            768,
            engine.bos_id,
            engine.eos_id,
        )
        if len(chunks) < 2 or dropped:
            raise AssertionError(
                f"long multiturn split failed: chunks={len(chunks)} dropped={dropped}"
            )
        tool_chunks = [
            messages
            for messages, _ in chunks
            if any(message["tool_calls"] is not None for message in messages)
        ]
        if len(tool_chunks) != 1:
            raise AssertionError("tool turn was duplicated or separated")
        tool_chunk = tool_chunks[0]
        call_index = next(
            index
            for index, message in enumerate(tool_chunk)
            if message["tool_calls"] is not None
        )
        response_index = next(
            index
            for index, message in enumerate(tool_chunk)
            if message["role"] == "tool"
        )
        if response_index <= call_index:
            raise AssertionError("tool response does not follow its tool call")

        for messages, metrics in chunks:
            if metrics["rendered_tokens"] > 768:
                raise AssertionError("self-test emitted an overlength chunk")
            if metrics["assistant_messages"] != metrics["assistant_markers"]:
                raise AssertionError("self-test assistant marker mismatch")
            if metrics["assistant_messages"] != metrics["closed_assistant_spans"]:
                raise AssertionError("self-test assistant closure mismatch")
            if metrics["valid_targets"] <= 0:
                raise AssertionError("self-test has no shifted valid targets")
            if any(tuple(message) != MESSAGE_KEYS for message in messages):
                raise AssertionError("self-test message schema mismatch")

        engine.add_conversation(profile, long_messages, "first")
        engine.add_conversation(profile, long_messages, "duplicate")
        if profile.counts["exact_duplicate_chunks"] != len(chunks):
            raise AssertionError("SQLite exact duplicate accounting failed")

        orphan_tool = [
            {"role": "user", "content": "bad"},
            {"role": "tool", "content": "orphan"},
            {"role": "assistant", "content": "done"},
        ]
        try:
            split_complete_turns(
                tokenizer,
                canonical_conversation(orphan_tool),
                768,
                engine.bos_id,
                engine.eos_id,
            )
        except ProfileError:
            pass
        else:
            raise AssertionError("orphan tool response was accepted")

        too_long = canonical_conversation(
            [
                {"role": "user", "content": "very long " * 1800},
                {"role": "assistant", "content": "answer " * 1800},
            ]
        )
        overlength_chunks, overlength_dropped = split_complete_turns(
            tokenizer,
            too_long,
            768,
            engine.bos_id,
            engine.eos_id,
        )
        if overlength_chunks or overlength_dropped != 1:
            raise AssertionError("single overlength turn was not dropped intact")

        if fixture_output is not None:
            with fixture_output.open("w", encoding="utf-8") as handle:
                for messages, _ in chunks:
                    handle.write(
                        json.dumps(
                            training_row(messages),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        index.close()

    result = {
        "self_test": "passed",
        "chunks": len(chunks),
        "tool_chunks": len(tool_chunks),
        "dropped_overlength_turns": overlength_dropped,
        "exact_duplicates_detected": profile.counts[
            "exact_duplicate_chunks"
        ],
        "canonical_message_keys": list(MESSAGE_KEYS),
        "max_rendered_tokens": max(
            metrics["rendered_tokens"] for _, metrics in chunks
        ),
        "network_used": False,
        "fixture_output": str(fixture_output) if fixture_output else None,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        tokenizer_path = args.tokenizer or (REPO_ROOT / "minimind/model")
        return run_self_test(tokenizer_path, args.fixture_output)
    if args.profile_config is None:
        raise ProfileError("--profile-config is required")
    profile_config_path = args.profile_config.resolve()
    profile_config = load_profile_config(profile_config_path)
    if args.validate_only:
        result, _, _ = validate_profile_inputs(
            profile_config_path, profile_config
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["validation"] == "passed" else 1
    if args.output is None:
        raise ProfileError("--output is required for a full profile")
    return run_profile(profile_config_path, args.output.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileError as exc:
        print(f"error={exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
