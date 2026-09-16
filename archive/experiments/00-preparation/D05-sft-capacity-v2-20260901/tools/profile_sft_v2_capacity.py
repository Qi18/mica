#!/usr/bin/env python3
"""Fail-closed SFT v2 capacity profiler.

This module deliberately separates three capacity layers:

1. gross candidates after schema/rendering gates;
2. origin-exclusive candidates after one stable variant claim per donor record;
3. exact-unique candidates after deterministic global exact deduplication.

The implementation imports only local v1 helpers.  It never downloads data and
does not write training shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from transformers import AutoTokenizer

import profile_sft_v1_capacity as v1
from generate_strict_format import GENERATORS as STRICT_GENERATORS
from normalize_glaive_tool import parse_chat as parse_glaive_chat
from normalize_sft_proxy import (
    adapt_alpaca_gpt4_zh,
    adapt_code_alpaca,
    adapt_gsm8k,
    adapt_mbpp,
)
from tokenizer_stats import create_chat_prompt, sha256_file, tokenizer_fingerprint


PROTOCOL_VERSION = "2026-09-01-sft-capacity-v2"
MESSAGE_KEYS = v1.MESSAGE_KEYS
REPO_ROOT = Path(__file__).resolve().parents[4]
ProfileError = v1.ProfileError

NUMINA_SOURCE_ALIASES = {"numinamath_cot", "numina_math_cot"}
TULU_SOURCE_ALIASES = {
    "tulu_3_sft_personas_instruction_following",
    "tulu_3_personas_instruction_following",
}
EXPLICIT_V2_SOURCE_IDS = {
    "tigerbot_alpaca_zh_0_5m",
    "numinamath_cot",
    "magicoder_evol_instruct_110k",
    "tulu_3_sft_personas_instruction_following",
    "cnn_dailymail_summary",
}


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


def stable_text(value: Any) -> str:
    return str(value or "").strip()


def normalized_compare(value: Any) -> str:
    return re.sub(r"\s+", " ", stable_text(value))


def stable_hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    partition = dict(value.get("ultrachat_partition") or {})
    if partition.get("modulus") != 5:
        raise ProfileError("UltraChat partition modulus must be five")
    if sorted(partition.get("english_general_residues") or []) != [0, 1, 2]:
        raise ProfileError("UltraChat english residues must be 0,1,2")
    if sorted(partition.get("multiturn_residues") or []) != [3, 4]:
        raise ProfileError("UltraChat multiturn residues must be 3,4")
    return value


def canonical_source_id(source_id: str) -> str:
    if source_id in NUMINA_SOURCE_ALIASES:
        return "numinamath_cot"
    if source_id in TULU_SOURCE_ALIASES:
        return "tulu_3_sft_personas_instruction_following"
    return source_id


def source_specs(profile_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs, evidence = v1.build_source_specs(profile_config)
    expected = set(profile_config.get("required_v2_source_ids") or EXPLICIT_V2_SOURCE_IDS)
    actual = {canonical_source_id(str(item["id"])) for item in specs}
    missing = sorted(expected - actual)
    if missing:
        raise ProfileError("v2 source roster is incomplete: " + ",".join(missing))
    return specs, evidence


def validate_profile_inputs(
    config_path: Path,
    profile_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specs, source_config = source_specs(profile_config)
    evidence = [v1.validate_done_evidence(source) for source in specs]
    tokenizer_path = resolve_repo_path(profile_config["tokenizer"])
    tokenizer_files = tokenizer_fingerprint(tokenizer_path)
    issues = [item for item in evidence if item["status"] not in {"ready", "missing"}]
    missing = [item for item in evidence if item["status"] == "missing"]
    source_ids = {canonical_source_id(str(source["id"])) for source in specs}
    alias_issues = []
    if "numinamath_cot" not in source_ids:
        alias_issues.append("numinamath_cot_missing")
    if "tulu_3_sft_personas_instruction_following" not in source_ids:
        alias_issues.append("tulu_3_sft_personas_instruction_following_missing")
    result = {
        "validation": "passed" if not issues and not alias_issues else "failed",
        "ready_for_full_profile": not issues and not missing and not alias_issues,
        "profile_config": str(config_path),
        "profile_config_sha256": sha256_file(config_path),
        "source_config": source_config,
        "tokenizer": str(tokenizer_path),
        "tokenizer_files": tokenizer_files,
        "source_evidence": evidence,
        "missing_sources": [item["source_id"] for item in missing],
        "issues": issues + [{"status": item} for item in alias_issues],
        "network_used": False,
    }
    return result, specs, evidence


class LayeredCandidateStore:
    """SQLite-backed, scan-order-independent origin claims and exact dedup."""

    def __init__(
        self,
        path: Path,
        source_priority: list[str],
        bucket_priority: list[str],
        variant_priority: list[str],
    ):
        self.path = path
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.source_rank = {value: index for index, value in enumerate(source_priority)}
        self.bucket_rank = {value: index for index, value in enumerate(bucket_priority)}
        self.variant_rank = {value: index for index, value in enumerate(variant_priority)}
        self.connection.executescript(
            """
            CREATE TABLE candidates (
              candidate_id TEXT PRIMARY KEY,
              exact_digest BLOB NOT NULL,
              origin_group_id TEXT NOT NULL,
              variant TEXT NOT NULL,
              claim_key TEXT NOT NULL,
              priority_key TEXT NOT NULL,
              source_id TEXT NOT NULL,
              bucket TEXT NOT NULL,
              assistant_target_tokens INTEGER NOT NULL,
              rendered_tokens INTEGER NOT NULL,
              assistant_messages INTEGER NOT NULL
            );
            CREATE INDEX candidates_origin ON candidates(origin_group_id, variant);
            CREATE INDEX candidates_digest ON candidates(exact_digest);
            CREATE TABLE origin_claims (
              origin_group_id TEXT PRIMARY KEY,
              variant TEXT NOT NULL,
              claim_key TEXT NOT NULL
            );
            CREATE TABLE exact_winners (
              exact_digest BLOB PRIMARY KEY,
              candidate_id TEXT NOT NULL UNIQUE,
              source_id TEXT NOT NULL,
              bucket TEXT NOT NULL,
              assistant_target_tokens INTEGER NOT NULL,
              rendered_tokens INTEGER NOT NULL,
              assistant_messages INTEGER NOT NULL
            );
            """
        )

    def _rank(self, mapping: dict[str, int], value: str) -> int:
        return mapping.get(value, 999_999)

    def add(
        self,
        *,
        source_id: str,
        bucket: str,
        origin_group_id: str,
        variant: str,
        chunk_index: int,
        exact_digest: bytes,
        metrics: dict[str, int],
    ) -> None:
        if not origin_group_id or not variant:
            raise ProfileError("origin_group_id and variant are required")
        variant_family = variant.split(":", 1)[0]
        claim_key = (
            f"{self._rank(self.variant_rank, variant_family):06d}:"
            f"{self._rank(self.bucket_rank, bucket):06d}:"
            f"{self._rank(self.source_rank, source_id):06d}:"
            f"{variant}:{source_id}"
        )
        digest_hex = exact_digest.hex()
        priority_key = (
            f"{claim_key}:{origin_group_id}:{chunk_index:08d}:{digest_hex}"
        )
        candidate_id = stable_hash_text(
            "\x1f".join(
                [source_id, bucket, origin_group_id, variant, str(chunk_index), digest_hex]
            )
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                sqlite3.Binary(exact_digest),
                origin_group_id,
                variant,
                claim_key,
                priority_key,
                source_id,
                bucket,
                int(metrics["valid_targets"]),
                int(metrics["rendered_tokens"]),
                int(metrics["assistant_messages"]),
            ),
        )
        self.connection.execute(
            """
            INSERT INTO origin_claims(origin_group_id, variant, claim_key)
            VALUES (?, ?, ?)
            ON CONFLICT(origin_group_id) DO UPDATE SET
              variant=excluded.variant,
              claim_key=excluded.claim_key
            WHERE excluded.claim_key < origin_claims.claim_key
            """,
            (origin_group_id, variant, claim_key),
        )

    def commit(self) -> None:
        self.connection.commit()

    def finalize(self) -> None:
        self.connection.execute("DELETE FROM exact_winners")
        cursor = self.connection.execute(
            """
            SELECT c.exact_digest, c.candidate_id, c.source_id, c.bucket,
                   c.assistant_target_tokens, c.rendered_tokens, c.assistant_messages
            FROM candidates c
            JOIN origin_claims o
              ON o.origin_group_id = c.origin_group_id
             AND o.variant = c.variant
            ORDER BY c.priority_key, c.candidate_id
            """
        )
        for row in cursor:
            self.connection.execute(
                "INSERT OR IGNORE INTO exact_winners VALUES (?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        self.connection.commit()

    def _aggregate(self, relation: str, join: str = "") -> dict[str, Any]:
        rows = self.connection.execute(
            f"""
            SELECT c.source_id, c.bucket, COUNT(*),
                   COALESCE(SUM(c.assistant_target_tokens), 0),
                   COALESCE(SUM(c.rendered_tokens), 0),
                   COALESCE(SUM(c.assistant_messages), 0)
            FROM {relation} c {join}
            GROUP BY c.source_id, c.bucket
            ORDER BY c.source_id, c.bucket
            """
        ).fetchall()
        per_source: dict[str, dict[str, dict[str, int]]] = {}
        per_bucket: dict[str, Counter[str]] = {}
        total = Counter()
        for source_id, bucket, chunks, targets, rendered, assistants in rows:
            metrics = {
                "chunks": int(chunks),
                "assistant_target_tokens": int(targets),
                "rendered_tokens": int(rendered),
                "assistant_messages": int(assistants),
            }
            per_source.setdefault(str(source_id), {})[str(bucket)] = metrics
            bucket_counter = per_bucket.setdefault(str(bucket), Counter())
            bucket_counter.update(metrics)
            total.update(metrics)
        return {
            "global": dict(total),
            "by_source": per_source,
            "by_bucket": {key: dict(value) for key, value in per_bucket.items()},
        }

    def layers(self) -> dict[str, Any]:
        gross = self._aggregate("candidates")
        origin = self._aggregate(
            "candidates",
            "JOIN origin_claims o ON o.origin_group_id=c.origin_group_id "
            "AND o.variant=c.variant",
        )
        exact = self._aggregate("exact_winners")
        candidate_cross_bucket = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT origin_group_id FROM candidates
                  GROUP BY origin_group_id HAVING COUNT(DISTINCT bucket) > 1
                )
                """
            ).fetchone()[0]
        )
        selected_cross_bucket = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT c.origin_group_id FROM candidates c
                  JOIN origin_claims o ON o.origin_group_id=c.origin_group_id
                                      AND o.variant=c.variant
                  GROUP BY c.origin_group_id HAVING COUNT(DISTINCT c.bucket) > 1
                )
                """
            ).fetchone()[0]
        )
        multi_variant_origins = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT origin_group_id FROM candidates
                  GROUP BY origin_group_id HAVING COUNT(DISTINCT variant) > 1
                )
                """
            ).fetchone()[0]
        )
        exact_duplicates_removed = (
            int(origin["global"].get("chunks", 0))
            - int(exact["global"].get("chunks", 0))
        )
        return {
            "gross": gross,
            "origin_exclusive": origin,
            "exact_unique": exact,
            "origin_claims": {
                "candidate_cross_bucket_origin_groups": candidate_cross_bucket,
                "selected_cross_bucket_origin_groups": selected_cross_bucket,
                "multi_variant_origin_groups": multi_variant_origins,
                "selected_cross_bucket_gate": selected_cross_bucket == 0,
            },
            "exact_duplicate_chunks_removed_after_origin_claim": exact_duplicates_removed,
            "stable_priority_is_scan_order_independent": True,
        }

    def exact_winner_sources(self) -> list[str]:
        return [
            str(row[0])
            for row in self.connection.execute(
                "SELECT source_id FROM exact_winners ORDER BY exact_digest"
            )
        ]

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class SourceProfile(v1.SourceProfile):
    pass


class CapacityEngine:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        store: LayeredCandidateStore,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.store = store
        self.digest_index = store
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        if not self.bos_id or not self.eos_id:
            raise ProfileError("assistant marker token sequences are empty")

    def add_conversation(
        self,
        profile: SourceProfile,
        messages: Iterable[dict[str, Any]],
        reference: Any,
        *,
        origin_group_id: str | None = None,
        variant: str = "original",
        bucket: str | None = None,
        minimum_assistant_messages: int = 1,
    ) -> dict[str, int]:
        profile.counts["adapted_conversations"] += 1
        try:
            canonical = v1.canonical_conversation(messages)
            chunks, dropped = v1.split_complete_turns(
                self.tokenizer,
                canonical,
                self.max_length,
                self.bos_id,
                self.eos_id,
            )
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            profile.invalid("invalid_conversations", reference, exc)
            return {"accepted_chunks": 0, "dropped_overlength_turns": 0, "tokens": 0}

        stable_reference = json.dumps(
            reference, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        origin = origin_group_id or (
            f"{profile.source_id}:" + stable_hash_text(stable_reference)
        )
        selected_bucket = bucket or profile.bucket
        accepted = 0
        accepted_tokens = 0
        profile.counts["dropped_overlength_turns"] += dropped
        for chunk_index, (chunk, metrics) in enumerate(chunks):
            if metrics["assistant_messages"] < minimum_assistant_messages:
                profile.counts["rejected_minimum_assistant_messages"] += 1
                continue
            digest = v1.exact_digest(chunk)
            self.store.add(
                source_id=profile.source_id,
                bucket=selected_bucket,
                origin_group_id=origin,
                variant=variant,
                chunk_index=chunk_index,
                exact_digest=digest,
                metrics=metrics,
            )
            accepted += 1
            accepted_tokens += metrics["valid_targets"]
            profile.counts["gross_chunks"] += 1
            profile.counts["gross_assistant_target_tokens"] += metrics["valid_targets"]
            profile.counts[f"gross_bucket:{selected_bucket}:chunks"] += 1
            profile.counts[
                f"gross_bucket:{selected_bucket}:assistant_target_tokens"
            ] += metrics["valid_targets"]
            profile.max_rendered_tokens = max(
                profile.max_rendered_tokens, metrics["rendered_tokens"]
            )
        if dropped:
            profile.counts["conversations_with_overlength_turns"] += 1
        if not accepted:
            profile.counts["conversations_without_capacity_chunks"] += 1
        return {
            "accepted_chunks": accepted,
            "dropped_overlength_turns": dropped,
            "tokens": accepted_tokens,
        }


def adapt_tigerbot(row: dict[str, Any]) -> list[dict[str, Any]]:
    instruction = stable_text(row["instruction"])
    input_text = stable_text(row.get("input"))
    output = stable_text(row["output"])
    if not instruction or not output:
        raise ProfileError("TigerBot instruction/output is empty")
    prompt = instruction
    if input_text:
        prompt += "\n\nInput:\n" + input_text
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output},
    ]


def adapt_magicoder(row: dict[str, Any]) -> list[dict[str, Any]]:
    instruction = stable_text(row["instruction"])
    response = stable_text(row["response"])
    if not instruction or not response:
        raise ProfileError("Magicoder instruction/response is empty")
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]


def parsed_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProfileError("messages is not valid JSON") from exc
    if not isinstance(value, list) or not value:
        raise ProfileError("messages must be a non-empty list")
    if not all(isinstance(item, dict) for item in value):
        raise ProfileError("messages contains a non-object")
    return [dict(item) for item in value]


def _balanced_boxed_span(solution: str) -> tuple[int, int] | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", solution)]
    for start in reversed(starts):
        open_brace = solution.find("{", start)
        depth = 0
        escaped = False
        for index in range(open_brace, len(solution)):
            char = solution[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    trailer = solution[index + 1 :]
                    if re.fullmatch(r"[\s.!。]*", trailer):
                        return start, len(solution)
                    break
            if depth < 0:
                break
    return None


def parse_numina_reasoning_final(solution: str) -> dict[str, str]:
    solution = stable_text(solution)
    if not solution:
        raise ProfileError("Numina solution is empty")
    span = _balanced_boxed_span(solution)
    policy = "balanced_boxed_terminal"
    if span is None:
        terminal = re.search(
            r"(?im)^(?:final\s+answer|answer)\s*[:：]\s*[^\r\n]+[.!。]?\s*$",
            solution,
        )
        if terminal is None or terminal.end() != len(solution):
            raise ProfileError("no high-confidence terminal final answer")
        span = (terminal.start(), terminal.end())
        policy = "explicit_terminal_marker"
    start, end = span
    reasoning_raw = solution[:start]
    final_raw = solution[start:end]
    reasoning = reasoning_raw.rstrip()
    separator = reasoning_raw[len(reasoning) :]
    final = final_raw.strip()
    if not reasoning or not final:
        raise ProfileError("Numina reasoning or final answer is empty")
    if reasoning + separator + final_raw != solution:
        raise ProfileError("Numina reasoning/final roundtrip failed")
    return {
        "reasoning_content": reasoning,
        "content": final,
        "separator": separator,
        "final_raw": final_raw,
        "solution": solution,
        "parse_policy": policy,
    }


def adapt_numina(
    row: dict[str, Any],
    excluded_source_tags: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    source_tag = stable_text(row.get("source")).casefold()
    if source_tag in excluded_source_tags:
        raise ProfileError(f"excluded_numina_source:{source_tag}")
    problem = stable_text(row["problem"])
    solution = stable_text(row["solution"])
    if not problem or not solution:
        raise ProfileError("Numina problem/solution is empty")
    messages = v1.canonical_conversation(parsed_messages(row["messages"]))
    body = [item for item in messages if item["role"] != "system"]
    if len(body) != 2 or body[0]["role"] != "user" or body[1]["role"] != "assistant":
        raise ProfileError("Numina messages must contain one user/assistant pair")
    if normalized_compare(body[0]["content"]) != normalized_compare(problem):
        raise ProfileError("Numina messages user does not match problem")
    message_solution = body[1]["content"]
    if body[1]["reasoning_content"]:
        message_solution = (
            body[1]["reasoning_content"].rstrip()
            + "\n"
            + body[1]["content"].lstrip()
        )
    if normalized_compare(message_solution) != normalized_compare(solution):
        raise ProfileError("Numina messages assistant does not match solution")
    parsed = parse_numina_reasoning_final(solution)
    output = []
    if messages and messages[0]["role"] == "system":
        output.append(messages[0])
    output.extend(
        [
            {"role": "user", "content": problem},
            {
                "role": "assistant",
                "content": parsed["content"],
                "reasoning_content": parsed["reasoning_content"],
            },
        ]
    )
    return output, parsed


def adapt_cnn_dailymail_summary(row: dict[str, Any]) -> list[dict[str, Any]]:
    article = stable_text(row["article"])
    highlights = stable_text(row["highlights"])
    record_id = stable_text(row["id"])
    if not article or not highlights or not record_id:
        raise ProfileError("CNN/DailyMail article/highlights/id is empty")
    return [
        {
            "role": "user",
            "content": (
                "Summarize the following news article faithfully and "
                "concisely.\n\nArticle:\n" + article
            ),
        },
        {"role": "assistant", "content": highlights},
    ]


def tulu_constraint_report(row: dict[str, Any]) -> dict[str, Any]:
    constraints = row.get("constraints")
    if isinstance(constraints, str):
        try:
            constraints = json.loads(constraints)
        except json.JSONDecodeError:
            constraints = [constraints] if constraints.strip() else []
    if not isinstance(constraints, list):
        raise ProfileError("Tulu constraints must be a list")
    messages = v1.canonical_conversation(parsed_messages(row["messages"]))
    prompt = stable_text(row["prompt"])
    user_messages = [item for item in messages if item["role"] == "user"]
    if not prompt or not user_messages:
        raise ProfileError("Tulu prompt/user message is empty")
    if normalized_compare(prompt) != normalized_compare(user_messages[0]["content"]):
        raise ProfileError("Tulu prompt does not match the first user message")
    # No natural-language constraint is treated as mechanically verified.  This
    # is intentional: declared and parser-supported are not proof of compliance.
    return {
        "declared": len(constraints),
        "supported": 0,
        "verified": 0,
        "fully_verified": False,
    }


def ultrachat_bucket(prompt_id: str, partition: dict[str, Any]) -> tuple[str, int]:
    if not prompt_id:
        raise ProfileError("UltraChat prompt_id is empty")
    residue = int.from_bytes(
        hashlib.sha256(prompt_id.encode("utf-8")).digest(), "big"
    ) % int(partition["modulus"])
    if residue in set(partition["english_general_residues"]):
        return "english_general", residue
    if residue in set(partition["multiturn_residues"]):
        return "multiturn", residue
    raise ProfileError("UltraChat residue is not assigned")


def profile_ultrachat(
    source: dict[str, Any],
    profile: SourceProfile,
    engine: CapacityEngine,
    partition: dict[str, Any],
) -> None:
    for line_number, row in v1.iter_verified_jsonl(source, profile):
        try:
            prompt_id = stable_text(row["prompt_id"])
            bucket, residue = ultrachat_bucket(prompt_id, partition)
            messages = parsed_messages(row["messages"])
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            profile.invalid("invalid_adapter_rows", line_number, exc)
            continue
        profile.counts[f"partition_residue:{residue}"] += 1
        profile.counts[f"partition_bucket:{bucket}:records"] += 1
        result = engine.add_conversation(
            profile,
            messages,
            {"prompt_id": prompt_id},
            origin_group_id=f"ultrachat_200k:{prompt_id}",
            variant="original",
            bucket=bucket,
            minimum_assistant_messages=2 if bucket == "multiturn" else 1,
        )
        if bucket == "multiturn":
            profile.counts["multiturn_chunks_eligible"] += result["accepted_chunks"]
    engine.store.commit()


def profile_explicit_source(
    source: dict[str, Any],
    profile: SourceProfile,
    engine: CapacityEngine,
    profile_config: dict[str, Any],
) -> None:
    source_id = canonical_source_id(str(source["id"]))
    excluded = {
        stable_text(item).casefold()
        for item in profile_config.get("numina", {}).get(
            "excluded_source_tags", ["gsm8k", "math"]
        )
    }
    for line_number, row in v1.iter_verified_jsonl(source, profile):
        origin = f"{source_id}:{line_number}"
        try:
            if source_id == "tigerbot_alpaca_zh_0_5m":
                messages = adapt_tigerbot(row)
                variant = "original"
            elif source_id == "numinamath_cot":
                messages, parsed = adapt_numina(row, excluded)
                profile.counts[f"numina_parse:{parsed['parse_policy']}"] += 1
                variant = "reasoning_final"
            elif source_id == "magicoder_evol_instruct_110k":
                messages = adapt_magicoder(row)
                variant = "original"
            elif source_id == "cnn_dailymail_summary":
                messages = adapt_cnn_dailymail_summary(row)
                variant = "original"
                profile.counts["summary_schema_eligible_conversations"] += 1
            else:
                raise ProfileError(f"unknown explicit adapter: {source_id}")
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            kind = "invalid_adapter_rows"
            if str(exc).startswith("excluded_numina_source:"):
                kind = "excluded_numina_source_rows"
            profile.invalid(kind, line_number, exc)
            continue
        if source_id == "cnn_dailymail_summary":
            origin = f"{source_id}:{stable_text(row['id'])}"
        result = engine.add_conversation(
            profile,
            messages,
            line_number,
            origin_group_id=origin,
            variant=variant,
        )
        if source_id == "cnn_dailymail_summary":
            profile.counts["summary_eligible_chunks"] += result["accepted_chunks"]
            profile.counts["summary_eligible_assistant_target_tokens"] += result["tokens"]
            profile.counts["summary_overlength_turns"] += result[
                "dropped_overlength_turns"
            ]
    engine.store.commit()


def profile_tulu(
    source: dict[str, Any],
    profile: SourceProfile,
) -> None:
    for line_number, row in v1.iter_verified_jsonl(source, profile):
        profile.counts["tulu_rows_declared"] += 1
        try:
            claim = tulu_constraint_report(row)
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            profile.invalid("invalid_tulu_rows", line_number, exc)
            continue
        profile.counts["constraints_declared"] += int(claim["declared"])
        profile.counts["constraints_supported"] += int(claim["supported"])
        profile.counts["constraints_verified"] += int(claim["verified"])
        if claim["fully_verified"]:
            profile.counts["rows_fully_verified"] += 1
        else:
            profile.counts["rows_fail_closed"] += 1
    profile.extra["constraint_verification"] = {
        "declared": profile.counts["constraints_declared"],
        "supported": profile.counts["constraints_supported"],
        "verified": profile.counts["constraints_verified"],
        "rows_fully_verified": profile.counts["rows_fully_verified"],
        "rows_fail_closed": profile.counts["rows_fail_closed"],
        "capacity_counted_assistant_target_tokens": 0,
        "status": "fail_closed_no_complete_constraint_verifier",
        "claim_semantics": (
            "declared is source metadata; supported is parser support; verified "
            "requires a mechanical postcondition. No natural-language claim is "
            "promoted to verified."
        ),
    }


def profile_existing_source(
    source: dict[str, Any],
    profile: SourceProfile,
    engine: CapacityEngine,
) -> None:
    source_id = str(source["id"])
    adapters = {
        "gsm8k": adapt_gsm8k,
        "mbpp": adapt_mbpp,
        "code_alpaca_20k": adapt_code_alpaca,
        "alpaca_gpt4_zh": adapt_alpaca_gpt4_zh,
    }
    for line_number, row in v1.iter_verified_jsonl(source, profile):
        try:
            if source_id == "glaive_function_calling_v2":
                messages, parse_errors = parse_glaive_chat(
                    str(row["chat"]), str(row["system"])
                )
                if parse_errors:
                    raise ProfileError(f"Glaive parse errors: {parse_errors}")
            elif source_id == "databricks_dolly_15k":
                category = stable_text(row.get("category"))
                profile.counts[f"dolly_category:{category}"] += 1
                if category != "summarization":
                    profile.counts["filtered_non_summarization_rows"] += 1
                    continue
                messages = v1.adapt_dolly(row)
            else:
                messages = adapters[source_id](row)
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            profile.invalid("invalid_adapter_rows", line_number, exc)
            continue
        engine.add_conversation(
            profile,
            messages,
            line_number,
            origin_group_id=f"{source_id}:{line_number}",
            variant="original",
        )
    engine.store.commit()


def profile_translation(
    source: dict[str, Any],
    profile: SourceProfile,
    engine: CapacityEngine,
) -> None:
    for line_number, row in v1.iter_verified_jsonl(source, profile):
        for field, left, right in (
            ("instruction", "instruction", "instruction_zh"),
            ("output", "output", "output_zh"),
        ):
            left_value = stable_text(row.get(left))
            right_value = stable_text(row.get(right))
            if not left_value or not right_value:
                profile.counts["missing_aligned_pairs"] += 1
                continue
            engine.add_conversation(
                profile,
                [
                    {
                        "role": "user",
                        "content": f"Translate the following text to Chinese:\n{left_value}",
                    },
                    {"role": "assistant", "content": right_value},
                ],
                {"line_number": line_number, "field": field},
                origin_group_id=f"alpaca_gpt4_zh:{line_number}",
                variant=f"translation:{field}",
                bucket="summarization_translation",
            )
    engine.store.commit()


def strict_projection_only(
    tokenizer: Any,
    engine: CapacityEngine,
    rows: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    names = list(STRICT_GENERATORS)
    exact_seen: set[bytes] = set()
    tokens = 0
    chunks = 0
    dropped = 0
    for index in range(rows):
        category = names[index % len(names)]
        prompt, answer = STRICT_GENERATORS[category](index, rng)
        canonical = v1.canonical_conversation(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        )
        candidate_chunks, dropped_turns = v1.split_complete_turns(
            tokenizer,
            canonical,
            engine.max_length,
            engine.bos_id,
            engine.eos_id,
        )
        dropped += dropped_turns
        for messages, metrics in candidate_chunks:
            digest = v1.exact_digest(messages)
            if digest in exact_seen:
                continue
            exact_seen.add(digest)
            chunks += 1
            tokens += metrics["valid_targets"]
    return {
        "projection_only": True,
        "current_generator_rows": rows,
        "measured_unique_chunks": chunks,
        "measured_assistant_target_tokens": tokens,
        "dropped_overlength_turns": dropped,
        "actual_capacity_chunks": 0,
        "actual_capacity_assistant_target_tokens": 0,
        "reason": (
            "The current 500-row generator sample is an estimator. It is not "
            "materialized, origin-claimed, audited training capacity."
        ),
    }


def bucket_report(
    weights: dict[str, Any],
    total_target: int,
    layers: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    exact_buckets = layers["exact_unique"]["by_bucket"]
    origin_buckets = layers["origin_exclusive"]["by_bucket"]
    gross_buckets = layers["gross"]["by_bucket"]
    for bucket, raw_weight in weights.items():
        target = int(round(total_target * float(raw_weight)))
        exact_tokens = int(
            exact_buckets.get(bucket, {}).get("assistant_target_tokens", 0)
        )
        result[bucket] = {
            "weight": float(raw_weight),
            "target_assistant_tokens": target,
            "gross_assistant_target_tokens": int(
                gross_buckets.get(bucket, {}).get("assistant_target_tokens", 0)
            ),
            "origin_exclusive_assistant_target_tokens": int(
                origin_buckets.get(bucket, {}).get("assistant_target_tokens", 0)
            ),
            "exact_unique_assistant_target_tokens": exact_tokens,
            "capacity_ratio": round(exact_tokens / target, 6) if target else None,
            "shortfall_assistant_tokens": max(0, target - exact_tokens),
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
        raise ProfileError("missing raw sources: " + ",".join(validation["missing_sources"]))

    tokenizer_path = resolve_repo_path(profile_config["tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True
    )
    sqlite_dir = Path(profile_config["sqlite_work_dir"])
    sqlite_dir.mkdir(parents=True, exist_ok=True)
    descriptor, sqlite_name = tempfile.mkstemp(
        prefix="sft-capacity-v2-", suffix=".sqlite", dir=sqlite_dir
    )
    os.close(descriptor)
    sqlite_path = Path(sqlite_name)
    sqlite_path.unlink()
    priority = dict(profile_config["dedup_priority"])
    store = LayeredCandidateStore(
        sqlite_path,
        source_priority=[str(item) for item in priority["sources"]],
        bucket_priority=[str(item) for item in priority["buckets"]],
        variant_priority=[str(item) for item in priority["variants"]],
    )
    engine = CapacityEngine(tokenizer, int(profile_config["sequence_length"]), store)
    profiles: dict[str, SourceProfile] = {}
    source_by_id = {canonical_source_id(str(item["id"])): item for item in specs}

    try:
        for source in specs:
            source_id = canonical_source_id(str(source["id"]))
            profile = SourceProfile(
                source_id=source_id,
                purpose=str(source["purpose"]),
                bucket=str(source["bucket"]),
            )
            profiles[source_id] = profile
            if source_id == "ultrachat_200k":
                profile_ultrachat(
                    source,
                    profile,
                    engine,
                    dict(profile_config["ultrachat_partition"]),
                )
            elif source_id == "oasst1":
                v1.profile_oasst(
                    source,
                    profile,
                    engine,
                    set(profile_config.get("oasst_languages", ["en", "zh"])),
                )
            elif source_id in {
                "tigerbot_alpaca_zh_0_5m",
                "numinamath_cot",
                "magicoder_evol_instruct_110k",
                "cnn_dailymail_summary",
            }:
                profile_explicit_source(source, profile, engine, profile_config)
            elif source_id == "tulu_3_sft_personas_instruction_following":
                profile_tulu(source, profile)
            else:
                profile_existing_source(source, profile, engine)

        translation_config = dict(profile_config["derived"]["translation"])
        translation_profile = SourceProfile(
            source_id=str(translation_config["id"]),
            purpose="translation",
            bucket="summarization_translation",
        )
        profiles[translation_profile.source_id] = translation_profile
        profile_translation(
            source_by_id[canonical_source_id(str(translation_config["input_source"]))],
            translation_profile,
            engine,
        )

        strict_config = dict(profile_config["derived"]["strict_format"])
        strict_projection = strict_projection_only(
            tokenizer,
            engine,
            int(strict_config["current_generator_rows"]),
            int(strict_config["seed"]),
        )
        strict_projection.update(
            {
                "formal_policy_id": str(strict_config["id"]),
                "donor_sources": list(strict_config["donor_sources"]),
                "origin_exclusive": bool(strict_config["origin_exclusive"]),
                "no_record_reuse": bool(strict_config["no_record_reuse"]),
                "verified_transforms_only": bool(
                    strict_config["verified_transforms_only"]
                ),
                "baseline_projection_id": str(
                    strict_config["baseline_projection_id"]
                ),
            }
        )
        store.finalize()
        layers = store.layers()
        if not layers["origin_claims"]["selected_cross_bucket_gate"]:
            raise ProfileError("origin-exclusive cross-bucket groups are non-zero")

        acceptance = validation["source_config"]["acceptance"]
        total_target = int(acceptance["total_eligible_assistant_tokens"])
        buckets = bucket_report(
            acceptance["bucket_weights"], total_target, layers
        )
        blockers = [
            {
                "id": f"capacity_shortfall:{bucket}",
                "severity": "fatal",
                "shortfall_assistant_tokens": item["shortfall_assistant_tokens"],
            }
            for bucket, item in buckets.items()
            if item["shortfall_assistant_tokens"]
        ]
        blockers.append(
            {
                "id": "strict_projection_is_not_actual_capacity",
                "severity": "fatal",
                "measured_projection_tokens": strict_projection[
                    "measured_assistant_target_tokens"
                ],
            }
        )
        tulu = profiles["tulu_3_sft_personas_instruction_following"].extra.get(
            "constraint_verification", {}
        )
        if tulu.get("rows_fail_closed", 0) or tulu.get("verified", 0) == 0:
            blockers.append(
                {
                    "id": "tulu_constraints_not_fully_verified",
                    "severity": "fatal",
                    "declared": tulu.get("declared", 0),
                    "supported": tulu.get("supported", 0),
                    "verified": tulu.get("verified", 0),
                }
            )
        blockers.append(
            {
                "id": "source_capacity_mix_not_frozen",
                "severity": "fatal",
                "reason": "v2 profiles capacity; a later audited builder freezes quotas.",
            }
        )
        report = {
            "schema_version": 2,
            "protocol_version": PROTOCOL_VERSION,
            "created_at": utc_now(),
            "status": "capacity_profiled_not_trainable",
            "trainable": False,
            "network_used": False,
            "profile_config": str(config_path),
            "profile_config_sha256": sha256_file(config_path),
            "profiler": str(Path(__file__).resolve()),
            "profiler_sha256": sha256_file(Path(__file__).resolve()),
            "v1_helper": str(Path(v1.__file__).resolve()),
            "v1_helper_sha256": sha256_file(Path(v1.__file__).resolve()),
            "source_config": validation["source_config"],
            "tokenizer": {
                "path": str(tokenizer_path),
                "class": type(tokenizer).__name__,
                "vocab_size": tokenizer.vocab_size,
                "files": tokenizer_fingerprint(tokenizer_path),
            },
            "sequence_length": int(profile_config["sequence_length"]),
            "target_assistant_tokens": int(profile_config["target_assistant_tokens"]),
            "capacity_layers": layers,
            "planned_mix_buckets": buckets,
            "sources": {
                source_id: profile.as_dict() for source_id, profile in profiles.items()
            },
            "ultrachat_partition": {
                **profile_config["ultrachat_partition"],
                "multiturn_minimum_assistant_messages_per_chunk": 2,
                "selected_cross_bucket_origin_groups": layers["origin_claims"][
                    "selected_cross_bucket_origin_groups"
                ],
            },
            "numina_reasoning_policy": {
                "excluded_source_tags_casefold_exact": sorted(
                    {
                        stable_text(item).casefold()
                        for item in profile_config["numina"]["excluded_source_tags"]
                    }
                ),
                "messages_problem_solution_roundtrip_required": True,
                "accepted_final_parsers": [
                    "balanced_boxed_terminal",
                    "explicit_terminal_marker",
                ],
                "unparsed_rows": "fail_closed",
            },
            "strict_format_projection": strict_projection,
            "tulu_constraint_verification": tulu,
            "final_build_blockers": blockers,
            "final_build_allowed": False,
        }
        v1.atomic_write_json(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": report["status"],
                    "exact_unique_assistant_target_tokens": layers["exact_unique"][
                        "global"
                    ].get("assistant_target_tokens", 0),
                    "blockers": len(blockers),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        store.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            candidate = Path(str(sqlite_path) + suffix)
            if candidate.exists():
                candidate.unlink()
    return 0


def minimind_shifted_target_sentinel(
    input_ids: list[int],
    bos_id: list[int],
    eos_id: list[int],
) -> int:
    labels = [-100] * len(input_ids)
    index = 0
    while index < len(input_ids):
        if input_ids[index : index + len(bos_id)] == bos_id:
            start = index + len(bos_id)
            end = start
            while end < len(input_ids) and input_ids[end : end + len(eos_id)] != eos_id:
                end += 1
            for label_index in range(start, min(end + len(eos_id), len(input_ids))):
                labels[label_index] = input_ids[label_index]
            index = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            index += 1
    return sum(value != -100 for value in labels[1:])


def _self_test_store(
    directory: Path,
    tokenizer: Any,
    order: list[tuple[str, str, str, str]],
) -> tuple[dict[str, Any], list[str]]:
    path = directory / ("store-" + stable_hash_text(repr(order))[:8] + ".sqlite")
    store = LayeredCandidateStore(
        path,
        source_priority=["source_a", "source_b"],
        bucket_priority=["english_general", "strict_format"],
        variant_priority=["original", "reasoning_final", "translation", "strict_derived"],
    )
    engine = CapacityEngine(tokenizer, 768, store)
    profiles = {
        "source_a": SourceProfile("source_a", "test", "english_general"),
        "source_b": SourceProfile("source_b", "test", "strict_format"),
    }
    for source_id, bucket, variant, origin_group_id in order:
        engine.add_conversation(
            profiles[source_id],
            [
                {"role": "user", "content": "same donor prompt"},
                {"role": "assistant", "content": "same exact answer"},
            ],
            source_id,
            origin_group_id=origin_group_id,
            variant=variant,
            bucket=bucket,
        )
    store.finalize()
    layers = store.layers()
    winners = store.exact_winner_sources()
    store.close()
    return layers, winners


def run_self_test(tokenizer_path: Path, fixture_output: Path | None) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True
    )
    with tempfile.TemporaryDirectory(prefix="sft-capacity-v2-self-test-") as raw_dir:
        directory = Path(raw_dir)
        forward = [
            ("source_b", "strict_format", "strict_derived:json", "donor:1"),
            ("source_a", "english_general", "original", "donor:1"),
        ]
        reverse = list(reversed(forward))
        layers_a, winners_a = _self_test_store(directory, tokenizer, forward)
        layers_b, winners_b = _self_test_store(directory, tokenizer, reverse)
        if winners_a != winners_b or layers_a["exact_unique"] != layers_b["exact_unique"]:
            raise AssertionError("stable dedup depends on scan order")
        if layers_a["origin_claims"]["selected_cross_bucket_origin_groups"] != 0:
            raise AssertionError("origin variant claim left a cross-bucket group")
        if layers_a["exact_unique"]["by_bucket"].get("strict_format"):
            raise AssertionError("strict derivative double-counted its original donor")

        exact_forward = [
            ("source_b", "strict_format", "original", "exact-origin:b"),
            ("source_a", "english_general", "original", "exact-origin:a"),
        ]
        exact_reverse = list(reversed(exact_forward))
        exact_layers_a, exact_winners_a = _self_test_store(
            directory, tokenizer, exact_forward
        )
        exact_layers_b, exact_winners_b = _self_test_store(
            directory, tokenizer, exact_reverse
        )
        if (
            exact_winners_a != exact_winners_b
            or exact_layers_a["exact_unique"] != exact_layers_b["exact_unique"]
        ):
            raise AssertionError("stable exact dedup depends on scan order")
        if exact_winners_a != ["source_a"]:
            raise AssertionError("stable exact dedup ignored configured priority")

        store = LayeredCandidateStore(
            directory / "main.sqlite",
            source_priority=["ultrachat_200k", "numinamath_cot", "cnn_dailymail_summary"],
            bucket_priority=["english_general", "multiturn", "math", "summarization_translation"],
            variant_priority=["original", "reasoning_final", "translation", "strict_derived"],
        )
        engine = CapacityEngine(tokenizer, 768, store)
        ultra_profile = SourceProfile("ultrachat_200k", "test", "english_general")
        partition = {
            "modulus": 5,
            "english_general_residues": [0, 1, 2],
            "multiturn_residues": [3, 4],
        }
        residue_prompts: dict[int, str] = {}
        probe = 0
        while len(residue_prompts) < 5:
            value = f"prompt-{probe}"
            bucket, residue = ultrachat_bucket(value, partition)
            residue_prompts.setdefault(residue, value)
            expected = "english_general" if residue < 3 else "multiturn"
            if bucket != expected:
                raise AssertionError("UltraChat 3:2 residue assignment drift")
            probe += 1

        one_turn = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "one answer"},
        ]
        rejected = engine.add_conversation(
            ultra_profile,
            one_turn,
            "one",
            origin_group_id="ultrachat:one",
            bucket="multiturn",
            minimum_assistant_messages=2,
        )
        if rejected["accepted_chunks"] != 0:
            raise AssertionError("single-assistant chunk counted as multiturn")
        two_turn = one_turn + [
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "two answer"},
        ]
        accepted = engine.add_conversation(
            ultra_profile,
            two_turn,
            "two",
            origin_group_id="ultrachat:two",
            bucket="multiturn",
            minimum_assistant_messages=2,
        )
        if accepted["accepted_chunks"] != 1:
            raise AssertionError("valid multiturn chunk was not accepted")

        numina_row = {
            "source": "olympiads",
            "problem": "Compute 6 times 7.",
            "solution": "We multiply six by seven to obtain forty-two.\n\\boxed{42}",
            "messages": [
                {"role": "user", "content": "Compute 6 times 7."},
                {
                    "role": "assistant",
                    "content": "We multiply six by seven to obtain forty-two.\n\\boxed{42}",
                },
            ],
        }
        numina_messages, parsed = adapt_numina(numina_row, {"gsm8k", "math"})
        if (
            parsed["reasoning_content"]
            + parsed["separator"]
            + parsed["final_raw"]
            != parsed["solution"]
        ):
            raise AssertionError("Numina reasoning/final roundtrip drift")
        canonical = v1.canonical_conversation(numina_messages)
        metrics = v1.render_metrics(tokenizer, canonical, engine.bos_id, engine.eos_id)
        prompt = create_chat_prompt(tokenizer, canonical)
        input_ids = tokenizer(prompt).input_ids
        sentinel = minimind_shifted_target_sentinel(
            input_ids, engine.bos_id, engine.eos_id
        )
        if sentinel != metrics["valid_targets"] or sentinel <= 0:
            raise AssertionError("MiniMind shifted target sentinel mismatch")
        if parsed["reasoning_content"] not in prompt or parsed["content"] not in prompt:
            raise AssertionError("Numina reasoning/final did not survive chat rendering")
        for excluded in ("gsm8k", "MATH"):
            rejected_row = dict(numina_row, source=excluded)
            try:
                adapt_numina(rejected_row, {"gsm8k", "math"})
            except ProfileError as exc:
                if not str(exc).startswith("excluded_numina_source:"):
                    raise
            else:
                raise AssertionError("excluded Numina source was accepted")
        try:
            parse_numina_reasoning_final("Reasoning only; \\boxed{42")
        except ProfileError:
            pass
        else:
            raise AssertionError("unbalanced Numina final was accepted")
        terminal = parse_numina_reasoning_final(
            "A short derivation.\nFinal Answer: 42"
        )
        if terminal["parse_policy"] != "explicit_terminal_marker":
            raise AssertionError("terminal Numina parser was not exercised")

        summary_profile = SourceProfile(
            "cnn_dailymail_summary",
            "summarization_translation",
            "summarization_translation",
        )
        summary = adapt_cnn_dailymail_summary(
            {
                "article": "A compact article about a local event.",
                "highlights": "A local event occurred.",
                "id": "cnn-fixture-1",
            }
        )
        summary_result = engine.add_conversation(
            summary_profile,
            summary,
            "summary",
            origin_group_id="summary:1",
        )
        if summary_result["accepted_chunks"] != 1:
            raise AssertionError("valid summary fixture was rejected")
        invalid_summaries = [
            {
                "article": "Missing a target summary.",
                "id": "cnn-fixture-missing",
            },
            {
                "article": "",
                "highlights": "An empty article must fail.",
                "id": "cnn-fixture-empty",
            },
        ]
        for invalid_summary in invalid_summaries:
            try:
                adapt_cnn_dailymail_summary(invalid_summary)
            except (KeyError, ProfileError):
                pass
            else:
                raise AssertionError("invalid CNN/DailyMail row was accepted")
        overlength = adapt_cnn_dailymail_summary(
            {
                "article": "article " * 2000,
                "highlights": "A short retained summary.",
                "id": "cnn-fixture-2",
            }
        )
        long_result = engine.add_conversation(
            summary_profile,
            overlength,
            "summary-long",
            origin_group_id="summary:2",
        )
        if long_result["accepted_chunks"] or long_result["dropped_overlength_turns"] != 1:
            raise AssertionError("overlength article record was not rejected intact")

        tulu = tulu_constraint_report(
            {
                "prompt": "Answer in exactly two bullets.",
                "messages": [
                    {"role": "user", "content": "Answer in exactly two bullets."},
                    {"role": "assistant", "content": "- one\n- two"},
                ],
                "constraints": ["Use exactly two bullets."],
            }
        )
        if tulu != {
            "declared": 1,
            "supported": 0,
            "verified": 0,
            "fully_verified": False,
        }:
            raise AssertionError("Tulu constraint claim is not fail-closed")

        store.commit()
        before_projection = store.layers()["gross"]["global"].get(
            "assistant_target_tokens", 0
        )
        projection = strict_projection_only(tokenizer, engine, 10, 42)
        after_projection = store.layers()["gross"]["global"].get(
            "assistant_target_tokens", 0
        )
        if before_projection != after_projection:
            raise AssertionError("strict projection was counted as actual capacity")
        if projection["actual_capacity_assistant_target_tokens"] != 0:
            raise AssertionError("strict projection claims actual tokens")

        store.finalize()
        layers = store.layers()
        if layers["origin_claims"]["selected_cross_bucket_origin_groups"] != 0:
            raise AssertionError("self-test left a cross-bucket origin group")

        if fixture_output is not None:
            with fixture_output.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        v1.training_row(canonical),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        store.close()

    result = {
        "self_test": "passed",
        "protocol_version": PROTOCOL_VERSION,
        "network_used": False,
        "ultrachat_partition_residues": sorted(residue_prompts),
        "multiturn_minimum_assistant_messages": 2,
        "numina_parse_policies": [
            parsed["parse_policy"],
            terminal["parse_policy"],
        ],
        "numina_shifted_target_sentinel": sentinel,
        "stable_origin_claim_winner_sources": winners_a,
        "stable_exact_dedup_winner_sources": exact_winners_a,
        "origin_exclusive_cross_bucket_groups": layers["origin_claims"][
            "selected_cross_bucket_origin_groups"
        ],
        "tulu_constraint_claim": tulu,
        "strict_projection_actual_capacity_tokens": projection[
            "actual_capacity_assistant_target_tokens"
        ],
        "summary_overlength_turns": long_result["dropped_overlength_turns"],
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
    config_path = args.profile_config.resolve()
    profile_config = load_profile_config(config_path)
    if args.validate_only:
        result, _, _ = validate_profile_inputs(config_path, profile_config)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["validation"] == "passed" else 1
    if args.output is None:
        raise ProfileError("--output is required for a full profile")
    return run_profile(config_path, args.output.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileError as exc:
        print(f"error={exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
