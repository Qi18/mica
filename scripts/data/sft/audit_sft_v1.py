#!/usr/bin/env python3
"""Independently audit a MiniMind SFT-v1 dataset build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from build_sft_v1 import BuildError as StrictBuildError, validate_strict

MESSAGE_KEYS = {"role", "content", "reasoning_content", "tools", "tool_calls"}
VALID_ROLES = {"system", "user", "assistant", "tool"}
SPLITS = ("train", "validation", "test")


class AuditError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--acceptance-config",
        type=Path,
        default=Path("configs/data/sft/acceptance_v1.yaml"),
    )
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalized_prompt(messages: list[dict[str, Any]]) -> str:
    return normalize(
        "\n".join(message["content"] for message in messages if message["role"] == "user")
    )


def create_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    prepared: list[dict[str, Any]] = []
    tools = None
    for original in messages:
        message = dict(original)
        if message["role"] == "system" and message["tools"]:
            tools = json.loads(message["tools"])
        if message["tool_calls"]:
            message["tool_calls"] = json.loads(message["tool_calls"])
        prepared.append(message)
    return tokenizer.apply_chat_template(
        prepared,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )


def find_sequence(values: list[int], needle: list[int], start: int = 0) -> int:
    for index in range(start, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def label_metrics(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    bos_id: list[int],
    eos_id: list[int],
) -> dict[str, int]:
    input_ids = tokenizer(create_chat_prompt(tokenizer, messages)).input_ids
    assistant_count = sum(message["role"] == "assistant" for message in messages)
    markers: list[int] = []
    cursor = 0
    while True:
        position = find_sequence(input_ids, bos_id, cursor)
        if position < 0:
            break
        markers.append(position)
        cursor = position + max(1, len(bos_id))
    if len(markers) != assistant_count:
        raise AuditError("assistant marker count mismatch")
    targets = 0
    for marker_index, marker in enumerate(markers):
        start = marker + len(bos_id)
        end = find_sequence(input_ids, eos_id, start)
        if end < 0:
            raise AuditError("assistant EOS is missing")
        if marker_index + 1 < len(markers) and end >= markers[marker_index + 1]:
            raise AuditError("assistant span crosses next marker")
        targets += max(0, end + len(eos_id) - max(1, start))
    if targets <= 0:
        raise AuditError("zero shifted assistant targets")
    return {
        "rendered_tokens": len(input_ids),
        "assistant_messages": assistant_count,
        "closed_assistant_spans": len(markers),
        "valid_targets": targets,
    }


def validate_message_schema(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise AuditError("conversations must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != MESSAGE_KEYS:
            raise AuditError("message keys are not exact")
        role = message["role"]
        if role not in VALID_ROLES:
            raise AuditError(f"invalid role: {role}")
        if not isinstance(message["content"], str):
            raise AuditError("content is not a string")
        if message["reasoning_content"] is not None and not isinstance(
            message["reasoning_content"], str
        ):
            raise AuditError("reasoning_content type is invalid")
        for key in ("tools", "tool_calls"):
            value = message[key]
            if value is not None:
                if not isinstance(value, str):
                    raise AuditError(f"{key} is not a canonical JSON string")
                parsed = json.loads(value)
                if not isinstance(parsed, list) or not parsed:
                    raise AuditError(f"{key} is not a non-empty JSON list")
        if message["tools"] is not None and role != "system":
            raise AuditError("tools outside system role")
        if message["tool_calls"] is not None and role != "assistant":
            raise AuditError("tool_calls outside assistant role")
        if role != "assistant" and not message["content"]:
            raise AuditError(f"empty {role} content")
        if role == "assistant" and not (
            message["content"]
            or message["reasoning_content"]
            or message["tool_calls"]
        ):
            raise AuditError("assistant has no supervised content")
        if role == "system" and index != 0:
            raise AuditError("system role is not first")
    return messages


def validate_turns(messages: list[dict[str, Any]]) -> None:
    body = messages[1:] if messages[0]["role"] == "system" else messages
    if not body or body[0]["role"] != "user":
        raise AuditError("conversation body does not start with user")
    has_tools = bool(messages[0]["tools"]) if messages[0]["role"] == "system" else False
    pending = 0
    assistants = 0
    in_turn = False
    for message in body:
        role = message["role"]
        if role == "user":
            if in_turn and assistants == 0:
                raise AuditError("turn has no assistant")
            if pending:
                raise AuditError("new turn before tool calls closed")
            in_turn = True
            assistants = 0
        elif not in_turn:
            raise AuditError("message before first user")
        elif role == "assistant":
            if pending:
                raise AuditError("assistant before tool responses closed")
            assistants += 1
            calls = json.loads(message["tool_calls"]) if message["tool_calls"] else []
            if calls and not has_tools:
                raise AuditError("tool call without system tools")
            pending = len(calls)
        elif role == "tool":
            if pending <= 0:
                raise AuditError("orphan tool response")
            pending -= 1
    if assistants == 0:
        raise AuditError("final turn has no assistant")
    if pending:
        raise AuditError("unclosed tool exchange")


def hashed_shingles(text: str, size: int = 5) -> set[int]:
    return {
        int.from_bytes(
            hashlib.blake2b(text[index : index + size].encode(), digest_size=8).digest(),
            "big",
        )
        for index in range(max(0, len(text) - size + 1))
    }


def simhash(values: set[int]) -> int:
    scores = [0] * 64
    for value in values:
        for bit in range(64):
            scores[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(scores):
        if score >= 0:
            result |= 1 << bit
    return result


class IndependentNearAudit:
    def __init__(self) -> None:
        self.bands: dict[tuple[int, int], list[str]] = defaultdict(list)
        self.values: dict[str, set[int]] = {}
        self.splits: dict[str, str] = {}

    def check_and_add(self, row_id: str, split: str, prompt: str) -> list[dict[str, Any]]:
        if len(prompt) < 50:
            return []
        values = hashed_shingles(prompt)
        signature = simhash(values)
        candidates: set[str] = set()
        for band in range(4):
            key = (band, (signature >> (band * 16)) & 0xFFFF)
            candidates.update(self.bands.get(key, ()))
        overlaps: list[dict[str, Any]] = []
        for other_id in sorted(candidates):
            other = self.values[other_id]
            score = len(values & other) / len(values | other)
            if score >= 0.85:
                overlaps.append(
                    {
                        "left": other_id,
                        "right": row_id,
                        "left_split": self.splits[other_id],
                        "right_split": split,
                        "jaccard": round(score, 6),
                    }
                )
        self.values[row_id] = values
        self.splits[row_id] = split
        for band in range(4):
            key = (band, (signature >> (band * 16)) & 0xFFFF)
            self.bands[key].append(row_id)
        return overlaps


def load_provenance(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    required = {
        "candidate_id", "source_id", "repo_id", "revision", "license",
        "source_record_locator", "source_record_sha256", "upstream_id",
        "origin_group_id", "raw_sha256", "transform_chain", "variant_family",
        "bucket", "split", "split_line_number", "rendered_tokens",
        "shifted_assistant_target_tokens",
    }
    result: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[str] = []
    origins: set[str] = set()
    candidates: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for sidecar_line, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                if not required.issubset(value):
                    raise AuditError(f"missing provenance fields: {sorted(required - set(value))}")
                key = (value["split"], int(value["split_line_number"]))
                if key in result:
                    raise AuditError(f"duplicate provenance location: {key}")
                if value["origin_group_id"] in origins:
                    raise AuditError("origin group reused")
                if value["candidate_id"] in candidates:
                    raise AuditError("candidate reused")
                if len(value["revision"]) != 40:
                    raise AuditError("source revision is not 40 chars")
                origins.add(value["origin_group_id"])
                candidates.add(value["candidate_id"])
                result[key] = value
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AuditError) as exc:
                failures.append(f"provenance:{sidecar_line}:{exc}")
    return result, failures


def load_strict_source_rows(
    manifest: dict[str, Any],
    provenance: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    failures: list[str] = []
    requested: dict[int, dict[str, Any]] = {}
    for sidecar in provenance.values():
        if sidecar.get("bucket") != "strict_format":
            continue
        locator = sidecar.get("source_record_locator")
        if not isinstance(locator, dict) or "line_number" not in locator:
            failures.append("strict provenance lacks source line number")
            continue
        requested[int(locator["line_number"])] = sidecar
    if not requested:
        return {}, failures
    source = next(
        (
            item
            for item in manifest.get("source_objects", [])
            if item.get("source_id") == "tulu_3_sft_personas_instruction_following"
        ),
        None,
    )
    if source is None:
        return {}, failures + ["strict source object is missing from manifest"]
    path = Path(source["path"])
    if not path.exists():
        return {}, failures + [f"strict source file is missing: {path}"]
    if path.stat().st_size != int(source["bytes"]):
        failures.append("strict source byte size differs from manifest")
    if file_sha256(path) != source["sha256"]:
        failures.append("strict source sha256 differs from manifest")
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for source_line_number, line in enumerate(handle, 1):
            sidecar = requested.get(source_line_number)
            if sidecar is None:
                continue
            raw_line = line.rstrip("\n")
            if sha256_bytes(raw_line.encode()) != sidecar["source_record_sha256"]:
                failures.append(
                    f"strict source line {source_line_number} sha256 differs from provenance"
                )
                continue
            try:
                rows[source_line_number] = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                failures.append(f"strict source line {source_line_number} invalid JSON: {exc}")
    missing = sorted(set(requested) - set(rows))
    if missing:
        failures.append(f"strict source rows unavailable={len(missing)}")
    return rows, failures


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = args.dataset_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance_path = args.acceptance_config.resolve()
    acceptance = yaml.safe_load(acceptance_path.read_text(encoding="utf-8"))
    tokenizer_path = Path.cwd() / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    bos_id = tokenizer(
        f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
    ).input_ids
    eos_id = tokenizer(
        f"{tokenizer.eos_token}\n", add_special_tokens=False
    ).input_ids
    provenance, failures = load_provenance(root / "provenance.jsonl")
    strict_source_rows, strict_source_failures = load_strict_source_rows(manifest, provenance)
    failures.extend(strict_source_failures)
    exact_conversations: dict[str, str] = {}
    exact_prompts: dict[str, str] = {}
    near_audit = IndependentNearAudit()
    near_overlaps: list[dict[str, Any]] = []
    split_rows = Counter()
    split_tokens = Counter()
    bucket_rows: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    strict_families: set[str] = set()
    seen_provenance: set[tuple[str, int]] = set()
    maximum_length = int(manifest["sequence_length"])
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        expected_file = manifest["files"][split]
        if path.stat().st_size != int(expected_file["bytes"]):
            failures.append(f"{split}:byte size differs from manifest")
        if file_sha256(path) != expected_file["sha256"]:
            failures.append(f"{split}:sha256 differs from manifest")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row_id = f"{split}:{line_number}"
                try:
                    row = json.loads(line)
                    if set(row) != {"conversations"}:
                        raise AuditError("top-level keys are not exact")
                    messages = validate_message_schema(row["conversations"])
                    validate_turns(messages)
                    metrics = label_metrics(tokenizer, messages, bos_id, eos_id)
                    if metrics["rendered_tokens"] > maximum_length:
                        raise AuditError("rendered sequence exceeds maximum")
                    key = (split, line_number)
                    if key not in provenance:
                        raise AuditError("provenance row is missing")
                    sidecar = provenance[key]
                    seen_provenance.add(key)
                    if metrics["valid_targets"] != int(
                        sidecar["shifted_assistant_target_tokens"]
                    ):
                        raise AuditError("assistant target count differs from sidecar")
                    if metrics["rendered_tokens"] != int(sidecar["rendered_tokens"]):
                        raise AuditError("rendered token count differs from sidecar")
                    exact_digest = sha256_bytes(stable_json(row).encode())
                    if exact_digest in exact_conversations:
                        raise AuditError(
                            f"exact conversation duplicate with {exact_conversations[exact_digest]}"
                        )
                    exact_conversations[exact_digest] = row_id
                    prompt = normalized_prompt(messages)
                    prompt_digest = sha256_bytes(prompt.encode())
                    if prompt_digest in exact_prompts:
                        raise AuditError(f"exact prompt duplicate with {exact_prompts[prompt_digest]}")
                    exact_prompts[prompt_digest] = row_id
                    near_overlaps.extend(near_audit.check_and_add(row_id, split, prompt))
                    bucket = sidecar["bucket"]
                    split_rows[split] += 1
                    split_tokens[split] += metrics["valid_targets"]
                    bucket_rows[split][bucket] += 1
                    bucket_tokens[split][bucket] += metrics["valid_targets"]
                    if bucket == "strict_format":
                        chains = sidecar["transform_chain"]
                        verifier = next(
                            (item for item in chains if item.get("id") == "tulu_constraint_verifier"),
                            None,
                        )
                        if not verifier or verifier.get("status") != "passed":
                            raise AuditError("strict row lacks passed verifier evidence")
                        if len(messages) != 2 or [item["role"] for item in messages] != [
                            "user",
                            "assistant",
                        ]:
                            raise AuditError("strict row is not one user/assistant pair")
                        strict_line_number = int(
                            sidecar["source_record_locator"]["line_number"]
                        )
                        source_row = strict_source_rows.get(strict_line_number)
                        if source_row is None:
                            raise AuditError("strict source row is unavailable")
                        if normalize(str(source_row["prompt"])) != normalize(
                            messages[0]["content"]
                        ):
                            raise AuditError("strict source prompt differs from payload")
                        try:
                            replayed = validate_strict(
                                messages[0]["content"],
                                messages[1]["content"],
                                source_row.get("constraints"),
                            )
                        except StrictBuildError as exc:
                            raise AuditError(f"strict verifier replay failed: {exc}") from exc
                        if replayed.get("verifier_version") != verifier.get(
                            "verifier_version"
                        ):
                            raise AuditError("strict verifier version differs from provenance")
                        if stable_json(replayed.get("evidence")) != stable_json(
                            verifier.get("evidence")
                        ):
                            raise AuditError("strict verifier evidence differs from provenance")
                        strict_families.update(verifier.get("families") or [])
                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    AuditError,
                ) as exc:
                    failures.append(f"{row_id}:{exc}")
    unused_provenance = sorted(set(provenance) - seen_provenance)
    if unused_provenance:
        failures.append(f"unused provenance rows={len(unused_provenance)}")
    if near_overlaps:
        failures.append(f"near duplicate pairs={len(near_overlaps)}")
    quotas = manifest["selection"]["quotas"]
    for split in SPLITS:
        for bucket, target in quotas[split].items():
            if bucket_tokens[split][bucket] < int(target):
                failures.append(
                    f"{split}:{bucket}:tokens={bucket_tokens[split][bucket]} below quota={target}"
                )
    if len(strict_families) < int(
        acceptance["derived_quality_gate"]["strict_format"][
            "minimum_verified_transform_families"
        ]
    ):
        failures.append(f"strict verifier families={len(strict_families)} below minimum")
    provenance_path = root / "provenance.jsonl"
    expected_provenance = manifest["files"]["provenance"]
    if file_sha256(provenance_path) != expected_provenance["sha256"]:
        failures.append("provenance sha256 differs from manifest")
    contamination = json.loads((root / "contamination_report.json").read_text(encoding="utf-8"))
    contamination_accepted = contamination.get("status") == "accepted"
    core_accepted = not failures
    status = (
        "accepted"
        if core_accepted and contamination_accepted
        else "core_accepted_contamination_pending"
        if core_accepted
        else "rejected"
    )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": status,
        "core_accepted": core_accepted,
        "contamination_accepted": contamination_accepted,
        "failures": failures,
        "near_overlap_examples": near_overlaps[:20],
        "split_rows": dict(split_rows),
        "split_assistant_target_tokens": dict(split_tokens),
        "bucket_rows": {key: dict(value) for key, value in bucket_rows.items()},
        "bucket_assistant_target_tokens": {
            key: dict(value) for key, value in bucket_tokens.items()
        },
        "strict_verified_families": sorted(strict_families),
        "bindings": {
            "auditor": "scripts/data/sft/audit_sft_v1.py",
            "auditor_sha256": file_sha256(Path(__file__)),
            "acceptance_config": str(args.acceptance_config),
            "acceptance_config_sha256": file_sha256(acceptance_path),
            "dataset_manifest_sha256": file_sha256(manifest_path),
        },
    }
    (root / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = root / "_SUCCESS"
    if marker.exists():
        marker.unlink()
    if args.finalize:
        contamination_config = yaml.safe_load(
            (Path.cwd() / "configs/data/sft/contamination_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        alignment = contamination_config["task_alignment"]
        if not alignment.get("harness_commit") or not alignment.get(
            "prompt_builder_sha256"
        ):
            raise AuditError("contamination harness bindings are not frozen")
        if status != "accepted":
            raise AuditError(f"dataset cannot be finalized: {status}")
        marker.write_text(
            json.dumps(
                {
                    "status": "accepted",
                    "created_at": utc_now(),
                    "audit_report_sha256": file_sha256(root / "audit_report.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report, 0 if core_accepted else 1


def main() -> int:
    args = parse_args()
    report, code = audit(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failures": len(report["failures"]),
                "rows": report["split_rows"],
                "assistant_target_tokens": report["split_assistant_target_tokens"],
            },
            ensure_ascii=False,
        )
    )
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
