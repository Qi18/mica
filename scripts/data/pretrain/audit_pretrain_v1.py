#!/usr/bin/env python3
"""Independently audit final MiniMind pretraining JSONL shards.

The audit reproduces MiniMind PretrainDataset tokenization exactly:
raw text is tokenized without special tokens, truncated to sequence_length - 2,
then BOS and EOS are added. The shifted loss therefore sees raw tokens plus EOS.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import math
import json
import os
import sqlite3
import struct
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


REPORT_SCHEMA_VERSION = 2
AUDITOR_VERSION = "1.3.0"
MAX_EXAMPLES = 10
PINNED_EVAL_EXPECTATIONS = {
    "ceval_valid": {
        "repo_id": "ceval/ceval-exam",
        "revision": "617524a00b307ff6f9933702f724131fe12ca7ce",
        "split": "val",
        "configs": 52,
        "rows": 1346,
    },
    "cmmlu": {
        "repo_id": "haonan-li/cmmlu",
        "revision": "efcc940752ea4a1ea94d2727f11f83858d64fc8e",
        "split": "test",
        "configs": 67,
        "rows": 11582,
    },
    "arc_easy": {
        "repo_id": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "split": "test",
        "configs": 1,
        "rows": 2376,
    },
    "piqa": {
        "repo_id": "baber/piqa",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "split": "validation",
        "configs": 1,
        "rows": 1838,
    },
    "openbookqa": {
        "repo_id": "allenai/openbookqa",
        "revision": "388097ea7776314e93a529163e0fea805b8a6454",
        "split": "test",
        "configs": 1,
        "rows": 500,
    },
    "hellaswag": {
        "repo_id": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "validation",
        "configs": 1,
        "rows": 10042,
    },
    "social_iqa": {
        "repo_id": "allenai/social_i_qa",
        "revision": "8835ceb9141d7896d9d968634a9b21ae440e3ec5",
        "split": "validation",
        "configs": 1,
        "rows": 1954,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/cache/huggingface"))
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--verification-report", type=Path)
    parser.add_argument("--sqlite-path", type=Path)
    parser.add_argument("--success-marker", type=Path)
    parser.add_argument("--no-write-success", action="store_true")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument(
        "--near-mode",
        choices=("stratified", "full"),
        default="stratified",
        help="stratified audits deterministic bottom-k documents per physical shard",
    )
    parser.add_argument("--near-sample-per-shard", type=int, default=512)
    parser.add_argument(
        "--allow-test-eval-config",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def absolute_path_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def success_marker_path(args: argparse.Namespace) -> Path:
    raw_path = (
        args.success_marker
        if args.success_marker is not None
        else args.data_dir / "_SUCCESS"
    )
    return absolute_path_without_symlink_resolution(raw_path)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_success_marker_absent(marker_path: Path) -> str:
    marker_path = absolute_path_without_symlink_resolution(marker_path)
    parent = marker_path.parent
    marker_path.unlink(missing_ok=True)
    if parent.is_dir():
        fsync_directory(parent)
        status = "confirmed_absent_parent_fsynced"
    else:
        status = "confirmed_absent_parent_missing"
    if os.path.lexists(marker_path):
        raise RuntimeError(f"success marker still exists: {marker_path}")
    return status


def initial_startup_fingerprints(
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    return {
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": None,
        },
        "config": {
            "path": str(args.config.resolve()),
            "sha256": None,
        },
        "eval_config": {
            "path": str(args.eval_config.resolve()),
            "sha256": None,
        },
    }


def load_startup_inputs(
    startup: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    auditor_bytes = Path(startup["auditor"]["path"]).read_bytes()
    startup["auditor"]["sha256"] = sha256_bytes(auditor_bytes)
    config_bytes = Path(startup["config"]["path"]).read_bytes()
    startup["config"]["sha256"] = sha256_bytes(config_bytes)
    eval_bytes = Path(startup["eval_config"]["path"]).read_bytes()
    startup["eval_config"]["sha256"] = sha256_bytes(eval_bytes)
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    eval_config = yaml.safe_load(eval_bytes.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config YAML root must be an object")
    if not isinstance(eval_config, dict):
        raise ValueError("eval config YAML root must be an object")
    return config, eval_config


def verify_startup_fingerprints(
    startup: dict[str, dict[str, Any]],
) -> None:
    mismatches: dict[str, dict[str, Any]] = {}
    for name in ("auditor", "config", "eval_config"):
        record = startup[name]
        observed = sha256_file(Path(str(record["path"])))
        if observed != record.get("sha256"):
            mismatches[name] = {
                "expected": record.get("sha256"),
                "observed": observed,
            }
    if mismatches:
        raise RuntimeError(
            "startup input TOCTOU detected before report publication: "
            f"{canonical_json(mismatches)}"
        )


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def tokenizer_fingerprint(path: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        candidate = path / name
        if candidate.exists():
            files[name] = {
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
    if "tokenizer.json" not in files:
        raise FileNotFoundError(path / "tokenizer.json")
    return {"path": str(path), "files": files}


def allocate(total: int, weighted: list[tuple[str, float]]) -> dict[str, int]:
    if total < 0 or not weighted:
        raise ValueError("bad quota allocation input")
    weight_sum = sum(weight for _, weight in weighted)
    if weight_sum <= 0:
        raise ValueError("quota weights must be positive")
    normalized = [(name, weight / weight_sum) for name, weight in weighted]
    raw = {name: total * weight for name, weight in normalized}
    output = {name: math.floor(value) for name, value in raw.items()}
    order = sorted(
        normalized,
        key=lambda item: (raw[item[0]] - output[item[0]], item[0]),
        reverse=True,
    )
    for name, _ in order[: total - sum(output.values())]:
        output[name] += 1
    return output


def source_targets(config: dict[str, Any], total: int) -> dict[str, int]:
    categories = config.get("active_v1_mix")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("active_v1_mix is empty")
    category_weights = [
        (str(name), float(settings["weight"]))
        for name, settings in categories.items()
    ]
    if not math.isclose(
        sum(weight for _, weight in category_weights),
        1.0,
        abs_tol=1e-9,
        rel_tol=0.0,
    ):
        raise ValueError("active_v1_mix weights must sum to 1")
    category_budgets = allocate(total, category_weights)
    output: dict[str, int] = {}
    for category, settings in categories.items():
        sources = [str(source) for source in settings.get("sources", [])]
        if not sources:
            raise ValueError(f"active_v1_mix.{category}.sources is empty")
        configured = settings.get("source_weights")
        weights = (
            [(source, float(configured[source])) for source in sources]
            if configured
            else [(source, 1.0) for source in sources]
        )
        for source, tokens in allocate(category_budgets[str(category)], weights).items():
            output[source] = output.get(source, 0) + tokens
    if sum(output.values()) != total:
        raise AssertionError((output, total))
    return output


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return "".join(character for character in value if character.isalnum())


def shingles(value: str, width: int) -> set[str]:
    if len(value) <= width:
        return {value} if value else set()
    return {value[index : index + width] for index in range(len(value) - width + 1)}


def extract_query(row: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        rendered = value.strip() if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, sort_keys=True
        )
        if rendered:
            parts.append(rendered)
    return " ".join(parts)


def matching_contract(eval_config: dict[str, Any]) -> dict[str, Any]:
    configured = eval_config.get("matching")
    if configured is not None and not isinstance(configured, dict):
        raise ValueError("matching must be an object")
    matching = configured or {}
    near = eval_config.get("near_duplicate", {})
    exact_min = int(matching.get("exact_min_normalized_chars", 1))
    containment_min = int(
        matching.get(
            "containment_min_normalized_chars",
            near.get("min_normalized_chars", 20),
        )
    )
    if exact_min != 1:
        raise ValueError(
            "matching.exact_min_normalized_chars must be 1 so every "
            "nonempty normalized query remains exact-match eligible"
        )
    if containment_min < exact_min:
        raise ValueError(
            "matching.containment_min_normalized_chars must be at least 1"
        )
    return {
        "exact_min_normalized_chars": exact_min,
        "containment_min_normalized_chars": containment_min,
        "configured_explicitly": configured is not None,
    }


class AhoCorasick:
    """Small dependency-free exact multi-pattern matcher."""

    def __init__(self) -> None:
        self.transitions: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.output_link: list[int] = [-1]
        self.terminals: list[list[int]] = [[]]

    def add(self, pattern: str, identifier: int) -> None:
        state = 0
        for character in pattern:
            next_state = self.transitions[state].get(character)
            if next_state is None:
                next_state = len(self.transitions)
                self.transitions[state][character] = next_state
                self.transitions.append({})
                self.fail.append(0)
                self.output_link.append(-1)
                self.terminals.append([])
            state = next_state
        self.terminals[state].append(identifier)

    def build(self) -> None:
        queue: deque[int] = deque()
        for child in self.transitions[0].values():
            queue.append(child)
            self.fail[child] = 0
            self.output_link[child] = -1
        while queue:
            state = queue.popleft()
            for character, child in self.transitions[state].items():
                queue.append(child)
                fallback = self.fail[state]
                while fallback and character not in self.transitions[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.transitions[fallback].get(character, 0)
                failure_state = self.fail[child]
                self.output_link[child] = (
                    failure_state
                    if self.terminals[failure_state]
                    else self.output_link[failure_state]
                )

    def find(self, text: str) -> set[int]:
        matches: set[int] = set()
        state = 0
        for character in text:
            while state and character not in self.transitions[state]:
                state = self.fail[state]
            state = self.transitions[state].get(character, 0)
            output_state = state
            while output_state != -1:
                if self.terminals[output_state]:
                    matches.update(self.terminals[output_state])
                output_state = self.output_link[output_state]
        return matches


def task_configs(source: dict[str, Any], get_dataset_config_names: Any) -> list[str | None]:
    if source.get("all_configs"):
        return list(
            get_dataset_config_names(
                source["repo_id"],
                revision=source["revision"],
            )
        )
    return [source.get("config")]


def load_with_retries(
    load_dataset: Any,
    kwargs: dict[str, Any],
    attempts: int = 3,
    on_failure: Callable[[int, int, Exception], None] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(**kwargs)
        except Exception as exc:
            last_error = exc
            if on_failure is not None:
                on_failure(attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    assert last_error is not None
    raise last_error


def load_eval_queries(
    eval_config: dict[str, Any],
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint = eval_config.get("metadata_endpoint", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from datasets import get_dataset_config_names, load_dataset

    audit_started_at = time.monotonic()
    pattern_map: dict[str, dict[str, Any]] = {}
    source_results: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    sources = list(eval_config.get("sources", []))
    for source_index, source in enumerate(sources, 1):
        source_started_at = time.monotonic()
        source_id = source.get("id")
        source_counter: Counter[str] = Counter()
        configs: list[str | None] = []
        configs_succeeded = 0
        configs_failed = 0
        discovery_started_at = time.monotonic()
        print(
            canonical_json(
                {
                    "event": "eval_source_configs_start",
                    "source": source_id,
                    "source_index": source_index,
                    "sources_total": len(sources),
                    "repo_id": source.get("repo_id"),
                    "revision": source.get("revision"),
                    "split": source.get("split"),
                }
            ),
            flush=True,
        )
        try:
            configs = task_configs(source, get_dataset_config_names)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            load_errors.append(
                {
                    "source": source_id,
                    "phase": "list_configs",
                    "error": error,
                }
            )
            print(
                canonical_json(
                    {
                        "event": "eval_source_configs_failure",
                        "source": source_id,
                        "attempt": 1,
                        "attempts": 1,
                        "error": error,
                        "elapsed_seconds": round(
                            time.monotonic() - discovery_started_at,
                            3,
                        ),
                    }
                ),
                flush=True,
            )
        else:
            print(
                canonical_json(
                    {
                        "event": "eval_source_configs_success",
                        "source": source_id,
                        "configs": len(configs),
                        "elapsed_seconds": round(
                            time.monotonic() - discovery_started_at,
                            3,
                        ),
                    }
                ),
                flush=True,
            )
        for config_index, config_name in enumerate(configs, 1):
            kwargs: dict[str, Any] = {
                "path": source["repo_id"],
                "split": source["split"],
                "streaming": True,
                "cache_dir": str(cache_dir),
            }
            if source.get("revision") and source["repo_id"] != "json":
                kwargs["revision"] = source["revision"]
            if config_name:
                kwargs["name"] = config_name
            if source.get("data_files") is not None:
                kwargs["data_files"] = source["data_files"]

            config_started_at = time.monotonic()
            config_counter: Counter[str] = Counter()
            load_attempts = 3
            attempt_state = {"last_failure": 0}
            dataset_loaded = False
            successful_attempt = 1
            print(
                canonical_json(
                    {
                        "event": "eval_config_load_start",
                        "source": source_id,
                        "config": config_name,
                        "config_index": config_index,
                        "configs_total": len(configs),
                        "repo_id": source.get("repo_id"),
                        "revision": source.get("revision"),
                        "split": source.get("split"),
                        "attempts": load_attempts,
                    }
                ),
                flush=True,
            )

            def log_load_failure(
                attempt: int,
                attempts: int,
                exc: Exception,
            ) -> None:
                attempt_state["last_failure"] = attempt
                print(
                    canonical_json(
                        {
                            "event": "eval_config_load_failure",
                            "phase": "load_dataset",
                            "source": source_id,
                            "config": config_name,
                            "attempt": attempt,
                            "attempts": attempts,
                            "will_retry": attempt < attempts,
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": round(
                                time.monotonic() - config_started_at,
                                3,
                            ),
                        }
                    ),
                    flush=True,
                )

            try:
                dataset = load_with_retries(
                    load_dataset,
                    kwargs,
                    attempts=load_attempts,
                    on_failure=log_load_failure,
                )
                dataset_loaded = True
                successful_attempt = attempt_state["last_failure"] + 1
                for row in dataset:
                    source_counter["rows"] += 1
                    config_counter["rows"] += 1
                    raw = extract_query(row, list(source["query_fields"]))
                    key = normalize(raw)
                    if not key:
                        source_counter["empty_queries"] += 1
                        config_counter["empty_queries"] += 1
                        continue
                    pattern = pattern_map.setdefault(
                        key,
                        {"normalized": key, "raw": raw[:500], "entries": []},
                    )
                    pattern["entries"].append(
                        {
                            "source_id": source_id,
                            "task": source["task"],
                            "raw": raw[:500],
                        }
                    )
                    source_counter["queries"] += 1
                    config_counter["queries"] += 1
                configs_succeeded += 1
                print(
                    canonical_json(
                        {
                            "event": "eval_config_load_success",
                            "source": source_id,
                            "config": config_name,
                            "attempt": successful_attempt,
                            "rows": config_counter["rows"],
                            "queries": config_counter["queries"],
                            "empty_queries": config_counter[
                                "empty_queries"
                            ],
                            "elapsed_seconds": round(
                                time.monotonic() - config_started_at,
                                3,
                            ),
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                configs_failed += 1
                error = f"{type(exc).__name__}: {exc}"
                load_errors.append(
                    {
                        "source": source_id,
                        "config": config_name,
                        "phase": "load",
                        "error": error,
                    }
                )
                if dataset_loaded:
                    print(
                        canonical_json(
                            {
                                "event": "eval_config_load_failure",
                                "phase": "iterate",
                                "source": source_id,
                                "config": config_name,
                                "attempt": successful_attempt,
                                "attempts": load_attempts,
                                "will_retry": False,
                                "error": error,
                                "rows": config_counter["rows"],
                                "queries": config_counter["queries"],
                                "elapsed_seconds": round(
                                    time.monotonic() - config_started_at,
                                    3,
                                ),
                            }
                        ),
                        flush=True,
                    )
        source_result = {
            "id": source_id,
            "task": source.get("task"),
            "repo_id": source.get("repo_id"),
            "revision": source.get("revision"),
            "split": source.get("split"),
            "configs": len(configs),
            "configs_succeeded": configs_succeeded,
            "configs_failed": configs_failed,
            "config_names": configs,
            "elapsed_seconds": round(
                time.monotonic() - source_started_at,
                3,
            ),
            **dict(source_counter),
        }
        source_results.append(source_result)
        print(
            canonical_json(
                {
                    "event": "eval_source_summary",
                    "eval_source": source_result["id"],
                    "configs": source_result["configs"],
                    "configs_succeeded": configs_succeeded,
                    "configs_failed": configs_failed,
                    "rows": source_result.get("rows", 0),
                    "queries": source_result.get("queries", 0),
                    "empty_queries": source_result.get(
                        "empty_queries", 0
                    ),
                    "load_errors_so_far": len(load_errors),
                    "elapsed_seconds": source_result[
                        "elapsed_seconds"
                    ],
                }
            ),
            flush=True,
        )
    patterns = list(pattern_map.values())
    for identifier, pattern in enumerate(patterns):
        pattern["id"] = identifier
    print(
        canonical_json(
            {
                "event": "eval_load_summary",
                "sources": len(source_results),
                "configs": sum(
                    int(source["configs"]) for source in source_results
                ),
                "configs_succeeded": sum(
                    int(source["configs_succeeded"])
                    for source in source_results
                ),
                "configs_failed": sum(
                    int(source["configs_failed"])
                    for source in source_results
                ),
                "rows": sum(
                    int(source.get("rows", 0))
                    for source in source_results
                ),
                "queries": sum(
                    int(source.get("queries", 0))
                    for source in source_results
                ),
                "patterns_unique": len(patterns),
                "load_errors": len(load_errors),
                "elapsed_seconds": round(
                    time.monotonic() - audit_started_at,
                    3,
                ),
            }
        ),
        flush=True,
    )
    return patterns, source_results, load_errors


def validate_pinned_eval_sources(
    eval_config: dict[str, Any],
    source_results: list[dict[str, Any]],
    load_errors: list[dict[str, Any]],
    allow_test_config: bool,
) -> list[str]:
    errors: list[str] = []
    if load_errors:
        errors.append(f"benchmark load errors: {len(load_errors)}")
    if len(source_results) != 7:
        errors.append(f"expected 7 benchmark sources, got {len(source_results)}")
    if allow_test_config:
        return errors
    configured = {
        str(source.get("id")): source
        for source in eval_config.get("sources", [])
    }
    observed = {str(source.get("id")): source for source in source_results}
    if set(configured) != set(PINNED_EVAL_EXPECTATIONS):
        errors.append(
            "benchmark source ids differ: "
            f"{sorted(configured)} != {sorted(PINNED_EVAL_EXPECTATIONS)}"
        )
    for source_id, expected in PINNED_EVAL_EXPECTATIONS.items():
        source = configured.get(source_id)
        result = observed.get(source_id)
        if source is None or result is None:
            continue
        for key in ("repo_id", "revision", "split"):
            if source.get(key) != expected[key]:
                errors.append(
                    f"{source_id} {key}: {source.get(key)!r} != {expected[key]!r}"
                )
        if int(result.get("configs", 0)) != int(expected["configs"]):
            errors.append(
                f"{source_id} configs: {result.get('configs')} != "
                f"{expected['configs']}"
            )
        if int(result.get("rows", 0)) != int(expected["rows"]):
            errors.append(
                f"{source_id} rows: {result.get('rows', 0)} != "
                f"{expected['rows']}"
            )
        if int(result.get("empty_queries", 0)) != 0:
            errors.append(
                f"{source_id} empty queries: {result.get('empty_queries')}"
            )
    expected_tasks = {
        str(source.get("task")) for source in eval_config.get("sources", [])
    }
    aligned_tasks = set(eval_config.get("task_alignment", {}).get("task_names", []))
    if expected_tasks != aligned_tasks:
        errors.append(
            f"task alignment differs: {sorted(expected_tasks)} != "
            f"{sorted(aligned_tasks)}"
        )
    return errors


def pattern_entries_by_source(
    pattern: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for entry in pattern["entries"]:
        source_id = str(entry["source_id"])
        output.setdefault(source_id, entry)
    return output


def build_containment_automaton(
    patterns: list[dict[str, Any]],
    min_normalized_chars: int,
) -> AhoCorasick:
    automaton = AhoCorasick()
    for pattern in patterns:
        if len(pattern["normalized"]) < min_normalized_chars:
            continue
        automaton.add(pattern["normalized"], int(pattern["id"]))
    automaton.build()
    return automaton


class ContaminationAudit:
    def __init__(
        self,
        patterns: list[dict[str, Any]],
        eval_config: dict[str, Any],
        matching: dict[str, Any],
    ) -> None:
        self.patterns = patterns
        self.exact_min_chars = int(
            matching["exact_min_normalized_chars"]
        )
        self.containment_min_chars = int(
            matching["containment_min_normalized_chars"]
        )
        self.pattern_by_text = {
            pattern["normalized"]: int(pattern["id"])
            for pattern in patterns
            if len(pattern["normalized"]) >= self.exact_min_chars
        }
        self.exact_patterns_indexed = len(self.pattern_by_text)
        self.exact_patterns_skipped_short = (
            len(patterns) - self.exact_patterns_indexed
        )
        self.containment_patterns_indexed = sum(
            len(pattern["normalized"]) >= self.containment_min_chars
            for pattern in patterns
        )
        self.containment_patterns_skipped_short = (
            len(patterns) - self.containment_patterns_indexed
        )
        self.automaton = build_containment_automaton(
            patterns, self.containment_min_chars
        )
        self.counts: Counter[str] = Counter()
        self.split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.source_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.example_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.max_examples = int(
            eval_config.get("near_duplicate", {}).get(
                "max_examples_per_source", MAX_EXAMPLES
            )
        )

    def _record(
        self,
        kind: str,
        pattern_id: int,
        split: str,
        shard: str,
        line_number: int,
        document: str,
    ) -> None:
        pattern = self.patterns[pattern_id]
        self.counts[kind] += 1
        self.split_counts[split][kind] += 1
        for source_id, entry in sorted(
            pattern_entries_by_source(pattern).items()
        ):
            self.source_counts[source_id][kind] += 1
            if self.example_counts[source_id][kind] < self.max_examples:
                self.examples[source_id].append(
                    {
                        "kind": kind,
                        "split": split,
                        "shard": shard,
                        "line": line_number,
                        "eval": entry["raw"],
                        "document": document[:500],
                    }
                )
                self.example_counts[source_id][kind] += 1

    def audit(
        self,
        normalized_document: str,
        raw_document: str,
        split: str,
        shard: str,
        line_number: int,
    ) -> None:
        self.counts["documents_audited"] += 1
        self.split_counts[split]["documents_audited"] += 1
        exact_id = self.pattern_by_text.get(normalized_document)
        if exact_id is not None:
            self._record(
                "exact_overlap",
                exact_id,
                split,
                shard,
                line_number,
                raw_document,
            )
        for pattern_id in sorted(self.automaton.find(normalized_document)):
            if pattern_id == exact_id:
                continue
            self._record(
                "containment_overlap",
                pattern_id,
                split,
                shard,
                line_number,
                raw_document,
            )


class NearAudit:
    def __init__(
        self,
        patterns: list[dict[str, Any]],
        eval_config: dict[str, Any],
    ) -> None:
        from datasketch import MinHash, MinHashLSH

        settings = eval_config["near_duplicate"]
        self.MinHash = MinHash
        self.width = int(settings["char_ngram"])
        self.min_chars = int(settings["min_normalized_chars"])
        self.num_perm = int(settings["num_perm"])
        self.seed = int(settings["seed"])
        self.threshold = float(settings["jaccard_threshold"])
        self.patterns = patterns
        self.grams: dict[str, set[str]] = {}
        self.index = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for pattern in patterns:
            normalized = pattern["normalized"]
            if len(normalized) < self.min_chars:
                continue
            identifier = str(pattern["id"])
            grams = shingles(normalized, self.width)
            signature = self._minhash(grams)
            self.index.insert(identifier, signature)
            self.grams[identifier] = grams
        self.counts: Counter[str] = Counter()
        self.source_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.example_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.max_examples = int(settings.get("max_examples_per_source", MAX_EXAMPLES))

    def _minhash(self, values: set[str]) -> Any:
        signature = self.MinHash(num_perm=self.num_perm, seed=self.seed)
        signature.update_batch(value.encode("utf-8") for value in sorted(values))
        return signature

    def audit_document(
        self,
        normalized_document: str,
        raw_document: str,
        split: str,
        shard: str,
        line_number: int,
    ) -> None:
        if len(normalized_document) < self.min_chars:
            return
        self.counts["documents_audited"] += 1
        document_grams = shingles(normalized_document, self.width)
        signature = self._minhash(document_grams)
        for identifier in sorted(
            self.index.query(signature), key=lambda value: int(value)
        ):
            pattern = self.patterns[int(identifier)]
            query = pattern["normalized"]
            if query == normalized_document or query in normalized_document:
                continue
            eval_grams = self.grams[identifier]
            union = len(document_grams | eval_grams)
            score = len(document_grams & eval_grams) / union if union else 0.0
            if score < self.threshold:
                continue
            kind = "near_duplicate_overlap"
            self.counts[kind] += 1
            for source_id, entry in sorted(
                pattern_entries_by_source(pattern).items()
            ):
                self.source_counts[source_id]["near_duplicate_overlap"] += 1
                if self.example_counts[source_id][kind] < self.max_examples:
                    self.examples[source_id].append(
                        {
                            "kind": kind,
                            "score": round(score, 6),
                            "split": split,
                            "shard": shard,
                            "line": line_number,
                            "eval": entry["raw"],
                            "document": raw_document[:500],
                        }
                    )
                    self.example_counts[source_id][kind] += 1


class StratifiedBottomK:
    def __init__(self, per_shard: int, seed: int, min_chars: int) -> None:
        if per_shard <= 0:
            raise ValueError("--near-sample-per-shard must be positive")
        self.per_shard = per_shard
        self.seed = seed
        self.min_chars = min_chars
        self.heaps: dict[str, list[tuple[int, int, str, str, str, str]]] = defaultdict(list)
        self.eligible: Counter[str] = Counter()

    def offer(
        self,
        normalized_document: str,
        raw_document: str,
        split: str,
        shard: str,
        line_number: int,
    ) -> None:
        if len(normalized_document) < self.min_chars:
            return
        key = f"{split}/{shard}"
        self.eligible[key] += 1
        payload = (
            f"{self.seed}\0{split}\0{shard}\0{line_number}\0{normalized_document}"
        ).encode("utf-8")
        score = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        item = (
            -score,
            line_number,
            split,
            shard,
            normalized_document,
            raw_document[:1000],
        )
        heap = self.heaps[key]
        if len(heap) < self.per_shard:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    def documents(self) -> Iterable[tuple[str, str, str, str, int]]:
        for key in sorted(self.heaps):
            for _, line_number, split, shard, normalized, raw in sorted(
                self.heaps[key], reverse=True
            ):
                yield normalized, raw, split, shard, line_number


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root is not an object")
    return value


def success_marker_bindings(marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": marker.get("status"),
        "audit_report_path": marker.get("audit_report", {}).get("path"),
        "audit_report_sha256": marker.get("audit_report", {}).get("sha256"),
        "auditor_path": marker.get("auditor", {}).get("path"),
        "auditor_version": marker.get("auditor", {}).get("version"),
        "auditor_sha256": marker.get("auditor", {}).get("sha256"),
        "manifest_path": marker.get("manifest", {}).get("path"),
        "manifest_sha256": marker.get("manifest", {}).get("sha256"),
        "manifest_fingerprint": marker.get("manifest", {}).get("fingerprint"),
        "verification_path": marker.get("verification", {}).get("path"),
        "verification_sha256": marker.get("verification", {}).get("sha256"),
        "builder_version": marker.get("builder", {}).get("version"),
        "candidate_manifest_sha256": marker.get("builder", {}).get(
            "candidate_manifest_sha256"
        ),
        "config_path": marker.get("config", {}).get("path"),
        "config_sha256": marker.get("config", {}).get("sha256"),
        "eval_config_path": marker.get("config", {}).get("eval_path"),
        "eval_config_sha256": marker.get("config", {}).get("eval_sha256"),
        "tokenizer_fingerprint_sha256": marker.get("tokenizer", {}).get(
            "fingerprint_sha256"
        ),
        "dataset_fingerprint": marker.get("dataset_fingerprint"),
        "repair_evidence": marker.get("repair_evidence"),
        "benchmark_near_mode": marker.get("benchmark_audit", {}).get("near_mode"),
        "benchmark_near_sample_per_shard": marker.get("benchmark_audit", {}).get(
            "near_sample_per_shard"
        ),
    }


def validate_success_marker(
    marker_path: Path,
    expected_bindings: dict[str, Any],
) -> dict[str, Any]:
    marker = read_json_object(marker_path)
    observed_bindings = success_marker_bindings(marker)
    if observed_bindings != expected_bindings:
        mismatches = {
            key: {
                "expected": expected_bindings.get(key),
                "observed": observed_bindings.get(key),
            }
            for key in sorted(set(expected_bindings) | set(observed_bindings))
            if expected_bindings.get(key) != observed_bindings.get(key)
        }
        raise ValueError(
            "success marker read-back binding mismatch: "
            f"{canonical_json(mismatches)}"
        )
    return marker


def safe_data_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes data root: {relative}") from exc
    return candidate


def safe_relative_path(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not str(value):
        raise ValueError(f"{label} must be a non-empty relative path")
    return safe_data_path(root, str(relative))


def repair_pattern_sha256(patterns: list[dict[str, Any]]) -> str:
    payload = []
    for pattern in patterns:
        entries = [{"raw": str(e["raw"]), "source_id": str(e["source_id"]), "task": str(e["task"])} for e in pattern["entries"]]
        entries.sort(key=canonical_json)
        payload.append({"entries": entries, "normalized": str(pattern["normalized"])})
    payload.sort(key=lambda item: item["normalized"])
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_repair_evidence(evidence, repo_root, data_dir, startup, matching, patterns) -> None:
    manifest = evidence["manifest"]
    repair, filt = manifest.get("repair_mixer"), manifest.get("contamination_filter")
    if repair is None and filt is None:
        evidence["repair_evidence"] = {"mode": "legacy_builder_corpus"}
        return
    if not isinstance(repair, dict) or not isinstance(filt, dict):
        add_evidence_error(evidence, "repair_mixer and contamination_filter must both be objects")
        return
    before = len(evidence["errors"])
    def eq(label, observed, expected):
        if observed != expected: add_evidence_error(evidence, f"{label}: {observed!r} != {expected!r}")
    script = repair.get("script") if isinstance(repair.get("script"), dict) else {}
    eq("scripts.repair_mixer", manifest.get("scripts", {}).get("repair_mixer"), script)
    eq("repair_mixer.version", repair.get("version"), script.get("version"))
    eq("repair_mixer.candidate_objects_reused", repair.get("candidate_objects_reused"), 54)
    try:
        path = safe_relative_path(repo_root, script.get("path"), "repair_mixer.script.path")
        if not path.is_file(): raise FileNotFoundError(path)
        eq("repair_mixer.script.sha256", sha256_file(path), script.get("sha256"))
    except Exception as exc: add_evidence_error(evidence, f"repair mixer script: {type(exc).__name__}: {exc}")
    eq("contamination_filter.status", filt.get("status"), "applied")
    eq("contamination_filter.eval_config_sha256", filt.get("eval_config_sha256"), startup["eval_config"]["sha256"])
    try:
        declared_eval = Path(str(filt.get("eval_config")))
        if not declared_eval.is_absolute():
            declared_eval = safe_relative_path(repo_root, filt.get("eval_config"), "contamination_filter.eval_config")
        eq("contamination_filter.eval_config", declared_eval.resolve(), Path(str(startup["eval_config"]["path"])).resolve())
    except Exception as exc:
        add_evidence_error(evidence, f"repair eval config path: {type(exc).__name__}: {exc}")
    policy = {"exact_min_normalized_chars": int(matching["exact_min_normalized_chars"]), "containment_min_normalized_chars": int(matching["containment_min_normalized_chars"]), "normalization": "NFKC_lower_keep_unicode_alnum", "empty_text": "reject_if_not_text.strip()", "piece_rechecks": ["original", "exact_prefix", "boundary_split"]}
    eq("contamination_filter.filter_policy", filt.get("filter_policy"), policy)
    pattern_sha = repair_pattern_sha256(patterns)
    eq("contamination_filter.pattern_sha256", filt.get("pattern_sha256"), pattern_sha)
    files = {}
    for key in ("benchmark_snapshot", "exclusion_ledger", "exclusion_snapshot"):
        try:
            path = safe_relative_path(data_dir, filt.get(key), f"contamination_filter.{key}")
            if not path.is_file(): raise FileNotFoundError(path)
            digest = sha256_file(path); eq(f"contamination_filter.{key}_sha256", filt.get(f"{key}_sha256"), digest)
            files[key] = {"path": str(path), "sha256": digest}
        except Exception as exc: add_evidence_error(evidence, f"repair {key}: {type(exc).__name__}: {exc}")
    candidate = evidence.get("candidate_manifest")
    candidate_value = candidate["value"] if candidate else {}
    eq("repair candidate manifest sha256", candidate["sha256"] if candidate else None, manifest.get("candidate_manifest_sha256"))
    sources_sha = manifest.get("sources_config_sha256")
    eq("candidate_manifest.sources_config_sha256", candidate_value.get("sources_config_sha256"), sources_sha)
    try:
        declared_sources = Path(str(manifest.get("sources_config")))
        if not declared_sources.is_absolute():
            declared_sources = safe_relative_path(
                repo_root,
                manifest.get("sources_config"),
                "manifest.sources_config",
            )
        path = declared_sources.resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        eq("manifest.sources_config_sha256", sha256_file(path), sources_sha)
        candidate_sources = Path(str(candidate_value.get("sources_config")))
        if not candidate_sources.is_absolute():
            candidate_sources = safe_relative_path(
                repo_root,
                candidate_value.get("sources_config"),
                "candidate_manifest.sources_config",
            )
        eq(
            "manifest.sources_config_path",
            path,
            candidate_sources.resolve(),
        )
    except Exception as exc: add_evidence_error(evidence, f"sources config binding: {type(exc).__name__}: {exc}")
    evidence["repair_evidence"] = {"mode": "repair_mixer_contamination_filter", "passed": len(evidence["errors"]) == before, "script": script, "sources_config_sha256": sources_sha, "candidate_manifest_sha256": manifest.get("candidate_manifest_sha256"), "candidate_roster_sha256": repair.get("candidate_roster_sha256"), "eval_config_sha256": filt.get("eval_config_sha256"), "filter_policy": filt.get("filter_policy"), "pattern_sha256": pattern_sha, "files": files}


def resolve_verification_report(
    explicit: Path | None,
    data_dir: Path,
    repo_root: Path,
    manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    if explicit is not None:
        candidates = [explicit.resolve()]
    else:
        candidates = [data_dir.resolve() / "verification.json"]
        candidates.extend(
            sorted(
                path.resolve()
                for path in (repo_root / "experiments").glob(
                    "**/pretrain_v1_verification.json"
                )
            )
        )
    existing = []
    matching = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        report = read_json_object(candidate)
        existing.append((candidate, report))
        if report.get("manifest_sha256") == manifest_sha256:
            matching.append((candidate, report))
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise ValueError(
            "no verification report is bound to current manifest; "
            f"examined={[str(path) for path, _ in existing]}"
        )
    raise ValueError(
        "multiple verification reports match current manifest: "
        f"{[str(path) for path, _ in matching]}"
    )


def add_evidence_error(
    evidence: dict[str, Any],
    message: str,
) -> None:
    evidence["errors"].append(message)


def load_build_evidence(
    args: argparse.Namespace,
    config: dict[str, Any],
    tokenizer_path: Path,
    repo_root: Path,
    train_paths: list[Path],
    validation_paths: list[Path],
    config_sha: str,
) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    manifest_path = (
        args.build_manifest.resolve()
        if args.build_manifest is not None
        else data_dir / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"build manifest is missing: {manifest_path}")
    manifest = read_json_object(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    tokenizer_fp = tokenizer_fingerprint(tokenizer_path)
    evidence: dict[str, Any] = {
        "errors": [],
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
        "tokenizer": tokenizer_fp,
        "shard_map": {},
        "expected_quotas": {},
        "declared_source_stats": {
            "train": defaultdict(Counter),
            "validation": defaultdict(Counter),
        },
        "observed_source_stats": {
            "train": defaultdict(Counter),
            "validation": defaultdict(Counter),
        },
        "runtime_error_counts": Counter(),
        "candidate_manifest": None,
    }

    def require_equal(label: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            add_evidence_error(
                evidence,
                f"{label}: {observed!r} != {expected!r}",
            )

    require_equal("manifest.schema_version", manifest.get("schema_version"), 2)
    require_equal(
        "manifest.status",
        manifest.get("status"),
        "pending_external_audit",
    )
    require_equal(
        "manifest.config_sha256",
        manifest.get("config_sha256"),
        config_sha,
    )
    require_equal(
        "manifest.sequence_length",
        manifest.get("sequence_length"),
        int(config["sequence_length"]),
    )
    require_equal(
        "manifest.tokenizer",
        manifest.get("tokenizer"),
        tokenizer_fp,
    )
    if not manifest.get("builder_version"):
        add_evidence_error(evidence, "manifest.builder_version is missing")
    require_equal(
        "manifest.external_audit.benchmark_contamination",
        manifest.get("external_audit", {}).get("benchmark_contamination"),
        "pending",
    )
    require_equal(
        "manifest.external_audit.required_before_release",
        manifest.get("external_audit", {}).get("required_before_release"),
        True,
    )
    if not manifest.get("fingerprint"):
        add_evidence_error(evidence, "manifest.fingerprint is missing")

    candidate_path_value = manifest.get("candidate_manifest")
    candidate_sha = manifest.get("candidate_manifest_sha256")
    if not candidate_path_value or not candidate_sha:
        add_evidence_error(evidence, "candidate manifest binding is missing")
    else:
        candidate_path = Path(str(candidate_path_value))
        if not candidate_path.is_absolute():
            candidate_path = (repo_root / candidate_path).resolve()
        try:
            candidate_manifest = read_json_object(candidate_path)
            observed_sha = sha256_file(candidate_path)
            require_equal(
                "candidate_manifest.sha256",
                observed_sha,
                candidate_sha,
            )
            require_equal(
                "candidate_manifest.status",
                candidate_manifest.get("status"),
                "materialized",
            )
            require_equal(
                "candidate_manifest.fingerprint",
                candidate_manifest.get("fingerprint"),
                manifest.get("candidate_fingerprint"),
            )
            require_equal(
                "candidate_manifest.config_sha256",
                candidate_manifest.get("config_sha256"),
                config_sha,
            )
            require_equal(
                "candidate_manifest.builder_version",
                candidate_manifest.get("builder_version"),
                manifest.get("builder_version"),
            )
            require_equal(
                "candidate_manifest.tokenizer",
                candidate_manifest.get("tokenizer"),
                tokenizer_fp,
            )
            evidence["candidate_manifest"] = {
                "path": candidate_path,
                "sha256": observed_sha,
                "value": candidate_manifest,
            }
        except Exception as exc:
            add_evidence_error(
                evidence,
                f"candidate manifest validation: {type(exc).__name__}: {exc}",
            )

    budgets = config.get("budgets", {})
    budget_by_split = {
        "train": int(budgets["full_loss_target_tokens"]),
        "validation": int(budgets["validation_loss_target_tokens"]),
    }
    paths_by_split = {
        "train": train_paths,
        "validation": validation_paths,
    }
    for split in ("train", "validation"):
        split_manifest = manifest.get(split)
        if not isinstance(split_manifest, dict):
            add_evidence_error(evidence, f"manifest.{split} is missing")
            continue
        budget = budget_by_split[split]
        quotas = {
            source: tokens
            for source, tokens in source_targets(config, budget).items()
            if tokens
        }
        evidence["expected_quotas"][split] = quotas
        require_equal(
            f"manifest.{split}.budget_loss_target_tokens",
            split_manifest.get("budget_loss_target_tokens"),
            budget,
        )
        selection = split_manifest.get("selection")
        if not isinstance(selection, dict):
            add_evidence_error(
                evidence,
                f"manifest.{split}.selection is missing",
            )
            selection = {}
        require_equal(
            f"manifest.{split}.selection.sources",
            sorted(selection),
            sorted(quotas),
        )
        for source, quota in quotas.items():
            selected = selection.get(source)
            if not isinstance(selected, dict):
                continue
            require_equal(
                f"manifest.{split}.selection.{source}.quota",
                selected.get("quota_loss_target_tokens"),
                quota,
            )
            require_equal(
                f"manifest.{split}.selection.{source}.loss_target_tokens",
                selected.get("loss_target_tokens"),
                quota,
            )
            require_equal(
                f"manifest.{split}.selection.{source}.status",
                selected.get("status"),
                "ok",
            )

        sidecars = split_manifest.get("shards")
        if not isinstance(sidecars, list):
            add_evidence_error(
                evidence,
                f"manifest.{split}.shards is missing",
            )
            sidecars = []
        manifest_names = [
            str(sidecar.get("file"))
            for sidecar in sidecars
            if isinstance(sidecar, dict)
        ]
        require_equal(
            f"manifest.{split}.shard_files",
            sorted(manifest_names),
            sorted(path.name for path in paths_by_split[split]),
        )
        for sidecar in sidecars:
            if not isinstance(sidecar, dict) or not sidecar.get("file"):
                add_evidence_error(
                    evidence,
                    f"manifest.{split} has invalid shard sidecar",
                )
                continue
            name = str(sidecar["file"])
            try:
                data_path = safe_data_path(data_dir, name)
                provenance_path = safe_data_path(
                    data_dir,
                    str(sidecar["provenance_file"]),
                )
                meta_path = safe_data_path(
                    data_dir,
                    str(sidecar["meta_file"]),
                )
                if not data_path.is_file():
                    raise FileNotFoundError(data_path)
                if not provenance_path.is_file():
                    raise FileNotFoundError(provenance_path)
                if not meta_path.is_file():
                    raise FileNotFoundError(meta_path)
                require_equal(
                    f"{name}.provenance_sha256",
                    sha256_file(provenance_path),
                    sidecar.get("provenance_sha256"),
                )
                require_equal(
                    f"{name}.provenance_size_bytes",
                    provenance_path.stat().st_size,
                    sidecar.get("provenance_size_bytes"),
                )
                require_equal(
                    f"{name}.meta_sha256",
                    sha256_file(meta_path),
                    sidecar.get("meta_sha256"),
                )
                meta = read_json_object(meta_path)
                expected_meta = {
                    key: value
                    for key, value in sidecar.items()
                    if key not in {"meta_file", "meta_sha256"}
                }
                require_equal(f"{name}.meta_content", meta, expected_meta)
                require_equal(
                    f"{name}.meta_sequence_length",
                    meta.get("sequence_length"),
                    int(config["sequence_length"]),
                )
                evidence["shard_map"][name] = {
                    "split": split,
                    "sidecar": sidecar,
                    "meta": meta,
                    "data_path": data_path,
                    "provenance_path": provenance_path,
                    "meta_path": meta_path,
                }
                for source, values in sidecar.get("sources", {}).items():
                    for key, value in values.items():
                        evidence["declared_source_stats"][split][source][
                            key
                        ] += int(value)
            except Exception as exc:
                add_evidence_error(
                    evidence,
                    f"{split}/{name} evidence: {type(exc).__name__}: {exc}",
                )

    verification_path, verification = resolve_verification_report(
        args.verification_report,
        data_dir,
        repo_root,
        manifest_sha,
    )
    verification_sha = sha256_file(verification_path)
    evidence["verification_path"] = verification_path
    evidence["verification_sha256"] = verification_sha
    evidence["verification"] = verification
    require_equal(
        "verification.status",
        verification.get("status"),
        "pending_external_audit",
    )
    require_equal(
        "verification.manifest_sha256",
        verification.get("manifest_sha256"),
        manifest_sha,
    )
    require_equal(
        "verification.external_audit",
        verification.get("external_audit", {}).get(
            "benchmark_contamination"
        ),
        "pending",
    )
    require_equal(
        "verification.loader_dry_run.status",
        verification.get("loader_dry_run", {}).get("status"),
        "ok",
    )
    recount = verification.get("independent_tokenizer_recount")
    if not isinstance(recount, dict):
        add_evidence_error(
            evidence,
            "verification independent_tokenizer_recount is missing",
        )
        recount = {}
    for split in ("train", "validation"):
        split_recount = recount.get(split)
        if not isinstance(split_recount, dict):
            add_evidence_error(
                evidence,
                f"verification recount {split} is missing",
            )
            continue
        require_equal(
            f"verification.{split}.status",
            split_recount.get("status"),
            "ok",
        )
        require_equal(
            f"verification.{split}.expected_loss_target_tokens",
            split_recount.get("expected_loss_target_tokens"),
            budget_by_split[split],
        )
        require_equal(
            f"verification.{split}.stats.loss_target_tokens",
            split_recount.get("stats", {}).get("loss_target_tokens"),
            budget_by_split[split],
        )
        expected_files = sorted(
            path.name for path in paths_by_split[split]
        )
        observed_files = sorted(
            str(shard.get("file"))
            for shard in split_recount.get("shards", [])
            if isinstance(shard, dict)
        )
        require_equal(
            f"verification.{split}.shard_files",
            observed_files,
            expected_files,
        )
    return evidence


def token_content_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def evidence_source_stats_json(
    value: dict[str, defaultdict[str, Counter[str]]],
) -> dict[str, Any]:
    return {
        split: {
            source: dict(stats)
            for source, stats in sorted(sources.items())
        }
        for split, sources in value.items()
    }


def evidence_public_report(evidence: dict[str, Any]) -> dict[str, Any]:
    candidate = evidence.get("candidate_manifest")
    return {
        "passed": not evidence["errors"],
        "errors": evidence["errors"],
        "builder_version": evidence["manifest"].get("builder_version"),
        "manifest": {
            "path": str(evidence["manifest_path"]),
            "sha256": evidence["manifest_sha256"],
            "fingerprint": evidence["manifest"].get("fingerprint"),
            "status": evidence["manifest"].get("status"),
        },
        "candidate_manifest": (
            {
                "path": str(candidate["path"]),
                "sha256": candidate["sha256"],
                "fingerprint": candidate["value"].get("fingerprint"),
            }
            if candidate
            else None
        ),
        "verification": {
            "path": str(evidence["verification_path"]),
            "sha256": evidence["verification_sha256"],
            "status": evidence["verification"].get("status"),
        },
        "tokenizer": evidence["tokenizer"],
        "repair_evidence": evidence.get("repair_evidence"),
        "expected_source_quotas": evidence["expected_quotas"],
        "declared_source_stats": evidence_source_stats_json(
            evidence["declared_source_stats"]
        ),
        "observed_source_stats": evidence_source_stats_json(
            evidence["observed_source_stats"]
        ),
        "runtime_error_counts": dict(evidence["runtime_error_counts"]),
    }


def new_stats() -> Counter[str]:
    return Counter(
        {
            "physical_lines": 0,
            "rows": 0,
            "valid_rows": 0,
            "blank_lines": 0,
            "invalid_utf8_lines": 0,
            "invalid_json_lines": 0,
            "invalid_schema_rows": 0,
            "empty_text_rows": 0,
            "loss_target_tokens": 0,
            "nonpad_input_tokens": 0,
            "padded_compute_tokens": 0,
            "exact_duplicate_chunks": 0,
            "train_validation_overlap_chunks": 0,
        }
    )


def merge_stats(target: Counter[str], source: Counter[str]) -> None:
    target.update(source)


def chunk_digest(token_ids: list[int], bos_id: int, eos_id: int) -> bytes:
    values = [bos_id, *token_ids, eos_id]
    packed = struct.pack(f"<{len(values)}I", *values)
    return hashlib.sha256(packed).digest()


def create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS train_chunks (
            digest BLOB NOT NULL,
            token_count INTEGER NOT NULL,
            first_shard TEXT NOT NULL,
            first_line INTEGER NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (digest, token_count)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS validation_chunks (
            digest BLOB NOT NULL,
            token_count INTEGER NOT NULL,
            first_shard TEXT NOT NULL,
            first_line INTEGER NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (digest, token_count)
        )
        """
    )
    return connection


def register_chunk(
    connection: sqlite3.Connection,
    split: str,
    digest: bytes,
    token_count: int,
    shard: str,
    line_number: int,
) -> tuple[bool, dict[str, Any] | None]:
    table = "train_chunks" if split == "train" else "validation_chunks"
    cursor = connection.execute(
        f"""
        INSERT OR IGNORE INTO {table}
            (digest, token_count, first_shard, first_line)
        VALUES (?, ?, ?, ?)
        """,
        (digest, token_count, shard, line_number),
    )
    duplicate = cursor.rowcount == 0
    if duplicate:
        connection.execute(
            f"""
            UPDATE {table}
            SET occurrences = occurrences + 1
            WHERE digest = ? AND token_count = ?
            """,
            (digest, token_count),
        )
    overlap: dict[str, Any] | None = None
    if split == "validation":
        row = connection.execute(
            """
            SELECT first_shard, first_line
            FROM train_chunks
            WHERE digest = ? AND token_count = ?
            """,
            (digest, token_count),
        ).fetchone()
        if row:
            overlap = {"train_shard": row[0], "train_line": row[1]}
    return duplicate, overlap


def record_runtime_evidence_error(
    evidence: dict[str, Any],
    kind: str,
    message: str,
) -> None:
    evidence["runtime_error_counts"][kind] += 1
    if len(evidence["errors"]) < 100:
        evidence["errors"].append(f"{kind}: {message}")


def validate_provenance_row(
    provenance: dict[str, Any] | None,
    token_ids: list[int],
    split: str,
    shard: str,
    line_number: int,
    sequence_length: int,
    evidence: dict[str, Any],
    shard_source_stats: defaultdict[str, Counter[str]],
) -> None:
    if not isinstance(provenance, dict):
        record_runtime_evidence_error(
            evidence,
            "invalid_provenance",
            f"{split}/{shard}:{line_number}",
        )
        return
    required = {
        "source",
        "content_sha256",
        "loss_target_tokens",
        "nonpad_input_tokens",
        "padded_compute_tokens",
        "raw_tokens",
        "text_tokens",
    }
    missing = sorted(required - set(provenance))
    if missing:
        record_runtime_evidence_error(
            evidence,
            "provenance_schema",
            f"{split}/{shard}:{line_number} missing={missing}",
        )
        return
    source = str(provenance["source"])
    expected_sources = evidence["expected_quotas"].get(split, {})
    if source not in expected_sources:
        record_runtime_evidence_error(
            evidence,
            "unknown_provenance_source",
            f"{split}/{shard}:{line_number} source={source}",
        )
    expected = {
        "content_sha256": token_content_sha256(token_ids),
        "loss_target_tokens": len(token_ids) + 1,
        "nonpad_input_tokens": len(token_ids) + 2,
        "padded_compute_tokens": sequence_length,
        "text_tokens": len(token_ids),
    }
    for key, value in expected.items():
        observed = provenance.get(key)
        if key != "content_sha256":
            try:
                observed = int(observed)
            except (TypeError, ValueError):
                pass
        if observed != value:
            record_runtime_evidence_error(
                evidence,
                "provenance_binding",
                f"{split}/{shard}:{line_number} {key}={observed!r} "
                f"expected={value!r}",
            )
    try:
        raw_tokens = int(provenance["raw_tokens"])
    except (TypeError, ValueError):
        raw_tokens = -1
        record_runtime_evidence_error(
            evidence,
            "provenance_schema",
            f"{split}/{shard}:{line_number} raw_tokens is invalid",
        )
    values = {
        "rows": 1,
        "loss_target_tokens": expected["loss_target_tokens"],
        "nonpad_input_tokens": expected["nonpad_input_tokens"],
        "padded_compute_tokens": expected["padded_compute_tokens"],
        "text_tokens": expected["text_tokens"],
        "raw_tokens": raw_tokens,
    }
    for key, value in values.items():
        evidence["observed_source_stats"][split][source][key] += value
        shard_source_stats[source][key] += value


def tokenize_pending(
    pending: list[
        tuple[str, str, int, Counter[str], dict[str, Any] | None]
    ],
    tokenizer: Any,
    max_text_tokens: int,
    sequence_length: int,
    split: str,
    connection: sqlite3.Connection,
    duplicate_examples: list[dict[str, Any]],
    overlap_examples: list[dict[str, Any]],
    evidence: dict[str, Any],
    shard_source_stats: defaultdict[str, Counter[str]],
) -> None:
    if not pending:
        return
    texts = [item[0] for item in pending]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        max_length=max_text_tokens,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    for (
        text,
        shard,
        line_number,
        stats,
        provenance,
    ), token_ids in zip(pending, encoded):
        token_ids = list(token_ids)
        loss_target = len(token_ids) + 1
        nonpad = len(token_ids) + 2
        stats["loss_target_tokens"] += loss_target
        stats["nonpad_input_tokens"] += nonpad
        stats["padded_compute_tokens"] += sequence_length
        validate_provenance_row(
            provenance,
            token_ids,
            split,
            shard,
            line_number,
            sequence_length,
            evidence,
            shard_source_stats,
        )
        digest = chunk_digest(
            token_ids,
            int(tokenizer.bos_token_id),
            int(tokenizer.eos_token_id),
        )
        duplicate, overlap = register_chunk(
            connection,
            split,
            digest,
            nonpad,
            shard,
            line_number,
        )
        if duplicate:
            stats["exact_duplicate_chunks"] += 1
            if len(duplicate_examples) < MAX_EXAMPLES:
                duplicate_examples.append(
                    {
                        "split": split,
                        "shard": shard,
                        "line": line_number,
                        "text": text[:300],
                    }
                )
        if overlap:
            stats["train_validation_overlap_chunks"] += 1
            if len(overlap_examples) < MAX_EXAMPLES:
                overlap_examples.append(
                    {
                        "validation_shard": shard,
                        "validation_line": line_number,
                        **overlap,
                        "text": text[:300],
                    }
                )
    pending.clear()


def audit_shard(
    path: Path,
    split: str,
    tokenizer: Any,
    sequence_length: int,
    strict_text_only: bool,
    batch_size: int,
    connection: sqlite3.Connection,
    contamination: ContaminationAudit,
    near_audit: NearAudit,
    near_mode: str,
    sampler: StratifiedBottomK | None,
    progress_every: int,
    duplicate_examples: list[dict[str, Any]],
    overlap_examples: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], Counter[str]]:
    stats = new_stats()
    digest = hashlib.sha256()
    pending: list[
        tuple[str, str, int, Counter[str], dict[str, Any] | None]
    ] = []
    shard_source_stats: defaultdict[str, Counter[str]] = defaultdict(Counter)
    max_text_tokens = sequence_length - 2
    evidence_shard = evidence["shard_map"].get(path.name)
    provenance_handle = None
    if evidence_shard is not None:
        provenance_handle = evidence_shard["provenance_path"].open("rb")
    else:
        record_runtime_evidence_error(
            evidence,
            "missing_shard_evidence",
            f"{split}/{path.name}",
        )
    try:
        with path.open("rb") as handle:
            pairs = zip_longest(
                handle,
                provenance_handle if provenance_handle is not None else (),
                fillvalue=None,
            )
            for line_number, pair in enumerate(pairs, 1):
                raw_line, raw_provenance = pair
                if raw_line is None:
                    record_runtime_evidence_error(
                        evidence,
                        "provenance_alignment",
                        f"{split}/{path.name}:{line_number} extra provenance row",
                    )
                    continue
                digest.update(raw_line)
                stats["physical_lines"] += 1
                provenance: dict[str, Any] | None = None
                if raw_provenance is None:
                    record_runtime_evidence_error(
                        evidence,
                        "provenance_alignment",
                        f"{split}/{path.name}:{line_number} missing provenance row",
                    )
                else:
                    try:
                        decoded_provenance = raw_provenance.decode("utf-8")
                        value = json.loads(decoded_provenance)
                        if isinstance(value, dict):
                            provenance = value
                        else:
                            raise ValueError("provenance root is not object")
                    except Exception as exc:
                        record_runtime_evidence_error(
                            evidence,
                            "invalid_provenance",
                            f"{split}/{path.name}:{line_number} "
                            f"{type(exc).__name__}: {exc}",
                        )
                if not raw_line.strip():
                    stats["blank_lines"] += 1
                    continue
                stats["rows"] += 1
                try:
                    decoded = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    stats["invalid_utf8_lines"] += 1
                    continue
                try:
                    row = json.loads(decoded)
                except json.JSONDecodeError:
                    stats["invalid_json_lines"] += 1
                    continue
                valid_schema = (
                    isinstance(row, dict)
                    and isinstance(row.get("text"), str)
                    and (not strict_text_only or set(row) == {"text"})
                )
                if not valid_schema:
                    stats["invalid_schema_rows"] += 1
                    continue
                text = row["text"]
                stats["valid_rows"] += 1
                if not text.strip():
                    stats["empty_text_rows"] += 1
                normalized_document = normalize(text)
                contamination.audit(
                    normalized_document,
                    text,
                    split,
                    path.name,
                    line_number,
                )
                if near_mode == "full":
                    near_audit.audit_document(
                        normalized_document,
                        text,
                        split,
                        path.name,
                        line_number,
                    )
                else:
                    assert sampler is not None
                    sampler.offer(
                        normalized_document,
                        text,
                        split,
                        path.name,
                        line_number,
                    )
                pending.append(
                    (text, path.name, line_number, stats, provenance)
                )
                if len(pending) >= batch_size:
                    tokenize_pending(
                        pending,
                        tokenizer,
                        max_text_tokens,
                        sequence_length,
                        split,
                        connection,
                        duplicate_examples,
                        overlap_examples,
                        evidence,
                        shard_source_stats,
                    )
                if progress_every and line_number % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "shard": path.name,
                                "line": line_number,
                                "loss_target_tokens": stats[
                                    "loss_target_tokens"
                                ],
                            }
                        ),
                        flush=True,
                    )
        tokenize_pending(
            pending,
            tokenizer,
            max_text_tokens,
            sequence_length,
            split,
            connection,
            duplicate_examples,
            overlap_examples,
            evidence,
            shard_source_stats,
        )
    finally:
        if provenance_handle is not None:
            provenance_handle.close()
    connection.commit()
    shard_report = {
        "path": str(path),
        "name": path.name,
        "split": split,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "source_stats": {
            source: dict(values)
            for source, values in sorted(shard_source_stats.items())
        },
        **dict(stats),
    }
    if evidence_shard is not None:
        sidecar = evidence_shard["sidecar"]
        if shard_report["sha256"] != sidecar.get("sha256"):
            record_runtime_evidence_error(
                evidence,
                "shard_sha",
                f"{path.name}: {shard_report['sha256']} != "
                f"{sidecar.get('sha256')}",
            )
        if int(shard_report["bytes"]) != int(
            sidecar.get("size_bytes", -1)
        ):
            record_runtime_evidence_error(
                evidence,
                "shard_size",
                f"{path.name}: {shard_report['bytes']} != "
                f"{sidecar.get('size_bytes')}",
            )
        expected_stats = sidecar.get("stats", {})
        observed_stats = {
            "rows": stats["valid_rows"],
            "loss_target_tokens": stats["loss_target_tokens"],
            "nonpad_input_tokens": stats["nonpad_input_tokens"],
            "padded_compute_tokens": stats["padded_compute_tokens"],
            "text_tokens": sum(
                values["text_tokens"]
                for values in shard_source_stats.values()
            ),
            "raw_tokens": sum(
                values["raw_tokens"]
                for values in shard_source_stats.values()
            ),
        }
        for key, value in observed_stats.items():
            if int(expected_stats.get(key, -1)) != int(value):
                record_runtime_evidence_error(
                    evidence,
                    "shard_meta_stats",
                    f"{path.name} {key}: {value} != "
                    f"{expected_stats.get(key)}",
                )
        declared_sources = {
            source: dict(values)
            for source, values in sidecar.get("sources", {}).items()
        }
        if shard_report["source_stats"] != declared_sources:
            record_runtime_evidence_error(
                evidence,
                "shard_meta_sources",
                f"{path.name} observed source stats differ",
            )
    return shard_report, stats



def finalize_build_evidence(
    evidence: dict[str, Any],
    totals: dict[str, Counter[str]],
) -> None:
    manifest = evidence["manifest"]
    verification = evidence["verification"]
    for split in ("train", "validation"):
        observed_sources = {
            source: dict(values)
            for source, values in sorted(
                evidence["observed_source_stats"][split].items()
            )
        }
        declared_sources = {
            source: dict(values)
            for source, values in sorted(
                evidence["declared_source_stats"][split].items()
            )
        }
        if observed_sources != declared_sources:
            add_evidence_error(
                evidence,
                f"{split} provenance source totals differ from manifest/meta",
            )
        quotas = evidence["expected_quotas"].get(split, {})
        observed_loss = {
            source: int(values.get("loss_target_tokens", 0))
            for source, values in observed_sources.items()
            if int(values.get("loss_target_tokens", 0))
        }
        if observed_loss != quotas:
            add_evidence_error(
                evidence,
                f"{split} exact source quotas: {observed_loss} != {quotas}",
            )
        manifest_stats = manifest.get(split, {}).get("stats", {})
        audit_stats = {
            "rows": int(totals[split]["valid_rows"]),
            "loss_target_tokens": int(
                totals[split]["loss_target_tokens"]
            ),
            "nonpad_input_tokens": int(
                totals[split]["nonpad_input_tokens"]
            ),
            "padded_compute_tokens": int(
                totals[split]["padded_compute_tokens"]
            ),
            "text_tokens": sum(
                int(values.get("text_tokens", 0))
                for values in observed_sources.values()
            ),
            "raw_tokens": sum(
                int(values.get("raw_tokens", 0))
                for values in observed_sources.values()
            ),
        }
        for key, value in audit_stats.items():
            if int(manifest_stats.get(key, -1)) != value:
                add_evidence_error(
                    evidence,
                    f"manifest {split} stats {key}: "
                    f"{manifest_stats.get(key)} != {value}",
                )
        recount = verification.get(
            "independent_tokenizer_recount", {}
        ).get(split, {})
        recount_stats = recount.get("stats", {})
        for key in (
            "rows",
            "loss_target_tokens",
            "nonpad_input_tokens",
            "padded_compute_tokens",
            "text_tokens",
        ):
            if int(recount_stats.get(key, -1)) != audit_stats[key]:
                add_evidence_error(
                    evidence,
                    f"verification {split} stats {key}: "
                    f"{recount_stats.get(key)} != {audit_stats[key]}",
                )


def active_source_count(config: dict[str, Any]) -> int:
    sources: set[str] = set()
    for group in config.get("active_v1_mix", {}).values():
        sources.update(str(source) for source in group.get("sources", []))
    return max(len(sources), 1)


def budget_gate(
    observed: int,
    target: int | None,
    max_overshoot: int,
) -> dict[str, Any]:
    if target is None:
        return {
            "enabled": False,
            "passed": True,
            "observed": observed,
            "target": None,
        }
    return {
        "enabled": True,
        "passed": target <= observed <= target + max_overshoot,
        "observed": observed,
        "target": target,
        "max_overshoot": max_overshoot,
        "delta": observed - target,
    }


def dataset_fingerprint(shards: list[dict[str, Any]]) -> str:
    payload = [
        {
            "name": shard["name"],
            "split": shard["split"],
            "bytes": shard["bytes"],
            "sha256": shard["sha256"],
        }
        for shard in shards
    ]
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def benchmark_matching_report(
    contamination: ContaminationAudit,
    near_audit: NearAudit,
) -> dict[str, Any]:
    return {
        "count_unit": "unique_normalized_pattern_document_pair",
        "source_count_unit": "unique_source_per_normalized_pattern_document_pair",
        "source_duplicate_rows_collapsed": True,
        "example_policy": {
            "exact_and_containment": (
                "limited independently per source per kind"
            ),
            "near_duplicate_overlap": (
                "limited independently from exact and containment per source"
            ),
            "max_examples_per_source_per_kind": contamination.max_examples,
            "near_max_examples_per_source": near_audit.max_examples,
        },
        "totals": {
            "exact_overlap": int(
                contamination.counts["exact_overlap"]
            ),
            "containment_overlap": int(
                contamination.counts["containment_overlap"]
            ),
            "near_duplicate_overlap": int(
                near_audit.counts["near_duplicate_overlap"]
            ),
            "exact_containment_documents_audited": int(
                contamination.counts["documents_audited"]
            ),
            "near_documents_audited": int(
                near_audit.counts["documents_audited"]
            ),
        },
        "exact_scope": {
            "min_normalized_chars": contamination.exact_min_chars,
            "patterns_indexed": contamination.exact_patterns_indexed,
            "patterns_skipped_short": (
                contamination.exact_patterns_skipped_short
            ),
            "boundary_note": (
                "Exact equality indexes every nonempty normalized benchmark "
                "query, including queries shorter than the containment boundary."
            ),
        },
        "containment_scope": {
            "min_normalized_chars": contamination.containment_min_chars,
            "patterns_indexed": contamination.containment_patterns_indexed,
            "patterns_skipped_short": (
                contamination.containment_patterns_skipped_short
            ),
            "boundary_note": (
                "Query-in-document containment indexes only normalized benchmark "
                "queries at or above min_normalized_chars; shorter queries are "
                "skipped to avoid ubiquitous-substring false positives."
            ),
        },
    }


def run(
    args: argparse.Namespace,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    startup: dict[str, dict[str, Any]],
    marker_path: Path,
) -> int:
    if args.tokenizer_batch_size <= 0:
        raise ValueError("--tokenizer-batch-size must be positive")
    matching = matching_contract(eval_config)
    auditor_path = Path(str(startup["auditor"]["path"]))
    auditor_sha256 = str(startup["auditor"]["sha256"])
    repo_root = Path(str(startup["config"]["path"])).parents[3]
    tokenizer_path = (args.tokenizer or (repo_root / config["tokenizer"])).resolve()
    sequence_length = int(config["sequence_length"])
    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3")

    train_paths = sorted(args.data_dir.glob("train-*.jsonl"))
    validation_paths = sorted(args.data_dir.glob("validation-*.jsonl"))
    all_paths = train_paths + validation_paths
    if not all_paths:
        raise FileNotFoundError(f"no train/validation JSONL shards under {args.data_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("MiniMind tokenizer must define BOS and EOS")

    evidence = load_build_evidence(
        args,
        config,
        tokenizer_path,
        repo_root,
        train_paths,
        validation_paths,
        str(startup["config"]["sha256"]),
    )

    patterns, eval_sources, eval_load_errors = load_eval_queries(
        eval_config,
        args.cache_dir,
    )
    eval_pin_errors = validate_pinned_eval_sources(
        eval_config,
        eval_sources,
        eval_load_errors,
        args.allow_test_eval_config,
    )
    validate_repair_evidence(evidence, repo_root, args.data_dir.resolve(), startup, matching, patterns)
    contamination = ContaminationAudit(patterns, eval_config, matching)
    near_audit = NearAudit(patterns, eval_config)
    near_settings = eval_config["near_duplicate"]
    sampler = (
        StratifiedBottomK(
            per_shard=args.near_sample_per_shard,
            seed=int(near_settings["seed"]),
            min_chars=int(near_settings["min_normalized_chars"]),
        )
        if args.near_mode == "stratified"
        else None
    )

    temporary_db = args.sqlite_path is None
    if temporary_db:
        descriptor, raw_db_path = tempfile.mkstemp(
            prefix="minimind-pretrain-audit-", suffix=".sqlite3"
        )
        os.close(descriptor)
        db_path = Path(raw_db_path)
    else:
        db_path = args.sqlite_path.resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()
    connection = create_database(db_path)

    shard_reports: list[dict[str, Any]] = []
    totals: dict[str, Counter[str]] = {
        "train": new_stats(),
        "validation": new_stats(),
        "all": new_stats(),
    }
    duplicate_examples: list[dict[str, Any]] = []
    overlap_examples: list[dict[str, Any]] = []
    try:
        for split, paths in (("train", train_paths), ("validation", validation_paths)):
            for path in paths:
                shard_report, stats = audit_shard(
                    path=path,
                    split=split,
                    tokenizer=tokenizer,
                    sequence_length=sequence_length,
                    strict_text_only=bool(
                        config.get("sharding", {}).get("final_rows_text_only", True)
                    ),
                    batch_size=args.tokenizer_batch_size,
                    connection=connection,
                    contamination=contamination,
                    near_audit=near_audit,
                    near_mode=args.near_mode,
                    sampler=sampler,
                    progress_every=args.progress_every,
                    duplicate_examples=duplicate_examples,
                    overlap_examples=overlap_examples,
                    evidence=evidence,
                )
                shard_reports.append(shard_report)
                merge_stats(totals[split], stats)
                merge_stats(totals["all"], stats)
                print(
                    json.dumps(
                        {
                            "audited": path.name,
                            "sha256": shard_report["sha256"],
                            "valid_rows": stats["valid_rows"],
                            "loss_target_tokens": stats["loss_target_tokens"],
                        }
                    ),
                    flush=True,
                )
        if sampler is not None:
            for normalized_document, raw, split, shard, line_number in sampler.documents():
                near_audit.audit_document(
                    normalized_document,
                    raw,
                    split,
                    shard,
                    line_number,
                )
    finally:
        connection.close()
        if temporary_db:
            for candidate in (
                db_path,
                Path(str(db_path) + "-wal"),
                Path(str(db_path) + "-shm"),
            ):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass

    finalize_build_evidence(evidence, totals)
    data_quality = config.get("quality_gates", {})
    eval_gates = eval_config.get("quality_gates", {})
    invalid_total = (
        totals["all"]["invalid_utf8_lines"]
        + totals["all"]["invalid_json_lines"]
        + totals["all"]["invalid_schema_rows"]
    )
    empty_total = totals["all"]["blank_lines"] + totals["all"]["empty_text_rows"]
    denominator = max(totals["all"]["physical_lines"], 1)
    expected_train_shards = config.get("sharding", {}).get("num_train_shards")
    source_count = active_source_count(config)
    overshoot_per_source = int(
        data_quality.get("per_source_overshoot_tokens_max", sequence_length - 1)
    )
    max_budget_overshoot = source_count * overshoot_per_source
    budgets = config.get("budgets", {})
    train_budget = budget_gate(
        totals["train"]["loss_target_tokens"],
        int(budgets["full_loss_target_tokens"])
        if budgets.get("full_loss_target_tokens") is not None
        else None,
        max_budget_overshoot,
    )
    validation_budget = budget_gate(
        totals["validation"]["loss_target_tokens"],
        int(budgets["validation_loss_target_tokens"])
        if budgets.get("validation_loss_target_tokens") is not None
        else None,
        max_budget_overshoot,
    )

    checks = {
        "train_shards_present": bool(train_paths),
        "validation_shards_present": bool(validation_paths),
        "train_shard_count": (
            True
            if expected_train_shards is None
            else len(train_paths) == int(expected_train_shards)
        ),
        "invalid_rate": invalid_total / denominator
        <= float(data_quality.get("invalid_json_rate_max", 0.0)),
        "empty_rate": empty_total / denominator
        <= float(data_quality.get("empty_text_rate_max", 0.0)),
        "train_exact_duplicate_chunks": totals["train"]["exact_duplicate_chunks"]
        <= int(data_quality.get("exact_training_visible_duplicate_max", 0)),
        "train_validation_exact_overlap": totals["validation"][
            "train_validation_overlap_chunks"
        ]
        <= int(data_quality.get("train_validation_exact_overlap_max", 0)),
        "build_evidence_chain": not evidence["errors"],
        "eval_sources_loaded_and_pinned": (
            not eval_pin_errors and len(patterns) > 0
        ),
        "eval_exact_overlap": contamination.counts["exact_overlap"]
        <= int(
            eval_gates.get(
                "exact_overlap_max",
                data_quality.get("exact_eval_overlap_max", 0),
            )
        ),
        "eval_query_in_document_containment": contamination.counts[
            "containment_overlap"
        ]
        <= int(eval_gates.get("containment_overlap_max", 0)),
        "eval_near_duplicate": near_audit.counts["near_duplicate_overlap"]
        <= int(eval_gates.get("near_duplicate_overlap_max", 0)),
        "train_budget": bool(train_budget["passed"]),
        "validation_budget": bool(validation_budget["passed"]),
    }
    data_gates_passed = all(checks.values())
    fingerprint = dataset_fingerprint(shard_reports)
    marker_required = bool(data_quality.get("success_marker_required", True))
    passed = data_gates_passed

    near_scope: dict[str, Any] = {
        "mode": args.near_mode,
        "all_long_eval_queries_indexed": len(near_audit.grams),
        "documents_audited": near_audit.counts["documents_audited"],
        "is_full_pretrain_document_scan": args.near_mode == "full",
    }
    if sampler is not None:
        near_scope.update(
            {
                "sampling_method": "deterministic SHA256 bottom-k independently per physical shard",
                "sample_per_shard": args.near_sample_per_shard,
                "eligible_documents_per_shard": dict(sampler.eligible),
                "eligible_documents": sum(sampler.eligible.values()),
                "boundary_note": (
                    "Near-duplicate MinHash is not a full corpus scan: every eligible "
                    "long benchmark query is indexed, while only deterministic bottom-k "
                    "eligible documents from each train/validation shard are queried. "
                    "Exact equality and query-in-document containment still scan every "
                    "valid pretraining document."
                ),
            }
        )
    else:
        near_scope["boundary_note"] = (
            "Near-duplicate MinHash scans every valid pretraining document whose "
            "normalized length meets min_normalized_chars."
        )

    matching_report = benchmark_matching_report(contamination, near_audit)
    source_contamination = []
    near_examples = near_audit.examples
    exact_examples = contamination.examples
    for source in eval_sources:
        source_id = source["id"]
        source_contamination.append(
            {
                **source,
                **dict(contamination.source_counts[source_id]),
                **dict(near_audit.source_counts[source_id]),
                "examples": exact_examples[source_id] + near_examples[source_id],
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "auditor_version": AUDITOR_VERSION,
        "created_at": utc_now(),
        "passed": passed,
        "data_gates_passed_before_success_marker": data_gates_passed,
        "build_evidence": evidence_public_report(evidence),
        "fingerprints": {
            "config_sha256": startup["config"]["sha256"],
            "auditor": {
                "path": str(auditor_path),
                "version": AUDITOR_VERSION,
                "sha256": auditor_sha256,
            },
            "eval_config_sha256": startup["eval_config"]["sha256"],
            "tokenizer": evidence["tokenizer"],
        },
        "input": {
            "data_dir": str(args.data_dir.resolve()),
            "config": str(startup["config"]["path"]),
            "eval_config": str(startup["eval_config"]["path"]),
            "tokenizer": str(tokenizer_path),
            "sequence_length": sequence_length,
            "max_text_tokens": sequence_length - 2,
            "tokenizer_class": tokenizer.__class__.__name__,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "token_accounting": {
            "primary_metric": "loss_target_tokens",
            "formula": {
                "loss_target_tokens": "len(truncated_raw_tokens) + 1 EOS",
                "nonpad_input_tokens": "len(truncated_raw_tokens) + BOS + EOS",
                "padded_compute_tokens": "sequence_length per valid row",
            },
            "implementation_alignment": (
                "Matches dataset/lm_dataset.py PretrainDataset with "
                "add_special_tokens=False, truncation at sequence_length-2, "
                "then BOS/EOS and right padding."
            ),
            "chunk_digest": "SHA256 over BOS/raw/EOS token ids encoded uint32 little-endian",
        },
        "totals": {key: dict(value) for key, value in totals.items()},
        "budgets": {
            "source_count_for_max_overshoot": source_count,
            "max_aggregate_overshoot": max_budget_overshoot,
            "train": train_budget,
            "validation": validation_budget,
        },
        "shards": shard_reports,
        "dataset_fingerprint": fingerprint,
        "duplicates": {
            "train_exact_duplicate_chunks": totals["train"][
                "exact_duplicate_chunks"
            ],
            "validation_exact_duplicate_chunks": totals["validation"][
                "exact_duplicate_chunks"
            ],
            "train_validation_overlap_chunks": totals["validation"][
                "train_validation_overlap_chunks"
            ],
            "duplicate_examples": duplicate_examples,
            "overlap_examples": overlap_examples,
            "storage": "temporary SQLite exact-digest index",
        },
        "benchmark_contamination": {
            "endpoint": eval_config.get("metadata_endpoint"),
            "matching_contract": matching,
            **matching_report,
            "task_alignment": eval_config.get("task_alignment"),
            "eval_patterns_unique": len(patterns),
            "eval_load_errors": eval_load_errors,
            "pin_validation_errors": eval_pin_errors,
            "formal_pinned_snapshot_enforced": not args.allow_test_eval_config,
            "split_counts": {
                split: dict(counts)
                for split, counts in contamination.split_counts.items()
            },
            "near_scope": near_scope,
            "sources": source_contamination,
        },
        "checks": checks,
        "success_marker": {
            "path": str(marker_path),
            "required": marker_required,
            "eligible": passed,
            "write_order": "atomic audit report first, then atomic marker",
            "read_back_bindings": [
                "audit_report_sha256",
                "auditor_sha256",
                "builder_version",
                "candidate_manifest_sha256",
                "config_sha256",
                "eval_config_sha256",
                "tokenizer_fingerprint_sha256",
                "manifest_sha256",
                "manifest_fingerprint",
                "verification_sha256",
                "dataset_fingerprint",
                "repair_evidence",
                "benchmark_near_mode",
                "benchmark_near_sample_per_shard",
            ],
        },
    }

    # The release marker must never exist for an audit whose complete report was
    # not durably and atomically published first.
    verify_startup_fingerprints(startup)
    atomic_json(args.output, report)
    report_sha256 = sha256_file(args.output)
    marker_status = "not_written_data_gates_failed"
    if passed and marker_required and not args.no_write_success:
        marker = {
            "schema_version": 2,
            "status": "accepted",
            "accepted_at": utc_now(),
            "audit_report": {
                "path": str(args.output.resolve()),
                "sha256": report_sha256,
            },
            "auditor": {
                "name": "audit_pretrain_v1.py",
                "path": str(auditor_path),
                "version": AUDITOR_VERSION,
                "sha256": startup["auditor"]["sha256"],
            },
            "builder": {
                "version": evidence["manifest"].get("builder_version"),
                "candidate_manifest_sha256": (
                    evidence["candidate_manifest"]["sha256"]
                    if evidence.get("candidate_manifest")
                    else None
                ),
            },
            "config": {
                "path": str(startup["config"]["path"]),
                "sha256": startup["config"]["sha256"],
                "eval_path": str(startup["eval_config"]["path"]),
                "eval_sha256": startup["eval_config"]["sha256"],
            },
            "tokenizer": {
                "fingerprint": evidence["tokenizer"],
                "fingerprint_sha256": hashlib.sha256(
                    canonical_json(evidence["tokenizer"]).encode("utf-8")
                ).hexdigest(),
            },
            "manifest": {
                "path": str(evidence["manifest_path"]),
                "sha256": evidence["manifest_sha256"],
                "fingerprint": evidence["manifest"].get("fingerprint"),
            },
            "verification": {
                "path": str(evidence["verification_path"]),
                "sha256": evidence["verification_sha256"],
            },
            "dataset_fingerprint": fingerprint,
            "repair_evidence": {
                "script_sha256": evidence.get("repair_evidence", {}).get("script", {}).get("sha256"),
                "sources_config_sha256": evidence.get("repair_evidence", {}).get("sources_config_sha256"),
                "candidate_manifest_sha256": evidence.get("repair_evidence", {}).get("candidate_manifest_sha256"),
                "candidate_roster_sha256": evidence.get("repair_evidence", {}).get("candidate_roster_sha256"),
                "eval_config_sha256": evidence.get("repair_evidence", {}).get("eval_config_sha256"),
                "pattern_sha256": evidence.get("repair_evidence", {}).get("pattern_sha256"),
                "benchmark_snapshot_sha256": evidence.get("repair_evidence", {}).get("files", {}).get("benchmark_snapshot", {}).get("sha256"),
                "exclusion_ledger_sha256": evidence.get("repair_evidence", {}).get("files", {}).get("exclusion_ledger", {}).get("sha256"),
                "exclusion_snapshot_sha256": evidence.get("repair_evidence", {}).get("files", {}).get("exclusion_snapshot", {}).get("sha256"),
            },
            "benchmark_audit": {
                "formal_pinned_snapshot": not args.allow_test_eval_config,
                "near_mode": args.near_mode,
                "near_sample_per_shard": (
                    args.near_sample_per_shard
                    if args.near_mode == "stratified"
                    else None
                ),
            },
        }
        tokenizer_fingerprint_sha256 = hashlib.sha256(
            canonical_json(evidence["tokenizer"]).encode("utf-8")
        ).hexdigest()
        expected_bindings = {
            "status": "accepted",
            "audit_report_path": str(args.output.resolve()),
            "audit_report_sha256": sha256_file(args.output),
            "auditor_path": str(auditor_path),
            "auditor_version": AUDITOR_VERSION,
            "auditor_sha256": startup["auditor"]["sha256"],
            "manifest_path": str(evidence["manifest_path"]),
            "manifest_sha256": sha256_file(evidence["manifest_path"]),
            "manifest_fingerprint": evidence["manifest"].get("fingerprint"),
            "verification_path": str(evidence["verification_path"]),
            "verification_sha256": sha256_file(
                evidence["verification_path"]
            ),
            "builder_version": evidence["manifest"].get("builder_version"),
            "candidate_manifest_sha256": (
                evidence["candidate_manifest"]["sha256"]
                if evidence.get("candidate_manifest")
                else None
            ),
            "config_path": str(startup["config"]["path"]),
            "config_sha256": startup["config"]["sha256"],
            "eval_config_path": str(startup["eval_config"]["path"]),
            "eval_config_sha256": startup["eval_config"]["sha256"],
            "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
            "dataset_fingerprint": fingerprint,
            "repair_evidence": marker["repair_evidence"],
            "benchmark_near_mode": args.near_mode,
            "benchmark_near_sample_per_shard": (
                args.near_sample_per_shard
                if args.near_mode == "stratified"
                else None
            ),
        }
        try:
            atomic_json(marker_path, marker)
            validate_success_marker(marker_path, expected_bindings)
        except Exception:
            marker_path.unlink(missing_ok=True)
            raise
        marker_status = "written_and_verified"
    elif passed and marker_required:
        marker_status = "eligible_but_disabled"
    elif passed:
        marker_status = "not_required"

    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": report_sha256,
                "passed": passed,
                "failed_checks": [
                    name for name, value in checks.items() if not value
                ],
                "dataset_fingerprint": fingerprint,
                "success_marker": marker_status,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if passed else 1


def fatal_startup_fingerprints(
    startup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        name: {
            "path": record.get("path"),
            "sha256": record.get("sha256"),
        }
        for name, record in startup.items()
    }


def main() -> int:
    args = parse_args()
    marker_path = success_marker_path(args)
    startup = initial_startup_fingerprints(args)
    marker_status = "preserved_no_write_success"
    try:
        if not args.no_write_success:
            marker_status = ensure_success_marker_absent(marker_path)
        config, eval_config = load_startup_inputs(startup)
        exit_code = run(
            args,
            config,
            eval_config,
            startup,
            marker_path,
        )
        if exit_code != 0 and not args.no_write_success:
            marker_status = ensure_success_marker_absent(marker_path)
        return exit_code
    except Exception as exc:
        if not args.no_write_success:
            try:
                marker_status = ensure_success_marker_absent(marker_path)
            except Exception as marker_exc:
                marker_status = (
                    "absence_check_failed: "
                    f"{type(marker_exc).__name__}: {marker_exc}"
                )
        failure = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "auditor_version": AUDITOR_VERSION,
            "created_at": utc_now(),
            "passed": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "startup_fingerprints": fatal_startup_fingerprints(startup),
            "success_marker": {
                "path": str(marker_path),
                "status": marker_status,
            },
        }
        try:
            atomic_json(args.output, failure)
        except Exception:
            pass
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())