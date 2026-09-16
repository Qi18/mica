#!/usr/bin/env python3
"""Build a deterministic, resumable MiniMind Pretrain v1 corpus.

The final dataset remains intentionally incomplete until the separately maintained
benchmark-contamination audit is attached.  This builder therefore never writes
an _SUCCESS marker.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import importlib.util
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer


SCHEMA_VERSION = 2
BUILDER_VERSION = "pretrain-v1-builder-3"
VENDOR_PATH_PARTS = {
    "node_modules",
    "third_party",
    "third-party",
    "vendor",
    "vendors",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_sha256(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        payload = part if isinstance(part, bytes) else str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_content_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    if int(value.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: expected schema_version={SCHEMA_VERSION}, "
            f"got {value.get('schema_version')!r}"
        )
    return value


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "object"


def allocate(total: int, weighted: list[tuple[str, float]]) -> dict[str, int]:
    if total < 0:
        raise ValueError("negative budget")
    if not weighted:
        if total:
            raise ValueError("nonzero budget without weighted entries")
        return {}
    weight_sum = sum(weight for _, weight in weighted)
    if weight_sum <= 0:
        raise ValueError("weights must be positive")
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


def source_targets(
    config: dict[str, Any],
    total: int,
    mix_key: str = "active_v1_mix",
) -> dict[str, int]:
    categories = config.get(mix_key)
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"{mix_key} is empty")
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
        raise ValueError(f"{mix_key} weights must sum to 1.0")
    category_budgets = allocate(total, category_weights)
    output: dict[str, int] = {}
    for category, settings in categories.items():
        sources = [str(source) for source in settings.get("sources", [])]
        if not sources:
            raise ValueError(f"{mix_key}.{category}.sources is empty")
        configured = settings.get("source_weights")
        weights = (
            [(source, float(configured[source])) for source in sources]
            if configured
            else [(source, 1.0) for source in sources]
        )
        for source, tokens in allocate(
            category_budgets[str(category)],
            weights,
        ).items():
            output[source] = output.get(source, 0) + tokens
    if sum(output.values()) != total:
        raise AssertionError((output, total))
    return output


def budget_value(config: dict[str, Any], name: str) -> int:
    budgets = config.get("budgets", {})
    aliases = {
        "full_loss_target_tokens": (
            "full_loss_target_tokens",
            "full_processed_tokens",
        ),
        "validation_loss_target_tokens": ("validation_loss_target_tokens",),
    }
    for key in aliases[name]:
        if key in budgets:
            return int(budgets[key])
    if name == "validation_loss_target_tokens":
        return 0
    raise KeyError(name)


def sampling_seed(config: dict[str, Any]) -> int:
    return int(config.get("sampling", {}).get("seed", config.get("seed", 42)))


def tokenizer_path(cli_value: Path | None, config: dict[str, Any]) -> Path:
    return (
        cli_value
        or Path(str(config.get("tokenizer", "tokenizer")))
    ).resolve()


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


def script_fingerprints() -> dict[str, Any]:
    builder = Path(__file__).resolve()
    auditor = builder.with_name("audit_pretrain_v1.py")
    if not auditor.exists():
        raise FileNotFoundError(auditor)
    return {
        "builder": {
            "path": "scripts/data/pretrain/build_pretrain_v1.py",
            "sha256": sha256_file(builder),
            "version": BUILDER_VERSION,
        },
        "auditor": {
            "path": "scripts/data/pretrain/audit_pretrain_v1.py",
            "sha256": sha256_file(auditor),
        },
    }


def hash_domains(config: dict[str, Any]) -> dict[str, str]:
    sampling = config.get("sampling", {})
    if str(sampling.get("sample_hash", "")).lower() != "sha256":
        raise ValueError("sampling.sample_hash must be sha256")
    domains = {
        "sample": str(sampling.get("sample_hash_domain", "")).strip(),
        "mix": str(sampling.get("mix_hash_domain", "")).strip(),
        "split": str(sampling.get("split_hash_domain", "")).strip(),
    }
    if any(not value for value in domains.values()):
        raise ValueError(
            "sampling sample/mix/split hash domains must be nonempty"
        )
    if len(set(domains.values())) != len(domains):
        raise ValueError(
            "sampling sample/mix/split hash domains must be distinct"
        )
    return domains


def sharding_contract(
    config: dict[str, Any],
    full_budget: int,
    validation_budget: int,
) -> dict[str, Any]:
    sharding = config.get("sharding", {})
    required = (
        "assignment",
        "num_train_shards",
        "num_validation_shards",
        "train_glob",
        "validation_glob",
        "filename_format",
        "expected_mean_train_loss_target_tokens_per_shard",
    )
    missing = [key for key in required if key not in sharding]
    if missing:
        raise ValueError(f"sharding keys missing: {missing}")
    num_train = int(sharding["num_train_shards"])
    num_validation = int(sharding["num_validation_shards"])
    if num_train <= 0 or num_validation <= 0:
        raise ValueError("train and validation shard counts must be positive")
    assignment = str(sharding["assignment"])
    if assignment != "deterministic_hash_order_row_round_robin":
        raise ValueError(
            "sharding.assignment must be "
            "deterministic_hash_order_row_round_robin"
        )

    filename_format = str(sharding["filename_format"])
    names: dict[str, list[str]] = {}
    for split, count in (
        ("train", num_train),
        ("validation", num_validation),
    ):
        split_names = [
            filename_format.format(
                split=split,
                index=index,
                num_shards=count,
            )
            for index in range(count)
        ]
        if (
            len(set(split_names)) != count
            or any(
                Path(name).name != name or not name.endswith(".jsonl")
                for name in split_names
            )
        ):
            raise ValueError(
                f"sharding.filename_format is invalid for {split}"
            )
        names[split] = split_names
        configured_glob = str(sharding[f"{split}_glob"])
        if not configured_glob or any(
            not Path(name).match(configured_glob)
            for name in split_names
        ):
            raise ValueError(
                f"sharding.{split}_glob does not match generated names"
            )
    expected_mean = full_budget / num_train
    configured_mean = float(
        sharding[
            "expected_mean_train_loss_target_tokens_per_shard"
        ]
    )
    if not math.isclose(
        configured_mean,
        expected_mean,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "expected mean train loss-target tokens per shard differs "
            f"from budget/count: {configured_mean} != {expected_mean}"
        )
    return {
        "assignment": assignment,
        "expected_mean_train_loss_target_tokens_per_shard": configured_mean,
        "filename_format": filename_format,
        "names": names,
        "num_train_shards": num_train,
        "num_validation_shards": num_validation,
        "train_glob": str(sharding["train_glob"]),
        "validation_glob": str(sharding["validation_glob"]),
        "validation_loss_target_tokens": validation_budget,
    }


def load_tokenizer(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
    )
    if (
        tokenizer.bos_token_id is None
        or tokenizer.eos_token_id is None
        or tokenizer.pad_token_id is None
    ):
        raise ValueError("tokenizer BOS/EOS/PAD ids are required")
    return tokenizer


def normalize_sources_config(value: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for raw_source in value.get("sources", []):
        source = dict(raw_source)
        objects = source.get("objects")
        if objects is None and "path" in source:
            objects = [{"path": source.pop("path")}]
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"source {source.get('id')} objects are empty")
        source["objects"] = [dict(item) for item in objects]
        output.append(source)
    if not output:
        raise ValueError("no configured sources")
    return output


def validate_source_coverage(
    sources: list[dict[str, Any]],
    required: set[str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_id = str(source.get("id", ""))
        if not source_id or source_id in output:
            raise ValueError(f"bad or duplicate source id: {source_id!r}")
        for key in ("repo_id", "revision", "format", "text_field"):
            if key not in source:
                raise ValueError(f"{source_id} is missing {key}")
        for item in source["objects"]:
            if "path" not in item and "url" not in item:
                raise ValueError(f"{source_id} object requires path or url")
        output[source_id] = source
    missing = sorted(required - set(output))
    if missing:
        raise ValueError(f"missing source configurations: {missing}")
    return output


def object_identity(
    source: dict[str, Any],
    obj: dict[str, Any],
    index: int,
) -> str:
    path = str(obj.get("path") or obj.get("url"))
    return str(
        obj.get("id")
        or f"{index:03d}-{safe_name(Path(path).name)}"
    )


def object_uri(
    endpoint: str,
    source: dict[str, Any],
    obj: dict[str, Any],
) -> str:
    if obj.get("url"):
        return str(obj["url"])
    path = str(obj["path"])
    if path.startswith(("file://", "http://", "https://")):
        return path
    if Path(path).is_absolute():
        return Path(path).as_uri()
    return (
        f"{endpoint.rstrip('/')}/datasets/{source['repo_id']}/resolve/"
        f"{source['revision']}/{quote(path, safe='/')}"
    )


def parquet_rows(
    uri: str,
    source: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], int]]:
    with fsspec.open(uri, "rb", block_size=16 * 1024 * 1024) as handle:
        parquet = pq.ParquetFile(handle)
        available = set(parquet.schema_arrow.names)
        if source.get("record_layout") == "stack_v3_files":
            if "files" not in available:
                raise ValueError(
                    f"Stack v3 object is missing files; "
                    f"available={sorted(available)}"
                )
            columns = ["files"]
        else:
            text_field = str(source["text_field"])
            if text_field not in available:
                raise ValueError(
                    f"missing field {text_field}; "
                    f"available={sorted(available)}"
                )
            columns = [text_field]
        row_index = 0
        for group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(group_index, columns=columns)
            for row in table.to_pylist():
                yield row, row_index
                row_index += 1


def gzip_rows(uri: str) -> Iterator[tuple[dict[str, Any], int]]:
    if uri.startswith("file://"):
        raw_handle = open(Path(uri[7:]), "rb")
    elif Path(uri).is_absolute():
        raw_handle = open(uri, "rb")
    else:
        request = urllib.request.Request(
            uri,
            headers={"User-Agent": "minimind-lab-data-v1/2.0"},
        )
        raw_handle = urllib.request.urlopen(request, timeout=120)
    with raw_handle:
        with gzip.GzipFile(fileobj=raw_handle) as stream:
            for row_index, raw in enumerate(stream):
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid gzip JSON row {row_index}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"gzip row {row_index} is not an object")
                yield row, row_index


def get_nested(value: Any, dotted: str) -> tuple[Any, bool]:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


def first_present(
    file_record: dict[str, Any],
    row: dict[str, Any],
    explicit: str | None,
    candidates: tuple[str, ...],
) -> tuple[Any, bool]:
    fields = (explicit,) if explicit else candidates
    for field in fields:
        for value in (file_record, row):
            result, present = get_nested(value, str(field))
            if present:
                return result, True
    return None, False


def normalize_language(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("language") or value.get("id")
    normalized = str(value or "").strip().lower()
    return {
        "c++": "cpp",
        "c#": "csharp",
    }.get(normalized, normalized)


def license_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith(("[", "{")):
            try:
                return license_values(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [
            item.strip()
            for item in re.split(r"[,;|]", stripped)
            if item.strip()
        ]
    if isinstance(value, dict):
        direct = (
            value.get("spdx_id")
            or value.get("spdx")
            or value.get("license")
            or value.get("name")
            or value.get("id")
        )
        if direct:
            return license_values(direct)
        output: list[str] = []
        for item in value.values():
            output.extend(license_values(item))
        return output
    if isinstance(value, (list, tuple, set)):
        output = []
        for item in value:
            output.extend(license_values(item))
        return output
    return [str(value)]


def vendor_truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "vendor",
    }


def stack_file_passes(
    row: dict[str, Any],
    file_record: dict[str, Any],
    filters: dict[str, Any],
    counters: dict[str, int],
) -> bool:
    allowed_languages = filters.get("allowed_languages")
    if not isinstance(allowed_languages, list):
        raise ValueError(
            "Stack v3 filters.allowed_languages must be a list"
        )
    configured_languages = {
        normalize_language(item)
        for item in allowed_languages
    }
    if (
        len(allowed_languages) != 14
        or len(configured_languages) != 14
        or int(filters.get("allowed_languages_count", 14)) != 14
    ):
        raise ValueError(
            "Stack v3 filters.allowed_languages must contain exactly "
            "14 unique normalized languages"
        )
    language, language_present = first_present(
        file_record,
        row,
        filters.get("language_field"),
        ("language", "lang", "language_name", "programming_language"),
    )
    if (
        not language_present
        or normalize_language(language) not in configured_languages
    ):
        counters["stack_reject_language"] += 1
        return False

    expected_license_type = filters.get("license_type")
    if expected_license_type is not None:
        license_type, type_present = first_present(
            file_record,
            row,
            filters.get("license_type_field"),
            ("license_type",),
        )
        if (
            not type_present
            or str(license_type).strip().lower()
            != str(expected_license_type).strip().lower()
        ):
            counters["stack_reject_license_type"] += 1
            return False

    detected, detected_present = first_present(
        file_record,
        row,
        filters.get("licenses_field"),
        ("detected_licenses", "licenses", "license"),
    )
    if filters.get("require_detected_licenses", False):
        if not detected_present or not license_values(detected):
            counters["stack_reject_missing_detected_license"] += 1
            return False

    vendor, vendor_present = first_present(
        file_record,
        row,
        filters.get("vendor_field"),
        ("is_vendor", "vendor", "is_vendor_file"),
    )
    path, path_present = first_present(
        file_record,
        row,
        filters.get("path_field"),
        ("file_path", "path", "name", "filename"),
    )
    expected_vendor = bool(filters.get("is_vendor", False))
    if vendor_present and vendor_truth(vendor) != expected_vendor:
        counters["stack_reject_vendor"] += 1
        return False
    path_vendor = False
    if path_present:
        parts = {
            item.lower()
            for item in re.split(r"[\\/]+", str(path))
            if item
        }
        path_vendor = bool(parts & VENDOR_PATH_PARTS)
    if not expected_vendor and path_vendor:
        counters["stack_reject_vendor_path"] += 1
        return False
    if not vendor_present and not path_present:
        counters["stack_reject_vendor_signal_missing"] += 1
        return False

    counters["stack_accepted_files"] += 1
    return True


def texts_from_row(
    row: dict[str, Any],
    source: dict[str, Any],
    counters: dict[str, int],
) -> Iterator[tuple[str, int]]:
    if source.get("record_layout") != "stack_v3_files":
        text = row.get(str(source["text_field"]))
        if text is None:
            counters["missing_text"] += 1
            return
        yield str(text), 0
        return

    files = row.get("files")
    if not isinstance(files, list):
        counters["stack_rows_without_files"] += 1
        return
    filters = dict(source.get("filters", {}))
    text_field = str(source.get("text_field", "content"))
    for file_index, file_record in enumerate(files):
        if not isinstance(file_record, dict):
            counters["stack_invalid_file_records"] += 1
            continue
        if not stack_file_passes(row, file_record, filters, counters):
            continue
        text, present = get_nested(file_record, text_field)
        if not present:
            counters["stack_missing_content"] += 1
            continue
        yield str(text), file_index


def source_rows(
    uri: str,
    source: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], int]]:
    source_format = str(source["format"]).lower()
    if source_format == "parquet":
        yield from parquet_rows(uri, source)
    elif source_format in {"jsonl_gzip", "jsonl.gz", "gzip_jsonl"}:
        yield from gzip_rows(uri)
    else:
        raise ValueError(f"unsupported source format: {source_format}")


def token_aligned_chunks(
    text: str,
    tokenizer,
    max_text_tokens: int,
) -> Iterator[tuple[str, list[int], int]]:
    raw_ids = tokenizer(text, add_special_tokens=False).input_ids
    offset = 0
    while offset < len(raw_ids):
        original = raw_ids[offset : offset + max_text_tokens]
        consumed = len(original)
        while consumed:
            decoded = tokenizer.decode(
                original[:consumed],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            actual = tokenizer(decoded, add_special_tokens=False).input_ids
            if decoded and actual and len(actual) <= max_text_tokens:
                yield decoded, actual, consumed
                offset += consumed
                break
            consumed -= 1
        if not consumed:
            raise ValueError("cannot form a nonempty tokenizer-aligned chunk")


def merge_run_lines(paths: list[Path]) -> Iterator[tuple[str, str]]:
    handles = [path.open("r", encoding="utf-8") for path in paths]
    heap: list[tuple[str, str, int]] = []
    try:
        for index, handle in enumerate(handles):
            line = handle.readline()
            if line:
                key, payload = line.rstrip("\n").split("\t", 1)
                heapq.heappush(heap, (key, payload, index))
        while heap:
            key, payload, index = heapq.heappop(heap)
            yield key, payload
            line = handles[index].readline()
            if line:
                next_key, next_payload = line.rstrip("\n").split("\t", 1)
                heapq.heappush(
                    heap,
                    (next_key, next_payload, index),
                )
    finally:
        for handle in handles:
            handle.close()


class ExternalSorter:
    def __init__(
        self,
        run_dir: Path,
        prefix: str,
        memory_bytes: int,
    ) -> None:
        self.run_dir = run_dir
        self.prefix = prefix
        self.limit = max(memory_bytes, 1024 * 1024)
        self.buffer: list[tuple[str, str]] = []
        self.size = 0
        self.runs: list[Path] = []
        run_dir.mkdir(parents=True, exist_ok=True)

    def add(self, key: str, record: dict[str, Any]) -> None:
        payload = canonical_json(record)
        self.buffer.append((key, payload))
        self.size += len(key) + len(payload) + 2
        if self.size >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.buffer.sort(key=lambda item: (item[0], item[1]))
        path = self.run_dir / f"{self.prefix}-{len(self.runs):05d}.run"
        with path.open("w", encoding="utf-8") as handle:
            for key, payload in self.buffer:
                handle.write(f"{key}\t{payload}\n")
        self.runs.append(path)
        self.buffer = []
        self.size = 0

    def finish_jsonl(self, path: Path) -> int:
        self.flush()
        rows = 0
        with path.open("w", encoding="utf-8") as output:
            for _, payload in merge_run_lines(self.runs):
                output.write(payload + "\n")
                rows += 1
            output.flush()
            os.fsync(output.fileno())
        return rows


def merge_sorted_candidate_files(
    paths: list[Path],
) -> Iterator[dict[str, Any]]:
    handles = [path.open("r", encoding="utf-8") for path in paths]
    heap: list[tuple[str, str, int]] = []
    try:
        for index, handle in enumerate(handles):
            line = handle.readline()
            if line:
                record = json.loads(line)
                heapq.heappush(
                    heap,
                    (
                        str(record["sample_key"]),
                        canonical_json(record),
                        index,
                    ),
                )
        while heap:
            _, payload, index = heapq.heappop(heap)
            yield json.loads(payload)
            line = handles[index].readline()
            if line:
                record = json.loads(line)
                heapq.heappush(
                    heap,
                    (
                        str(record["sample_key"]),
                        canonical_json(record),
                        index,
                    ),
                )
    finally:
        for handle in handles:
            handle.close()


def externally_sort_records(
    paths: list[Path],
    key_function,
    run_dir: Path,
    prefix: str,
    memory_bytes: int,
) -> tuple[list[Path], Any]:
    sorter = ExternalSorter(run_dir, prefix, memory_bytes)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                sorter.add(key_function(record), record)
    sorter.flush()

    def cleanup() -> None:
        for run in sorter.runs:
            run.unlink(missing_ok=True)

    return sorter.runs, cleanup


def iter_records_from_runs(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for _, payload in merge_run_lines(paths):
        yield json.loads(payload)


def object_fingerprint(
    source: dict[str, Any],
    obj: dict[str, Any],
    target: int,
    sequence_length: int,
    seed: int,
    tokenizer_fp: dict[str, Any],
    sources_sha: str,
    config_sha: str,
    builder_fp: dict[str, Any],
    domains: dict[str, str],
) -> str:
    return stable_sha256(
        BUILDER_VERSION,
        canonical_json(source),
        canonical_json(obj),
        target,
        sequence_length,
        seed,
        canonical_json(tokenizer_fp),
        sources_sha,
        config_sha,
        canonical_json(builder_fp),
        canonical_json(domains),
    )


def valid_done(
    done_path: Path,
    output_path: Path,
    fingerprint: str,
) -> bool:
    if not done_path.exists() or not output_path.exists():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        done.get("status") == "ok"
        and done.get("fingerprint") == fingerprint
        and done.get("output_sha256") == sha256_file(output_path)
    )


def materialize_object(
    source: dict[str, Any],
    obj: dict[str, Any],
    object_id: str,
    uri: str,
    target: int,
    output: Path,
    done_path: Path,
    tokenizer,
    sequence_length: int,
    seed: int,
    fingerprint: str,
    sample_hash_domain: str,
    builder_fp: dict[str, Any],
    domains: dict[str, str],
    sort_memory_bytes: int,
) -> dict[str, Any]:
    if valid_done(done_path, output, fingerprint):
        done = json.loads(done_path.read_text(encoding="utf-8"))
        done["resume_status"] = "verified_and_reused"
        print(
            canonical_json(
                {
                    "source": source["id"],
                    "object": object_id,
                    "status": "reused",
                }
            ),
            flush=True,
        )
        return done

    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    run_dir = output.parent / f".{output.stem}.runs-{os.getpid()}"
    temporary.unlink(missing_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    sorter = ExternalSorter(run_dir, "candidate", sort_memory_bytes)
    counters: dict[str, int] = defaultdict(int)
    source_id = str(source["id"])
    max_text_tokens = sequence_length - 2
    current_loss_tokens = 0
    try:
        stop = False
        for row, row_index in source_rows(uri, source):
            counters["raw_rows"] += 1
            for raw_text, nested_index in texts_from_row(
                row,
                source,
                counters,
            ):
                if not raw_text or raw_text.isspace():
                    counters["empty_text"] += 1
                    continue
                chunk_index = 0
                for text, token_ids, raw_chunk_tokens in token_aligned_chunks(
                    raw_text,
                    tokenizer,
                    max_text_tokens,
                ):
                    content_hash = token_content_sha256(token_ids)
                    candidate_id = stable_sha256(
                        "candidate-identity-v1",
                        source_id,
                        source["repo_id"],
                        source["revision"],
                        obj.get("path") or obj.get("url"),
                        row_index,
                        nested_index,
                        chunk_index,
                        content_hash,
                    )
                    sample_key = stable_sha256(
                        sample_hash_domain,
                        seed,
                        candidate_id,
                    )
                    record = {
                        "candidate_id": candidate_id,
                        "content_sha256": content_hash,
                        "loss_target_tokens": len(token_ids) + 1,
                        "nonpad_input_tokens": len(token_ids) + 2,
                        "object_id": object_id,
                        "padded_compute_tokens": sequence_length,
                        "raw_tokens": raw_chunk_tokens,
                        "repo_id": source["repo_id"],
                        "revision": source["revision"],
                        "sample_key": sample_key,
                        "source": source_id,
                        "source_path": obj.get("path") or obj.get("url"),
                        "text": text,
                        "text_tokens": len(token_ids),
                    }
                    sorter.add(sample_key, record)
                    counters["candidate_rows"] += 1
                    counters["raw_tokens"] += raw_chunk_tokens
                    counters["text_tokens"] += len(token_ids)
                    counters["loss_target_tokens"] += len(token_ids) + 1
                    counters["nonpad_input_tokens"] += len(token_ids) + 2
                    counters["padded_compute_tokens"] += sequence_length
                    current_loss_tokens += len(token_ids) + 1
                    chunk_index += 1
                    if current_loss_tokens >= target:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        if current_loss_tokens < target:
            raise RuntimeError(
                f"object {source_id}/{object_id} is short: "
                f"{current_loss_tokens} < {target}"
            )
        rows = sorter.finish_jsonl(temporary)
        temporary.replace(output)
        result = {
            "builder_version": BUILDER_VERSION,
            "builder": builder_fp,
            "candidate_target_loss_tokens": target,
            "created_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "fingerprint": fingerprint,
            "hash_domains": domains,
            "object_id": object_id,
            "output": str(output),
            "output_rows": rows,
            "output_sha256": sha256_file(output),
            "output_size_bytes": output.stat().st_size,
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "source": source_id,
            "source_path": obj.get("path") or obj.get("url"),
            "stats": dict(counters),
            "status": "ok",
            "uri": uri,
        }
        atomic_json(done_path, result)
        print(
            canonical_json(
                {
                    "source": source_id,
                    "object": object_id,
                    "status": "ok",
                    "loss_target_tokens": current_loss_tokens,
                }
            ),
            flush=True,
        )
        return result
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def materialize(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    sources_raw = load_yaml(args.sources_config)
    sources = normalize_sources_config(sources_raw)
    full_budget = budget_value(config, "full_loss_target_tokens")
    validation_budget = budget_value(
        config,
        "validation_loss_target_tokens",
    )
    train_targets = source_targets(config, full_budget, args.mix_key)
    validation_targets = source_targets(
        config,
        validation_budget,
        args.mix_key,
    )
    by_id = validate_source_coverage(
        sources,
        set(train_targets) | set(validation_targets),
    )
    quality_gates = config.get("quality_gates", {})
    if quality_gates.get("declared_license_required", False):
        missing_licenses = sorted(
            source_id
            for source_id in set(train_targets) | set(validation_targets)
            if not by_id[source_id].get("declared_license")
        )
        if missing_licenses:
            raise ValueError(
                "sources missing declared_license: "
                f"{missing_licenses}"
            )
    token_path = tokenizer_path(args.tokenizer, config)
    tokenizer = load_tokenizer(token_path)
    tokenizer_fp = tokenizer_fingerprint(token_path)
    sequence_length = int(config["sequence_length"])
    if sequence_length < 4:
        raise ValueError("sequence_length must be at least 4")
    seed = sampling_seed(config)
    multiplier = float(
        config.get("sampling", {}).get("candidate_multiplier", 1.15)
    )
    if multiplier <= 1.0:
        raise ValueError("candidate_multiplier must exceed 1.0")
    domains = hash_domains(config)
    scripts_fp = script_fingerprints()
    builder_fp = scripts_fp["builder"]
    sharding = sharding_contract(config, full_budget, validation_budget)
    config_sha = sha256_file(args.config)
    sources_sha = sha256_file(args.sources_config)
    endpoint = str(
        sources_raw.get("endpoint", "https://hf-mirror.com")
    )
    candidates_root = args.work_root / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    build_fp = stable_sha256(
        BUILDER_VERSION,
        config_sha,
        sources_sha,
        canonical_json(tokenizer_fp),
        full_budget,
        validation_budget,
        multiplier,
        args.mix_key,
        canonical_json(domains),
        canonical_json(builder_fp),
        canonical_json(sharding),
    )

    expected_object_count = sum(
        len(by_id[source_id]["objects"])
        for source_id in sorted(train_targets)
    )
    manifest_path = candidates_root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") == build_fp:
            old_objects = old.get("objects", [])
            valid = (
                isinstance(old_objects, list)
                and len(old_objects) == expected_object_count
                and all(
                    valid_done(
                        Path(str(item["done"])),
                        Path(str(item["output"])),
                        str(item["fingerprint"]),
                    )
                    for item in old_objects
                )
            )
            if valid:
                print(
                    canonical_json(
                        {
                            "status": "already_materialized",
                            "path": str(candidates_root),
                        }
                    )
                )
                return 0
        print(
            canonical_json(
                {
                    "status": "candidate_manifest_invalidated",
                    "path": str(manifest_path),
                    "action": "rebuild_without_reusing_mismatched_done",
                }
            ),
            flush=True,
        )

    results = []
    for source_id in sorted(train_targets):
        source = by_id[source_id]
        objects = source["objects"]
        weights = [
            (str(index), float(obj.get("weight", 1.0)))
            for index, obj in enumerate(objects)
        ]
        candidate_total = math.ceil(
            (
                train_targets[source_id]
                + validation_targets.get(source_id, 0)
            )
            * multiplier
        )
        targets = allocate(candidate_total, weights)
        for index, obj in enumerate(objects):
            object_id = object_identity(source, obj, index)
            directory = candidates_root / safe_name(source_id)
            output = directory / f"{safe_name(object_id)}.jsonl"
            done_path = directory / f"{safe_name(object_id)}.done.json"
            target = targets[str(index)]
            fingerprint = object_fingerprint(
                source,
                obj,
                target,
                sequence_length,
                seed,
                tokenizer_fp,
                sources_sha,
                config_sha,
                builder_fp,
                domains,
            )
            result = materialize_object(
                source,
                obj,
                object_id,
                object_uri(endpoint, source, obj),
                target,
                output,
                done_path,
                tokenizer,
                sequence_length,
                seed,
                fingerprint,
                domains["sample"],
                builder_fp,
                domains,
                args.sort_memory_mb * 1024 * 1024,
            )
            result["done"] = str(done_path)
            results.append(result)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "config": str(args.config),
        "config_sha256": config_sha,
        "created_at": utc_now(),
        "fingerprint": build_fp,
        "full_loss_target_tokens": full_budget,
        "hash_domains": domains,
        "mix_key": args.mix_key,
        "objects": results,
        "sampling": {
            "candidate_multiplier": multiplier,
            "seed": seed,
            "hash_domains": domains,
            "sample_key": (
                "SHA256(domain, seed, stable candidate identity)"
            ),
        },
        "schema_version": SCHEMA_VERSION,
        "scripts": {"builder": builder_fp},
        "sequence_length": sequence_length,
        "sharding": sharding,
        "sources_config": str(args.sources_config),
        "sources_config_sha256": sources_sha,
        "status": "materialized",
        "tokenizer": tokenizer_fp,
        "train_source_quotas": train_targets,
        "validation_loss_target_tokens": validation_budget,
        "validation_source_quotas": validation_targets,
    }
    atomic_json(manifest_path, manifest)
    print(
        canonical_json(
            {
                "status": "materialized",
                "candidates": str(candidates_root),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    )
    return 0


class SeenStore:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE content "
            "(hash TEXT PRIMARY KEY, split TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE candidate "
            "(id TEXT PRIMARY KEY, split TEXT NOT NULL)"
        )

    def reserve(self, record: dict[str, Any], split: str) -> bool:
        content_hash = str(record["content_sha256"])
        candidate_id = str(record["candidate_id"])
        if self.connection.execute(
            "SELECT 1 FROM content WHERE hash=?",
            (content_hash,),
        ).fetchone():
            return False
        if self.connection.execute(
            "SELECT 1 FROM candidate WHERE id=?",
            (candidate_id,),
        ).fetchone():
            return False
        self.connection.execute(
            "INSERT INTO content(hash, split) VALUES (?, ?)",
            (content_hash, split),
        )
        self.connection.execute(
            "INSERT INTO candidate(id, split) VALUES (?, ?)",
            (candidate_id, split),
        )
        return True

    def replace_pending_content(
        self,
        old: dict[str, Any],
        pieces: list[dict[str, Any]],
        split: str,
    ) -> bool:
        old_hash = str(old["content_sha256"])
        new_hashes = [str(piece["content_sha256"]) for piece in pieces]
        if len(set(new_hashes)) != len(new_hashes):
            return False
        for content_hash in new_hashes:
            row = self.connection.execute(
                "SELECT split FROM content WHERE hash=?",
                (content_hash,),
            ).fetchone()
            if row and content_hash != old_hash:
                return False
        self.connection.execute(
            "DELETE FROM content WHERE hash=?",
            (old_hash,),
        )
        for content_hash in new_hashes:
            self.connection.execute(
                "INSERT INTO content(hash, split) VALUES (?, ?)",
                (content_hash, split),
            )
        return True

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def piece_record(
    record: dict[str, Any],
    text: str,
    token_ids: list[int],
    sequence_length: int,
    index: int,
) -> dict[str, Any]:
    output = dict(record)
    output.update(
        {
            "content_sha256": token_content_sha256(token_ids),
            "loss_target_tokens": len(token_ids) + 1,
            "nonpad_input_tokens": len(token_ids) + 2,
            "padded_compute_tokens": sequence_length,
            "piece_index": index,
            "piece_of": record["candidate_id"],
            "raw_tokens": len(token_ids),
            "text": text,
            "text_tokens": len(token_ids),
        }
    )
    return output


def exact_prefix_piece(
    record: dict[str, Any],
    desired_text_tokens: int,
    tokenizer,
    sequence_length: int,
) -> dict[str, Any] | None:
    token_ids = tokenizer(
        record["text"],
        add_special_tokens=False,
    ).input_ids
    for cut in sorted(
        range(1, len(token_ids) + 1),
        key=lambda value: (abs(value - desired_text_tokens), value),
    ):
        text = tokenizer.decode(
            token_ids[:cut],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        actual = tokenizer(text, add_special_tokens=False).input_ids
        if len(actual) == desired_text_tokens:
            return piece_record(
                record,
                text,
                actual,
                sequence_length,
                0,
            )
    return None


def split_for_one_extra_target(
    record: dict[str, Any],
    tokenizer,
    sequence_length: int,
) -> list[dict[str, Any]] | None:
    token_ids = tokenizer(
        record["text"],
        add_special_tokens=False,
    ).input_ids
    if len(token_ids) < 2:
        return None
    for at in sorted(
        range(1, len(token_ids)),
        key=lambda value: abs(value - len(token_ids) // 2),
    ):
        left_text = tokenizer.decode(
            token_ids[:at],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        right_text = tokenizer.decode(
            token_ids[at:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        left_ids = tokenizer(
            left_text,
            add_special_tokens=False,
        ).input_ids
        right_ids = tokenizer(
            right_text,
            add_special_tokens=False,
        ).input_ids
        if (
            left_ids
            and right_ids
            and len(left_ids) + len(right_ids) == len(token_ids)
        ):
            return [
                piece_record(
                    record,
                    left_text,
                    left_ids,
                    sequence_length,
                    0,
                ),
                piece_record(
                    record,
                    right_text,
                    right_ids,
                    sequence_length,
                    1,
                ),
            ]
    return None


def select_source(
    records: Iterator[dict[str, Any]],
    quota: int,
    split: str,
    output: Path,
    tokenizer,
    sequence_length: int,
    seen: SeenStore,
) -> dict[str, Any]:
    if quota == 1:
        raise ValueError(
            f"{split}/{output.stem}: a one-token loss quota is "
            "not representable by a nonempty MiniMind row"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    totals: dict[str, int] = defaultdict(int)
    pending: dict[str, Any] | None = None

    def write(handle, record: dict[str, Any]) -> None:
        handle.write(canonical_json(record) + "\n")
        totals["rows"] += 1
        for key in (
            "loss_target_tokens",
            "nonpad_input_tokens",
            "padded_compute_tokens",
            "raw_tokens",
            "text_tokens",
        ):
            totals[key] += int(record[key])

    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            totals["candidates_considered"] += 1
            loss_tokens = int(record["loss_target_tokens"])
            remaining = quota - totals["selected_budget"]
            if remaining <= 0:
                break
            if loss_tokens <= remaining:
                if not seen.reserve(record, split):
                    totals["dedup_rejected"] += 1
                    continue
                if pending is not None:
                    write(handle, pending)
                pending = record
                totals["selected_budget"] += loss_tokens
                if totals["selected_budget"] == quota:
                    break
                continue
            if remaining >= 2:
                piece = exact_prefix_piece(
                    record,
                    remaining - 1,
                    tokenizer,
                    sequence_length,
                )
                if piece is None:
                    totals["unrepresentable_prefix"] += 1
                    continue
                if not seen.reserve(piece, split):
                    totals["dedup_rejected"] += 1
                    continue
                if pending is not None:
                    write(handle, pending)
                pending = piece
                totals["selected_budget"] += int(
                    piece["loss_target_tokens"]
                )
                break
            if remaining == 1 and pending is not None:
                pieces = split_for_one_extra_target(
                    pending,
                    tokenizer,
                    sequence_length,
                )
                if (
                    pieces is not None
                    and seen.replace_pending_content(
                        pending,
                        pieces,
                        split,
                    )
                ):
                    write(handle, pieces[0])
                    pending = pieces[1]
                    totals["selected_budget"] += 1
                    break
                totals["unrepresentable_boundary_split"] += 1

        if pending is not None:
            write(handle, pending)
        handle.flush()
        os.fsync(handle.fileno())

    if (
        totals["selected_budget"] != quota
        or totals["loss_target_tokens"] != quota
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{split}/{output.stem} quota short: {dict(totals)}; "
            f"expected={quota}"
        )
    temporary.replace(output)
    seen.commit()
    result = dict(totals)
    result.update(
        {
            "output": str(output),
            "output_sha256": sha256_file(output),
            "quota_loss_target_tokens": quota,
            "split": split,
            "status": "ok",
        }
    )
    return result


def candidate_paths_by_source(
    manifest: dict[str, Any],
) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = defaultdict(list)
    for obj in manifest["objects"]:
        path = Path(str(obj["output"]))
        done_path = Path(str(obj["done"]))
        if not valid_done(done_path, path, str(obj["fingerprint"])):
            raise ValueError(f"candidate fingerprint or SHA failed: {path}")
        output[str(obj["source"])].append(path)
    for paths in output.values():
        paths.sort()
    return dict(output)


def provenance_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "text"
    }


def write_final_shards(
    selected_paths: list[Path],
    split: str,
    num_shards: int,
    stage: Path,
    seed: int,
    sequence_length: int,
    memory_bytes: int,
    mix_hash_domain: str,
    filename_format: str,
) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError(f"{split}: num_shards must be positive")
    run_dir = stage / f".{split}-sort-runs"

    def final_key(record: dict[str, Any]) -> str:
        return stable_sha256(
            mix_hash_domain,
            seed,
            split,
            record["candidate_id"],
            record.get("piece_index", -1),
            record["content_sha256"],
        )

    runs, cleanup = externally_sort_records(
        selected_paths,
        final_key,
        run_dir,
        split,
        memory_bytes,
    )
    names = [
        filename_format.format(
            split=split,
            index=index,
            num_shards=num_shards,
        )
        for index in range(num_shards)
    ]
    temporary_data = [stage / f".{name}.tmp" for name in names]
    provenance_dir = stage / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    temporary_provenance = [
        provenance_dir / f".{name}.provenance.tmp" for name in names
    ]
    data_handles = [
        path.open("w", encoding="utf-8") for path in temporary_data
    ]
    provenance_handles = [
        path.open("w", encoding="utf-8")
        for path in temporary_provenance
    ]
    stats = [defaultdict(int) for _ in names]
    sources = [
        defaultdict(lambda: defaultdict(int)) for _ in names
    ]
    total: dict[str, int] = defaultdict(int)
    try:
        for row_index, record in enumerate(
            iter_records_from_runs(runs)
        ):
            shard_index = row_index % num_shards
            data_handles[shard_index].write(
                canonical_json({"text": record["text"]}) + "\n"
            )
            provenance_handles[shard_index].write(
                canonical_json(provenance_row(record)) + "\n"
            )
            stats[shard_index]["rows"] += 1
            total["rows"] += 1
            source_id = str(record["source"])
            for key in (
                "loss_target_tokens",
                "nonpad_input_tokens",
                "padded_compute_tokens",
                "raw_tokens",
                "text_tokens",
            ):
                value = int(record[key])
                stats[shard_index][key] += value
                sources[shard_index][source_id][key] += value
                total[key] += value
            sources[shard_index][source_id]["rows"] += 1
        for handle in data_handles + provenance_handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    finally:
        for handle in data_handles + provenance_handles:
            if not handle.closed:
                handle.close()
        cleanup()
        shutil.rmtree(run_dir, ignore_errors=True)

    shard_records = []
    for index, name in enumerate(names):
        final = stage / name
        provenance = provenance_dir / f"{name}.provenance.jsonl"
        temporary_data[index].replace(final)
        temporary_provenance[index].replace(provenance)
        sidecar = {
            "created_at": utc_now(),
            "file": name,
            "provenance_file": str(provenance.relative_to(stage)),
            "provenance_sha256": sha256_file(provenance),
            "provenance_size_bytes": provenance.stat().st_size,
            "sequence_length": sequence_length,
            "sha256": sha256_file(final),
            "size_bytes": final.stat().st_size,
            "sources": {
                source: dict(values)
                for source, values in sorted(sources[index].items())
            },
            "split": split,
            "stats": dict(stats[index]),
        }
        atomic_json(stage / f"{name}.meta.json", sidecar)
        sidecar["meta_file"] = f"{name}.meta.json"
        sidecar["meta_sha256"] = sha256_file(
            stage / sidecar["meta_file"]
        )
        shard_records.append(sidecar)
    return {"shards": shard_records, "stats": dict(total)}


def mix(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    config_sha = sha256_file(args.config)
    domains = hash_domains(config)
    scripts_fp = script_fingerprints()
    candidate_manifest_path = args.work_root / "candidates" / "manifest.json"
    if not candidate_manifest_path.exists():
        raise RuntimeError("candidate manifest is missing")
    candidate_manifest = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    if candidate_manifest.get("status") != "materialized":
        raise ValueError("candidate materialization is not complete")
    if candidate_manifest["config_sha256"] != config_sha:
        raise ValueError("candidate config fingerprint differs")
    if (
        candidate_manifest.get("scripts", {}).get("builder")
        != scripts_fp["builder"]
    ):
        raise ValueError("candidate builder fingerprint differs")
    if candidate_manifest.get("hash_domains") != domains:
        raise ValueError("candidate hash domains differ")

    token_path = tokenizer_path(args.tokenizer, config)
    tokenizer = load_tokenizer(token_path)
    tokenizer_fp = tokenizer_fingerprint(token_path)
    if tokenizer_fp != candidate_manifest["tokenizer"]:
        raise ValueError("candidate tokenizer fingerprint differs")

    full_budget = budget_value(config, "full_loss_target_tokens")
    validation_budget = budget_value(
        config,
        "validation_loss_target_tokens",
    )
    train_quotas = source_targets(config, full_budget, args.mix_key)
    validation_quotas = source_targets(
        config,
        validation_budget,
        args.mix_key,
    )
    sharding = sharding_contract(config, full_budget, validation_budget)
    sequence_length = int(config["sequence_length"])
    seed = sampling_seed(config)
    num_train_shards = int(sharding["num_train_shards"])
    num_validation_shards = int(sharding["num_validation_shards"])
    fingerprint = stable_sha256(
        BUILDER_VERSION,
        candidate_manifest["fingerprint"],
        config_sha,
        canonical_json(tokenizer_fp),
        args.mix_key,
        num_train_shards,
        num_validation_shards,
        canonical_json(domains),
        canonical_json(scripts_fp["builder"]),
        canonical_json(sharding),
    )

    final_manifest_path = args.output_root / "manifest.json"
    if final_manifest_path.exists():
        old = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        if (
            old.get("fingerprint") == fingerprint
            and old.get("status") == "pending_external_audit"
        ):
            print(
                canonical_json(
                    {
                        "status": "already_mixed_pending_external_audit",
                        "path": str(args.output_root),
                    }
                )
            )
            return 0
        raise RuntimeError("different output already exists")
    if args.output_root.exists():
        raise RuntimeError("incomplete output root already exists")

    stage = args.output_root.with_name(
        f".{args.output_root.name}.tmp-{fingerprint[:12]}"
    )
    if stage.exists():
        if not args.restart_incomplete:
            raise RuntimeError(f"incomplete mix stage exists: {stage}")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    selected = stage / ".selected"
    selected.mkdir()
    seen = SeenStore(stage / ".seen.sqlite3")
    paths = candidate_paths_by_source(candidate_manifest)
    train_results: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    try:
        for source_id in sorted(train_quotas):
            train_results[source_id] = select_source(
                merge_sorted_candidate_files(paths[source_id]),
                train_quotas[source_id],
                "train",
                selected / f"train-{safe_name(source_id)}.jsonl",
                tokenizer,
                sequence_length,
                seen,
            )
        for source_id in sorted(validation_quotas):
            quota = validation_quotas[source_id]
            if not quota:
                continue
            run_dir = stage / f".validation-{safe_name(source_id)}-runs"

            def validation_key(
                record: dict[str, Any],
                sid: str = source_id,
            ) -> str:
                return stable_sha256(
                    domains["split"],
                    seed,
                    sid,
                    record["candidate_id"],
                )

            runs, cleanup = externally_sort_records(
                paths[source_id],
                validation_key,
                run_dir,
                f"validation-{safe_name(source_id)}",
                args.sort_memory_mb * 1024 * 1024,
            )
            try:
                validation_results[source_id] = select_source(
                    iter_records_from_runs(runs),
                    quota,
                    "validation",
                    selected
                    / f"validation-{safe_name(source_id)}.jsonl",
                    tokenizer,
                    sequence_length,
                    seen,
                )
            finally:
                cleanup()
                shutil.rmtree(run_dir, ignore_errors=True)

        train_shards = write_final_shards(
            [
                Path(result["output"])
                for _, result in sorted(train_results.items())
            ],
            "train",
            num_train_shards,
            stage,
            seed,
            sequence_length,
            args.sort_memory_mb * 1024 * 1024,
            domains["mix"],
            str(sharding["filename_format"]),
        )
        validation_shards = (
            write_final_shards(
                [
                    Path(result["output"])
                    for _, result in sorted(validation_results.items())
                ],
                "validation",
                num_validation_shards,
                stage,
                seed,
                sequence_length,
                args.sort_memory_mb * 1024 * 1024,
                domains["mix"],
                str(sharding["filename_format"]),
            )
            if validation_budget
            else {"shards": [], "stats": {}}
        )

        manifest = {
            "builder_version": BUILDER_VERSION,
            "candidate_fingerprint": candidate_manifest["fingerprint"],
            "candidate_manifest": str(candidate_manifest_path),
            "candidate_manifest_sha256": sha256_file(
                candidate_manifest_path
            ),
            "config": str(args.config),
            "config_sha256": config_sha,
            "created_at": utc_now(),
            "external_audit": {
                "benchmark_contamination": "pending",
                "required_before_release": True,
            },
            "fingerprint": fingerprint,
            "hash_domains": domains,
            "loader_data_path": str(
                args.output_root / str(sharding["train_glob"])
            ),
            "loader_validation_path": str(
                args.output_root / str(sharding["validation_glob"])
            ),
            "mix_key": args.mix_key,
            "schema_version": SCHEMA_VERSION,
            "scripts": scripts_fp,
            "sequence_length": sequence_length,
            "sharding": sharding,
            "status": "pending_external_audit",
            "tokenizer": tokenizer_fp,
            "train": {
                "budget_loss_target_tokens": full_budget,
                "selection": train_results,
                **train_shards,
            },
            "validation": {
                "budget_loss_target_tokens": validation_budget,
                "selection": validation_results,
                **validation_shards,
            },
        }
        atomic_json(stage / "manifest.json", manifest)
        seen.close()
        shutil.rmtree(selected)
        (stage / ".seen.sqlite3").unlink(missing_ok=True)
        stage.replace(args.output_root)
    except Exception:
        try:
            seen.close()
        except sqlite3.ProgrammingError:
            pass
        raise

    print(
        canonical_json(
            {
                "status": "pending_external_audit",
                "output": str(args.output_root),
                "manifest_sha256": sha256_file(
                    args.output_root / "manifest.json"
                ),
            }
        )
    )
    return 0


def loader_dry_run(
    hook: str | None,
    output_root: Path,
    data_glob: str,
    tokenizer,
    sequence_length: int,
) -> dict[str, Any]:
    quoted_glob = str(output_root / data_glob)
    if hook:
        file_name, separator, class_name = hook.partition(":")
        if not separator:
            raise ValueError("--loader-hook must be path.py:Class")
        module_path = Path(file_name).resolve()
        spec = importlib.util.spec_from_file_location(
            "minimind_pretrain_loader_hook",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dataset_class = getattr(module, class_name)
        dataset = dataset_class(
            quoted_glob,
            tokenizer,
            max_length=sequence_length,
        )
        if len(dataset) <= 0:
            raise ValueError("MiniMind loader returned an empty dataset")
        input_ids, labels = dataset[0]
        if (
            tuple(input_ids.shape) != (sequence_length,)
            or tuple(labels.shape) != (sequence_length,)
        ):
            raise ValueError(
                f"MiniMind loader shapes: "
                f"{input_ids.shape}, {labels.shape}"
            )
        return {
            "hook": hook,
            "quoted_glob": quoted_glob,
            "rows": len(dataset),
            "sample_shape": list(input_ids.shape),
            "status": "ok",
        }

    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files=quoted_glob,
        split="train",
        streaming=True,
    )
    first = next(iter(dataset))
    if set(first) != {"text"}:
        raise ValueError(f"loader columns: {sorted(first)}")
    return {
        "hook": "datasets-streaming-json",
        "quoted_glob": quoted_glob,
        "status": "ok",
    }


def verify(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    config_sha = sha256_file(args.config)
    scripts_fp = script_fingerprints()
    domains = hash_domains(config)
    full_budget = budget_value(config, "full_loss_target_tokens")
    validation_budget = budget_value(
        config,
        "validation_loss_target_tokens",
    )
    sharding = sharding_contract(
        config,
        full_budget,
        validation_budget,
    )
    manifest_path = args.output_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("final manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pending_external_audit":
        raise ValueError(
            "manifest must remain pending_external_audit"
        )
    if (args.output_root / "_SUCCESS").exists():
        raise ValueError("unexpected _SUCCESS marker")
    if manifest["config_sha256"] != config_sha:
        raise ValueError("config SHA differs")
    if (
        manifest.get("scripts", {}).get("builder")
        != scripts_fp["builder"]
    ):
        raise ValueError("builder fingerprint differs")
    if manifest.get("hash_domains") != domains:
        raise ValueError("hash domains differ")
    if manifest.get("sharding") != sharding:
        raise ValueError("sharding contract differs")

    token_path = tokenizer_path(args.tokenizer, config)
    tokenizer = load_tokenizer(token_path)
    if tokenizer_fingerprint(token_path) != manifest["tokenizer"]:
        raise ValueError("tokenizer fingerprint differs")
    sequence_length = int(config["sequence_length"])
    max_text_tokens = sequence_length - 2
    seen_db = sqlite3.connect(args.output_root / ".verify-seen.sqlite3")
    seen_db.execute(
        "CREATE TABLE IF NOT EXISTS content "
        "(hash TEXT PRIMARY KEY, split TEXT NOT NULL)"
    )
    seen_db.execute("DELETE FROM content")
    reports: dict[str, Any] = {}
    try:
        for split in ("train", "validation"):
            expected_names = set(sharding["names"][split])
            actual_names = {
                str(sidecar["file"])
                for sidecar in manifest[split]["shards"]
            }
            if actual_names != expected_names:
                raise ValueError(
                    f"{split} shard names differ: "
                    f"{sorted(actual_names)} != {sorted(expected_names)}"
                )
            expected = int(
                manifest[split]["budget_loss_target_tokens"]
            )
            total: dict[str, int] = defaultdict(int)
            shard_reports = []
            for sidecar in manifest[split]["shards"]:
                path = args.output_root / sidecar["file"]
                provenance = (
                    args.output_root / sidecar["provenance_file"]
                )
                meta = args.output_root / sidecar["meta_file"]
                if sha256_file(path) != sidecar["sha256"]:
                    raise ValueError(f"shard SHA differs: {path}")
                if (
                    sha256_file(provenance)
                    != sidecar["provenance_sha256"]
                ):
                    raise ValueError(
                        f"provenance SHA differs: {provenance}"
                    )
                if sha256_file(meta) != sidecar["meta_sha256"]:
                    raise ValueError(f"meta SHA differs: {meta}")
                counts: dict[str, int] = defaultdict(int)
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as data_handle, provenance.open(
                    "r",
                    encoding="utf-8",
                ) as provenance_handle:
                    for line_number, pair in enumerate(
                        zip(data_handle, provenance_handle, strict=True),
                        1,
                    ):
                        data_line, provenance_line = pair
                        try:
                            row = json.loads(data_line)
                            provenance_row_value = json.loads(
                                provenance_line
                            )
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"{path}:{line_number} invalid JSON"
                            ) from error
                        if (
                            set(row) != {"text"}
                            or not isinstance(row["text"], str)
                        ):
                            raise ValueError(
                                f"{path}:{line_number} is not text-only"
                            )
                        token_ids = tokenizer(
                            row["text"],
                            add_special_tokens=False,
                        ).input_ids
                        if (
                            not token_ids
                            or len(token_ids) > max_text_tokens
                        ):
                            raise ValueError(
                                f"{path}:{line_number} text tokens="
                                f"{len(token_ids)}"
                            )
                        content_hash = token_content_sha256(token_ids)
                        if (
                            content_hash
                            != provenance_row_value["content_sha256"]
                        ):
                            raise ValueError(
                                f"{path}:{line_number} provenance hash"
                            )
                        try:
                            seen_db.execute(
                                "INSERT INTO content(hash, split) "
                                "VALUES (?, ?)",
                                (content_hash, split),
                            )
                        except sqlite3.IntegrityError as error:
                            prior = seen_db.execute(
                                "SELECT split FROM content WHERE hash=?",
                                (content_hash,),
                            ).fetchone()
                            raise ValueError(
                                f"duplicate training-visible tokens: "
                                f"{path}:{line_number}; prior={prior}"
                            ) from error
                        counts["rows"] += 1
                        counts["text_tokens"] += len(token_ids)
                        counts["loss_target_tokens"] += len(token_ids) + 1
                        counts["nonpad_input_tokens"] += len(token_ids) + 2
                        counts["padded_compute_tokens"] += sequence_length
                for key, value in counts.items():
                    total[key] += value
                for key in (
                    "rows",
                    "text_tokens",
                    "loss_target_tokens",
                    "nonpad_input_tokens",
                    "padded_compute_tokens",
                ):
                    if int(counts[key]) != int(
                        sidecar["stats"].get(key, 0)
                    ):
                        raise ValueError(
                            f"{path} recount {key}: {counts[key]} != "
                            f"{sidecar['stats'].get(key, 0)}"
                        )
                shard_reports.append(
                    {
                        "file": sidecar["file"],
                        "stats": dict(counts),
                        "status": "ok",
                    }
                )
            if total["loss_target_tokens"] != expected:
                raise ValueError(
                    f"{split} budget: "
                    f"{total['loss_target_tokens']} != {expected}"
                )
            reports[split] = {
                "expected_loss_target_tokens": expected,
                "shards": shard_reports,
                "stats": dict(total),
                "status": "ok",
            }
        seen_db.commit()
    finally:
        seen_db.close()
        (args.output_root / ".verify-seen.sqlite3").unlink(
            missing_ok=True
        )

    loader_hook = args.loader_hook
    if loader_hook is None:
        default_loader = (
            Path(__file__).resolve().parents[3]
            / "dataset/lm_dataset.py"
        )
        if not default_loader.exists():
            raise FileNotFoundError(default_loader)
        loader_hook = f"{default_loader}:PretrainDataset"
    loader = loader_dry_run(
        loader_hook,
        args.output_root,
        str(sharding["train_glob"]),
        tokenizer,
        sequence_length,
    )
    report = {
        "created_at": utc_now(),
        "external_audit": {
            "benchmark_contamination": "pending",
        },
        "hash_domains": domains,
        "independent_tokenizer_recount": reports,
        "loader_dry_run": loader,
        "manifest_sha256": sha256_file(manifest_path),
        "scripts": scripts_fp,
        "sharding": sharding,
        "status": "pending_external_audit",
    }
    report_path = args.report or args.output_root / "verification.json"
    atomic_json(report_path, report)
    print(
        canonical_json(
            {
                "status": "pending_external_audit",
                "report": str(report_path),
            }
        )
    )
    return 0


def write_gzip_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def self_test(args: argparse.Namespace) -> int:
    token_path = args.tokenizer.resolve()
    with tempfile.TemporaryDirectory(
        prefix="minimind-pretrain-v1-selftest-"
    ) as raw:
        root = Path(raw)
        os.environ.setdefault("HF_HOME", str(root / "hf-cache"))
        languages = [
            "Python",
            "C++",
            "C",
            "JavaScript",
            "Java",
            "SQL",
            "PHP",
            "C#",
            "TypeScript",
            "Shell",
            "Swift",
            "Go",
            "Rust",
            "Ruby",
        ]
        write_gzip_jsonl(
            root / "web-a.jsonl.gz",
            [
                {
                    "text": (
                        f"Web sample {index}, deterministic chunk "
                        "and resume test. "
                    )
                    * 4
                }
                for index in range(500)
            ],
        )
        write_gzip_jsonl(
            root / "web-b.jsonl.gz",
            [
                {
                    "text": (
                        f"Second web object {index}, unique "
                        "multi-object content. "
                    )
                    * 4
                }
                for index in range(500)
            ],
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "text": (
                            f"Exercise {index}: prove {index}+1 is "
                            f"greater than {index}. "
                        )
                        * 3
                    }
                    for index in range(700)
                ]
            ),
            root / "math.parquet",
        )
        stack_rows = []
        for index in range(700):
            language = languages[index % len(languages)]
            stack_rows.append(
                {
                    "files": [
                        {
                            "content": (
                                f"// unique {index}\n"
                                f"function value_{index}() "
                                f"{{ return {index}; }}\n"
                            )
                            * 4,
                            "detected_licenses": ["MIT"],
                            "is_vendor": False,
                            "language": language,
                            "license_type": "permissive",
                            "file_path": f"src/example_{index}.txt",
                        },
                        {
                            "content": "vendored content",
                            "detected_licenses": ["MIT"],
                            "is_vendor": True,
                            "language": language,
                            "license_type": "permissive",
                            "file_path": "vendor/dependency.txt",
                        },
                    ]
                }
            )
        pq.write_table(
            pa.Table.from_pylist(stack_rows),
            root / "stack.parquet",
        )

        config = {
            "schema_version": 2,
            "tokenizer": str(token_path),
            "sequence_length": 64,
            "budgets": {
                "full_loss_target_tokens": 6000,
                "validation_loss_target_tokens": 600,
            },
            "active_v1_mix": {
                "web": {"weight": 0.4, "sources": ["web"]},
                "math": {"weight": 0.3, "sources": ["math"]},
                "code": {"weight": 0.3, "sources": ["code"]},
            },
            "sampling": {
                "seed": 17,
                "sample_hash": "sha256",
                "sample_hash_domain": "selftest-sample",
                "mix_hash_domain": "selftest-mix",
                "split_hash_domain": "selftest-split",
                "candidate_multiplier": 2.0,
            },
            "sharding": {
                "assignment": (
                    "deterministic_hash_order_row_round_robin"
                ),
                "expected_mean_train_loss_target_tokens_per_shard": 2000,
                "filename_format": (
                    "{split}-{index:05d}-of-{num_shards:05d}.jsonl"
                ),
                "num_train_shards": 3,
                "num_validation_shards": 1,
                "train_glob": "train-*-of-00003.jsonl",
                "validation_glob": "validation-*-of-00001.jsonl",
            },
        }
        sources = {
            "schema_version": 2,
            "endpoint": "file://",
            "sources": [
                {
                    "id": "web",
                    "repo_id": "local/web",
                    "revision": "selftest",
                    "format": "jsonl_gzip",
                    "text_field": "text",
                    "objects": [
                        {
                            "id": "a",
                            "path": str(root / "web-a.jsonl.gz"),
                        },
                        {
                            "id": "b",
                            "path": str(root / "web-b.jsonl.gz"),
                        },
                    ],
                },
                {
                    "id": "math",
                    "repo_id": "local/math",
                    "revision": "selftest",
                    "format": "parquet",
                    "text_field": "text",
                    "objects": [
                        {"path": str(root / "math.parquet")}
                    ],
                },
                {
                    "id": "code",
                    "repo_id": "local/stack",
                    "revision": "selftest",
                    "format": "parquet",
                    "record_layout": "stack_v3_files",
                    "text_field": "content",
                    "filters": {
                        "license_type": "permissive",
                        "is_vendor": False,
                        "require_detected_licenses": True,
                        "allowed_languages": languages,
                        "allowed_languages_count": 14,
                    },
                    "objects": [
                        {"path": str(root / "stack.parquet")}
                    ],
                },
            ],
        }
        config_path = root / "config.yaml"
        sources_path = root / "sources.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        sources_path.write_text(
            yaml.safe_dump(sources, sort_keys=False),
            encoding="utf-8",
        )
        work_root = root / "work"
        output_root = root / "output"
        materialize_args = argparse.Namespace(
            config=config_path,
            sources_config=sources_path,
            tokenizer=token_path,
            work_root=work_root,
            mix_key="active_v1_mix",
            sort_memory_mb=4,
        )
        materialize(materialize_args)
        mix(
            argparse.Namespace(
                config=config_path,
                tokenizer=token_path,
                work_root=work_root,
                output_root=output_root,
                mix_key="active_v1_mix",
                sort_memory_mb=4,
                restart_incomplete=False,
            )
        )
        default_loader = (
            Path.cwd() / "dataset/lm_dataset.py"
        )
        hook = (
            f"{default_loader}:PretrainDataset"
            if default_loader.exists()
            else None
        )
        verify(
            argparse.Namespace(
                config=config_path,
                tokenizer=token_path,
                output_root=output_root,
                loader_hook=hook,
                report=root / "verification.json",
            )
        )
        if materialize(materialize_args) != 0:
            raise AssertionError("resume test failed")
        if (output_root / "_SUCCESS").exists():
            raise AssertionError("self-test created forbidden _SUCCESS")
        print(
            canonical_json(
                {
                    "formats": [
                        "jsonl_gzip",
                        "parquet",
                        "stack_v3_files",
                    ],
                    "resume": "verified",
                    "status": "self_test_ok",
                }
            )
        )
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--mix-key", default="active_v1_mix")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic, resumable MiniMind Pretrain v1 data builder"
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    materialize_parser = subparsers.add_parser("materialize")
    add_common(materialize_parser)
    materialize_parser.add_argument(
        "--sources-config",
        "--shards-config",
        dest="sources_config",
        type=Path,
        required=True,
    )
    materialize_parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
    )
    materialize_parser.add_argument(
        "--sort-memory-mb",
        type=int,
        default=256,
    )
    materialize_parser.set_defaults(function=materialize)

    mix_parser = subparsers.add_parser("mix")
    add_common(mix_parser)
    mix_parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
    )
    mix_parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    mix_parser.add_argument(
        "--sort-memory-mb",
        type=int,
        default=256,
    )
    mix_parser.add_argument(
        "--restart-incomplete",
        action="store_true",
    )
    mix_parser.set_defaults(function=mix)

    verify_parser = subparsers.add_parser("verify")
    add_common(verify_parser)
    verify_parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    verify_parser.add_argument("--loader-hook")
    verify_parser.add_argument("--report", type=Path)
    verify_parser.set_defaults(function=verify)

    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
    )
    self_test_parser.set_defaults(function=self_test)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())