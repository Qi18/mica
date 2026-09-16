#!/usr/bin/env python3
"""Full, read-only audit of the historical P02 MiniMind pretraining file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import time
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from transformers import AutoTokenizer


SCHEMA_VERSION = 1
AUDITOR_VERSION = "p02-full-audit-1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--shared-auditor", type=Path, required=True)
    parser.add_argument("--benchmark-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--near-sample", type=int, default=20480)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_shared(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("shared_pretrain_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import shared auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def chunk_digest(token_ids: list[int], bos_id: int, eos_id: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", bos_id))
    values = array("I", token_ids)
    if sys.byteorder != "little":
        values.byteswap()
    digest.update(values.tobytes())
    digest.update(struct.pack("<I", eos_id))
    return digest.digest()


def percentile(histogram: Counter[int], probability: float) -> int | None:
    total = sum(histogram.values())
    if total == 0:
        return None
    rank = max(1, math.ceil(probability * total))
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= rank:
            return value
    raise AssertionError("unreachable percentile")


def describe(histogram: Counter[int]) -> dict[str, Any]:
    count = sum(histogram.values())
    total = sum(value * occurrences for value, occurrences in histogram.items())
    return {
        "count": count,
        "sum": total,
        "min": min(histogram) if histogram else None,
        "p50": percentile(histogram, 0.50),
        "p90": percentile(histogram, 0.90),
        "p95": percentile(histogram, 0.95),
        "p99": percentile(histogram, 0.99),
        "max": max(histogram) if histogram else None,
        "mean": total / count if count else None,
        "percentile_method": "nearest_rank",
    }


def json_safe_counter(value: Counter[str]) -> dict[str, int]:
    return {key: int(count) for key, count in sorted(value.items())}


def text_script_stats(normalized: str, totals: Counter[str]) -> None:
    has_cjk = False
    has_latin = False
    for character in normalized:
        codepoint = ord(character)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            totals["cjk_chars"] += 1
            has_cjk = True
        elif "a" <= character <= "z":
            totals["ascii_latin_chars"] += 1
            has_latin = True
        elif character.isdigit():
            totals["digit_chars"] += 1
        else:
            totals["other_unicode_alnum_chars"] += 1
    if has_cjk:
        totals["rows_with_cjk"] += 1
    if has_latin:
        totals["rows_with_ascii_latin"] += 1
    if has_cjk and has_latin:
        totals["rows_with_both_cjk_and_ascii_latin"] += 1


def iter_valid_texts(
    path: Path,
    max_rows: int,
    quality: Counter[str],
    file_digest: Any,
) -> Iterable[tuple[int, str, int]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            file_digest.update(raw_line)
            quality["physical_lines"] += 1
            if not raw_line.strip():
                quality["blank_lines"] += 1
                continue
            quality["nonblank_rows"] += 1
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                quality["invalid_utf8_lines"] += 1
                continue
            try:
                row = json.loads(decoded)
            except json.JSONDecodeError:
                quality["invalid_json_lines"] += 1
                continue
            if not isinstance(row, dict) or set(row) != {"text"} or not isinstance(row.get("text"), str):
                quality["invalid_schema_rows"] += 1
                continue
            text = row["text"]
            quality["valid_rows"] += 1
            if not text.strip():
                quality["empty_text_rows"] += 1
            yield line_number, text, len(raw_line)
            if max_rows and quality["valid_rows"] >= max_rows:
                break


def batches(iterator: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_validation_digests(
    path: Path,
    tokenizer: Any,
    batch_size: int,
    sequence_length: int,
) -> tuple[set[bytes], dict[str, Any]]:
    digests: set[bytes] = set()
    rows = 0
    duplicate_chunks = 0
    invalid_rows = 0
    bos_id = int(tokenizer.bos_token_id)
    eos_id = int(tokenizer.eos_token_id)
    pending: list[str] = []

    def consume() -> None:
        nonlocal rows, duplicate_chunks
        if not pending:
            return
        encoded = tokenizer(
            pending,
            add_special_tokens=False,
            max_length=sequence_length - 2,
            truncation=True,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        for token_ids in encoded:
            digest = chunk_digest(list(token_ids), bos_id, eos_id)
            if digest in digests:
                duplicate_chunks += 1
            digests.add(digest)
            rows += 1
        pending.clear()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                invalid_rows += 1
                continue
            pending.append(value["text"])
            if len(pending) >= batch_size:
                consume()
    consume()
    return digests, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "unique_training_visible_chunks": len(digests),
        "duplicate_chunks": duplicate_chunks,
        "invalid_rows": invalid_rows,
    }


def source_counts(value: Any) -> dict[str, dict[str, int]]:
    return {
        source: json_safe_counter(counts)
        for source, counts in sorted(value.items())
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.near_sample <= 0:
        raise ValueError("batch size and near sample must be positive")
    started_at = utc_now()
    started = time.monotonic()
    shared = load_shared(args.shared_auditor)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.model_max_length = 1_000_000_000
    sequence_length = 768
    bos_id = int(tokenizer.bos_token_id)
    eos_id = int(tokenizer.eos_token_id)
    validation_digests, validation_report = load_validation_digests(
        args.validation, tokenizer, args.batch_size, sequence_length
    )

    eval_config = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    patterns: list[dict[str, Any]] = []
    benchmark_sources: list[dict[str, Any]] = []
    benchmark_load_errors: list[dict[str, Any]] = []
    benchmark_validation_errors: list[str] = []
    contamination = None
    near_audit = None
    near_sampler = None
    matching = shared.matching_contract(eval_config)
    benchmark_cache_report: dict[str, Any] | None = None
    if not args.skip_benchmark:
        eval_config_sha256 = sha256_file(args.eval_config)
        shared_auditor_sha256 = sha256_file(args.shared_auditor)
        cache_loaded = False
        if args.benchmark_cache is not None and args.benchmark_cache.is_file():
            cached = json.loads(args.benchmark_cache.read_text(encoding="utf-8"))
            if cached.get("eval_config_sha256") != eval_config_sha256:
                raise RuntimeError("benchmark cache eval-config SHA mismatch")
            if cached.get("shared_auditor_sha256") != shared_auditor_sha256:
                raise RuntimeError("benchmark cache shared-auditor SHA mismatch")
            patterns = cached["patterns"]
            benchmark_sources = cached["sources"]
            benchmark_load_errors = cached["load_errors"]
            cache_loaded = True
        else:
            patterns, benchmark_sources, benchmark_load_errors = shared.load_eval_queries(
                eval_config, Path("/data/cache/huggingface")
            )
            if args.benchmark_cache is not None:
                atomic_json(
                    args.benchmark_cache,
                    {
                        "schema_version": 1,
                        "created_at": utc_now(),
                        "eval_config": str(args.eval_config),
                        "eval_config_sha256": eval_config_sha256,
                        "shared_auditor": str(args.shared_auditor),
                        "shared_auditor_sha256": shared_auditor_sha256,
                        "patterns": patterns,
                        "sources": benchmark_sources,
                        "load_errors": benchmark_load_errors,
                    },
                )
        benchmark_validation_errors = shared.validate_pinned_eval_sources(
            eval_config, benchmark_sources, benchmark_load_errors, False
        )
        if benchmark_validation_errors:
            raise RuntimeError(
                "pinned benchmark validation failed: "
                + "; ".join(benchmark_validation_errors)
            )
        benchmark_cache_report = {
            "path": str(args.benchmark_cache) if args.benchmark_cache else None,
            "loaded_existing": cache_loaded,
            "sha256": (
                sha256_file(args.benchmark_cache)
                if args.benchmark_cache is not None
                else None
            ),
        }
        contamination = shared.ContaminationAudit(patterns, eval_config, matching)
        near_audit = shared.NearAudit(patterns, eval_config)
        near_sampler = shared.StratifiedBottomK(
            args.near_sample,
            int(eval_config["near_duplicate"]["seed"]),
            int(eval_config["near_duplicate"]["min_normalized_chars"]),
        )

    quality: Counter[str] = Counter(
        {
            "physical_lines": 0,
            "blank_lines": 0,
            "nonblank_rows": 0,
            "invalid_utf8_lines": 0,
            "invalid_json_lines": 0,
            "invalid_schema_rows": 0,
            "valid_rows": 0,
            "empty_text_rows": 0,
        }
    )
    token_totals: Counter[str] = Counter(
        {
            "raw_text_tokens": 0,
            "truncated_text_tokens": 0,
            "loss_target_tokens": 0,
            "nonpad_input_tokens": 0,
            "padded_compute_tokens": 0,
            "truncated_rows": 0,
            "discarded_tail_tokens": 0,
            "exact_duplicate_chunks": 0,
            "p03_validation_overlap_occurrences": 0,
        }
    )
    script_totals: Counter[str] = Counter(
        {
            "cjk_chars": 0,
            "ascii_latin_chars": 0,
            "digit_chars": 0,
            "other_unicode_alnum_chars": 0,
            "rows_with_cjk": 0,
            "rows_with_ascii_latin": 0,
            "rows_with_both_cjk_and_ascii_latin": 0,
        }
    )
    raw_token_lengths: Counter[int] = Counter()
    truncated_token_lengths: Counter[int] = Counter()
    character_lengths: Counter[int] = Counter()
    utf8_text_bytes: Counter[int] = Counter()
    token_frequency: Counter[int] = Counter()
    seen_chunks: set[bytes] = set()
    overlap_unique: set[bytes] = set()
    duplicate_examples: list[dict[str, Any]] = []
    validation_overlap_examples: list[dict[str, Any]] = []
    file_digest = hashlib.sha256()
    next_progress = args.progress_every

    iterator = iter_valid_texts(args.data, args.max_rows, quality, file_digest)
    for batch in batches(iterator, args.batch_size):
        texts = [text for _, text, _ in batch]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        for (line_number, text, _), raw_ids in zip(batch, encoded):
            raw_ids = list(raw_ids)
            text_ids = raw_ids[: sequence_length - 2]
            raw_length = len(raw_ids)
            text_length = len(text_ids)
            raw_token_lengths[raw_length] += 1
            truncated_token_lengths[text_length] += 1
            character_lengths[len(text)] += 1
            utf8_text_bytes[len(text.encode("utf-8"))] += 1
            token_totals["raw_text_tokens"] += raw_length
            token_totals["truncated_text_tokens"] += text_length
            token_totals["loss_target_tokens"] += text_length + 1
            token_totals["nonpad_input_tokens"] += text_length + 2
            token_totals["padded_compute_tokens"] += sequence_length
            if raw_length > sequence_length - 2:
                token_totals["truncated_rows"] += 1
                token_totals["discarded_tail_tokens"] += raw_length - (sequence_length - 2)
            token_frequency.update(text_ids)

            digest = chunk_digest(text_ids, bos_id, eos_id)
            if digest in seen_chunks:
                token_totals["exact_duplicate_chunks"] += 1
                if len(duplicate_examples) < 10:
                    duplicate_examples.append(
                        {"line": line_number, "text": text[:300]}
                    )
            else:
                seen_chunks.add(digest)
            if digest in validation_digests:
                token_totals["p03_validation_overlap_occurrences"] += 1
                overlap_unique.add(digest)
                if len(validation_overlap_examples) < 10:
                    validation_overlap_examples.append(
                        {"line": line_number, "text": text[:300]}
                    )

            normalized = shared.normalize(text)
            text_script_stats(normalized, script_totals)
            if contamination is not None:
                contamination.audit(
                    normalized, text, "train", args.data.name, line_number
                )
                near_sampler.offer(
                    normalized, text, "train", args.data.name, line_number
                )

        if args.progress_every and quality["valid_rows"] >= next_progress:
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "rows": quality["valid_rows"],
                        "rows_per_second": quality["valid_rows"] / elapsed,
                        "elapsed_seconds": elapsed,
                        "unique_chunks": len(seen_chunks),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_progress += args.progress_every

    near_sample_report: dict[str, Any] | None = None
    if near_audit is not None:
        for normalized, raw, split, shard, line_number in near_sampler.documents():
            near_audit.audit_document(normalized, raw, split, shard, line_number)
        near_sample_report = {
            "method": "deterministic_bottom_k_single_physical_file",
            "requested_documents": args.near_sample,
            "eligible_documents": sum(near_sampler.eligible.values()),
            "audited_documents": int(near_audit.counts["documents_audited"]),
            "counts": json_safe_counter(near_audit.counts),
            "counts_by_source": source_counts(near_audit.source_counts),
            "examples_by_source": dict(near_audit.examples),
            "boundary": (
                "Near-duplicate matching is sampled, not a full-corpus claim. "
                "The sample size equals 40 P03 train shards x 512 rows."
            ),
        }

    observed_sha256 = file_digest.hexdigest()
    complete_scan = args.max_rows == 0
    expected_sha256 = "31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d"
    expected_rows = 8_468_827
    integrity = {
        "complete_scan": complete_scan,
        "expected_rows": expected_rows,
        "observed_physical_lines": int(quality["physical_lines"]),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "rows_match": complete_scan and quality["physical_lines"] == expected_rows,
        "sha256_match": complete_scan and observed_sha256 == expected_sha256,
        "size_bytes": args.data.stat().st_size,
    }
    if complete_scan and (not integrity["rows_match"] or not integrity["sha256_match"]):
        raise RuntimeError(f"P02 integrity gate failed: {integrity}")

    total_rows = int(quality["valid_rows"])
    loss_targets = int(token_totals["loss_target_tokens"])
    nonpad = int(token_totals["nonpad_input_tokens"])
    padded = int(token_totals["padded_compute_tokens"])
    raw_total = int(token_totals["raw_text_tokens"])
    character_total = sum(length * count for length, count in character_lengths.items())
    utf8_bytes_total = sum(length * count for length, count in utf8_text_bytes.items())
    seq380_loss_targets = sum(
        (min(length, 378) + 1) * count
        for length, count in raw_token_lengths.items()
    )
    seq380_nonpad = sum(
        (min(length, 378) + 2) * count
        for length, count in raw_token_lengths.items()
    )
    seq380_truncated_rows = sum(
        count for length, count in raw_token_lengths.items() if length > 378
    )

    top_tokens = []
    for token_id, count in token_frequency.most_common(30):
        top_tokens.append(
            {
                "token_id": token_id,
                "count": count,
                "decoded": tokenizer.decode([token_id]),
            }
        )
    benchmark_report = None
    if contamination is not None:
        benchmark_report = {
            "matching": matching,
            "patterns_unique": len(patterns),
            "pattern_entries": sum(len(item["entries"]) for item in patterns),
            "sources": benchmark_sources,
            "load_errors": benchmark_load_errors,
            "validation_errors": benchmark_validation_errors,
            "full_scan": {
                "counts": json_safe_counter(contamination.counts),
                "counts_by_source": source_counts(contamination.source_counts),
                "examples_by_source": dict(contamination.examples),
                "scope": "exact and containment over every valid P02 row",
            },
            "near_duplicate": near_sample_report,
        }

    wall_seconds = 17_147.030945
    report = {
        "schema_version": SCHEMA_VERSION,
        "auditor_version": AUDITOR_VERSION,
        "status": "completed" if complete_scan else "smoke_completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "audit_wall_seconds": time.monotonic() - started,
        "inputs": {
            "auditor": str(Path(__file__).resolve()),
            "auditor_sha256": sha256_file(Path(__file__).resolve()),
            "data": str(args.data),
            "tokenizer": str(args.tokenizer),
            "tokenizer_fingerprint": shared.tokenizer_fingerprint(args.tokenizer),
            "validation": validation_report,
            "eval_config": str(args.eval_config),
            "eval_config_sha256": sha256_file(args.eval_config),
            "benchmark_cache": benchmark_cache_report,
            "shared_auditor": str(args.shared_auditor),
            "shared_auditor_sha256": sha256_file(args.shared_auditor),
            "sequence_length": sequence_length,
            "tokenizer_batch_size": args.batch_size,
            "max_rows": args.max_rows,
        },
        "integrity": integrity,
        "quality": json_safe_counter(quality),
        "lengths": {
            "raw_text_tokens": describe(raw_token_lengths),
            "training_visible_text_tokens": describe(truncated_token_lengths),
            "unicode_codepoints": describe(character_lengths),
            "utf8_text_bytes": describe(utf8_text_bytes),
        },
        "token_accounting_actual_seq768": {
            **json_safe_counter(token_totals),
            "unique_training_visible_chunks": len(seen_chunks),
            "unique_vocabulary_tokens_observed": len(token_frequency),
            "vocabulary_size": len(tokenizer),
            "vocabulary_coverage": len(token_frequency) / len(tokenizer),
            "loss_target_utilization": loss_targets / padded if padded else None,
            "nonpad_input_utilization": nonpad / padded if padded else None,
            "discarded_tail_fraction_of_raw_tokens": (
                token_totals["discarded_tail_tokens"] / raw_total if raw_total else None
            ),
            "truncated_row_fraction": (
                token_totals["truncated_rows"] / total_rows if total_rows else None
            ),
            "p03_validation_overlap_unique_chunks": len(overlap_unique),
            "p03_validation_overlap_examples": validation_overlap_examples,
            "exact_duplicate_examples": duplicate_examples,
            "utf8_bytes_per_raw_token": utf8_bytes_total / raw_total if raw_total else None,
            "unicode_codepoints_per_raw_token": character_total / raw_total if raw_total else None,
            "top_training_visible_tokens": top_tokens,
        },
        "counterfactual_seq380_not_trained": {
            "loss_target_tokens": seq380_loss_targets,
            "nonpad_input_tokens": seq380_nonpad,
            "padded_compute_tokens": total_rows * 380,
            "loss_target_utilization": seq380_loss_targets / (total_rows * 380) if total_rows else None,
            "nonpad_input_utilization": seq380_nonpad / (total_rows * 380) if total_rows else None,
            "truncated_rows": seq380_truncated_rows,
            "truncated_row_fraction": seq380_truncated_rows / total_rows if total_rows else None,
            "boundary": "Counterfactual recomputation only; P02 was trained at seq_len=768.",
        },
        "script_composition_proxy": {
            **json_safe_counter(script_totals),
            "boundary": (
                "Unicode-script counts are objective content proxies, not language "
                "labels and not source-mixture provenance."
            ),
        },
        "source_provenance": {
            "repo_id": "jingyaogong/minimind_dataset",
            "revision": "312afb4f76391145c6902f765bb51691c09a12f5",
            "file": "pretrain_t2t.jsonl",
            "dataset_card_description": [
                "general text corpora",
                "reorganized dialogue corpora",
                "distillation supplements",
                "public sources including Jiangshu and Magpie-Align",
            ],
            "per_row_source_labels": False,
            "exact_source_weights": None,
            "status": "not_recoverable_from_the_pinned_artifact",
            "boundary": (
                "The pinned file has only a text field. Exact constituent datasets, "
                "weights, row-level lineage, and per-source licenses cannot be reconstructed."
            ),
        },
        "split_and_overlap": {
            "training_time_independent_validation_split": False,
            "training_time_split_status": "absent",
            "retroactive_shared_validation": validation_report,
            "p02_occurrences_overlapping_shared_validation": int(
                token_totals["p03_validation_overlap_occurrences"]
            ),
            "p02_unique_chunks_overlapping_shared_validation": len(overlap_unique),
            "boundary": (
                "The P03 validation set was built after P02. This overlap audit improves "
                "comparability but does not create a historical P02 validation split."
            ),
        },
        "benchmark_contamination": benchmark_report,
        "derived_training_efficiency": {
            "historical_wall_seconds": wall_seconds,
            "padded_token_slot_throughput_per_second": padded / wall_seconds,
            "effective_loss_target_throughput_per_second": loss_targets / wall_seconds,
            "effective_nonpad_input_throughput_per_second": nonpad / wall_seconds,
            "boundary": (
                "Dataset totals exclude the five sampler-padding repetitions needed to "
                "make 8,468,827 rows divisible across 8 ranks; impact is negligible but "
                "the values are not claimed as exact runtime token counters."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "event": "completed",
                "output": str(args.output),
                "rows": total_rows,
                "loss_target_tokens": loss_targets,
                "duplicates": int(token_totals["exact_duplicate_chunks"]),
                "validation_overlap": int(token_totals["p03_validation_overlap_occurrences"]),
                "wall_seconds": report["audit_wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
