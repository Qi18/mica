#!/usr/bin/env python3
"""Build an 8M-target control set from MiniMind's official SFT corpus."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from build_sft_v1 import (
    BuildError,
    canonical_conversation,
    choose_chunk,
    file_sha256,
    initialize_database,
    normalize_prompt,
    select_candidates,
    sha256_bytes,
    stable_json,
    split_for_prompt,
    tokenizer_digest,
    utc_now,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/sft/build_official_pilot_v1.yaml"),
    )
    return parser.parse_args()


def sampled(seed: int, line_number: int, rate: float) -> bool:
    digest = sha256_bytes(f"{seed}|{line_number}".encode())
    return int(digest[:16], 16) / float(2**64) < rate


def source_metadata_sha(config: dict[str, Any]) -> str:
    source = config["source"]
    path = Path(source["path"])
    metadata = path.parent / ".cache" / "huggingface" / "download" / (
        path.name + ".metadata"
    )
    if not metadata.exists():
        raise BuildError(f"source metadata missing: {metadata}")
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise BuildError(f"source metadata is incomplete: {metadata}")
    if lines[0] != source["revision"]:
        raise BuildError("source revision differs from cached metadata")
    if lines[1] != source["blob_sha256"]:
        raise BuildError("source blob sha256 differs from cached metadata")
    return file_sha256(metadata)


def populate(
    connection: sqlite3.Connection,
    config: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    source = config["source"]
    path = Path(source["path"])
    if not path.exists():
        raise BuildError(f"official source is missing: {path}")
    if path.stat().st_size != int(source["bytes"]):
        raise BuildError("official source byte size differs from frozen config")
    if file_sha256(path) != source["blob_sha256"]:
        raise BuildError("official source sha256 differs from frozen config")
    seed = int(config["seed"])
    rate = float(config["sampling"]["rate"])
    maximum_length = int(config["sequence_length"])
    bos_id = tokenizer(
        f"{tokenizer.bos_token}assistant\n",
        add_special_tokens=False,
    ).input_ids
    eos_id = tokenizer(
        f"{tokenizer.eos_token}\n",
        add_special_tokens=False,
    ).input_ids
    report: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            report["raw_rows_scanned"] += 1
            if not sampled(seed, line_number, rate):
                report["sampling_filtered"] += 1
                continue
            report["sampled_rows"] += 1
            try:
                row = json.loads(line)
                if set(row) != {"conversations"}:
                    raise BuildError("top-level keys are not exact")
                origin = f"official_sft_t2t:{line_number}"
                messages = canonical_conversation(row["conversations"])
                chunk, metrics, dropped = choose_chunk(
                    tokenizer,
                    messages,
                    origin,
                    "official",
                    maximum_length,
                    bos_id,
                    eos_id,
                )
                prompt = normalize_prompt(chunk)
                if not prompt:
                    raise BuildError("empty normalized user prompt")
                payload_json = stable_json({"conversations": chunk})
                exact_digest = sha256_bytes(payload_json.encode())
                prompt_digest = sha256_bytes(prompt.encode())
                candidate_id = sha256_bytes(
                    f"{origin}|{payload_json}".encode()
                )
                selection_rank = sha256_bytes(
                    f"rank|{seed}|{candidate_id}".encode()
                )
                split = split_for_prompt(prompt)
                provenance = {
                    "candidate_id": candidate_id,
                    "source_id": "minimind_official_sft_t2t",
                    "repo_id": source["repo_id"],
                    "revision": source["revision"],
                    "license": source["license_status"],
                    "source_record_locator": f"line:{line_number}",
                    "source_record_sha256": sha256_bytes(line.encode()),
                    "upstream_id": str(line_number),
                    "origin_group_id": origin,
                    "raw_sha256": source["blob_sha256"],
                    "transform_chain": [
                        {
                            "id": "official_canonical_complete_turn",
                            "version": "1.0",
                            "status": "passed",
                        }
                    ],
                    "variant_family": "official_complete_turn",
                    "bucket": "official",
                    "split": split,
                    "rendered_tokens": metrics["rendered_tokens"],
                    "shifted_assistant_target_tokens": metrics["valid_targets"],
                }
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        candidate_id,
                        origin,
                        "official",
                        split,
                        selection_rank,
                        metrics["valid_targets"],
                        metrics["rendered_tokens"],
                        prompt,
                        prompt_digest,
                        exact_digest,
                        payload_json,
                        stable_json(provenance),
                    ),
                )
                report["candidate_rows"] += 1
                report["candidate_assistant_tokens"] += metrics["valid_targets"]
                report["dropped_overlength_turns"] += dropped
                split_rows[split] += 1
                split_tokens[split] += metrics["valid_targets"]
            except (
                BuildError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                sqlite3.IntegrityError,
            ) as exc:
                key = f"rejected:{type(exc).__name__}:{str(exc)[:100]}"
                report[key] += 1
            if line_number % 100_000 == 0:
                connection.commit()
    connection.commit()
    if report["raw_rows_scanned"] != int(source["rows"]):
        raise BuildError(
            "official source row count differs from frozen config: "
            f"{report['raw_rows_scanned']} != {source['rows']}"
        )
    return {
        "by_source": {"minimind_official_sft_t2t": dict(report)},
        "candidate_rows_by_split": dict(split_rows),
        "candidate_assistant_tokens_by_split": dict(split_tokens),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_path = (repo_root / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise BuildError("official pilot config schema_version must be 1")
    source_metadata_digest = source_metadata_sha(config)
    tokenizer_path = repo_root / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    database_path = Path(config["paths"]["work_database"])
    output_root = Path(config["paths"]["final_root"])
    connection = initialize_database(database_path)
    try:
        capacity = populate(connection, config, tokenizer)
        budgets = {
            name: int(config["budgets"][name])
            for name in ("train", "validation", "test")
        }
        selected, selection = select_candidates(
            connection,
            budgets,
            {"official": 1.0},
        )
    finally:
        connection.close()
    acceptance_path = repo_root / config["contracts"]["acceptance"]
    manifest = {
        "schema_version": 1,
        "name": config["name"],
        "created_at": utc_now(),
        "status": "pending_external_audit",
        "trainable": False,
        "stage": "official_pilot",
        "seed": int(config["seed"]),
        "sequence_length": int(config["sequence_length"]),
        "budget_unit": "shifted_assistant_loss_target_tokens",
        "budgets": budgets,
        "planned_mix": {"official": 1.0},
        "selection": selection,
        "capacity_profile": capacity,
        "bindings": {
            "build_config": str(args.config),
            "build_config_sha256": file_sha256(config_path),
            "acceptance_config": config["contracts"]["acceptance"],
            "acceptance_config_sha256": file_sha256(acceptance_path),
            "builder": "scripts/data/sft/build_official_pilot_v1.py",
            "builder_sha256": file_sha256(Path(__file__)),
            "tokenizer": "tokenizer",
            "tokenizer_sha256": tokenizer_digest(tokenizer_path),
            "source_metadata_sha256": source_metadata_digest,
        },
        "source_objects": [
            {
                "source_id": "minimind_official_sft_t2t",
                **config["source"],
            }
        ],
        "audit": {
            "required": True,
            "status": "pending",
            "success_marker_may_only_be_written_by": (
                "scripts/data/sft/audit_sft_v1.py"
            ),
        },
    }
    final_path = write_outputs(output_root, selected, manifest)
    print(
        json.dumps(
            {"status": "pending_external_audit", "output": str(final_path)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
