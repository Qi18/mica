#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path("/data/projects/minimind-lab-data-v1")
DATA = Path("/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1")
EXP = REPO / "experiments/00-preparation/D01-training-data-v1-20260828"
DEFAULT_OUTPUT = Path("/data/datasets/minimind-lab/releases/modelscope/minimind-pretrain-v1-1b28-metadata")

SOURCE_BLOCKERS = {
    "chinese_webtext2": [
        "The declared Apache-2.0 tag does not establish a document-level rights chain for all upstream components.",
        "Publication of payload requires an explicit redistribution review and takedown path.",
    ],
    "fineweb_edu": [
        "ODC-By covers database rights, not automatically every source web page.",
        "Publication of payload requires content-rights and privacy review.",
    ],
    "wikipedia_zh": [
        "Current chunks lack article URL, title, page revision, attribution history, and modification notice.",
    ],
    "wikipedia_en": [
        "Current chunks lack article URL, title, page revision, attribution history, and modification notice.",
    ],
    "finemath": [
        "ODC-By covers database rights, not automatically every source web page.",
        "Publication of payload requires content-rights and privacy review.",
    ],
    "stack_v3_code": [
        "Current chunks lack repository, commit, file path, detected license, LICENSE, and NOTICE linkage.",
    ],
}

LICENSE_URLS = {
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "ODC-By-1.0": "https://opendatacommons.org/licenses/by/1-0/",
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "GFDL": "https://www.gnu.org/licenses/fdl-1.3.html",
    "Common-Crawl-ToU": "https://commoncrawl.org/terms-of-use",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def logical_evidence() -> dict[str, Path]:
    return {
        "accepted/_SUCCESS": DATA / "_SUCCESS",
        "accepted/manifest.json": DATA / "manifest.json",
        "reports/pretrain_v1_remix_audit.json": EXP / "pretrain_v1_remix_audit.json",
        "reports/pretrain_v1_remix_verification.json": EXP / "pretrain_v1_remix_verification.json",
        "configs/pretrain_shards_v1.yaml": REPO / "configs/data/pretrain/pretrain_shards_v1.yaml",
        "scripts/audit_pretrain_v1.py": REPO / "scripts/data/pretrain/audit_pretrain_v1.py",
        "scripts/build_pretrain_v1.py": REPO / "scripts/data/pretrain/build_pretrain_v1.py",
        "scripts/remix_pretrain_v1.py": REPO / "scripts/data/pretrain/remix_pretrain_v1.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output

    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    required = logical_evidence()
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        raise SystemExit("missing evidence: " + ", ".join(missing))

    manifest = load_json(DATA / "manifest.json")
    marker = load_json(DATA / "_SUCCESS")
    audit = load_json(EXP / "pretrain_v1_remix_audit.json")
    verification = load_json(EXP / "pretrain_v1_remix_verification.json")
    sources = yaml.safe_load((REPO / "configs/data/pretrain/pretrain_shards_v1.yaml").read_text(encoding="utf-8"))

    if marker.get("status") != "accepted":
        raise SystemExit("accepted dataset marker is not accepted")
    if audit["budgets"]["train"]["passed"] is not True or audit["budgets"]["validation"]["passed"] is not True:
        raise SystemExit("token budget audit did not pass")

    train_stats = verification["independent_tokenizer_recount"]["train"]["stats"]
    validation_stats = verification["independent_tokenizer_recount"]["validation"]["stats"]
    contamination = audit["benchmark_contamination"]
    contamination_totals = contamination["totals"]

    roster = {
        "schema_version": 1,
        "publication_scope": "metadata-only",
        "payload_included": False,
        "composite_license": "other",
        "sources": [],
    }
    for source in sources["sources"]:
        roster["sources"].append(
            {
                "id": source["id"],
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "release": source.get("release"),
                "declared_license": source["declared_license"],
                "format": source["format"],
                "object_count": len(source["objects"]),
                "publication_status": "blocked-pending-source-specific-release-gate",
                "blockers": SOURCE_BLOCKERS[source["id"]],
            }
        )

    public_manifest = {
        "schema_version": 1,
        "release_name": "minimind-pretrain-v1-1b28",
        "publication_scope": {
            "status": "metadata-only",
            "payload_included": False,
            "training_text_included": False,
            "provenance_sidecars_included": False,
            "reason": "Source-specific redistribution, attribution, and privacy gates are incomplete.",
        },
        "technical_acceptance": {
            "status": marker["status"],
            "accepted_at": marker.get("accepted_at"),
            "dataset_fingerprint": marker["dataset_fingerprint"],
            "sequence_length": manifest["sequence_length"],
        },
        "sharding": {
            "train_shards": manifest["sharding"]["num_train_shards"],
            "validation_shards": manifest["sharding"]["num_validation_shards"],
            "assignment": manifest["sharding"]["assignment"],
        },
        "train": train_stats,
        "validation": validation_stats,
        "source_quotas_loss_target_tokens": audit["build_evidence"]["expected_source_quotas"],
    }

    verification_summary = {
        "schema_version": 1,
        "source_report_sha256": sha256(EXP / "pretrain_v1_remix_verification.json"),
        "technical_status": marker["status"],
        "verification_report_status": verification["status"],
        "manifest_sha256": verification["manifest_sha256"],
        "independent_tokenizer_recount": {
            "train": {"status": verification["independent_tokenizer_recount"]["train"]["status"], "stats": train_stats},
            "validation": {
                "status": verification["independent_tokenizer_recount"]["validation"]["status"],
                "stats": validation_stats,
            },
        },
        "loader_dry_run": {
            "status": verification["loader_dry_run"]["status"],
            "rows": verification["loader_dry_run"]["rows"],
            "sample_shape": verification["loader_dry_run"]["sample_shape"],
        },
        "sharding": manifest["sharding"],
    }

    audit_summary = {
        "schema_version": 1,
        "source_report_sha256": sha256(EXP / "pretrain_v1_remix_audit.json"),
        "auditor_version": audit["auditor_version"],
        "token_budgets": audit["budgets"],
        "build_evidence_passed": audit["build_evidence"]["passed"],
        "benchmark_contamination": {
            "formal_pinned_snapshot_enforced": contamination["formal_pinned_snapshot_enforced"],
            "eval_patterns_unique": contamination["eval_patterns_unique"],
            "exact_containment_documents_audited": contamination_totals["exact_containment_documents_audited"],
            "exact_overlap": contamination_totals["exact_overlap"],
            "containment_overlap": contamination_totals["containment_overlap"],
            "near_duplicate_overlap": contamination_totals["near_duplicate_overlap"],
            "near_scope": {
                "is_full_pretrain_document_scan": contamination["near_scope"]["is_full_pretrain_document_scan"],
                "mode": contamination["near_scope"]["mode"],
                "sample_per_shard": contamination["near_scope"]["sample_per_shard"],
                "documents_audited": contamination["near_scope"]["documents_audited"],
                "eligible_documents": contamination["near_scope"]["eligible_documents"],
            },
        },
        "release_gate": {
            "redistribution_review": "blocked",
            "record_level_attribution": "blocked",
            "privacy_and_pii_review": "blocked",
            "payload_publishable": False,
        },
    }

    readme = f"""---
license: other
task:
  - text-generation
tags:
  - MiniMind
  - pretraining
  - metadata-only
---

# MiniMind Pretrain v1 1.28B — Metadata-only Release

这是 MiniMind-Lab 预训练数据实验的可审阅元数据仓。当前版本只发布构建配置的摘要、固定来源版本、技术验收结果和证据哈希；不包含训练或验证正文，也不包含现有 provenance sidecar。

原始实验产物已经通过技术验收：训练集 {train_stats["loss_target_tokens"]:,} 个 loss-target tokens、{train_stats["rows"]:,} 行、40 个 shard；验证集 {validation_stats["loss_target_tokens"]:,} 个 loss-target tokens、{validation_stats["rows"]:,} 行、1 个 shard，序列长度为 {manifest["sequence_length"]}。固定 benchmark 污染审计对 exact equality 和 query-in-document containment 执行了全量扫描，命中均为 0；near-duplicate 结果也是 0，但 near 部分仅按每个物理 shard 512 条做确定性抽样，不是全库扫描。

## 为什么没有数据正文

当前混合语料包含 ChineseWebText2.0、FineWeb-Edu、Wikipedia zh/en、FineMath 和 The Stack v3。现有 text-only shard 没有保留完成再分发所需的逐记录字段，例如 Wikipedia 的文章 URL/修订历史，Stack 的仓库、commit、文件路径、许可证和 NOTICE，以及网页语料的原始 URL。私有仓库不替代版权、归因或隐私审查，因此正文上传被发布门禁阻止。

## 本仓内容

- metadata/source_roster.public.yaml：来源、revision、声明许可和发布阻断项
- metadata/manifest.public.json：无机器绝对路径的规模与技术验收摘要
- reports：独立 tokenizer recount、loader dry-run、污染审计边界
- checksums：原始 L20 证据及本发布包的 SHA256

## 后续发布条件

Wikipedia 需要重建 article-level attribution；Stack 需要重建 file-level license/NOTICE provenance；网页语料需要完成内容权利、PII、opt-out 和 takedown 门禁。所有条件通过后，数据正文应按来源拆分发布，而不是给整个混合库套用单一许可证。

本仓的 license 字段设为 other，只描述组合发布状态，不改变任何上游内容的许可。详细链接见 THIRD_PARTY_NOTICES.md。
"""

    notices = """# Third-party notices

This repository contains metadata only. It does not redistribute source text.

| Source | Pinned repository | Declared license or terms |
|---|---|---|
| ChineseWebText2.0 | CASIA-LM/ChineseWebText2.0 | Apache-2.0; dataset-level declaration requires further rights-chain review |
| FineWeb-Edu | HuggingFaceFW/fineweb-edu | ODC-By-1.0 and Common Crawl Terms of Use |
| Wikipedia zh/en | wikimedia/wikipedia | CC BY-SA and GFDL; article-level attribution is required |
| FineMath | HuggingFaceTB/finemath | ODC-By-1.0 and Common Crawl Terms of Use |
| The Stack v3 | HuggingFaceCode/stack-v3-train | ODC-By-1.0 plus each original file license and NOTICE |

## Authoritative terms

- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- Open Data Commons Attribution License 1.0: https://opendatacommons.org/licenses/by/1-0/
- Creative Commons Attribution-ShareAlike 3.0: https://creativecommons.org/licenses/by-sa/3.0/
- GNU Free Documentation License: https://www.gnu.org/licenses/fdl-1.3.html
- Common Crawl Terms of Use: https://commoncrawl.org/terms-of-use

No statement in this repository grants rights beyond the applicable upstream terms.
"""

    report = f"""# D01 build and release summary

## Technical acceptance

- Accepted marker: {marker["status"]}
- Dataset fingerprint: {marker["dataset_fingerprint"]}
- Sequence length: {manifest["sequence_length"]}
- Train: {train_stats["rows"]:,} rows, {train_stats["loss_target_tokens"]:,} loss-target tokens, 40 shards
- Validation: {validation_stats["rows"]:,} rows, {validation_stats["loss_target_tokens"]:,} loss-target tokens, 1 shard
- Loader dry-run: {verification["loader_dry_run"]["status"]}
- Exact overlap: {contamination_totals["exact_overlap"]}
- Containment overlap: {contamination_totals["containment_overlap"]}
- Near-duplicate overlap: {contamination_totals["near_duplicate_overlap"]}

## Audit boundary

Exact and containment checks scanned every valid pretraining document. Near-duplicate checking used deterministic bottom-k sampling of 512 eligible documents per physical shard and is not a full-corpus near-duplicate scan.

## Publication decision

Technical acceptance does not imply redistribution approval. The current ModelScope release is metadata-only because source-specific attribution, rights, PII, opt-out, and takedown gates remain incomplete. No training text, validation text, current provenance sidecar, raw input, cache, command log, PID, credential, or machine-local absolute path is included.
"""

    write_text(output / "README.md", readme)
    write_text(output / "THIRD_PARTY_NOTICES.md", notices)
    write_text(
        output / "LICENSES/README.md",
        "This metadata-only release uses license: other. Authoritative upstream license and terms links are listed in THIRD_PARTY_NOTICES.md.",
    )
    write_text(
        output / "metadata/source_roster.public.yaml",
        yaml.safe_dump(roster, sort_keys=False, allow_unicode=True),
    )
    dump_json(output / "metadata/manifest.public.json", public_manifest)
    dump_json(output / "reports/pretrain_v1_remix_verification.summary.json", verification_summary)
    dump_json(output / "reports/pretrain_v1_remix_audit.summary.json", audit_summary)
    write_text(output / "reports/D01_build_report.public.md", report)

    original_checksums = [f"{sha256(path)}  {label}" for label, path in sorted(required.items())]
    write_text(output / "checksums/original_evidence_sha256.txt", "\n".join(original_checksums))

    forbidden = (b"/data/", b"/root/", b"xuzKg")
    modelscope_token = re.compile(rb"\bms-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
    for path in sorted(output.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            for marker_bytes in forbidden:
                if marker_bytes in payload:
                    raise SystemExit(f"forbidden marker in release file {path.name}")
            if modelscope_token.search(payload):
                raise SystemExit(f"ModelScope token pattern in release file {path.name}")

    release_files = [
        path for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "release_sha256.txt"
    ]
    release_checksums = [
        f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in release_files
    ]
    write_text(output / "checksums/release_sha256.txt", "\n".join(release_checksums))

    print(json.dumps({
        "output": str(output),
        "files": sum(1 for path in output.rglob("*") if path.is_file()),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "payload_included": False,
        "technical_status": marker["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
