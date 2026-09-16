#!/usr/bin/env python3
"""Reproducible Stage1 audit of MiniMind tokenizer and dataset semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import (  # noqa: E402
    AgentRLDataset,
    DPODataset,
    PretrainDataset,
    RLAIFDataset,
    SFTDataset,
)


DOMAIN_NAMES = ("chinese", "english", "code", "math", "news", "domain")
CODE_MARKERS = ("def ", "class ", "import ", "return ", "function(", "```", "#include", "console.log")
MATH_MARKERS = ("\\frac", "\\sum", "\\sqrt", "证明", "方程", "函数", "概率", "theorem", "equation")
NEWS_MARKERS = ("记者", "报道", "新华社", "新闻", "近日", "北京时间", "reuters", "reported", "according to")
DOMAIN_MARKERS = ("量子", "细胞", "蛋白", "基因", "医学", "数据库", "算法", "神经网络", "physics", "biology", "protein", "clinical", "database", "algorithm")


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def scan_jsonl(path: Path, sample_size: int, rng: random.Random) -> tuple[dict[str, int], list[dict[str, Any]]]:
    line_count = 0
    invalid = 0
    empty = 0
    reservoir: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line_count += 1
            if not line.strip():
                empty += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if len(reservoir) < sample_size:
                reservoir.append(record)
            else:
                slot = rng.randrange(line_count)
                if slot < sample_size:
                    reservoir[slot] = record
    return {"line_count": line_count, "invalid_json_lines": invalid, "empty_lines": empty}, reservoir


def conversation_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations") or []
    return "\n".join(str(message.get("content") or "") for message in conversations)


def record_text(filename: str, record: dict[str, Any]) -> str:
    if filename.startswith("pretrain"):
        return str(record.get("text") or "")
    if filename == "dpo.jsonl":
        messages = (record.get("chosen") or []) + (record.get("rejected") or [])
        return "\n".join(str(message.get("content") or "") for message in messages)
    text = conversation_text(record)
    if filename.startswith("agent_rl"):
        text += "\n" + str(record.get("gt") or "")
    return text


def classify_domain(text: str) -> str | None:
    lowered = text.lower()
    if any(marker in lowered for marker in CODE_MARKERS):
        return "code"
    if any(marker in lowered for marker in MATH_MARKERS):
        return "math"
    if any(marker in lowered for marker in NEWS_MARKERS):
        return "news"
    if any(marker in lowered for marker in DOMAIN_MARKERS):
        return "domain"
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return None
    cjk = sum("\u4e00" <= character <= "\u9fff" for character in visible)
    latin = sum(character.isascii() and character.isalpha() for character in visible)
    if cjk / len(visible) >= 0.35:
        return "chinese"
    if latin / len(visible) >= 0.55:
        return "english"
    return None


def parse_tools(messages: list[dict[str, Any]]) -> Any:
    if messages and messages[0].get("role") == "system" and messages[0].get("tools"):
        value = messages[0]["tools"]
        return json.loads(value) if isinstance(value, str) else value
    return None


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for raw_message in messages:
        message = dict(raw_message)
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        normalized.append(message)
    return normalized


def render_record(tokenizer: Any, filename: str, record: dict[str, Any]) -> str:
    if filename.startswith("pretrain"):
        return str(record.get("text") or "")
    if filename == "dpo.jsonl":
        return tokenizer.apply_chat_template(record.get("chosen") or [], tokenize=False)
    messages = normalize_messages(list(record.get("conversations") or []))
    if filename == "rlaif.jsonl":
        return tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    if filename.startswith("agent_rl"):
        return tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, tools=parse_tools(messages)
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=parse_tools(messages)
    )


def summarize_lengths(lengths: list[int], max_length: int) -> dict[str, float | int]:
    return {
        "sample_count": len(lengths),
        "mean_tokens": round(statistics.fmean(lengths), 3) if lengths else 0.0,
        "p50_tokens": percentile(lengths, 0.50),
        "p90_tokens": percentile(lengths, 0.90),
        "p99_tokens": percentile(lengths, 0.99),
        "max_tokens": max(lengths, default=0),
        "truncated_count": sum(length > max_length for length in lengths),
        "truncated_ratio": round(sum(length > max_length for length in lengths) / len(lengths), 6) if lengths else 0.0,
        "mean_padding_tokens": round(statistics.fmean(max(0, max_length - min(length, max_length)) for length in lengths), 3) if lengths else 0.0,
    }


def tensor_list(value: Any) -> list[int]:
    return [int(item) for item in value.tolist()]


def fixture_transformations(tokenizer: Any, fixture_dir: Path, max_length: int) -> dict[str, Any]:
    random.seed(42)
    pretrain = PretrainDataset(str(fixture_dir / "pretrain.jsonl"), tokenizer, max_length=max_length)
    pretrain_ids, pretrain_labels = pretrain[0]

    random.seed(42)
    sft = SFTDataset(str(fixture_dir / "sft.jsonl"), tokenizer, max_length=max_length)
    sft_record = dict(sft.samples[0])
    sft_prompt = sft.create_chat_prompt(sft_record["conversations"])
    sft_ids, sft_labels = sft[0]

    dpo = DPODataset(str(fixture_dir / "dpo.jsonl"), tokenizer, max_length=max_length)
    dpo_item = dpo[0]

    random.seed(42)
    rlaif = RLAIFDataset(str(fixture_dir / "rlaif.jsonl"), tokenizer, max_length=max_length, thinking_ratio=0.0)
    agent = AgentRLDataset(str(fixture_dir / "agent_rl.jsonl"), tokenizer, max_length=max_length)

    return {
        "pretrain": {
            "source": json.loads((fixture_dir / "pretrain.jsonl").read_text(encoding="utf-8").splitlines()[0]),
            "input_ids": tensor_list(pretrain_ids),
            "labels": tensor_list(pretrain_labels),
            "valid_label_tokens": int((pretrain_labels != -100).sum()),
        },
        "sft": {
            "source": json.loads((fixture_dir / "sft.jsonl").read_text(encoding="utf-8").splitlines()[0]),
            "rendered_prompt": sft_prompt,
            "input_ids": tensor_list(sft_ids),
            "labels": tensor_list(sft_labels),
            "valid_label_tokens": int((sft_labels != -100).sum()),
            "ignored_label_tokens": int((sft_labels == -100).sum()),
            "assistant_only_labels": bool((sft_labels != -100).any()),
        },
        "dpo": {key: tensor_list(value) for key, value in dpo_item.items()},
        "rlaif": rlaif[0],
        "agent_rl": agent[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--fixture-max-length", type=int, default=128)
    parser.add_argument("--audit-sample-size", type=int, default=2000)
    parser.add_argument("--domain-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {}
    sampled: dict[str, list[dict[str, Any]]] = {}
    for item in manifest["files"]:
        filename = item["name"]
        path = Path(item["path"])
        scan, records = scan_jsonl(path, args.audit_sample_size, rng)
        item.update(scan)
        sampled[filename] = records
        lengths = []
        for record in records:
            try:
                rendered = render_record(tokenizer, filename, record)
                lengths.append(len(tokenizer(rendered, add_special_tokens=False).input_ids))
            except Exception:
                continue
        audit[filename] = {**scan, **summarize_lengths(lengths, args.max_length)}

    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in DOMAIN_NAMES}
    candidates: Iterable[tuple[str, dict[str, Any]]] = (
        [("pretrain_t2t_mini.jsonl", record) for record in sampled.get("pretrain_t2t_mini.jsonl", [])]
        + [("sft_t2t_mini.jsonl", record) for record in sampled.get("sft_t2t_mini.jsonl", [])]
    )
    for filename, record in candidates:
        text = record_text(filename, record).strip()
        domain = classify_domain(text)
        if domain and len(buckets[domain]) < args.domain_samples:
            token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
            if token_count:
                buckets[domain].append(
                    {
                        "source_file": filename,
                        "text_sha256": digest_text(text),
                        "characters": len(text),
                        "utf8_bytes": len(text.encode("utf-8")),
                        "tokens": token_count,
                        "chars_per_token": round(len(text) / token_count, 6),
                        "bytes_per_token": round(len(text.encode("utf-8")) / token_count, 6),
                    }
                )

    compression = {}
    for domain, rows in buckets.items():
        ratios = [row["chars_per_token"] for row in rows]
        compression[domain] = {
            "sample_count": len(rows),
            "mean_chars_per_token": round(statistics.fmean(ratios), 6) if ratios else 0.0,
            "samples": rows,
        }

    transformations = fixture_transformations(tokenizer, args.fixture_dir, args.fixture_max_length)
    (args.output_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "compression_metrics.json").write_text(json.dumps(compression, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "sample_transformations.json").write_text(json.dumps(transformations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = [domain for domain, rows in buckets.items() if len(rows) < args.domain_samples]
    result = {
        "status": "pass" if not missing else "partial",
        "missing_or_short_domains": missing,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_bos": tokenizer.bos_token,
        "tokenizer_eos": tokenizer.eos_token,
        "tokenizer_pad": tokenizer.pad_token,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
