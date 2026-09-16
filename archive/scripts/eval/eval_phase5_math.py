#!/usr/bin/env python3
"""Evaluate Phase5 verifier math with greedy pass@1 and empirical sampled pass@k."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase5_math_common import parse_choice, strict_choice  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode_batch(model, tokenizer, prompts, max_new_tokens, sample_k, seed):
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True, return_token_type_ids=False).to(model.device)
    torch.manual_seed(seed)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample_k > 1,
        "num_return_sequences": sample_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample_k > 1:
        kwargs.update(temperature=0.8, top_p=0.95)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(**inputs, **kwargs)
    completion = output[:, inputs.input_ids.shape[1]:]
    decoded = []
    for ids in completion:
        eos = (ids == tokenizer.eos_token_id).nonzero()
        length = int(eos[0].item() + 1) if len(eos) else len(ids)
        decoded.append({"text": tokenizer.decode(ids[:length], skip_special_tokens=True).strip(), "tokens": length})
    return decoded


def rate(rows, key):
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def grouped(rows, key):
    result = {}
    for value in sorted({str(row[key]) for row in rows}):
        part = [row for row in rows if str(row[key]) == value]
        result[value] = {"samples": len(part), "pass_at_1": rate(part, "greedy_correct")}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    rows = read_jsonl(args.data / f"{args.split}.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda().eval()
    details = []
    for start in range(0, len(rows), args.batch_size):
        part = rows[start:start + args.batch_size]
        prompts = [row["prompt"] for row in part]
        greedy = decode_batch(model, tokenizer, prompts, args.max_new_tokens, 1, args.seed + start)
        sampled = decode_batch(model, tokenizer, prompts, args.max_new_tokens, args.sample_k, args.seed + 100000 + start)
        for i, row in enumerate(part):
            g = greedy[i]
            samples = sampled[i * args.sample_k:(i + 1) * args.sample_k]
            g_choice = parse_choice(g["text"])
            s_choices = [parse_choice(item["text"]) for item in samples]
            details.append({
                "task_id": row["task_id"],
                "operation": row["operation"],
                "template_id": row["template_id"],
                "answer_position": row["answer"],
                "answer": row["answer"],
                "greedy_text": g["text"],
                "greedy_choice": g_choice,
                "greedy_tokens": g["tokens"],
                "choice_valid_at_1": g_choice is not None,
                "strict_format_at_1": strict_choice(g["text"]) is not None,
                "greedy_correct": g_choice == row["answer"],
                "sample_texts": [item["text"] for item in samples],
                "sample_choices": s_choices,
                "sample_valid": [item is not None for item in s_choices],
                "sample_strict_format": [strict_choice(item["text"]) is not None for item in samples],
                "sample_correct": [item == row["answer"] for item in s_choices],
                "pass_at_k": any(item == row["answer"] for item in s_choices),
            })
    sampled_correct = [flag for row in details for flag in row["sample_correct"]]
    sampled_valid = [flag for row in details for flag in row["sample_valid"]]
    sampled_strict = [flag for row in details for flag in row["sample_strict_format"]]
    summary = {
        "status": "completed",
        "candidate": args.candidate,
        "split": args.split,
        "samples": len(details),
        "protocol": {
            "correctness": "one unambiguous leading/correct-answer option label equals programmatic answer",
            "strict_format": "entire normalized completion is one A/B/C/D label",
            "greedy_pass_at_1": True,
            "sample_k": args.sample_k,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "precision": "float32 parameters with BF16 autocast",
        },
        "pass_at_1": rate(details, "greedy_correct"),
        "choice_valid_at_1": rate(details, "choice_valid_at_1"),
        "strict_format_at_1": rate(details, "strict_format_at_1"),
        "sample_accuracy": sum(sampled_correct) / len(sampled_correct),
        "sample_choice_valid": sum(sampled_valid) / len(sampled_valid),
        "sample_strict_format": sum(sampled_strict) / len(sampled_strict),
        "pass_at_k": rate(details, "pass_at_k"),
        "mean_greedy_tokens": statistics.mean(row["greedy_tokens"] for row in details),
        "greedy_choice_distribution": Counter(row["greedy_choice"] or "INVALID" for row in details),
        "by_operation": grouped(details, "operation"),
        "by_template": grouped(details, "template_id"),
        "by_answer_position": grouped(details, "answer_position"),
        "data_sha256": sha(args.data / f"{args.split}.jsonl"),
        "model_config_sha256": sha(args.model / "config.json"),
        "limitations": [
            "pass@k is empirical any-correct over fixed sampled generations, not an unbiased estimator",
            "multiple-choice arithmetic does not measure free-form GSM8K reasoning",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "samples.jsonl").open("w") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
