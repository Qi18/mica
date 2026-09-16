#!/usr/bin/env python3
"""Controlled Phase5 SFT, GRPO, and CISPO trainer for an exact-verifier task."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase5_math_common import parse_choice  # noqa: E402
from phase5_numeric_common import parse_integer  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def completion_mask(ids, eos_id):
    mask = torch.ones_like(ids, dtype=torch.bool)
    for row in range(ids.size(0)):
        eos = (ids[row] == eos_id).nonzero()
        if len(eos):
            mask[row, int(eos[0].item()) + 1:] = False
    return mask


def token_logps(model, sequences, attention_mask, completion_ids, input_width):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(sequences, attention_mask=attention_mask).logits
    logits = logits[:, input_width - 1:input_width - 1 + completion_ids.size(1)].float()
    all_logps = F.log_softmax(logits, dim=-1)
    selected = all_logps.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    entropy = -(all_logps.exp() * all_logps).sum(dim=-1)
    return selected, entropy


def sample_rollouts(model, tokenizer, prompts, num_generations, max_new_tokens, seed, temperature, top_p):
    model.eval()
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
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    sequences = sequences.clone()
    input_width = inputs.input_ids.size(1)
    completion_ids = sequences[:, input_width:]
    mask = completion_mask(completion_ids, tokenizer.eos_token_id)
    repeated_input_mask = inputs.attention_mask.repeat_interleave(num_generations, dim=0)
    attention_mask = torch.cat([repeated_input_mask, mask.long()], dim=1)
    texts = []
    for ids, valid in zip(completion_ids, mask):
        texts.append(tokenizer.decode(ids[valid], skip_special_tokens=True).strip())
    with torch.inference_mode():
        old_logps, entropy = token_logps(model, sequences, attention_mask, completion_ids, input_width)
    return {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "completion_ids": completion_ids,
        "completion_mask": mask,
        "texts": texts,
        "old_logps": old_logps.detach(),
        "entropy": entropy.detach(),
        "input_width": input_width,
    }


def sft_batch(tokenizer, rows, device, mask_eos=False):
    input_ids, labels = [], []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(prompt + row["answer"] + tokenizer.eos_token, add_special_tokens=False).input_ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError("assistant answer changed prompt tokenization boundary")
        completion_labels = full_ids[len(prompt_ids):]
        if mask_eos:
            if not completion_labels or completion_labels[-1] != tokenizer.eos_token_id:
                raise RuntimeError("completion does not end with eos token")
            completion_labels = completion_labels[:-1] + [-100]
        input_ids.append(full_ids)
        labels.append([-100] * len(prompt_ids) + completion_labels)
    padded = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    max_len = padded.input_ids.size(1)
    padded_labels = [label + [-100] * (max_len - len(label)) for label in labels]
    attention_mask = (padded.input_ids != tokenizer.pad_token_id).long()
    return padded.input_ids.to(device), attention_mask.to(device), torch.tensor(padded_labels, device=device)


def save_model(model, tokenizer, output):
    target = output / "model-fp32"
    target.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    model.save_pretrained(target, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(target)
    return target


def make_batches(rows, batch_size, seed, stratify_by_answer):
    rows = list(rows)
    if not stratify_by_answer:
        random.Random(seed).shuffle(rows)
        return [rows[start:start + batch_size] for start in range(0, len(rows), batch_size)]
    if batch_size % 10:
        raise ValueError("--stratify-by-answer requires batch-prompts divisible by 10")
    buckets = {str(value): [] for value in range(10)}
    for row in rows:
        if row.get("answer") not in buckets:
            raise ValueError("stratified numeric training requires answers 0-9")
        buckets[row["answer"]].append(row)
    for value, bucket in buckets.items():
        random.Random(seed + int(value)).shuffle(bucket)
    groups_per_batch = batch_size // 10
    batch_count = min(len(bucket) for bucket in buckets.values()) // groups_per_batch
    batches = []
    for index in range(batch_count):
        batch = []
        for group in range(groups_per_batch):
            for value in range(10):
                batch.append(buckets[str(value)][index * groups_per_batch + group])
        random.Random(seed + 100000 + index).shuffle(batch)
        batches.append(batch)
    if sum(len(batch) for batch in batches) != len(rows):
        raise ValueError("stratified batching would drop rows; balance the dataset first")
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sft", "grpo", "cispo"], required=True)
    parser.add_argument("--reward-type", choices=["choice", "integer"], default="choice")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--batch-prompts", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--rollout-temperature", type=float, default=0.8)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--inner-updates", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--stratify-by-answer", action="store_true")
    parser.add_argument(
        "--sft-mask-eos",
        action="store_true",
        help="Exclude the final EOS token from SFT loss; useful for exact one-token tasks.",
    )
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-outer-steps", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--disable-swanlab", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = read_jsonl(args.data / args.train_file)
    batches = make_batches(rows, args.batch_prompts, args.seed, args.stratify_by_answer)
    if args.max_outer_steps:
        batches = batches[:args.max_outer_steps]
    total_outer = len(batches)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).cuda()
    reference = None
    if args.method in {"grpo", "cispo"}:
        reference = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).cuda().eval()
        reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metrics_path = args.output / "metrics.jsonl"
    run = None
    if not args.disable_swanlab:
        import swanlab
        run = swanlab.init(
            project="MiniMind-Lab",
            experiment_name=args.run_name,
            group="Phase5-Verifiable-RL",
            job_type="training",
            config={
                **vars(args),
                "base_model": str(args.base_model),
                "data": str(args.data),
                "output": str(args.output),
                "train_rows": len(rows),
                "optimizer_updates": total_outer * args.inner_updates,
                "precision": "float32 parameters with BF16 autocast",
                "rollout_temperature": args.rollout_temperature,
                "rollout_top_p": args.rollout_top_p,
            },
        )
    started = time.monotonic()
    update = 0
    completion_tokens = 0
    correct_rollouts = 0
    total_rollouts = 0
    for outer, batch in enumerate(batches):
        last = {}
        if args.method == "sft":
            model.train()
            ids, attn, labels = sft_batch(tokenizer, batch, model.device, args.sft_mask_eos)
            for inner in range(args.inner_updates):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                update += 1
                completion_tokens += int((labels != -100).sum())
                last = {"loss": float(loss.detach()), "grad_norm": float(grad_norm)}
        else:
            rollout_started = time.monotonic()
            rollout = sample_rollouts(
                model, tokenizer, [row["prompt"] for row in batch],
                args.num_generations, args.max_new_tokens,
                args.seed * 1000000 + outer,
                args.rollout_temperature, args.rollout_top_p,
            )
            parser_fn = parse_choice if args.reward_type == "choice" else parse_integer
            targets = [row["answer"] if args.reward_type == "choice" else int(row["answer"]) for row in batch]
            rewards = torch.tensor(
                [float(parser_fn(text) == targets[i]) for i, row in enumerate(batch) for text in rollout["texts"][i * args.num_generations:(i + 1) * args.num_generations]],
                device=model.device,
            )
            grouped_rewards = rewards.view(len(batch), args.num_generations)
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)
            advantages = ((grouped_rewards - group_mean) / (group_std + 1e-4)).flatten().detach()
            mask = rollout["completion_mask"]
            with torch.inference_mode():
                ref_logps, _ = token_logps(
                    reference, rollout["sequences"], rollout["attention_mask"],
                    rollout["completion_ids"], rollout["input_width"],
                )
            rollout_seconds = time.monotonic() - rollout_started
            for inner in range(args.inner_updates):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                new_logps, entropy = token_logps(
                    model, rollout["sequences"], rollout["attention_mask"],
                    rollout["completion_ids"], rollout["input_width"],
                )
                logratio = new_logps - rollout["old_logps"]
                ratio = torch.exp(logratio.clamp(-10, 10))
                ref_delta = ref_logps.detach() - new_logps
                per_token_kl = torch.exp(ref_delta.clamp(-10, 10)) - ref_delta - 1
                if args.method == "grpo":
                    unclipped = ratio * advantages.unsqueeze(1)
                    clipped = ratio.clamp(1 - args.epsilon, 1 + args.epsilon) * advantages.unsqueeze(1)
                    objective = torch.minimum(unclipped, clipped)
                    clip_fraction = ((ratio < 1 - args.epsilon) | (ratio > 1 + args.epsilon))[mask].float().mean()
                else:
                    weight = ratio.clamp(max=args.epsilon_high).detach()
                    objective = weight * advantages.unsqueeze(1) * new_logps
                    clip_fraction = (ratio > args.epsilon_high)[mask].float().mean()
                per_token_loss = -(objective - args.beta * per_token_kl + args.entropy_coef * entropy)
                loss = ((per_token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                update += 1
                last = {
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    "approx_kl": float((per_token_kl.detach() * mask).sum() / mask.sum()),
                    "sampled_logratio_to_ref": float(((new_logps.detach() - ref_logps) * mask).sum() / mask.sum()),
                    "entropy": float((entropy.detach() * mask).sum() / mask.sum()),
                    "clip_fraction": float(clip_fraction),
                }
            completion_tokens += int(mask.sum())
            correct_rollouts += int(rewards.sum())
            total_rollouts += len(rewards)
            group_sums = grouped_rewards.sum(dim=1)
            last.update({
                "reward": float(rewards.mean()),
                "group_reward_std": float(group_std.mean()),
                "degenerate_group_rate": float(((group_sums == 0) | (group_sums == args.num_generations)).float().mean()),
                "all_zero_group_rate": float((group_sums == 0).float().mean()),
                "all_one_group_rate": float((group_sums == args.num_generations).float().mean()),
                "pass_at_group": float((group_sums > 0).float().mean()),
                "mean_completion_length": float(mask.sum(1).float().mean()),
                "rollout_seconds": rollout_seconds,
            })
        record = {
            "event": "train",
            "method": args.method,
            "outer_step": outer + 1,
            "optimizer_update": update,
            "wall_seconds": time.monotonic() - started,
            "completion_tokens": completion_tokens,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            **last,
        }
        if (outer + 1) % args.log_interval == 0 or outer + 1 == total_outer:
            append(metrics_path, record)
            if run:
                import swanlab
                swanlab.log({key: value for key, value in record.items() if isinstance(value, (int, float))}, step=update)
    model_path = save_model(model, tokenizer, args.output)
    finished = {
        "event": "completed",
        "method": args.method,
        "outer_steps": total_outer,
        "optimizer_updates": update,
        "wall_seconds": time.monotonic() - started,
        "completion_tokens": completion_tokens,
        "correct_rollouts": correct_rollouts,
        "total_rollouts": total_rollouts,
        "mean_training_reward": correct_rollouts / total_rollouts if total_rollouts else None,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "model_path": str(model_path),
    }
    append(metrics_path, finished)
    url = None
    if run:
        import swanlab
        swanlab.log({f"final/{key}": value for key, value in finished.items() if isinstance(value, (int, float))}, step=update)
        url = run.url
        swanlab.finish()
        (args.output / "swanlab-url.txt").write_text(url + "\n")
    (args.output / "run.json").write_text(json.dumps({
        "status": "completed",
        "run_name": args.run_name,
        "method": args.method,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training": finished,
        "swanlab_url": url,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"training": finished, "swanlab_url": url}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
