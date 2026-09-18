#!/usr/bin/env python3
"""Merge a MiniMind LoRA adapter and verify logits against base+adapter."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", "--minimind-dir", dest="minimind_dir", type=Path, default=Path("."))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=0.30)
    args = parser.parse_args()

    sys.path.insert(0, str(args.minimind_dir.resolve()))
    from model.model_lora import apply_lora, load_lora
    from model.model_mica import MicaConfig, MicaForCausalLM

    config = MicaConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    base_state = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    model = MicaForCausalLM(config)
    model.load_state_dict(base_state, strict=True)
    model = model.to(args.device).eval()
    apply_lora(model, rank=args.rank)
    load_lora(model, args.adapter)

    tokenizer = AutoTokenizer.from_pretrained(args.minimind_dir / "tokenizer")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a Python function add(a, b)."}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        adapter_logits = model(input_ids).logits.float().cpu()
        adapter_output = model.generate(input_ids, max_new_tokens=64, do_sample=False)

    merged_state = {
        key: value.detach().cpu().half()
        for key, value in model.state_dict().items()
        if ".lora." not in key
    }
    injected = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "lora"):
            merged_state[f"{name}.weight"] = (
                module.weight.detach().cpu()
                + module.lora.B.weight.detach().cpu() @ module.lora.A.weight.detach().cpu()
            ).half()
            injected.append(name)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state, args.output_checkpoint)
    merged = MicaForCausalLM(config)
    merged.load_state_dict(torch.load(args.output_checkpoint, map_location="cpu", weights_only=True), strict=True)
    merged = merged.to(args.device).eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        merged_logits = merged(input_ids).logits.float().cpu()
        merged_output = merged.generate(input_ids, max_new_tokens=64, do_sample=False)

    difference = (adapter_logits - merged_logits).abs()
    argmax_match_ratio = float(
        (adapter_logits.argmax(dim=-1) == merged_logits.argmax(dim=-1)).float().mean()
    )
    generated_tokens_equal = bool(torch.equal(adapter_output, merged_output))
    report = {
        "base_checkpoint": str(args.base_checkpoint),
        "base_sha256": sha256(args.base_checkpoint),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter),
        "adapter_bytes": args.adapter.stat().st_size,
        "merged_checkpoint": str(args.output_checkpoint),
        "merged_sha256": sha256(args.output_checkpoint),
        "merged_bytes": args.output_checkpoint.stat().st_size,
        "rank": args.rank,
        "injected_module_count": len(injected),
        "injected_modules": injected,
        "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()),
        "argmax_match_ratio": argmax_match_ratio,
        "generated_tokens_equal": generated_tokens_equal,
        "generated_text": tokenizer.decode(merged_output[0, input_ids.shape[1]:], skip_special_tokens=True),
        "atol": args.atol,
        "allclose": bool(torch.allclose(adapter_logits, merged_logits, atol=args.atol, rtol=0.0)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["allclose"] or not generated_tokens_equal:
        raise SystemExit("merge consistency check failed")


if __name__ == "__main__":
    main()
