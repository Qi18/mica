#!/usr/bin/env python3
"""Inspect MiniMind Dense/MoE parameters, tensor shapes, backward, and KV cache."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.modeling_mica import MicaConfig, MicaForCausalLM  # noqa: E402


def shapes(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shapes(item) for item in value]
    if hasattr(value, "logits"):
        return {"logits": shapes(value.logits), "hidden_states": shapes(value.hidden_states)}
    return type(value).__name__


def parameter_group(name: str) -> str:
    if "embed_tokens" in name or name == "lm_head.weight":
        return "embedding_lm_head"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def inspect_variant(use_moe: bool, batch_size: int, seq_len: int) -> dict[str, Any]:
    config = MicaConfig(
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=use_moe,
        flash_attn=True,
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = MicaForCausalLM(config).cuda().train()
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        groups[group] = groups.get(group, 0) + parameter.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    expert = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".mlp.experts.0." in name
    )
    active = total - expert * config.num_experts + expert * config.num_experts_per_tok
    if not use_moe:
        active = total

    captured: dict[str, Any] = {}
    handles = []
    if not use_moe:
        modules = {
            "embedding": model.model.embed_tokens,
            "layer0.input_norm": model.model.layers[0].input_layernorm,
            "layer0.attention": model.model.layers[0].self_attn,
            "layer0.post_attention_norm": model.model.layers[0].post_attention_layernorm,
            "layer0.mlp": model.model.layers[0].mlp,
            "final_norm": model.model.norm,
            "lm_head": model.lm_head,
        }
        for module_name, module in modules.items():
            handles.append(
                module.register_forward_hook(
                    lambda _module, inputs, output, key=module_name: captured.update(
                        {key: {"input": shapes(inputs), "output": shapes(output)}}
                    )
                )
            )

    input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels=labels)
        loss = output.loss + output.aux_loss
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    grad_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    for handle in handles:
        handle.remove()

    kv_cache = None
    if not use_moe:
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefix = model(input_ids[:, :8], use_cache=True, logits_to_keep=1)
            continued = model(
                input_ids[:, 8:9],
                past_key_values=prefix.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        kv_cache = {
            "layers": len(prefix.past_key_values),
            "prefix_layer0_key": list(prefix.past_key_values[0][0].shape),
            "continued_layer0_key": list(continued.past_key_values[0][0].shape),
            "continued_logits": list(continued.logits.shape),
        }

    result = {
        "variant": "moe" if use_moe else "dense",
        "config": {
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "attention_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.intermediate_size,
            "experts": config.num_experts if use_moe else 0,
            "experts_per_token": config.num_experts_per_tok if use_moe else 0,
            "flash_sdpa_enabled": model.model.layers[0].self_attn.flash,
        },
        "parameters": {
            "total": total,
            "active_per_token_estimate": active,
            "by_group": groups,
            "single_expert": expert,
        },
        "probe": {
            "input_ids": list(input_ids.shape),
            "logits": list(output.logits.shape),
            "hidden_states": list(output.hidden_states.shape),
            "loss": float(output.loss.detach()),
            "aux_loss": float(output.aux_loss.detach()),
            "loss_finite": math.isfinite(float(loss.detach())),
            "grad_finite": grad_finite,
            "forward_backward_seconds": elapsed,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        },
        "tensor_shapes": captured,
        "kv_cache": kv_cache,
    }
    del output, loss, input_ids, labels, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA with BF16 support is required")
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dense": inspect_variant(False, args.batch_size, args.seq_len),
        "moe": inspect_variant(True, args.batch_size, args.seq_len),
    }
    payload["comparison"] = {
        "total_parameter_ratio_moe_over_dense": (
            payload["moe"]["parameters"]["total"]
            / payload["dense"]["parameters"]["total"]
        ),
        "active_parameter_ratio_moe_over_dense": (
            payload["moe"]["parameters"]["active_per_token_estimate"]
            / payload["dense"]["parameters"]["active_per_token_estimate"]
        ),
        "peak_memory_ratio_moe_over_dense": (
            payload["moe"]["probe"]["peak_allocated_mib"]
            / payload["dense"]["probe"]["peak_allocated_mib"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["comparison"], ensure_ascii=False))


if __name__ == "__main__":
    main()
