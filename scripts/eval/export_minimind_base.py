#!/usr/bin/env python3
"""Export a MiniMind Dense PyTorch checkpoint as a Qwen3-compatible HF model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-dir", "--minimind-dir", dest="minimind_dir", type=Path, default=Path("."))
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()

    sys.path.insert(0, str(args.minimind_dir.resolve()))
    from model.modeling_mica import MicaConfig

    source_config = MicaConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )
    target_config = Qwen3Config(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.hidden_size,
        intermediate_size=source_config.intermediate_size,
        num_hidden_layers=source_config.num_hidden_layers,
        num_attention_heads=source_config.num_attention_heads,
        num_key_value_heads=source_config.num_key_value_heads,
        head_dim=source_config.hidden_size // source_config.num_attention_heads,
        max_position_embeddings=source_config.max_position_embeddings,
        rms_norm_eps=source_config.rms_norm_eps,
        rope_theta=source_config.rope_theta,
        tie_word_embeddings=source_config.tie_word_embeddings,
        use_sliding_window=False,
        sliding_window=None,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Qwen3ForCausalLM(target_config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(getattr(torch, args.dtype))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.minimind_dir / "tokenizer")
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "output_dir": str(args.output_dir),
        "architecture": "Qwen3ForCausalLM (MiniMind-compatible weights)",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizer_vocab_size": len(tokenizer),
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
