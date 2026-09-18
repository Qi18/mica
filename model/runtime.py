"""Portable token-level training, evaluation and checkpoint operations."""
from dataset.indexed import file_sha256
import json
import math
from pathlib import Path

import torch
from .model_mica import MicaConfig, MicaForCausalLM


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_config(config):
    if min(config.num_attention_heads, config.num_key_value_heads, config.head_dim) <= 0:
        raise ValueError("attention dimensions must be positive")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if config.head_dim % 2:
        raise ValueError("RoPE head_dim must be even")
    if min(config.hidden_size, config.num_hidden_layers, config.vocab_size) <= 0:
        raise ValueError("model dimensions must be positive")
    if config.use_moe and not 1 <= config.num_experts_per_tok <= config.num_experts:
        raise ValueError("invalid MoE top-k")


def resolve_checkpoint(path):
    path = Path(path)
    if not (path / "model.pt").exists() and (path / "latest.json").exists():
        target = (path / read_json(path / "latest.json")["checkpoint"]).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("checkpoint pointer escapes output directory")
        return target
    return path


def load_model(path, device="cpu"):
    path = resolve_checkpoint(path)
    config = MicaConfig(**read_json(path / "config.json"))
    validate_config(config)
    model = MicaForCausalLM(config)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to(device)


def load_data(path, vocab_size, max_length):
    from dataset.indexed import JsonlDataset
    return JsonlDataset(path, vocab_size, max_length)


def batch(rows, device):
    length = max(len(row[0]) for row in rows)
    ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, (tokens, targets) in enumerate(rows):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        labels[i, :len(targets)] = torch.tensor(targets, device=device)
        mask[i, :len(tokens)] = 1
    return ids, labels, mask


def train(recipe_path, output, resume=None):
    from trainer.common.engine import train as run
    return run(recipe_path, output, resume)


def evaluate(model_path, data, device="cpu"):
    from evaluation.loss import evaluate as run
    return run(model_path, data, device)


def import_legacy(checkpoint, config_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty")
    config = MicaConfig(**read_json(config_path))
    validate_config(config)
    model = MicaForCausalLM(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    torch.save(model.state_dict(), output / "model.pt")
    write_json(output / "source.json", {"checkpoint_sha256": file_sha256(checkpoint),
                                      "source": "legacy MiniMind state_dict", "architecture_changed": False})
    return {"output": str(output), "strict_load": True}
