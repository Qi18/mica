"""Portable token-level training, evaluation and checkpoint operations."""
import hashlib
import json
import math
from pathlib import Path

import torch
from .modeling_mica import MicaConfig, MicaForCausalLM


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_config(config):
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if config.head_dim % 2:
        raise ValueError("RoPE head_dim must be even")
    if min(config.hidden_size, config.num_hidden_layers, config.vocab_size) <= 0:
        raise ValueError("model dimensions must be positive")
    if config.use_moe and not 1 <= config.num_experts_per_tok <= config.num_experts:
        raise ValueError("invalid MoE top-k")


def load_model(path, device="cpu"):
    path = Path(path)
    config = MicaConfig(**read_json(path / "config.json"))
    validate_config(config)
    model = MicaForCausalLM(config)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to(device)


def load_data(path, vocab_size, max_length):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        ids = row["input_ids"]
        labels = row.get("labels", ids)
        if len(ids) != len(labels) or not 2 <= len(ids) <= max_length:
            raise ValueError(f"invalid lengths at row {number}")
        if any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
            raise ValueError(f"invalid token at row {number}")
        if any(type(x) is not int or (x != -100 and not 0 <= x < vocab_size) for x in labels):
            raise ValueError(f"invalid label at row {number}")
        if all(x == -100 for x in labels[1:]):
            raise ValueError(f"no supervised next-token targets at row {number}")
        rows.append((ids, labels))
    if not rows:
        raise ValueError("dataset is empty")
    return rows


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
    recipe_path = Path(recipe_path).resolve()
    recipe = read_json(recipe_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; resume into a new output directory")
    torch.manual_seed(recipe.get("seed", 42))
    torch.set_num_threads(recipe.get("cpu_threads", 2))
    device = recipe.get("device", "cpu")
    config = MicaConfig(**recipe["model"])
    validate_config(config)
    data_path = (recipe_path.parent / recipe["data"]).resolve()
    fingerprint = hashlib.sha256(data_path.read_bytes()).hexdigest()
    rows = load_data(data_path, config.vocab_size, config.max_position_embeddings)
    steps, size = recipe["steps"], recipe.get("batch_size", 1)
    if steps < 1 or size < 1:
        raise ValueError("steps and batch_size must be positive")
    model = load_model(resume, device) if resume else MicaForCausalLM(config).to(device)
    if resume and read_json(Path(resume) / "recipe.json")["model"] != recipe["model"]:
        raise ValueError("resume architecture differs from recipe")
    if not resume and recipe.get("initialize_from"):
        initial = load_model(recipe_path.parent / recipe["initialize_from"], device)
        model.load_state_dict(initial.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.get("learning_rate", 0.001))
    start = 0
    if resume:
        state = torch.load(Path(resume) / "training.pt", map_location="cpu", weights_only=True)
        old = read_json(Path(resume) / "recipe.json")
        if {k: v for k, v in old.items() if k != "steps"} != {k: v for k, v in recipe.items() if k != "steps"}:
            raise ValueError("resume may change only total steps")
        if state["data_sha256"] != fingerprint:
            raise ValueError("resume data fingerprint mismatch")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if start >= steps:
        raise ValueError("total steps must exceed saved step")
    model.train()
    metrics = []
    for step in range(start, steps):
        selected = [rows[(step * size + i) % len(rows)] for i in range(size)]
        ids, labels, mask = batch(selected, device)
        result = model(ids, labels=labels, attention_mask=mask)
        loss = result.loss + result.aux_loss
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        metrics.append({"step": step + 1, "loss": loss.item()})
    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    torch.save(model.state_dict(), output / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": steps, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                "data_sha256": fingerprint}, output / "training.pt")
    write_json(output / "recipe.json", recipe)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run.json", {"steps": steps, "start_step": start, "data_sha256": fingerprint,
                                    "parameters": sum(p.numel() for p in model.parameters()), "device": device})
    return {"output": str(output), "step": steps, "loss": metrics[-1]["loss"]}


@torch.inference_mode()
def evaluate(model_path, data, device="cpu"):
    torch.set_num_threads(2)
    model = load_model(model_path, device).eval()
    rows = load_data(data, model.config.vocab_size, model.config.max_position_embeddings)
    total, targets = 0., 0
    for row in rows:
        ids, labels, mask = batch([row], device)
        n = int((labels[:, 1:] != -100).sum())
        loss = model(ids, labels=labels, attention_mask=mask).loss.item()
        total += loss * n
        targets += n
    nll = total / targets
    return {"nll": nll, "perplexity": math.exp(nll), "targets": targets, "rows": len(rows)}


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
    write_json(output / "source.json", {"checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                                      "source": "legacy MiniMind state_dict", "architecture_changed": False})
    return {"output": str(output), "strict_load": True}
