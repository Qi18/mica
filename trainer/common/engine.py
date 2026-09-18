"""Token-normalized FP32 training with DDP and atomic step checkpoints."""
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import tempfile

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dataset.indexed import JsonlDataset
from model.model_mica import MicaConfig, MicaForCausalLM
from model.runtime import read_json, write_json, validate_config, load_model, batch, resolve_checkpoint


def _atomic_json(path, value):
    # Readers see either the old pointer or the complete new pointer.
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rng(device):
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _save(output, model, optimizer, recipe, step, data, device, rank, world, start, interrupted):
    own_rng = _rng(device)
    rngs = [None] * world if world > 1 else [own_rng]
    if world > 1:
        dist.all_gather_object(rngs, own_rng)
    error = [None]
    if rank == 0:
        try:
            parent = output / "checkpoints"
            parent.mkdir(exist_ok=True)
            final = parent / f"step-{step:08d}"
            if final.exists():
                raise ValueError(f"checkpoint already exists: {final}")
            temporary = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=parent))
            model.config.save_pretrained(temporary)
            torch.save(model.state_dict(), temporary / "model.pt")
            torch.save({"format_version": 2, "optimizer": optimizer.state_dict(),
                        "step": step, "rngs": rngs, "world_size": world,
                        "data_sha256": data.fingerprint}, temporary / "training.pt")
            write_json(temporary / "recipe.json", recipe)
            # Rename only after all assets are written. Incomplete directories never become latest.
            os.replace(temporary, final)
            _atomic_json(output / "latest.json", {"checkpoint": str(final.relative_to(output))})
            _atomic_json(output / "run.json", {
                "steps": step, "requested_steps": recipe["steps"], "start_step": start,
                "status": "interrupted" if interrupted else ("completed" if step == recipe["steps"] else "running"),
                "data_sha256": data.fingerprint, "rows": len(data), "world_size": world,
                "global_batch_size": world * recipe.get("batch_size", 1) * recipe.get("gradient_accumulation_steps", 1),
                "parameters": sum(p.numel() for p in model.parameters()), "device": str(device)})
        except Exception as exc:
            error[0] = str(exc)
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError("checkpoint save failed: " + error[0])


def train(recipe_path, output, resume=None):
    recipe_path = Path(recipe_path).resolve()
    recipe = read_json(recipe_path)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    raw_device = recipe.get("device", "cpu")
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if world > 1 and raw_device.startswith("cuda") else raw_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=120))
    old_handlers = {}
    stop = [False]
    try:
        output = Path(output)
        # All ranks check before rank zero creates anything.
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty; resume into a new output directory")
        if world > 1:
            dist.barrier()
        steps, size = recipe["steps"], recipe.get("batch_size", 1)
        accumulation = recipe.get("gradient_accumulation_steps", 1)
        interval = recipe.get("save_every", 100)
        if any(type(x) is not int or x < 1 for x in (steps, size, accumulation, interval)):
            raise ValueError("steps, batch_size, gradient_accumulation_steps and save_every must be positive integers")
        torch.set_num_threads(recipe.get("cpu_threads", 2))
        torch.manual_seed(recipe.get("seed", 42))
        config = MicaConfig(**recipe["model"])
        validate_config(config)
        data = JsonlDataset(recipe_path.parent / recipe["data"], config.vocab_size, config.max_position_embeddings)
        resume = resolve_checkpoint(resume) if resume else None
        model = load_model(resume, device) if resume else MicaForCausalLM(config).to(device)
        if resume and read_json(resume / "recipe.json")["model"] != recipe["model"]:
            raise ValueError("resume architecture differs from recipe")
        if not resume and recipe.get("initialize_from"):
            initial = load_model(recipe_path.parent / recipe["initialize_from"], device)
            model.load_state_dict(initial.state_dict(), strict=True)
            del initial
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.get("learning_rate", 0.001))
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                          find_unused_parameters=config.use_moe) if world > 1 else model
        start = 0
        torch.manual_seed(recipe.get("seed", 42) + rank)
        if resume:
            state = torch.load(resume / "training.pt", map_location="cpu", weights_only=True)
            if state.get("format_version") != 2:
                raise ValueError("v0.1 optimizer checkpoints require initialize_from; exact resume requires format v2")
            old = read_json(resume / "recipe.json")
            if {k: v for k, v in old.items() if k != "steps"} != {k: v for k, v in recipe.items() if k != "steps"}:
                raise ValueError("resume may change only total steps")
            if state["data_sha256"] != data.fingerprint:
                raise ValueError("resume data fingerprint mismatch")
            if state["world_size"] != world:
                raise ValueError("exact resume requires the same world_size")
            optimizer.load_state_dict(state["optimizer"])
            start = state["step"]
            torch.set_rng_state(state["rngs"][rank]["cpu"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["rngs"][rank]["cuda"], device)
        if start >= steps:
            raise ValueError("total steps must exceed saved step")
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, lambda *_: stop.__setitem__(0, True))
        wrapped.train()
        last = {}
        for step in range(start, steps):
            groups = []
            for micro in range(accumulation):
                base = (step * accumulation + micro) * world * size + rank * size
                groups.append([data[(base + i) % len(data)] for i in range(size)])
            count = sum(sum(x != -100 for x in labels[1:]) for group in groups for _, labels in group)
            global_count = torch.tensor(count, dtype=torch.long, device=device)
            if world > 1:
                dist.all_reduce(global_count)
            optimizer.zero_grad()
            values = torch.zeros(2, dtype=torch.float64, device=device)
            for micro, rows in enumerate(groups):
                ids, labels, mask = batch(rows, device)
                targets = (labels[:, 1:] != -100).sum()
                sync = wrapped.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with sync:
                    result = wrapped(ids, labels=labels, attention_mask=mask)
                    # DDP averages gradients across ranks, so multiply by world to obtain
                    # a true global token mean, including uneven prompt masks.
                    loss = result.loss * (targets * world / global_count)
                    loss = loss + result.aux_loss / accumulation
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise ValueError(f"non-finite loss at step {step}")
                    loss.backward()
                values[0] += result.loss.detach().double() * targets
                aux = result.aux_loss
                values[1] += float(aux.detach() if torch.is_tensor(aux) else aux) / accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(values)
            last = {"step": step + 1, "loss": (values[0] / global_count).item(),
                    "aux_loss": (values[1] / world).item(), "targets": global_count.item()}
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(last) + "\n")
            stopping = torch.tensor(int(stop[0]), device=device)
            if world > 1:
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if (step + 1) % interval == 0 or step + 1 == steps or stopping.item():
                _save(output, model, optimizer, recipe, step + 1, data, device, rank, world, start, bool(stopping.item()))
            if stopping.item():
                break
        return {"output": str(output), "step": last["step"], "loss": last["loss"],
                "status": "interrupted" if stopping.item() else "completed"}
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
