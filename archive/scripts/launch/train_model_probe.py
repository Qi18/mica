#!/usr/bin/env python3
"""Bounded DDP training probe with SwanLab and exact resume evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "minimind"))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class GpuSampler:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.samples: list[list[tuple[float, float]]] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            parsed = []
            for line in result.stdout.splitlines():
                try:
                    utilization, memory = (float(value.strip()) for value in line.split(","))
                    parsed.append((utilization, memory))
                except (TypeError, ValueError):
                    continue
            if parsed:
                self.samples.append(parsed)
            self.stop_event.wait(0.5)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        if not self.samples:
            return {"sample_count": 0, "mean_utilization": None, "peak_memory_mib": None}
        utilization = [value[0] for sample in self.samples for value in sample]
        memory = [value[1] for sample in self.samples for value in sample]
        return {
            "sample_count": len(self.samples),
            "mean_utilization": sum(utilization) / len(utilization),
            "peak_memory_mib": max(memory),
        }


def reduce_value(value: float, device: torch.device, operation: dist.ReduceOp) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=operation)
    return float(tensor.item())


def save_checkpoint(
    path: Path,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    swanlab_run_id: str,
) -> None:
    payload = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "seed": args.seed,
        "world_size": dist.get_world_size(),
        "total_steps": args.total_steps,
        "swanlab_run_id": swanlab_run_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-id-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--project", default="MiniMind-Lab")
    parser.add_argument("--group", default="E02-model-probe-20260824")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise SystemExit(f"Stage2 requires 8 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    input_ids = data["input_ids"]
    labels = data["labels"]
    if input_ids.shape != labels.shape or input_ids.ndim != 2:
        raise SystemExit("invalid probe data")
    if args.total_steps * args.batch_size * world_size > input_ids.shape[0]:
        raise SystemExit("probe data is too small for non-repeating batches")

    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False, flash_attn=True)
    model = MiniMindForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_step = 0
    swanlab_run_id = ""
    optimizer_state_entries_after_resume = 0
    if args.resume:
        if not args.checkpoint or not args.checkpoint.is_file():
            raise SystemExit("resume checkpoint not found")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["seed"] != args.seed or checkpoint["world_size"] != world_size:
            raise SystemExit("checkpoint seed/world-size mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        swanlab_run_id = str(checkpoint.get("swanlab_run_id") or "")
        optimizer_state_entries_after_resume = len(optimizer.state)
        if start_step >= args.max_steps or not swanlab_run_id:
            raise SystemExit("checkpoint cannot continue requested phase")
    elif args.metrics.exists() or args.run_id_file.exists():
        raise SystemExit("fresh probe refuses to overwrite existing metrics/run id")

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=True,
    )
    permutation = torch.randperm(input_ids.shape[0], generator=torch.Generator().manual_seed(args.seed))

    swanlab = None
    swanlab_run = None
    if rank == 0:
        import swanlab as swanlab_module

        swanlab = swanlab_module
        init_kwargs = {
            "project": args.project,
            "experiment_name": f"E02-model-probe-seed{args.seed}",
            "group": args.group,
            "tags": ["stage2", "model-probe", "8xL20", f"seed-{args.seed}"],
            "config": {
                "seed": args.seed,
                "world_size": world_size,
                "batch_size_per_gpu": args.batch_size,
                "sequence_length": int(input_ids.shape[1]),
                "total_steps": args.total_steps,
                "learning_rate": args.learning_rate,
                "hidden_size": config.hidden_size,
                "layers": config.num_hidden_layers,
                "dtype": "bfloat16",
                "data": str(args.data),
            },
            "mode": "cloud",
        }
        if args.resume:
            init_kwargs.update({"id": swanlab_run_id, "resume": "must"})
        swanlab_run = swanlab.init(**init_kwargs)
        swanlab_run_id = str(swanlab_run.id)
        args.run_id_file.parent.mkdir(parents=True, exist_ok=True)
        args.run_id_file.write_text(swanlab_run_id + "\n", encoding="utf-8")

    run_id_holder = [swanlab_run_id]
    dist.broadcast_object_list(run_id_holder, src=0)
    swanlab_run_id = run_id_holder[0]
    sampler = GpuSampler() if rank == 0 else None
    if sampler:
        sampler.start()

    torch.cuda.reset_peak_memory_stats(device)
    phase_records: list[dict[str, Any]] = []
    for step in range(start_step + 1, args.max_steps + 1):
        global_offset = (step - 1) * args.batch_size * world_size + rank * args.batch_size
        indices = permutation[global_offset : global_offset + args.batch_size]
        batch_inputs = input_ids[indices].to(device=device, dtype=torch.long, non_blocking=True)
        batch_labels = labels[indices].to(device=device, dtype=torch.long, non_blocking=True)
        learning_rate = args.learning_rate * (
            0.1 + 0.45 * (1 + math.cos(math.pi * step / args.total_steps))
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch_inputs, labels=batch_labels)
            loss = output.loss + output.aux_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        reduced_loss = reduce_value(float(loss.detach()), device, dist.ReduceOp.SUM) / world_size
        reduced_grad = reduce_value(float(grad_norm.detach()), device, dist.ReduceOp.SUM) / world_size
        max_elapsed = reduce_value(elapsed, device, dist.ReduceOp.MAX)
        valid_tokens = reduce_value(
            float((batch_labels[:, 1:] != -100).sum().item()),
            device,
            dist.ReduceOp.SUM,
        )
        record = {
            "step": step,
            "loss": reduced_loss,
            "learning_rate": learning_rate,
            "grad_norm": reduced_grad,
            "step_seconds": max_elapsed,
            "valid_tokens": valid_tokens,
            "tokens_per_second": valid_tokens / max_elapsed,
            "samples_per_second": args.batch_size * world_size / max_elapsed,
            "peak_allocated_mib_rank0": torch.cuda.max_memory_allocated(device) / 1024**2,
            "phase_resumed": args.resume,
        }
        if rank == 0:
            phase_records.append(record)
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            with args.metrics.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            swanlab.log(record, step=step)
            if step == start_step + 1 or step % args.log_interval == 0 or step == args.max_steps:
                print(json.dumps(record), flush=True)

        should_save = bool(args.checkpoint) and (
            step == args.max_steps or (args.save_every > 0 and step % args.save_every == 0)
        )
        if should_save:
            dist.barrier()
            if rank == 0:
                save_checkpoint(
                    args.checkpoint,
                    model,
                    optimizer,
                    step,
                    args,
                    swanlab_run_id,
                )
            dist.barrier()

    gpu_summary = sampler.finish() if sampler else None
    if rank == 0:
        all_records = [
            json.loads(line)
            for line in args.metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        steady = [record for record in all_records if record["step"] > 5]
        summary = {
            "status": "completed",
            "seed": args.seed,
            "world_size": world_size,
            "completed_step": args.max_steps,
            "total_steps": args.total_steps,
            "resumed_this_phase": args.resume,
            "resumed_from_step": start_step if args.resume else None,
            "optimizer_state_entries_after_resume": optimizer_state_entries_after_resume,
            "swanlab_run_id": swanlab_run_id,
            "first_loss": all_records[0]["loss"],
            "last_loss": all_records[-1]["loss"],
            "mean_tokens_per_second_after_warmup": sum(r["tokens_per_second"] for r in steady) / len(steady),
            "mean_samples_per_second_after_warmup": sum(r["samples_per_second"] for r in steady) / len(steady),
            "peak_allocated_mib_rank0": max(r["peak_allocated_mib_rank0"] for r in all_records),
            "gpu_sampler": gpu_summary,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(args.summary, summary)
        swanlab.log({"probe/completed_step": args.max_steps}, step=args.max_steps)
        swanlab_run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
