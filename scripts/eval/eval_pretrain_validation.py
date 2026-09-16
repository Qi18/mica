#!/usr/bin/env python3
"""Evaluate a raw MiniMind Base checkpoint on deterministic JSONL validation data.

The native MiniMind model and raw inference state_dict are deliberate: this
keeps the held-out NLL/PPL label shift identical to pretraining and avoids
adding the Hugging Face export layer to the quality metric.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def expand_data_files(patterns: list[str]) -> list[Path]:
    resolved: dict[str, Path] = {}
    for pattern in patterns:
        matches = [Path(item).resolve() for item in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern).resolve()]
        if not matches:
            raise FileNotFoundError(f"data pattern matched no files: {pattern}")
        for path in matches:
            if not path.is_file():
                raise FileNotFoundError(f"not a regular file: {path}")
            resolved[str(path)] = path
    return [resolved[key] for key in sorted(resolved)]


def find_success_marker(files: list[Path], explicit: Path | None) -> Path | None:
    if explicit is not None:
        marker = explicit.resolve()
        if not marker.is_file():
            raise FileNotFoundError(f"dataset success marker not found: {marker}")
        return marker
    common_parent = Path(os.path.commonpath([str(path.parent) for path in files]))
    for parent in (common_parent, *common_parent.parents):
        marker = parent / "_SUCCESS"
        if marker.is_file():
            return marker
    return None


def load_dataset_identity(
    files: list[Path], marker_arg: Path | None, fingerprint_arg: str | None
) -> dict[str, Any]:
    marker = find_success_marker(files, marker_arg)
    marker_payload: dict[str, Any] = {}
    if marker is not None:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    fingerprint = fingerprint_arg or marker_payload.get("dataset_fingerprint")
    if not fingerprint:
        raise ValueError(
            "dataset fingerprint unavailable; provide --dataset-success or "
            "--dataset-fingerprint"
        )
    return {
        "fingerprint": str(fingerprint),
        "status": marker_payload.get("status", "not_recorded"),
        "success_marker": str(marker) if marker else None,
        "success_marker_sha256": sha256(marker) if marker else None,
    }


def init_runtime(device_arg: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid distributed rank/world size: {rank}/{world_size}")
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return rank, world_size, local_rank, device


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name))


def atomic_write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-weighted NLL/PPL for a raw MiniMind Base checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", action="append", required=True,
        help="JSONL path or glob; repeat the option for multiple patterns.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-dir", "--minimind-dir", dest="minimind_dir", type=Path, default=Path("."))
    parser.add_argument("--dataset-success", type=Path)
    parser.add_argument("--dataset-fingerprint")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--expected-rows", type=int,
        help="Required for formal evaluation; exact global validation row count.",
    )
    parser.add_argument(
        "--expected-target-tokens", type=int,
        help=(
            "Required for formal evaluation; exact shifted labels[:,1:] != -100 count."
        ),
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Evaluate only the first N global rows; smoke-only, never a formal score.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers non-negative")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    for name in ("expected_rows", "expected_target_tokens"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    formal_run = args.max_samples is None
    if formal_run and (
        args.expected_rows is None or args.expected_target_tokens is None
    ):
        parser.error(
            "formal evaluation requires --expected-rows and --expected-target-tokens"
        )

    checkpoint = args.checkpoint.resolve()
    minimind_dir = args.minimind_dir.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not (minimind_dir / "model" / "modeling_mica.py").is_file():
        raise FileNotFoundError(f"MiniMind source not found: {minimind_dir}")
    data_files = expand_data_files(args.data)
    dataset_identity = load_dataset_identity(
        data_files, args.dataset_success, args.dataset_fingerprint
    )

    rank, world_size, local_rank, device = init_runtime(args.device)
    total_started_at = utc_now()
    total_started_clock = time.perf_counter()

    sys.path.insert(0, str(minimind_dir))
    from dataset.lm_dataset import PretrainDataset
    from model.modeling_mica import MicaConfig, MicaForCausalLM

    tokenizer_path = minimind_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    dataset = PretrainDataset(
        [str(path) for path in data_files], tokenizer, max_length=args.max_seq_len
    )
    dataset_rows = len(dataset)
    if formal_run and dataset_rows != args.expected_rows:
        raise RuntimeError(
            f"formal dataset row mismatch: {dataset_rows} != {args.expected_rows}"
        )
    evaluation_rows = min(dataset_rows, args.max_samples or dataset_rows)
    rank_indices = range(rank, evaluation_rows, world_size)
    loader = DataLoader(
        Subset(dataset, rank_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    config = MicaConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )
    model = MicaForCausalLM(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError("checkpoint is not a non-empty raw inference state_dict")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError(
            "checkpoint contains non-tensor state; pass the raw inference checkpoint, "
            "not a resume checkpoint"
        )
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.to(device).eval()

    if world_size > 1:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_started_at = utc_now()
    evaluation_started_clock = time.perf_counter()
    local_nll_sum = 0.0
    local_target_tokens = 0
    local_rows = 0
    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # MiniMind shifts labels internally. labels[:, 0] is BOS and is not
            # a prediction target; -100 targets are ignored by cross entropy.
            target_tokens = int(labels[:, 1:].ne(-100).sum().item())
            if target_tokens <= 0:
                raise RuntimeError("batch contains no valid next-token targets")
            with autocast_context(device, args.dtype):
                loss = model(input_ids, labels=labels).loss
            loss_value = float(loss.detach().float().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite validation loss: {loss_value}")
            local_nll_sum += loss_value * target_tokens
            local_target_tokens += target_tokens
            local_rows += int(input_ids.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    evaluation_elapsed = time.perf_counter() - evaluation_started_clock
    total_wall_elapsed = time.perf_counter() - total_started_clock
    stats_device = device if world_size > 1 and device.type == "cuda" else torch.device("cpu")
    totals = torch.tensor(
        [local_nll_sum, float(local_target_tokens), float(local_rows)],
        dtype=torch.float64,
        device=stats_device,
    )
    durations = torch.tensor(
        [evaluation_elapsed, total_wall_elapsed],
        dtype=torch.float64,
        device=stats_device,
    )
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(durations, op=dist.ReduceOp.MAX)
    total_nll_sum = float(totals[0].item())
    total_target_tokens = int(totals[1].item())
    total_rows = int(totals[2].item())
    if total_rows != evaluation_rows:
        raise RuntimeError(f"rank-stride row mismatch: {total_rows} != {evaluation_rows}")
    if total_target_tokens <= 0:
        raise RuntimeError("evaluation produced zero target tokens")
    if formal_run and total_rows != args.expected_rows:
        raise RuntimeError(
            f"formal evaluated row mismatch: {total_rows} != {args.expected_rows}"
        )
    if formal_run and total_target_tokens != args.expected_target_tokens:
        raise RuntimeError(
            "formal target-token mismatch: "
            f"{total_target_tokens} != {args.expected_target_tokens}"
        )
    nll = total_nll_sum / total_target_tokens
    ppl = math.exp(nll) if nll < 700 else None

    if rank == 0:
        evaluation_duration = float(durations[0].item())
        payload = {
            "status": "completed" if formal_run else "smoke",
            "started_at": evaluation_started_at,
            "finished_at": utc_now(),
            "total_started_at": total_started_at,
            "checkpoint": str(checkpoint),
            "checkpoint_format": "raw_inference_state_dict",
            "checkpoint_sha256": sha256(checkpoint),
            "dataset_fingerprint": dataset_identity["fingerprint"],
            "dataset_status": dataset_identity["status"],
            "dataset_success_marker": dataset_identity["success_marker"],
            "dataset_success_marker_sha256": dataset_identity["success_marker_sha256"],
            "data_files": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in data_files
            ],
            "dataset_rows": dataset_rows,
            "rows": total_rows,
            "expected_rows": args.expected_rows,
            "expected_target_tokens": args.expected_target_tokens,
            "target_tokens": total_target_tokens,
            "nll_sum": total_nll_sum,
            "nll": nll,
            "ppl": ppl,
            "ppl_overflow": ppl is None,
            "loss_semantics": "MiniMind shifted next-token CE; labels[:,1:] != -100",
            "rank_assignment": "global_index_mod_world_size",
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "dtype": args.dtype,
            "device": str(device),
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_json_sha256": sha256(tokenizer_path / "tokenizer.json"),
            "elapsed_seconds": evaluation_duration,
            "evaluation_elapsed_seconds": evaluation_duration,
            "total_wall_elapsed_seconds": float(durations[1].item()),
            "timing_scope": (
                "evaluation starts after tokenizer/dataset/model load, distributed barrier, "
                "and CUDA synchronize; durations are max across ranks"
            ),
            "rows_per_second": total_rows / evaluation_duration,
            "target_tokens_per_second": total_target_tokens / evaluation_duration,
            "torch_version": torch.__version__,
            "cuda_device_name": (
                torch.cuda.get_device_name(local_rank) if device.type == "cuda" else None
            ),
        }
        atomic_write_json(output, payload, args.overwrite)
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
