"""Auditable MiniMind Dense pretraining with exact validation and bounded probes.

This entry is deliberately separate from train_pretrain.py so historical P01/P02
commands keep their original behavior. It preserves that entry's model,
optimizer, cosine-by-full-epoch learning-rate schedule, and DDP train sampler.
Pure inference exports remain FP16, while resumable state keeps original-dtype
model tensors plus optimizer/scaler state. This entry adds:

- an independent validation JSON/JSONL glob;
- exact rank-stride validation without DistributedSampler padding;
- token-weighted train and validation metrics;
- explicit SwanLab project/run names;
- optimizer-step-bounded probes through --max_steps;
- correct normalization and checkpoint ordering for a final partial
  gradient-accumulation window;
- isolated, parameterized resume directories.
"""

import argparse
import signal
STOP_SIGNAL = False
def request_stop(*_):
    global STOP_SIGNAL
    STOP_SIGNAL = True
signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
import json
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets  # noqa: F401
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    setup_seed,
)

warnings.filterwarnings("ignore")


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def reduce_sum(values, device):
    if torch.is_tensor(values):
        tensor = values.detach().to(device=device, dtype=torch.float64).clone()
    else:
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def all_ranks_finite(value, device):
    tensor = value.detach().to(device=device, dtype=torch.int32).clone()
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def assert_all_ranks_finite(value, message):
    """All-reduce one device sentinel and abort before an unsafe update."""

    tensor = value.detach().to(dtype=torch.int32).clone()
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    if hasattr(torch, "_assert_async"):
        torch._assert_async(tensor.bool(), message)
    elif not bool(tensor.item()):
        raise FloatingPointError(message)


def reduce_max(value, device):
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float64).clone()
    else:
        tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def append_metric(path, payload, swanlab=None):
    if is_main_process():
        record = {
            **payload,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        if swanlab:
            swanlab.log(
                {
                    key: value
                    for key, value in record.items()
                    if isinstance(value, (int, float))
                    and math.isfinite(float(value))
                }
            )
    if dist.is_initialized():
        dist.barrier()


def save_weight(model, path):
    if is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            key: value.half().cpu()
            for key, value in unwrap_model(model).state_dict().items()
        }
        temporary_path = path + ".tmp"
        torch.save(state_dict, temporary_path)
        os.replace(temporary_path, path)
        del state_dict
    if dist.is_initialized():
        dist.barrier()


def move_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


def get_swanlab_run_id(swanlab):
    if not swanlab or not hasattr(swanlab, "get_run"):
        return None
    run = swanlab.get_run()
    return getattr(run, "id", None) if run else None


def resume_path(lm_config, args):
    suffix = "_moe" if lm_config.use_moe else ""
    return os.path.join(
        args.resume_dir,
        f"{args.save_weight}_{lm_config.hidden_size}{suffix}_resume.pth",
    )


def load_training_state(lm_config, args):
    path = resume_path(lm_config, args)
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def save_training_state(
    model,
    optimizer,
    scaler,
    lm_config,
    args,
    epoch,
    micro_step,
    optimizer_step,
    best_validation_loss,
    swanlab,
    resume_contract,
):
    suffix = "_moe" if lm_config.use_moe else ""
    weight_path = os.path.join(
        args.save_dir,
        f"{args.save_weight}_{lm_config.hidden_size}{suffix}.pth",
    )
    save_weight(model, weight_path)

    if is_main_process():
        os.makedirs(args.resume_dir, exist_ok=True)
        model_state = {
            key: value.detach().cpu()
            for key, value in unwrap_model(model).state_dict().items()
        }
        resume_data = {
            "resume_format_version": 1,
            "model": model_state,
            "optimizer": move_to_cpu(optimizer.state_dict()),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": micro_step,
            "optimizer_step": optimizer_step,
            "best_validation_loss": best_validation_loss,
            "world_size": world_size(),
            "swanlab_id": get_swanlab_run_id(swanlab),
            "resume_contract": resume_contract,
        }
        path = resume_path(lm_config, args)
        temporary_path = path + ".tmp"
        torch.save(resume_data, temporary_path)
        os.replace(temporary_path, path)
        del model_state, resume_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def evaluate_exact(
    model,
    loader,
    autocast_ctx,
    device,
    expected_rows=0,
    expected_tokens=0,
):
    """Evaluate disjoint rank-stride subsets and reduce exact token totals."""

    was_training = model.training
    model.eval()
    raw_model = unwrap_model(model)
    local_totals = torch.zeros(3, dtype=torch.float64, device=device)
    local_finite = torch.ones((), dtype=torch.bool, device=device)
    started = time.perf_counter()

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid_tokens = labels[..., 1:].ne(-100).sum()
        with autocast_ctx:
            result = raw_model(input_ids, labels=labels)
        local_finite.logical_and_(torch.isfinite(result.loss.detach()))
        local_totals[0].add_(result.loss.detach().double() * valid_tokens)
        local_totals[1].add_(valid_tokens)
        local_totals[2].add_(input_ids.shape[0])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not all_ranks_finite(local_finite, device):
        raise FloatingPointError("non-finite validation loss")
    global_nll, global_tokens, global_rows = reduce_sum(local_totals, device)
    global_seconds = reduce_max(elapsed, device)

    global_rows = int(global_rows)
    global_tokens = int(global_tokens)
    if expected_rows and global_rows != expected_rows:
        raise ValueError(
            f"validation row count mismatch: expected {expected_rows}, "
            f"found {global_rows}"
        )
    if expected_tokens and global_tokens != expected_tokens:
        raise ValueError(
            f"validation target-token mismatch: expected {expected_tokens}, "
            f"found {global_tokens}"
        )

    if was_training:
        model.train()
    validation_loss = global_nll / max(global_tokens, 1)
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 80.0)),
        "validation_rows": global_rows,
        "validation_tokens": global_tokens,
        "validation_seconds": global_seconds,
    }


def build_validation_loader(validation_dataset, args):
    indices = range(rank(), len(validation_dataset), world_size())
    rank_subset = Subset(validation_dataset, indices)
    return DataLoader(
        rank_subset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def validate_resume_contract(checkpoint, expected_contract):
    saved_world_size = int(checkpoint.get("world_size", 1))
    if saved_world_size != world_size():
        raise ValueError(
            "P03 resume requires the same world size: "
            f"saved={saved_world_size}, current={world_size()}"
        )
    saved_contract = checkpoint.get("resume_contract")
    if saved_contract != expected_contract:
        raise ValueError(
            "resume contract mismatch; dataset fingerprint, trainer/protocol, "
            "batch, accumulation, sequence, epochs, and schedule must match"
        )


def main():
    parser = argparse.ArgumentParser(
        description="MiniMind pretraining with exact validation and bounded probes"
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--resume_dir", required=True)
    parser.add_argument("--save_weight", default="pretrain_v1")
    parser.add_argument("--best_weight", default="pretrain_v1_best_val")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Metrics interval in optimizer steps; 0 disables periodic logging.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=250,
        help="Resume/checkpoint interval in optimizer steps; 0 disables it.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Validation interval in optimizer steps; final validation always runs.",
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--use_moe", type=int, default=0, choices=(0, 1))
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--validation_path", required=True)
    parser.add_argument("--validation_batch_size", type=int, default=64)
    parser.add_argument("--expected_validation_rows", type=int, default=0)
    parser.add_argument("--expected_validation_tokens", type=int, default=0)
    parser.add_argument("--from_weight", default="none")
    parser.add_argument("--from_resume", type=int, default=0, choices=(0, 1))
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument(
        "--experiment_id",
        default="pretrain-with-validation",
    )
    parser.add_argument("--dataset_fingerprint", default="")
    parser.add_argument("--trainer_sha256", default="")
    parser.add_argument("--protocol_sha256", default="")
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="MiniMind-Lab")
    parser.add_argument("--swanlab_group", default="")
    parser.add_argument("--swanlab_tags", default="")
    parser.add_argument(
        "--swanlab_run_name",
        default="MiniMind-Pretrain-With-Validation",
    )
    parser.add_argument("--use_compile", type=int, default=0, choices=(0, 1))
    args = parser.parse_args()

    for name in (
        "epochs",
        "batch_size",
        "num_workers",
        "accumulation_steps",
        "validation_batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("log_interval", "save_interval", "eval_interval", "max_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup_seed(42 + rank())
    device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    checkpoint = load_training_state(lm_config, args) if args.from_resume else None
    if args.from_resume and checkpoint is None:
        raise FileNotFoundError(
            "from_resume=1 but no matching resume checkpoint exists in "
            f"{args.resume_dir}"
        )

    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu" or dtype == torch.float32
        else torch.cuda.amp.autocast(dtype=dtype)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device,
    )
    train_dataset = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
    )
    validation_dataset = PretrainDataset(
        args.validation_path,
        tokenizer,
        max_length=args.max_seq_len,
    )

    train_sampler = (
        DistributedSampler(train_dataset)
        if dist.is_initialized()
        else None
    )
    samples_per_rank = (
        len(train_sampler) if train_sampler is not None else len(train_dataset)
    )
    micro_steps_per_epoch = math.ceil(samples_per_rank / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_steps_per_epoch / args.accumulation_steps
    )
    resolved_total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    resolved_total_micro_steps = micro_steps_per_epoch * args.epochs
    resolved_target_optimizer_steps = (
        min(args.max_steps, resolved_total_optimizer_steps)
        if args.max_steps
        else resolved_total_optimizer_steps
    )

    resume_contract = {
        "experiment_id": args.experiment_id,
        "dataset_fingerprint": args.dataset_fingerprint,
        "trainer_sha256": args.trainer_sha256,
        "protocol_sha256": args.protocol_sha256,
        "data_path": args.data_path,
        "validation_path": args.validation_path,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "max_seq_len": args.max_seq_len,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "use_moe": args.use_moe,
        "dtype": args.dtype,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "epochs": args.epochs,
        "world_size": world_size(),
        "micro_steps_per_epoch": micro_steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "resolved_total_optimizer_steps": resolved_total_optimizer_steps,
    }

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    start_micro_step = 0
    optimizer_step = 0
    best_validation_loss = float("inf")
    if checkpoint:
        validate_resume_contract(checkpoint, resume_contract)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        start_micro_step = int(checkpoint.get("step", 0))
        optimizer_step = int(
            checkpoint.get(
                "optimizer_step",
                start_epoch * optimizer_steps_per_epoch
                + math.ceil(start_micro_step / args.accumulation_steps),
            )
        )
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )

    if args.use_compile:
        model = torch.compile(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    route_counts = {}
    observe_route = False
    def route_hook(layer):
        def capture(module, inputs, logits):
            if observe_route:
                counts = torch.bincount(logits.detach().argmax(-1).flatten(), minlength=4).double()
                route_counts[layer] = counts
        return capture
    if lm_config.use_moe:
        for layer, block in enumerate(unwrap_model(model).model.layers):
            block.mlp.gate.register_forward_hook(route_hook(layer))
    validation_loader = build_validation_loader(validation_dataset, args)

    swanlab = None
    if args.use_swanlab and is_main_process():
        import swanlab

        run_id = checkpoint.get("swanlab_id") if checkpoint else None
        tags = [
            tag.strip()
            for tag in args.swanlab_tags.split(",")
            if tag.strip()
        ]
        swanlab_config = {
            **vars(args),
            "world_size": world_size(),
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "micro_steps_per_epoch": micro_steps_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "resolved_total_micro_steps": resolved_total_micro_steps,
            "resolved_total_optimizer_steps": resolved_total_optimizer_steps,
            "resolved_target_optimizer_steps": resolved_target_optimizer_steps,
        }
        init_kwargs = {
            "project": args.swanlab_project,
            "experiment_name": args.swanlab_run_name,
            "config": swanlab_config,
        }
        if args.swanlab_group:
            init_kwargs["group"] = args.swanlab_group
        if tags:
            init_kwargs["tags"] = tags
        if run_id:
            init_kwargs.update({"id": run_id, "resume": "must"})
        init_kwargs["mode"] = "online"
        run = swanlab.init(**init_kwargs)
        with open(os.path.join(args.save_dir, "swanlab-url.txt"), "w") as handle:
            handle.write(run.url + "\n")

    schedule_record = {
        "event": "schedule",
        "experiment_id": args.experiment_id,
        "dataset_fingerprint": args.dataset_fingerprint,
        "trainer_sha256": args.trainer_sha256,
        "protocol_sha256": args.protocol_sha256,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "world_size": world_size(),
        "per_gpu_batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "global_sequence_batch": (
            args.batch_size * args.accumulation_steps * world_size()
        ),
        "micro_steps_per_epoch": micro_steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "resolved_total_micro_steps": resolved_total_micro_steps,
        "resolved_total_optimizer_steps": resolved_total_optimizer_steps,
        "resolved_target_optimizer_steps": resolved_target_optimizer_steps,
        "scheduler_total_optimizer_steps": resolved_total_optimizer_steps,
        "max_steps_is_exit_bound_only": True,
    }
    Logger(json.dumps(schedule_record, ensure_ascii=False, sort_keys=True))
    append_metric(args.metrics_path, schedule_record, swanlab)

    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stop_requested = optimizer_step >= resolved_target_optimizer_steps
    final_epoch = start_epoch
    final_micro_step = start_micro_step
    window_totals = torch.zeros(4, dtype=torch.float64, device=device)
    window_finite = torch.ones((), dtype=torch.bool, device=device)
    window_micro_steps = 0
    window_optimizer_steps = 0
    cumulative_local_padded_tokens = torch.zeros(
        (), dtype=torch.float64, device=device
    )
    local_active_training_seconds = 0.0
    training_loop_started = time.perf_counter()
    window_started = training_loop_started

    for epoch in range(start_epoch, args.epochs):
        if stop_requested:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        epoch_indices = (
            train_sampler
            if train_sampler is not None
            else torch.randperm(len(train_dataset)).tolist()
        )
        skip = start_micro_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(
            epoch_indices,
            args.batch_size,
            skip,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if skip:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: skip {skip} consumed "
                f"micro-steps and resume at {skip + 1}"
            )

        for micro_step, (input_ids, labels) in enumerate(
            train_loader,
            start=skip + 1,
        ):
            final_epoch = epoch
            final_micro_step = micro_step
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid_tokens = labels[..., 1:].ne(-100).sum()
            padded_tokens = input_ids.numel()

            # AdamW consumes the LR only at the optimizer boundary. Keep one
            # full-epoch cosine schedule in optimizer-step units so changing
            # only the micro-batch/accumulation decomposition leaves the
            # effective LR curve unchanged. --max_steps is never the
            # denominator: a 100-step probe samples the formal run's first
            # 100 updates.
            scheduler_optimizer_step = optimizer_step + 1
            current_lr = get_lr(
                scheduler_optimizer_step,
                resolved_total_optimizer_steps,
                args.learning_rate,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr

            window_start = (
                ((micro_step - 1) // args.accumulation_steps)
                * args.accumulation_steps
                + 1
            )
            actual_window_size = min(
                args.accumulation_steps,
                micro_steps_per_epoch - window_start + 1,
            )

            observe_route = (optimizer_step + 1) % 100 == 0 and (micro_step - 1) % args.accumulation_steps == 0
            with autocast_ctx:
                result = model(input_ids, labels=labels)
                auxiliary_loss = (
                    result.aux_loss
                    if result.aux_loss is not None
                    else result.loss.new_zeros(())
                )
                combined_loss = result.loss + auxiliary_loss
                objective = combined_loss / actual_window_size
            window_finite.logical_and_(torch.isfinite(combined_loss.detach()))
            scaler.scale(objective).backward()

            window_totals[0].add_(result.loss.detach().double() * valid_tokens)
            window_totals[1].add_(valid_tokens)
            window_totals[2].add_(padded_tokens)
            window_totals[3].add_(auxiliary_loss.detach().double() * valid_tokens)
            cumulative_local_padded_tokens.add_(padded_tokens)
            window_micro_steps += 1

            end_of_accumulation = (
                micro_step % args.accumulation_steps == 0
                or micro_step == micro_steps_per_epoch
            )
            if not end_of_accumulation:
                del input_ids, labels, result, combined_loss, objective
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            window_finite.logical_and_(torch.isfinite(grad_norm.detach()))
            assert_all_ranks_finite(
                window_finite,
                f"non-finite loss or gradient before optimizer update "
                f"epoch={epoch}, micro_step={micro_step}, "
                f"optimizer_step={optimizer_step + 1}",
            )
            scaler.step(optimizer)
            scaler.update()
            window_finite.fill_(True)
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            window_optimizer_steps += 1

            # Test injection sends a real signal to rank0; all ranks stop at one boundary.
            if rank() == 0 and optimizer_step == int(os.environ.get("PHASE7_SIGNAL_STEP", "-1")):
                os.kill(os.getpid(), signal.SIGTERM)
            signal_requested = bool(reduce_max(int(STOP_SIGNAL), device))
            reached_bound = optimizer_step >= resolved_target_optimizer_steps or signal_requested
            should_evaluate = (
                args.eval_interval > 0
                and optimizer_step % args.eval_interval == 0
            )
            should_save = (
                args.save_interval > 0
                and optimizer_step % args.save_interval == 0
            )
            should_log = (
                (args.log_interval > 0 and optimizer_step % args.log_interval == 0)
                or should_evaluate
                or should_save
                or reached_bound
                or micro_step == micro_steps_per_epoch
            )

            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - window_started
                local_active_training_seconds += elapsed
                global_nll, global_tokens, global_padded_tokens, global_aux = reduce_sum(
                    window_totals,
                    device,
                )
                global_seconds = reduce_max(elapsed, device)
                grad_norm_max = reduce_max(grad_norm, device)
                peak_memory_mib = reduce_max(
                    (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                    device,
                )
                train_loss = global_nll / max(global_tokens, 1)
                remaining_updates = max(
                    resolved_target_optimizer_steps - optimizer_step,
                    0,
                )
                optimizer_step_seconds = (
                    global_seconds / max(window_optimizer_steps, 1)
                )
                metric = {
                    "event": "train",
                    "epoch": epoch + 1,
                    "micro_step": micro_step,
                    "optimizer_step": optimizer_step,
                    "train_loss": train_loss,
                    "aux_loss": global_aux / max(global_tokens, 1),
                    "total_loss_token_weighted_diagnostic": (global_nll + global_aux) / max(global_tokens, 1),
                    "learning_rate": current_lr,
                    "grad_norm": grad_norm_max,
                    "valid_tokens_per_second": (
                        global_tokens / max(global_seconds, 1e-9)
                    ),
                    "padded_tokens_per_second": (
                        global_padded_tokens / max(global_seconds, 1e-9)
                    ),
                    "window_valid_tokens": int(global_tokens),
                    "window_padded_tokens": int(global_padded_tokens),
                    "window_micro_steps": window_micro_steps,
                    "window_optimizer_steps": window_optimizer_steps,
                    "micro_step_seconds": (
                        global_seconds / max(window_micro_steps, 1)
                    ),
                    "optimizer_step_seconds": optimizer_step_seconds,
                    "peak_memory_mib": peak_memory_mib,
                    "eta_minutes_to_target": (
                        optimizer_step_seconds * remaining_updates / 60.0
                    ),
                }
                Logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}] "
                    f"micro={micro_step}/{micro_steps_per_epoch} "
                    f"optimizer={optimizer_step}/"
                    f"{resolved_target_optimizer_steps} "
                    f"loss={train_loss:.4f} lr={current_lr:.8f} "
                    f"valid_tok_s={metric['valid_tokens_per_second']:.1f} "
                    f"padded_tok_s={metric['padded_tokens_per_second']:.1f} "
                    f"step_s={optimizer_step_seconds:.4f} "
                    f"peak_mem={peak_memory_mib:.0f}MiB"
                )
                for layer, counts in route_counts.items():
                    if dist.is_initialized():
                        dist.all_reduce(counts)
                    fractions = counts / counts.sum()
                    for expert, value in enumerate(fractions.tolist()):
                        metric[f"router/layer{layer}/expert{expert}_fraction"] = value
                    metric[f"router/layer{layer}/max_mean_load"] = float(fractions.max() * 4)
                route_counts.clear()
                append_metric(args.metrics_path, metric, swanlab)
                window_totals.zero_()
                window_finite.fill_(True)
                window_micro_steps = 0
                window_optimizer_steps = 0
                window_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

            if should_evaluate:
                validation = evaluate_exact(
                    model,
                    validation_loader,
                    autocast_ctx,
                    device,
                    args.expected_validation_rows,
                    args.expected_validation_tokens,
                )
                validation.update(
                    {
                        "event": "validation",
                        "phase": "periodic",
                        "epoch": epoch + 1,
                        "micro_step": micro_step,
                        "optimizer_step": optimizer_step,
                    }
                )
                append_metric(args.metrics_path, validation, swanlab)
                Logger(
                    f"Validation optimizer={optimizer_step}: "
                    f"loss={validation['validation_loss']:.6f} "
                    f"ppl={validation['validation_perplexity']:.4f} "
                    f"rows={validation['validation_rows']} "
                    f"tokens={validation['validation_tokens']}"
                )
                if validation["validation_loss"] < best_validation_loss:
                    best_validation_loss = validation["validation_loss"]
                    best_path = os.path.join(
                        args.save_dir,
                        f"{args.best_weight}_{lm_config.hidden_size}.pth",
                    )
                    save_weight(model, best_path)
                window_started = time.perf_counter()

            if should_save:
                save_training_state(
                    model,
                    optimizer,
                    scaler,
                    lm_config,
                    args,
                    epoch,
                    micro_step,
                    optimizer_step,
                    best_validation_loss,
                    swanlab,
                    resume_contract,
                )
                window_started = time.perf_counter()

            del input_ids, labels, result, combined_loss, objective
            if reached_bound:
                stop_requested = True
                break

        start_micro_step = 0

    final_validation = evaluate_exact(
        model,
        validation_loader,
        autocast_ctx,
        device,
        args.expected_validation_rows,
        args.expected_validation_tokens,
    )
    final_validation.update(
        {
            "event": "validation",
            "phase": "final",
            "epoch": final_epoch + 1,
            "micro_step": final_micro_step,
            "optimizer_step": optimizer_step,
        }
    )
    append_metric(args.metrics_path, final_validation, swanlab)
    Logger(
        f"Final validation optimizer={optimizer_step}: "
        f"loss={final_validation['validation_loss']:.6f} "
        f"ppl={final_validation['validation_perplexity']:.4f} "
        f"rows={final_validation['validation_rows']} "
        f"tokens={final_validation['validation_tokens']}"
    )
    if final_validation["validation_loss"] < best_validation_loss:
        best_validation_loss = final_validation["validation_loss"]
        best_path = os.path.join(
            args.save_dir,
            f"{args.best_weight}_{lm_config.hidden_size}.pth",
        )
        save_weight(model, best_path)

    save_training_state(
        model,
        optimizer,
        scaler,
        lm_config,
        args,
        final_epoch,
        final_micro_step,
        optimizer_step,
        best_validation_loss,
        swanlab,
        resume_contract,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_loop_wall_seconds = reduce_max(
        time.perf_counter() - training_loop_started,
        device,
    )
    global_padded_tokens = int(
        reduce_sum(cumulative_local_padded_tokens.reshape(1), device)[0]
    )
    active_training_seconds = reduce_max(
        local_active_training_seconds,
        device,
    )

    completed_record = {
        "event": "completed",
        "status": (
            "interrupted_checkpointed" if STOP_SIGNAL else "max_steps_reached"
            if args.max_steps and optimizer_step >= args.max_steps
            else "epochs_complete"
        ),
        "epoch": final_epoch + 1,
        "micro_step": final_micro_step,
        "optimizer_step": optimizer_step,
        "resolved_total_optimizer_steps": resolved_total_optimizer_steps,
        "resolved_target_optimizer_steps": resolved_target_optimizer_steps,
        "best_validation_loss": best_validation_loss,
        "active_training_seconds": active_training_seconds,
        "training_loop_wall_seconds": training_loop_wall_seconds,
        "padded_tokens": global_padded_tokens,
        "active_padded_tokens_per_second": (
            global_padded_tokens / max(active_training_seconds, 1e-9)
        ),
        "training_loop_padded_tokens_per_second": (
            global_padded_tokens / max(training_loop_wall_seconds, 1e-9)
        ),
        "final_validation_loss": final_validation["validation_loss"],
        "final_validation_perplexity": final_validation[
            "validation_perplexity"
        ],
        "final_validation_rows": final_validation["validation_rows"],
        "final_validation_tokens": final_validation["validation_tokens"],
    }
    append_metric(args.metrics_path, completed_record, swanlab)
    Logger(json.dumps(completed_record, ensure_ascii=False, sort_keys=True))

    if swanlab:
        swanlab.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
