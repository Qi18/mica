import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext

__package__ = "trainer.sft"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import datasets  # noqa: F401
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataset.lm_dataset import SFTDataset
from model.model_mica import MicaConfig
from model.model_lora import apply_lora, save_lora
from trainer.common.utils import Logger, get_lr, init_distributed_mode, init_model, is_main_process, setup_seed


def reduce_sum(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def reduce_max(value, device):
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def append_metric(path, payload, swanlab=None):
    if not is_main_process():
        return
    payload = {**payload, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if swanlab:
        swanlab.log({key: value for key, value in payload.items() if isinstance(value, (int, float))})


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def save_weight(model, path, lora_only=False):
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if lora_only:
        save_lora(model, path)
    else:
        state_dict = unwrap_model(model).state_dict()
        torch.save({key: value.half().cpu() for key, value in state_dict.items()}, path)
    Logger(f"Saved {'LoRA adapter' if lora_only else 'full checkpoint'}: {path}")


@torch.no_grad()
def evaluate(model, loader, autocast_ctx, device):
    model.eval()
    local_nll = 0.0
    local_tokens = 0
    started = time.time()
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid_tokens = labels[..., 1:].ne(-100).sum().item()
        if valid_tokens == 0:
            continue
        with autocast_ctx:
            result = model(input_ids, labels=labels)
        local_nll += result.loss.item() * valid_tokens
        local_tokens += valid_tokens
    global_nll, global_tokens = reduce_sum([local_nll, local_tokens], device)
    elapsed = reduce_max(time.time() - started, device)
    model.train()
    return global_nll / max(global_tokens, 1), int(global_tokens), elapsed


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(total_size, validation_size, seed, manifest_path):
    if validation_size <= 0 or validation_size >= total_size:
        raise ValueError(f"validation_size must be in [1, {total_size - 1}]")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    validation_indices = sorted(permutation[:validation_size])
    train_indices = permutation[validation_size:]
    digest = hashlib.sha256(",".join(map(str, validation_indices)).encode()).hexdigest()
    if is_main_process() and manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "algorithm": "torch.randperm",
                "seed": seed,
                "dataset_size": total_size,
                "train_size": len(train_indices),
                "validation_size": len(validation_indices),
                "validation_indices_sha256": digest,
                "validation_indices": validation_indices,
            }, handle, ensure_ascii=False, indent=2)
    return train_indices, validation_indices, digest


def main():
    parser = argparse.ArgumentParser(description="MiniMind Full SFT with deterministic validation")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--save_weight", default="sft_last")
    parser.add_argument("--best_weight", default="sft_best_val")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--train_augment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--validation_data_path")
    parser.add_argument("--from_weight", default="pretrain")
    parser.add_argument("--lora_rank", type=int, default=0, help="0=full fine-tuning; positive=LoRA rank")
    parser.add_argument("--parameter_manifest")
    parser.add_argument("--validation_size", type=int, default=10000)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="MiniMind-Lab")
    parser.add_argument("--swanlab_run_name", default="dense-sft")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.split_seed + (dist.get_rank() if dist.is_initialized() else 0))
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = nullcontext() if device_type == "cpu" or dtype == torch.float32 else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    lm_config = MicaConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    lora_enabled = args.lora_rank > 0
    if lora_enabled:
        apply_lora(model, rank=args.lora_rank)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora" in name
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    injected_modules = [
        name for name, module in model.named_modules() if hasattr(module, "lora")
    ]
    parameter_payload = {
        "method": "lora" if lora_enabled else "full",
        "lora_rank": args.lora_rank,
        "total_parameters": total_parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "trainable_ratio": trainable_parameter_count / total_parameter_count,
        "injected_module_count": len(injected_modules),
        "injected_modules": injected_modules,
    }
    if is_main_process() and args.parameter_manifest:
        os.makedirs(os.path.dirname(args.parameter_manifest), exist_ok=True)
        with open(args.parameter_manifest, "w", encoding="utf-8") as handle:
            json.dump(parameter_payload, handle, ensure_ascii=False, indent=2)
    Logger(
        f"method={parameter_payload['method']} total_params={total_parameter_count} "
        f"trainable_params={trainable_parameter_count} "
        f"trainable_ratio={parameter_payload['trainable_ratio']:.6f}"
    )

    train_base = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, augment=bool(args.train_augment))
    if args.validation_data_path:
        train_dataset = train_base
        validation_dataset = SFTDataset(
            args.validation_data_path,
            tokenizer,
            max_length=args.max_seq_len,
            augment=False,
        )
        split_sha = sha256_file(args.validation_data_path)
        if is_main_process():
            os.makedirs(os.path.dirname(args.split_manifest), exist_ok=True)
            with open(args.split_manifest, "w", encoding="utf-8") as handle:
                json.dump({
                    "algorithm": "explicit_validation_file",
                    "seed": None,
                    "train_path": os.path.abspath(args.data_path),
                    "train_size": len(train_dataset),
                    "validation_path": os.path.abspath(args.validation_data_path),
                    "validation_size": len(validation_dataset),
                    "validation_sha256": split_sha,
                }, handle, ensure_ascii=False, indent=2)
    else:
        validation_base = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, augment=False)
        train_indices, validation_indices, split_sha = build_split(
            len(train_base), args.validation_size, args.split_seed, args.split_manifest
        )
        train_dataset = Subset(train_base, train_indices)
        validation_dataset = Subset(validation_base, validation_indices)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.split_seed) if dist.is_initialized() else None
    validation_sampler = DistributedSampler(validation_dataset, shuffle=False) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate)
    if args.use_compile and lora_enabled:
        raise ValueError("LoRA monkey-patched forwards are incompatible with torch.compile")
    if args.use_compile:
        model = torch.compile(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    swanlab = None
    if args.use_swanlab and is_main_process():
        import swanlab
        swanlab.init(project=args.swanlab_project, name=args.swanlab_run_name, config={
            **vars(args),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "global_batch_size": args.batch_size * (dist.get_world_size() if dist.is_initialized() else 1) * args.accumulation_steps,
            "train_size": len(train_dataset),
            "validation_size_actual": len(validation_dataset),
            "validation_indices_sha256": split_sha,
            **parameter_payload,
        })

    total_optimizer_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
    training_started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    baseline_loss, baseline_tokens, baseline_seconds = evaluate(model, validation_loader, autocast_ctx, device)
    append_metric(args.metrics_path, {
        "event": "validation",
        "phase": "sft_baseline",
        "global_step": 0,
        "validation_loss": baseline_loss,
        "validation_tokens": baseline_tokens,
        "validation_seconds": baseline_seconds,
    }, swanlab)
    Logger(f"SFT baseline validation_loss={baseline_loss:.6f}, tokens={baseline_tokens}")
    best_validation_loss = baseline_loss
    save_weight(model, os.path.join(args.save_dir, f"{args.best_weight}_{args.hidden_size}.pth"), lora_enabled)

    window_nll = 0.0
    window_tokens = 0
    window_started = time.time()
    last_grad_norm = 0.0

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        for micro_step, (input_ids, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid_tokens = labels[..., 1:].ne(-100).sum().item()
            global_valid_tokens = valid_tokens
            if dist.is_initialized():
                token_count = torch.tensor(float(valid_tokens), dtype=torch.float64, device=device)
                dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
                global_valid_tokens = token_count.item()
            next_global_step = global_step + 1
            lr = get_lr(next_global_step, total_optimizer_steps, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with autocast_ctx:
                result = model(input_ids, labels=labels)
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                token_weight = valid_tokens * world_size / max(global_valid_tokens, 1)
                objective = (
                    result.loss * token_weight + result.aux_loss
                ) / args.accumulation_steps
            scaler.scale(objective).backward()
            window_nll += result.loss.item() * valid_tokens
            window_tokens += valid_tokens

            if micro_step % args.accumulation_steps != 0 and micro_step != len(train_loader):
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            last_grad_norm = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            should_log = global_step % args.log_interval == 0 or global_step == total_optimizer_steps
            if should_log:
                global_nll, global_tokens = reduce_sum([window_nll, window_tokens], device)
                elapsed = reduce_max(time.time() - window_started, device)
                grad_norm_max = reduce_max(last_grad_norm, device)
                train_loss = global_nll / max(global_tokens, 1)
                tokens_per_second = global_tokens / max(elapsed, 1e-9)
                peak_memory = reduce_max(torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0, device)
                append_metric(args.metrics_path, {
                    "event": "train",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "learning_rate": lr,
                    "grad_norm": grad_norm_max,
                    "valid_tokens_per_second": tokens_per_second,
                    "window_valid_tokens": int(global_tokens),
                    "peak_memory_mib": peak_memory,
                }, swanlab)
                Logger(f"Epoch {epoch + 1}/{args.epochs} step {global_step}/{total_optimizer_steps}: train_loss={train_loss:.6f}, lr={lr:.8f}, grad_norm={grad_norm_max:.4f}, valid_tok_s={tokens_per_second:.1f}, peak_mem={peak_memory:.0f}MiB")
                window_nll = 0.0
                window_tokens = 0
                window_started = time.time()

            should_evaluate = global_step % args.eval_interval == 0 or global_step == total_optimizer_steps
            if should_evaluate:
                validation_loss, validation_tokens, validation_seconds = evaluate(model, validation_loader, autocast_ctx, device)
                append_metric(args.metrics_path, {
                    "event": "validation",
                    "phase": "sft",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "validation_loss": validation_loss,
                    "validation_tokens": validation_tokens,
                    "validation_seconds": validation_seconds,
                }, swanlab)
                Logger(f"Validation step {global_step}: loss={validation_loss:.6f}, tokens={validation_tokens}")
                window_started = time.time()
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    save_weight(model, os.path.join(args.save_dir, f"{args.best_weight}_{args.hidden_size}.pth"), lora_enabled)

            if global_step % args.save_interval == 0:
                save_weight(model, os.path.join(args.save_dir, f"{args.save_weight}_{args.hidden_size}.pth"), lora_enabled)

    save_weight(model, os.path.join(args.save_dir, f"{args.save_weight}_{args.hidden_size}.pth"), lora_enabled)
    append_metric(args.metrics_path, {
        "event": "completed",
        "global_step": global_step,
        "baseline_validation_loss": baseline_loss,
        "best_validation_loss": best_validation_loss,
        "wall_seconds": time.time() - training_started,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0,
        "total_parameters": total_parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "trainable_ratio": trainable_parameter_count / total_parameter_count,
    }, swanlab)
    if swanlab:
        swanlab.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
