import argparse
import json
import os
import sys
import time

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets  # noqa: F401
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import SFTDataset
from model.modeling_mica import MicaConfig
from trainer.trainer_utils import init_distributed_mode, init_model, setup_seed


def main():
    parser = argparse.ArgumentParser(description="8-GPU SFT batch-size throughput probe")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--from_weight", default="pretrain")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--measure_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_seed(42 + rank)

    config = MicaConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=False)
    model, tokenizer = init_model(config, args.from_weight, device=str(device))
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, augment=True)
    sampler = DistributedSampler(dataset, shuffle=True, seed=42, drop_last=True) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    total_steps = args.warmup_steps + args.measure_steps
    measured_tokens = 0
    started = None
    for step, (input_ids, labels) in enumerate(loader, start=1):
        if step > total_steps:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx:
            result = model(input_ids, labels=labels)
            loss = result.loss + result.aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == args.warmup_steps:
            torch.cuda.synchronize(device)
            started = time.time()
        elif step > args.warmup_steps:
            measured_tokens += labels[..., 1:].ne(-100).sum().item()

    if started is None:
        raise RuntimeError("probe loader ended during warmup")
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    values = torch.tensor([measured_tokens, elapsed, torch.cuda.max_memory_allocated(device) / 1024 ** 2], dtype=torch.float64, device=device)
    if dist.is_initialized():
        token_value = values[0].clone()
        dist.all_reduce(token_value, op=dist.ReduceOp.SUM)
        elapsed_value = values[1:].clone()
        dist.all_reduce(elapsed_value, op=dist.ReduceOp.MAX)
        global_tokens = token_value.item()
        elapsed = elapsed_value[0].item()
        peak_memory_mib = elapsed_value[1].item()
    else:
        global_tokens = values[0].item()
        peak_memory_mib = values[2].item()

    if rank == 0:
        result = {
            "status": "passed",
            "per_gpu_batch_size": args.batch_size,
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
            "sequence_length": args.max_seq_len,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "elapsed_seconds": elapsed,
            "mean_step_seconds": elapsed / args.measure_steps,
            "sequences_per_second": args.batch_size * world_size * args.measure_steps / elapsed,
            "valid_tokens_per_second": global_tokens / elapsed,
            "peak_memory_mib": peak_memory_mib,
        }
        print("PROBE_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
