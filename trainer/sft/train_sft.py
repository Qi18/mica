"""Mica SFT 入口：沿用 train_pretrain.py 的训练骨架，增加监督微调能力。

主体流程与预训练保持相同的 9 段顺序：初始化环境 -> 配置模型与 checkpoint ->
混合精度 -> 实验记录 -> 模型/数据/优化器 -> 恢复 -> compile/DDP ->
逐 epoch 训练与验证 -> 清理。阅读时可以直接和 train_pretrain.py 对照。

两者共用的部分是学习率、梯度累积、AMP、checkpoint、DDP 和 epoch 主循环；
SFT 只在对应阶段增加 assistant label mask、LoRA、验证集切分和 best 权重。

关键约定：
- --lora_rank 0 为全参数 SFT，大于 0 时只训练并导出 LoRA adapter。
- 不传验证参数时保持普通 SFT；传 --validation_data_path 使用独立验证集，传
  --validation_size 则从训练文件确定性切分。
- DDP 验证按 rank 步长切分，不补重复样本；loss 按有效 assistant token 加权。
- save_dir 保存给推理或下游使用的权重，resume_dir 保存 optimizer/scaler/训练位置。
"""

import argparse  # 命令行参数
import hashlib  # 为验证文件或切分索引生成可审计摘要
import json  # 写 metrics JSONL 和 manifest
import math  # validation loss 转 perplexity
import os  # 路径、目录和原子替换
import sys  # 直接运行脚本时注入项目根目录
import time  # 训练 ETA 和指标时间戳
import warnings  # 控制训练日志中的 warning
from contextlib import nullcontext  # CPU/float32 下替代 autocast

__package__ = "trainer.sft"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# 副作用导入：注册项目依赖的数据集能力，与其他阶段入口保持一致。
import datasets  # noqa: F401
import torch
import torch.distributed as dist  # rank、all_reduce、barrier 和进程组清理
from torch import optim  # AdamW
from torch.nn.parallel import DistributedDataParallel  # 多卡梯度同步包装
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataset.lm_dataset import SFTDataset  # 产出 input_ids 和只监督 assistant 的 labels
from model.model_lora import apply_lora, save_lora  # 注入和导出 LoRA adapter
from model.model_mica import MicaConfig  # 模型结构配置
from trainer.common.utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")  # 与预训练入口一致，避免 AMP 弃用提示淹没训练日志


def rank():
    """返回当前 DDP rank；单进程固定为 0。"""

    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    """返回 DDP 进程数；单进程固定为 1。"""

    return dist.get_world_size() if dist.is_initialized() else 1


def unwrap_model():
    """去掉 DDP / torch.compile 包装，返回真正的 Mica 模型。"""

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def append_metric(payload, swanlab=None):
    """由主进程记录指标；JSONL 保留完整字段，SwanLab 只接收有限数值。"""

    if not is_main_process():
        return
    record = {
        **payload,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.metrics_path:
        parent = os.path.dirname(args.metrics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if swanlab:
        swanlab.log({
            key: value
            for key, value in record.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        })


def export_path(weight_name):
    """生成完整模型或 LoRA adapter 的导出路径。"""

    moe_suffix = "_moe" if lm_config.use_moe else ""
    return os.path.join(
        args.save_dir,
        f"{weight_name}_{lm_config.hidden_size}{moe_suffix}.pth",
    )


def save_export_weight(path):
    """原子导出推理权重；全参数保存 FP16 state_dict，LoRA 只保存 adapter。

    先写临时文件再 replace，避免中断后留下看似完整的半成品。
    """

    if not is_main_process():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    raw_model = unwrap_model()
    if lora_enabled:
        save_lora(raw_model, temporary_path)
    else:
        state_dict = {
            key: value.half().cpu()
            for key, value in raw_model.state_dict().items()
        }
        torch.save(state_dict, temporary_path)
        del state_dict
    os.replace(temporary_path, path)


def save_training_checkpoint(epoch, step, swanlab):
    """同时保存部署权重和续训状态；调用方负责 DDP 同步。

    save_dir 面向推理；resume_dir 还保存 optimizer、scaler 和训练位置。
    """

    if not is_main_process():
        return
    model.eval()
    save_export_weight(export_path(args.save_weight))
    # resume 中保留完整模型状态，即使 LoRA 模式也能恢复 optimizer 和训练位置。
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=step,
        wandb=swanlab,
        save_dir=args.resume_dir,
        best_validation_loss=best_validation_loss,
        lora_rank=args.lora_rank,
        split_sha256=split_sha256,
    )
    model.train()


@torch.no_grad()
def validate(loader):
    """精确计算全局 token 加权 validation loss/perplexity。

    各 rank 处理不重复的 rank-stride 子集，并在末尾汇总 NLL、token 和样本数。
    使用解包模型 forward，允许不同 rank 拥有不同数量的验证 batch。
    """

    was_training = model.training
    model.eval()
    raw_model = unwrap_model()  # 不走 DDP forward，允许各 rank 拥有不同数量的验证 batch
    totals = torch.zeros(3, dtype=torch.float64, device=args.device)  # NLL、token、row

    for input_ids, labels in loader:
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        valid_tokens = labels[..., 1:].ne(-100).sum()
        if valid_tokens == 0:
            continue
        with autocast_ctx:
            result = raw_model(input_ids, labels=labels)
        if not torch.isfinite(result.loss):
            raise FloatingPointError("validation loss is not finite")
        totals[0] += result.loss.detach().double() * valid_tokens
        totals[1] += valid_tokens
        totals[2] += input_ids.shape[0]

    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_nll, total_tokens, total_rows = totals.tolist()
    if total_tokens == 0:
        raise ValueError("validation dataset has no assistant targets")

    if was_training:
        model.train()
    validation_loss = total_nll / total_tokens
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 80.0)),
        "validation_tokens": int(total_tokens),
        "validation_rows": int(total_rows),
    }


def build_split(total_size, validation_size, seed):
    """确定性切分训练/验证索引，并返回可审计摘要。

    独立 Generator 不消耗训练随机状态；摘要用于检查是否仍是同一验证子集。
    """

    if validation_size <= 0 or validation_size >= total_size:
        raise ValueError(f"validation_size must be in [1, {total_size - 1}]")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    validation_indices = sorted(permutation[:validation_size])
    train_indices = permutation[validation_size:]
    digest = hashlib.sha256(
        ",".join(map(str, validation_indices)).encode()
    ).hexdigest()
    return train_indices, validation_indices, digest


def write_json(path, payload):
    """可选写入 manifest；留空路径时不创建文件。"""

    if not path or not is_main_process():
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None):
    """执行一个 SFT epoch，整体步骤与预训练的 train_epoch 一致。

    loader 产出两个 [B, T] 张量。input_ids 是完整对话，labels 中非 assistant
    token 已由 SFTDataset 标为 -100，因此模型仍执行 next-token prediction，
    但梯度只来自 assistant 回复。iters 包含续训时跳过的 batch；start_step 是
    当前 epoch 已完成的 batch 数。

    每步依次执行：设置 cosine 学习率 -> AMP 前向 -> 按真实累积窗口缩放 loss ->
    反向 -> 在累积边界裁剪并更新 -> 记录指标 -> 在安全边界保存 checkpoint。
    """

    started = time.time()  # 用已运行 step 的平均耗时估算本 epoch 剩余时间
    last_step = start_step  # 循环结束后作为 checkpoint 中的恢复位置
    optimizer.zero_grad(set_to_none=True)  # 新 epoch 或续训入口都从空梯度开始

    # 续训时 step 从 start_step + 1 接着编号，学习率和日志进度不会回退。
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device, non_blocking=True)  # [B, T] 完整对话 token
        labels = labels.to(args.device, non_blocking=True)  # 非 assistant 与 padding 位置均为 -100
        last_step = step

        # 学习率沿用阶段脚本原有的整轮 cosine 曲线。
        learning_rate = get_lr(
            epoch * iters + step,
            args.epochs * iters,
            args.learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate  # 不额外创建 lr_scheduler，与预训练一致

        # epoch 尾部不足 accumulation_steps 时按真实窗口归一化，避免梯度被额外缩小。
        window_start = (
            ((step - 1) // args.accumulation_steps)
            * args.accumulation_steps
            + 1
        )
        actual_window_size = min(
            args.accumulation_steps,
            iters - window_start + 1,
        )
        with autocast_ctx:  # CUDA bf16/fp16 使用 autocast；CPU/float32 是空上下文
            result = model(input_ids, labels=labels)  # 内部完成 shift 后的交叉熵
            auxiliary_loss = (
                result.aux_loss
                if result.aux_loss is not None
                else result.loss.new_zeros(())
            )
            combined_loss = result.loss + auxiliary_loss  # Dense 模型的 aux_loss 为 0
            objective = combined_loss / actual_window_size  # 累积后等价于窗口平均梯度
        if not torch.isfinite(combined_loss):
            raise FloatingPointError(
                f"non-finite SFT loss at epoch={epoch + 1}, step={step}"
            )
        # scaler 只在 float16 真正缩放；bfloat16、float32 和 CPU 下是空操作。
        scaler.scale(objective).backward()

        update_boundary = (
            step % args.accumulation_steps == 0 or step == iters
        )
        if update_boundary:
            # 先反缩放再裁剪，否则 float16 下的 grad_norm 没有实际意义。
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient at epoch={epoch + 1}, step={step}"
                )
            scaler.step(optimizer)  # float16 溢出时 GradScaler 会跳过本次 AdamW 更新
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        should_log = (
            args.log_interval > 0
            and (step % args.log_interval == 0 or step == iters)
        )
        if should_log:
            elapsed = time.time() - started
            eta_minutes = (
                elapsed / max(step - start_step, 1) * (iters - step) / 60
            )
            metric = {
                "event": "train",
                "epoch": epoch + 1,
                "step": step,
                "train_loss": float(result.loss.item()),
                "aux_loss": float(auxiliary_loss.item()),
                "learning_rate": learning_rate,
                "grad_norm": (
                    float(grad_norm.item()) if update_boundary else None
                ),
                "eta_minutes": eta_minutes,
            }
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {metric['train_loss']:.4f}, "
                f"aux_loss: {metric['aux_loss']:.4f}, "
                f"lr: {learning_rate:.8f}, "
                f"eta: {eta_minutes:.1f}min"
            )
            append_metric(metric, swanlab)  # JSONL 始终可选，SwanLab 只由主进程上报

        # 只在 optimizer 边界保存，resume 时不会丢失尚未更新的累积梯度。
        should_save = (
            args.save_interval > 0
            and step % args.save_interval == 0
            and step < iters
            and update_boundary
        )
        if should_save:
            save_training_checkpoint(epoch, step, swanlab)
            if dist.is_initialized():
                dist.barrier()  # 等主进程写完 checkpoint，再继续下一批 DDP

        # 主动断开大张量引用，降低大 batch 下进入下一步前的显存峰值。
        del input_ids, labels, result, combined_loss, objective

    return last_step  # 主循环用它写入 epoch 末 resume checkpoint


if __name__ == "__main__":
    # 参数按输出、训练、模型、数据、初始化和观测的顺序排列。
    parser = argparse.ArgumentParser(
        description="Mica SFT with optional validation and LoRA"
    )
    # save_dir 给推理使用；resume_dir 给继续训练使用。
    parser.add_argument("--save_dir", default="../out", help="导出权重目录")
    parser.add_argument("--resume_dir", default="../checkpoints", help="续训状态目录")
    parser.add_argument("--save_weight", default="sft", help="末尾权重前缀")
    parser.add_argument("--best_weight", default="sft_best_val", help="验证集最优权重前缀")

    # batch_size 是单进程 batch；DDP 全局 batch 还要乘 world_size。
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 模型结构必须和 from_weight 指向的初始权重兼容。
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--use_moe", type=int, choices=(0, 1), default=0)

    # 独立验证文件和 validation_size 二选一；都不传则不验证。
    parser.add_argument("--data_path", default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--validation_data_path", default="", help="可选独立验证文件")
    parser.add_argument("--validation_size", type=int, default=0, help="从训练文件切出的验证样本数；0 表示不切分")
    parser.add_argument("--validation_batch_size", type=int, default=64)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest", default="")
    parser.add_argument("--parameter_manifest", default="")
    parser.add_argument("--train_augment", type=int, choices=(0, 1), default=1)

    # from_weight 决定初始化权重；from_resume 恢复完整训练状态。
    parser.add_argument("--from_weight", default="pretrain")
    parser.add_argument("--from_resume", type=int, choices=(0, 1), default=0)
    parser.add_argument("--lora_rank", type=int, default=0, help="0=全参数 SFT；大于 0=LoRA rank")
    parser.add_argument("--metrics_path", default="", help="可选 JSONL 指标文件")
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", default="Mica-SFT")
    parser.add_argument("--swanlab_run_name", "--wandb_run_name", dest="swanlab_run_name", default="")
    parser.add_argument("--use_compile", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    for name in ("epochs", "batch_size", "accumulation_steps", "validation_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.lora_rank < 0:
        raise ValueError("lora_rank must be non-negative")
    for name in ("log_interval", "save_interval", "validation_size"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.validation_data_path and args.validation_size:
        raise ValueError(
            "use either --validation_data_path or --validation_size, not both"
        )

    # ========== 1. 初始化环境和随机种子 ==========
    # torchrun 下绑定 LOCAL_RANK；每个 rank 使用独立训练随机流。
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.split_seed + rank())

    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    # 与预训练相同：先确定结构并探测 resume；SFT 额外允许自定义 resume_dir。
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)
    # 先读取 checkpoint 元数据；LoRA 注入后才加载模型 state。
    lm_config = MicaConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    checkpoint = (
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            save_dir=args.resume_dir,
        )
        if args.from_resume
        else None
    )
    if args.from_resume and checkpoint is None:
        raise FileNotFoundError(
            f"no resume checkpoint for {args.save_weight} in {args.resume_dir}"
        )

    # ========== 3. 设置混合精度 ==========
    # 与预训练相同：根据 device/dtype 复用一个 autocast 上下文。
    # CPU/float32 关闭 autocast；只有 float16 需要 GradScaler。
    device_type = "cuda" if "cuda" in args.device else "cpu"
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
    scaler = torch.cuda.amp.GradScaler(enabled=args.dtype == "float16")

    # ========== 4. 准备实验记录 ==========
    # 预训练可立即初始化；SFT 还要记录 LoRA 参数量和切分规模，因此在第 7 段启动。
    swanlab = None

    # ========== 5. 定义模型、数据和优化器 ==========
    # 与预训练相同的主干是 init_model -> Dataset -> Sampler -> AdamW。
    # SFT 的扩展只包括 LoRA 冻结、assistant labels、验证切分和 manifest。
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device,
    )
    lora_enabled = args.lora_rank > 0
    if lora_enabled:
        # 先注入 adapter，再冻结基础模型，使 optimizer 只持有 LoRA 参数。
        apply_lora(model, rank=args.lora_rank)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora." in name
    if lora_enabled and args.use_compile:
        raise ValueError("LoRA patched forwards are incompatible with torch.compile")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    parameter_summary = {
        "method": "lora" if lora_enabled else "full",
        "lora_rank": args.lora_rank,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameter_count,
        "trainable_ratio": trainable_parameter_count / total_parameters,
    }
    write_json(args.parameter_manifest, parameter_summary)
    Logger(
        f"method={parameter_summary['method']} "
        f"trainable={trainable_parameter_count}/{total_parameters} "
        f"ratio={parameter_summary['trainable_ratio']:.6f}"
    )

    # 训练可增强；验证始终关闭增强，保证多次评估可比。
    train_base = SFTDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        augment=bool(args.train_augment),
    )
    validation_dataset = None
    split_sha256 = ""
    split_payload = None

    if args.validation_data_path:
        # 独立验证文件用内容哈希标识，便于发现文件被替换。
        train_dataset = train_base
        validation_dataset = SFTDataset(
            args.validation_data_path,
            tokenizer,
            max_length=args.max_seq_len,
            augment=False,
        )
        with open(args.validation_data_path, "rb") as handle:
            split_sha256 = hashlib.sha256(handle.read()).hexdigest()
        split_payload = {
            "algorithm": "explicit_validation_file",
            "train_size": len(train_dataset),
            "validation_size": len(validation_dataset),
            "validation_sha256": split_sha256,
        }
    elif args.validation_size:
        # 同一原始文件按固定索引切分，验证侧另建无增强 Dataset。
        validation_base = SFTDataset(
            args.data_path,
            tokenizer,
            max_length=args.max_seq_len,
            augment=False,
        )
        train_indices, validation_indices, split_sha256 = build_split(
            len(train_base),
            args.validation_size,
            args.split_seed,
        )
        train_dataset = Subset(train_base, train_indices)
        validation_dataset = Subset(validation_base, validation_indices)
        split_payload = {
            "algorithm": "torch.randperm",
            "seed": args.split_seed,
            "dataset_size": len(train_base),
            "train_size": len(train_indices),
            "validation_size": len(validation_indices),
            "validation_indices_sha256": split_sha256,
            "validation_indices": validation_indices,
        }
    else:
        train_dataset = train_base
    write_json(args.split_manifest, split_payload)

    train_sampler = (
        DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=args.split_seed,
        )
        if dist.is_initialized()
        else None
    )
    # 不使用会补齐尾部样本的 DistributedSampler，避免验证重复计数。
    validation_loader = None
    if validation_dataset is not None:
        # rank-stride 子集不会像 DistributedSampler 那样补重复验证样本。
        validation_indices = range(rank(), len(validation_dataset), world_size())
        validation_subset = Subset(validation_dataset, validation_indices)
        validation_loader = DataLoader(
            validation_subset,
            batch_size=args.validation_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    optimizer = optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )

    # ========== 6. 从 checkpoint 恢复状态 ==========
    # 顺序与预训练一致，且必须发生在 compile/DDP 包装之前。
    start_epoch, start_step = 0, 0
    best_validation_loss = float("inf")
    if checkpoint:
        # LoRA rank 改变参数形状，加载 state_dict 前先检查兼容性。
        if int(checkpoint.get("lora_rank", 0)) != args.lora_rank:
            raise ValueError(
                "resume lora_rank does not match current --lora_rank"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        start_step = int(checkpoint.get("step", 0))
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )

    # ========== 7. 初始化实验记录、compile 和 DDP ==========
    # compile/DDP 的包装顺序与预训练一致；SwanLab 延迟到这里仅为补齐 SFT 元数据。
    if args.use_swanlab and is_main_process():
        import swanlab

        run_id = checkpoint.get("wandb_id") if checkpoint else None
        run_name = args.swanlab_run_name or (
            f"Mica-SFT-{parameter_summary['method']}-"
            f"Epoch-{args.epochs}-Batch-{args.batch_size}"
        )
        swanlab.init(
            project=args.swanlab_project,
            name=run_name,
            id=run_id,
            resume="must" if run_id else None,
            config={
                **vars(args),
                **parameter_summary,
                "world_size": world_size(),
                "train_size": len(train_dataset),
                "validation_size_actual": (
                    len(validation_dataset)
                    if validation_dataset is not None
                    else 0
                ),
            },
        )
    if args.use_compile:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练，并在每个 epoch 后可选验证 ==========
    # epoch 主循环与预训练相同：set_epoch -> 随机索引 -> SkipBatchSampler ->
    # train_epoch -> validation -> checkpoint。baseline best 是 SFT 的附加保护。
    # 新训练先评估初始权重并纳入 best；resume 沿用 checkpoint 中的门槛。
    if validation_loader is not None and start_epoch == 0 and start_step == 0:
        baseline = validate(validation_loader)
        append_metric(
            {"event": "validation", "phase": "baseline", **baseline},
            swanlab,
        )
        Logger(
            f"Baseline validation loss={baseline['validation_loss']:.6f}, "
            f"ppl={baseline['validation_perplexity']:.4f}"
        )
        # baseline 也参与 best 比较；若 SFT 让验证集变差，就保留训练前权重。
        best_validation_loss = baseline["validation_loss"]
        baseline_best_path = export_path(args.best_weight)
        save_export_weight(baseline_best_path)
        Logger(f"Baseline best weight saved to {baseline_best_path}")
        if dist.is_initialized():
            dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        setup_seed(args.split_seed + epoch)
        indices = torch.randperm(len(train_dataset)).tolist()
        # 只在恢复后的首个 epoch 跳过已完成 batch，之后自动回到 skip=0。
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or indices,
            args.batch_size,
            skip,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if skip:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: skip {skip} batches, "
                f"resume at {skip + 1}"
            )
            epoch_step = train_epoch(
                epoch,
                loader,
                len(loader) + skip,
                skip,
                swanlab,
            )
        else:
            epoch_step = train_epoch(
                epoch,
                loader,
                len(loader),
                0,
                swanlab,
            )

        if validation_loader is not None:
            validation = validate(validation_loader)
            append_metric(
                {
                    "event": "validation",
                    "phase": "epoch",
                    "epoch": epoch + 1,
                    **validation,
                },
                swanlab,
            )
            Logger(
                f"Validation Epoch:[{epoch + 1}/{args.epochs}], "
                f"loss: {validation['validation_loss']:.6f}, "
                f"ppl: {validation['validation_perplexity']:.4f}, "
                f"rows: {validation['validation_rows']}, "
                f"tokens: {validation['validation_tokens']}"
            )
            if validation["validation_loss"] < best_validation_loss:
                best_validation_loss = validation["validation_loss"]
                best_path = export_path(args.best_weight)
                save_export_weight(best_path)
                Logger(f"Validation improved; best weight saved to {best_path}")

        # 验证后再保存，使 best 门槛、模型和 optimizer 属于同一快照。
        save_training_checkpoint(epoch, epoch_step, swanlab)
        if dist.is_initialized():
            dist.barrier()

    append_metric(
        {
            "event": "completed",
            "best_validation_loss": (
                best_validation_loss
                if validation_loader is not None
                else None
            ),
            **parameter_summary,
        },
        swanlab,
    )
    if swanlab:
        swanlab.finish()

    # ========== 9. 清理分布式进程组 ==========
    # 与预训练相同：所有 rank 对齐后释放 NCCL/Gloo 资源。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
