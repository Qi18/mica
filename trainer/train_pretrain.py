"""MiniMind 预训练入口：单卡与 DDP 通用的 next-token 预测训练脚本。

主体流程写在 __main__ 的 9 个编号段里，顺序为：初始化分布式与随机种子 -> 构造 MicaConfig 并探测
续训 checkpoint -> 配置混合精度 -> 初始化 wandb（实际 import 的是 swanlab，且只在主进程）-> 建模型、
数据集与优化器 -> 从 checkpoint 恢复状态 -> torch.compile 与 DDP 包装 -> 逐 epoch 调用 train_epoch ->
销毁进程组。单卡直接 python 运行，多卡用 torchrun 拉起，靠环境变量里的 RANK 自动区分。

几个贯穿全局的约定：
- args、lm_config、model、optimizer、scaler、autocast_ctx 都定义在 __main__ 里，train_epoch 通过全局
  变量访问它们，因此本文件只能作为脚本运行，import 后单独调用 train_epoch 会 NameError。
- 学习率由 trainer_utils.get_lr 按 cos 曲线从 learning_rate 衰减到它的 10%，没有 warmup，每个 step 都更新。
- checkpoint 落两份：args.save_dir 下的 {save_weight}_{hidden_size}[_moe].pth 是纯半精度权重，供推理和
  下游 SFT 加载；../checkpoints 下的 *_resume.pth 额外含优化器、scaler、epoch、step、wandb_id，供
  --from_resume 1 续训。
- 续训用 SkipBatchSampler 跳过本 epoch 已消费的 batch，step 编号接着往下排；GPU 数量变化时
  lm_checkpoint 会按 world_size 折算 step。
"""
import os  # 路径拼接与建目录
import sys  # 用于往 sys.path 注入上级目录

__package__ = "trainer"  # 伪装成包内模块，使下面的 from trainer.xxx 绝对导入在直接运行脚本时也成立
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 把项目根目录加入搜索路径，model / dataset / trainer 三个子包才能 import

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse  # 命令行参数
import time  # 统计 step 耗时
import warnings  # 配合下面的 filterwarnings
import torch
import torch.distributed as dist  # DDP 的进程组、rank、barrier
from contextlib import nullcontext  # CPU 上用空上下文替代 autocast
from torch import optim, nn  # 只用到 optim.AdamW，nn 是上游遗留的未用导入
from torch.nn.parallel import DistributedDataParallel  # 多卡数据并行包装
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载与按 rank 分片
from model.modeling_mica import MicaConfig  # 模型超参容器
from dataset.lm_dataset import PretrainDataset  # 预训练数据集，产出 (input_ids, labels)
# 训练工具：cos 学习率、主进程日志、主进程判断、checkpoint 读写、DDP 初始化、随机种子、建模、跳批采样器
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')  # 屏蔽全部 warning（含 torch.cuda.amp 的弃用提示），让训练日志干净


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """跑完一个 epoch：前向、反向、按累积步更新参数，并顺带打日志与定期存档。

    loader 每次产出 (input_ids, labels) 两个 [B, T] 张量；iters 是本 epoch 的总 step 数（续训时为
    len(loader) 加上已跳过的 step），仅用于日志与 cos 学习率的进度换算；start_step 是续训时已完成的
    step 数，本轮 step 从 start_step + 1 接着编号；wandb 仅主进程为非 None。

    - loss = res.loss + res.aux_loss，Dense 模型的 aux_loss 恒为 0；先除以 accumulation_steps 再反向，
      日志里乘回来还原成单 step 的真实 loss，logits_loss 则是扣掉 aux 的部分。
    - scaler 只在 float16 下真正启用，bfloat16 与 CPU 时是空操作，所以 unscale_ / step 可以无条件调用。
    - 每 accumulation_steps 步才裁剪梯度并 optimizer.step()；epoch 结束时若还剩不满一轮的累积梯度，
      在循环外补一次更新，避免这段梯度被直接丢掉。
    - 存档只在主进程执行，先切 eval 再切回 train；取 state_dict 前会剥掉 DDP 的 .module 与
      torch.compile 的 ._orig_mod 包装。
    - 日志里的 epoch_time 是按当前平均 step 耗时推算的本 epoch 剩余分钟数，不是已经消耗的时间。
    """
    start_time = time.time()  # 本 epoch 起点，用于推算剩余时间
    last_step = start_step  # 记住循环走到的最后一个 step，供循环外补更新时判断
    # 遍历本 epoch 的 batch；step 从 start_step + 1 起编号，续训时接着上次往下排
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)  # [B, T] 的 token id，尾部已被数据集 pad 到 max_seq_len
        labels = labels.to(args.device)  # 与 input_ids 同形，pad 位是 -100，交叉熵会忽略
        last_step = step  # 每步刷新
        # 全局进度 = epoch * iters + step，总步数 = epochs * iters；cos 从 learning_rate 衰减到它的 10%
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:  # 直接写入 lr，没有用 lr_scheduler
            param_group['lr'] = lr

        with autocast_ctx:  # bf16 / fp16 自动混合精度；CPU 下是 nullcontext
            res = model(input_ids, labels=labels)  # 前向；模型内部做移位交叉熵，返回 loss / aux_loss / logits
            loss = res.loss + res.aux_loss  # 语言建模损失加 MoE 负载均衡损失；Dense 时 aux_loss 是 0 标量
            loss = loss / args.accumulation_steps  # 除以累积步数，使累积后的梯度等价于大 batch 的平均梯度

        scaler.scale(loss).backward()  # fp16 下先放大 loss 再反向以防梯度下溢；bf16 / CPU 时 scale 是恒等操作

        if step % args.accumulation_steps == 0:  # 攒够累积步才真正更新参数
            scaler.unscale_(optimizer)  # 把梯度还原回真实尺度，之后的裁剪阈值才有意义
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 按全局范数裁剪到 grad_clip，抑制 loss 尖峰

            scaler.step(optimizer)  # 执行 AdamW 更新；fp16 下若检测到 inf / nan 会跳过这次更新
            scaler.update()  # 根据本步是否溢出调整放大系数

            optimizer.zero_grad(set_to_none=True)  # 清梯度，set_to_none 更省显存也更快

        if step % args.log_interval == 0 or step == iters:  # 到打印间隔或本 epoch 最后一步
            spend_time = time.time() - start_time  # 本 epoch 已耗时（秒）
            current_loss = loss.item() * args.accumulation_steps  # 乘回累积步数，还原成单 step 的真实 loss
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0  # MoE 的负载均衡项，Dense 恒为 0
            current_logits_loss = current_loss - current_aux_loss  # 扣掉 aux 的纯语言建模损失，只有这项能换算困惑度
            current_lr = optimizer.param_groups[-1]['lr']  # 读回当前学习率（只有一个 param_group）
            # 平均单 step 耗时 × 剩余 step ÷ 60，即本 epoch 预计还要多少分钟（注意是剩余量，不是已用时间）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            # 只有主进程会真正 print
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 上报到 swanlab；非主进程或未开启时 wandb 为 None，这里直接短路
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 到存档间隔或本 epoch 最后一步，且只允许主进程写盘，避免多卡互相覆盖
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 存档前切 eval，关掉 dropout
            moe_suffix = '_moe' if lm_config.use_moe else ''  # MoE 权重单独命名，避免和 Dense 权重互相覆盖
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'  # 形如 ../out/pretrain_768.pth
            # 剥掉 DDP 包装，否则 state_dict 的 key 会多一层 module. 前缀
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 再剥掉 torch.compile 的 _orig_mod 包装，同理是为了 key 干净
            state_dict = raw_model.state_dict()  # 此时 key 与原始模型一致
            # 转半精度再落盘，体积减半；这份只含权重，供推理和下游 SFT 加载
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另存一份续训档到 ../checkpoints，含优化器 / scaler / epoch / step / wandb_id；内部用 tmp + os.replace 原子写
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()  # 切回训练态
            del state_dict  # 立刻释放这份 CPU 副本

        # 主动断开引用，压低峰值显存（大 batch 下比较明显）
        del input_ids, labels, res, loss

    # epoch 结束时若还剩不满一轮的累积梯度，在循环外补一次更新，避免这部分梯度被直接丢掉
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":  # 以下所有状态都是全局量，train_epoch 直接读取它们
    # 下面每个参数的含义见 help 文本，这里只补充 help 没说清的部分
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")  # 这是初始值，实际每步被 cos 曲线覆盖
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")  # 只区分 bfloat16 与其他；非 bfloat16 一律按 float16 处理并启用 GradScaler
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")  # 等效 batch = batch_size × accumulation_steps × 卡数
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")  # 定长训练：不足补 pad，超出直接截断
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")  # 只加载权重、不恢复优化器，用于换阶段接续（如 SFT 接预训练）
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")  # 读 ../checkpoints 下的 *_resume.pth，连优化器和 step 一起恢复
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")  # 首个 step 有编译开销，存档时需额外剥 _orig_mod
    args = parser.parse_args()  # 解析命令行

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 无 RANK 环境变量（非 torchrun 启动）时返回 0 且不建进程组
    # DDP 下忽略 --device，每个进程绑定自己的那张卡
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 各 rank 用不同种子，避免多卡 dropout 完全同步；函数内还会把 cudnn 设成确定性模式
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)  # 权重输出目录
    # 只从命令行取 3 个结构参数，其余超参走 MicaConfig 的默认值
    lm_config = MicaConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 不传 model 即为读取模式：无续训档时返回 None；GPU 数变化时内部会按 world_size 折算 step
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 判断在 CUDA 还是 CPU 上跑
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16  # 只认这两种精度
    # 建一个上下文对象反复 with：torch 的 autocast 实例可重入，所以不必每步新建
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None  # 未开启或非主进程时保持 None，train_epoch 里靠它短路
    if args.use_wandb and is_main_process():  # 只主进程上报，避免多卡产生多个 run
        import swanlab as wandb  # 变量名叫 wandb，实际用的是 swanlab（接口兼容）
        # 续训时复用同一个 run id，曲线才能接在原来那条后面
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 有 id 就强制续接，否则新建 run
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"  # 把关键超参写进 run 名，便于在面板里区分
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)  # 初始化实验记录
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # from_weight='none' 表示从头初始化，否则从 ../out 加载同名权重（strict=False，允许结构微调）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 读 jsonl 的 text 字段，前后加 BOS / EOS，pad 到 max_seq_len，labels 的 pad 位置为 -100
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 多卡时按 rank 切分样本，单卡为 None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 只有 fp16 需要 loss 缩放；bf16 动态范围够用，此时 scaler 是不做事的空壳
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 未显式设 weight_decay，取 AdamW 默认的 0.01；这里的 lr 每步都会被 get_lr 覆盖
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0  # 默认从头开始
    if ckp_data:
        # 必须在 torch.compile / DDP 包装之前加载，否则 key 前缀对不上
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])  # 恢复 AdamW 的一阶二阶动量
        scaler.load_state_dict(ckp_data['scaler'])  # 恢复 fp16 的放大系数
        start_epoch = ckp_data['epoch']  # 从中断的那个 epoch 继续
        start_step = ckp_data.get('step', 0)  # 该 epoch 内已完成的 step 数
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)  # 图编译加速，换来首步的编译等待
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 反向时自动 all-reduce 梯度；注意这里没用 no_sync，梯度累积期间每个 micro step 都会同步一次
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):  # 续训时从中断的 epoch 起算
        # 让每个 epoch 的分片顺序不同；单卡时 train_sampler 为 None，该表达式直接短路
        train_sampler and train_sampler.set_epoch(epoch)
        # 单卡路径自己造乱序索引（多卡时用不上）；种子固定，保证续训时的样本顺序可复现
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只有续训命中的那个 epoch 需要跳批，后面的 epoch 都从头跑
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 多卡传 sampler、单卡传 indices，内部按 batch_size 组批并丢掉前 skip 批
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 用了 batch_sampler 就不能再传 batch_size / shuffle / sampler，三者互斥
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # iters 把跳过的批数补回来，使日志分母和 cos 学习率进度与不中断时一致
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)  # 正常路径
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()  # 等所有 rank 跑完，避免主进程先退导致其他 rank 卡在通信上
        dist.destroy_process_group()  # 释放 NCCL 进程组资源