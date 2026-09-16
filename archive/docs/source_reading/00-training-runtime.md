# MiniMind 训练运行时结构：Stage 0 阅读笔记

## 入口的共同结构

train_pretrain.py、train_full_sft.py、DPO、LoRA、蒸馏和 RL 入口大体遵循同一顺序：

1. argparse 读取训练参数；
2. init_distributed_mode() 根据 RANK 判断单进程或 NCCL DDP；
3. 构建 MiniMindConfig；
4. 可选读取 resume checkpoint；
5. 配置 BF16/FP16 autocast 与 GradScaler；
6. 可选初始化 SwanLab；
7. 创建模型、Dataset、Sampler、Optimizer；
8. 恢复模型和优化器状态；
9. 可选 torch.compile，再封装 DDP；
10. 进入 epoch/step 训练并定期保存；
11. barrier 后销毁进程组。

## DDP 与设备

trainer_utils.init_distributed_mode() 只在环境变量 RANK 存在时调用 dist.init_process_group(backend="nccl")，再用 LOCAL_RANK 绑定 GPU。训练入口本身不负责拉起多进程，因此正式 8 卡命令必须由 torchrun --nproc_per_node=8 启动。

种子首先设置为 42 + rank，每个 epoch 又调用 setup_seed(42 + epoch)。DDP 使用 DistributedSampler.set_epoch(epoch)；非 DDP 才使用本地 randperm 索引。

## dtype

参数 --dtype 在训练入口中映射为：

- bfloat16 → BF16 autocast，GradScaler disabled；
- 其他值 → FP16 autocast，GradScaler enabled。

当前 L20 原生支持 BF16，因此主线固定 --dtype bfloat16。

## Checkpoint 与恢复

lm_checkpoint() 写两类文件：

- weight_hidden[_moe].pth：模型半精度 state dict；
- weight_hidden[_moe]_resume.pth：模型、optimizer、epoch、step、world size、SwanLab run id，以及 scaler/scheduler/critic 等额外状态。

正式权重与 resume 文件先写 .tmp 再由 os.replace() 原子替换。训练脚本还会向 save_dir 写一份模型权重。resume 目录在多个入口中写死为相对路径 ../checkpoints，所以启动命令必须明确工作目录，不能在任意目录执行。

当 world size 变化时，源码会按比例转换 step；这不等于自动证明全局 batch、采样顺序和优化轨迹完全等价，恢复报告仍需记录 world size 和 batch 配置。

## SwanLab

参数名仍是 --use_wandb / --wandb_project，实际代码执行 import swanlab as wandb。resume checkpoint 会保存 SwanLab run id，并用 resume="must" 尝试续接。因此：

- 正式 run 启动前必须通过 swanlab verify；
- run id 必须写入实验 run.json；
- API key 只能通过 masked interactive prompt 登录；
- 不能把 API key 写入 shell 参数、环境清单、日志或 Git。

## SIGTERM 边界

当前训练入口没有 SIGTERM handler，也不会在收到终止信号时立即调用 lm_checkpoint()。Lab 启动器负责：

- 训练前检查 GPU 空闲；
- 使用单实例锁避免两个任务竞争 8 卡；
- 把 INT/TERM/HUP 转发给训练子进程组；
- 等待子进程退出，避免遗留 worker。

它不能承诺终止时产生新 checkpoint。安全恢复依赖最近一次 save_interval 原子 checkpoint，因此正式实验必须根据可接受的最大重算窗口设置保存间隔。
