# Mica 训练与恢复

## 单进程与多进程

```bash
mica train --recipe recipes/smoke/cpu.json --output outputs/smoke
torchrun --standalone --nproc_per_node=2 -m mica_llm train \
  --recipe recipes/smoke/cpu-ddp.json --output outputs/ddp
```

CUDA 配置将 device 设为 cuda；torchrun 的 LOCAL_RANK 决定每进程使用哪张可见卡。
GPU 使用 NCCL，CPU 使用 Gloo。当前自动验收覆盖双进程 CPU；
CUDA/DDP 代码尚未做真实多卡验收。

| recipe 字段 | 含义 |
|---|---|
| batch_size | 每个进程、每个 microbatch 的行数 |
| gradient_accumulation_steps | 一个 optimizer update 累积多少个 microbatch，默认 1 |
| steps | 总 optimizer updates |
| save_every | 每多少个 updates 保存一次，默认 100 |
| seed | 初始化与每 rank 随机状态的种子 |
| device | cpu 或 cuda |
| data | 相对 recipe 路径的 token JSONL |

全局 batch = batch_size × gradient_accumulation_steps × world_size。
所有 rank 按确定顺序读取全局数据，文件尾部循环回头；当前不做 shuffle。
训练开始逐行校验数据并计算 SHA256，每行仅保留一个 8 字节偏移量。
token 内容按需解析；梯度累积会保留当前 update 的本 rank 数据。
因此内存随行数的偏移索引和当前 batch 增长，不再保留所有 token。
训练期间数据文件必须保持不变。每个 rank 都会做启动扫描，尚未实现共享磁盘索引缓存。

## 损失口径

next-token CE 按当前 update 的全局有效 label token 数归一化。
SFT prompt/padding 的 -100 不计入分母。DDP 的梯度平均通过乘 world_size 补偿；
变长样本下的梯度累积和单个大 batch 在浮点容差内一致。
MoE aux loss 单独按 microbatch/rank 平均，不能将这部分解释为 token 加权 CE。

当前为 FP32 AdamW 和梯度裁剪；混合精度、学习率调度和分布式 optimizer 不在当前入口中。

## Checkpoint

```text
outputs/run/
  latest.json
  run.json
  metrics.jsonl
  checkpoints/
    step-00000100/
      config.json
      model.pt
      training.pt
      recipe.json
```

只有写完 config、模型、optimizer、随机状态和 recipe 后，临时目录才会被重命名
为 step 目录，随后原子替换 latest.json。残留的 .incomplete-* 不参与加载。
只由 rank 0 写模型，每个 rank 的 RNG 都保存在 training.pt 中。
模型加载、评测、生成和 --resume 均接受运行根目录或具体 step 目录。
保留所有定期 checkpoint，当前不自动清理；长训练需规划磁盘空间。

## 中断与恢复

SIGTERM/SIGINT 在当前 optimizer update 完成后触发 checkpoint。
DDP 在 update 边界同步停止请求，保存成功后 CLI 返回 130；
run.json 的 status 为 interrupted。SIGKILL 或机器故障只能恢复最后一次完整保存。

```bash
mica train --recipe /path/to/continued.json \
  --resume outputs/run --output outputs/resumed
```

精确恢复要求相同 world_size、数据 SHA256 和 recipe；只允许增加总 steps。
恢复模型、optimizer、step 及每 rank CPU/CUDA RNG。
改变卡数应作为新的训练运行，用 initialize_from 初始化权重。
v0.1 旧 optimizer checkpoint 没有格式 v2 的信息，使用 initialize_from 迁移；
旧模型目录的 config.json/model.pt 仍可加载。

CPU 验证覆盖：单进程连续/恢复逐 tensor 完全相同、双进程 DDP 连续/恢复
逐 tensor 完全相同、SIGTERM 保存后恢复逐 tensor 完全相同。
跨 batch 划分与 DDP/单进程比较使用浮点容差，不声称按位一致。
