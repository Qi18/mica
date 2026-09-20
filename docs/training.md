# 训练指南

## 两类入口

| 项目 | 统一 CLI | 分阶段脚本 |
|---|---|---|
| 入口 | mica train | trainer/阶段/train_*.py |
| 实现 | common/engine.py | 各阶段循环，共用 common/utils.py |
| 输入 | token JSONL + JSON recipe | 原始 JSONL + 命令参数 |
| 保存 | config.json/model.pt/training.pt | .pth 与脚本各自的恢复格式 |
| 验证 | CPU、DDP、精确恢复测试 | 主要入口帮助检查，非全量训练验收 |

**下文统一引擎的保存、恢复保证不适用于所有阶段脚本。**
最小可运行例子见 [README](../README.md)。

## 数据格式

统一 CLI 数据转换接受以下两种原始 JSONL：
```json
{"text":"预训练文本"}
{"prompt":"你好","response":"你好！"}
```

mica data 输出 input_ids/labels；SFT 前缀 labels 为 -100，只对回答计 loss。
截断后没有有效 next-token target 会报错，输出文件必须不存在。原始数据经 tokenizer 编码后，模型词表必须与其匹配。
直接提供 token JSONL 时，输入和 labels 长度须一致，token ID 不能超出词表。

阶段脚本的输入不同：
- 预训练/MoE：text。
- SFT/LoRA/蒸馏：conversations 消息列表。
- DPO：chosen/rejected，各为完整消息列表。
- GRPO/PPO：按 RLAIFDataset 与奖励逻辑提供 conversations 等字段。
- Agent：conversations、gt，以及匹配的工具协议。

不要把 CLI token JSONL 直接传给这些阶段 Dataset。具体字段类型见 [lm_dataset.py](../dataset/lm_dataset.py)。

## 分阶段脚本

| 目录 | 入口与说明 |
|---|---|
| tokenizer/ | train_tokenizer.py；由文件顶部常量控制数据与输出，不提供通用 --help |
| pretrain/ | train_pretrain.py；验证集可选，启用后保存 best 权重 |
| sft/ | train_full_sft.py、train_sft_with_validation.py、probe_sft_batch.py |
| lora/ | train_lora.py；adapter 推理需要匹配的 base |
| dpo/ | train_dpo.py；偏好数据、策略与参考模型 |
| grpo/ | train_grpo.py；分组采样与奖励 |
| ppo/ | train_ppo.py；策略、价值估计与奖励 |
| agent/ | train_agent.py；工具轨迹，loss_type 可选 grpo/cispo |
| moe/ | train_moe.py；use_moe 选择 Dense/MoE，非无损权重转换 |
| distill/ | train_distillation.py；Mica 教师/学生，不是任意模型适配器 |

常见路线是预训练 → SFT → 可选偏好/RL；LoRA 是微调方式，MoE 是结构选择，不是必经步骤。

预训练只有一个入口。不传 --validation_path 时执行普通预训练；传入验证集后会在每个
epoch 结束计算精确的 token 加权 loss/perplexity，并保存 --best_weight：

~~~bash
python pretrain/train_pretrain.py --data_path ../dataset/train.jsonl
python pretrain/train_pretrain.py \
  --data_path ../dataset/train.jsonl \
  --validation_path ../dataset/validation.jsonl
~~~

从仓库根目录切换到 trainer/，**不要再 cd 到阶段子目录**：
```bash
cd trainer
python pretrain/train_pretrain.py --help
python sft/train_full_sft.py --help
python lora/train_lora.py --help
python dpo/train_dpo.py --help
```

默认 ../dataset、../tokenizer、../out、../checkpoints 相对 trainer/ 解析。
基础预训练从零开始使用 --from_weight none。
多个阶段的 init_model 默认从 ../out 加载；--from_weight 是前缀，文件名还包含 hidden_size 和 MoE 后缀。
**--save_dir 不一定控制输入权重或恢复目录**，运行前核对脚本调用。pretrain 可选传入 --validation_path，并在每个 epoch 后记录验证指标、更新 best 权重；MoE 入口需显式提供训练、验证、指标和恢复路径。
蒸馏须准备匹配的教师与学生配置/权重及兼容词表；RL 还需验证奖励与可选外部 rollout 服务。

CLI 目录不能直接作为阶段脚本的 .pth 输入。反向导入时 mica import-legacy 只接受与配置匹配的原始 state_dict，而非包含 optimizer 的整个恢复对象。

## 统一引擎配置与恢复

### 单进程与多进程

先完成 [README 快速开始](../README.md)的示例创建，在同一终端使用 MICA_RUN。输出目录必须不存在或为空，每次重跑换一个新目录。

```bash
mica train --recipe "$MICA_RUN/recipe.json" --output "$MICA_RUN/new-run"
torchrun --standalone --nproc_per_node=2 -m mica train \
  --recipe "$MICA_RUN/recipe.json" --output "$MICA_RUN/new-ddp"
```

CUDA 配置将 device 设为 cuda；torchrun 的 LOCAL_RANK 决定每进程使用哪张可见卡。
GPU 使用 NCCL，CPU 使用 Gloo。当前自动验收覆盖双进程 CPU；
CUDA/DDP 代码尚未做真实多卡验收。

| recipe 字段 | 含义 |
|---|---|
| model | 必填：MicaConfig 字段对象；完整示例见快速开始 |
| learning_rate | AdamW 学习率，默认 0.001 |
| cpu_threads | 每进程 torch CPU 线程数，默认 2 |
| initialize_from | 可选：相对 recipe 的模型目录，只初始化模型，不恢复 optimizer |
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

### 损失口径

next-token CE 按当前 update 的全局有效 label token 数归一化。
SFT prompt/padding 的 -100 不计入分母。DDP 的梯度平均通过乘 world_size 补偿；
变长样本下的梯度累积和单个大 batch 在浮点容差内一致。
MoE aux loss 单独按 microbatch/rank 平均，不能将这部分解释为 token 加权 CE。

当前为 FP32 AdamW 和梯度裁剪；混合精度、学习率调度和分布式 optimizer 不在当前入口中。

### Checkpoint

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

### 中断与恢复

--resume 接受旧运行目录，--output 必须是新的或空目录，不能原地覆盖已有运行。

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

### 实际续训示例

完成快速开始的单进程运行后，在同一终端执行；只增加 steps：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["MICA_RUN"])
recipe = json.loads((root / "recipe.json").read_text())
recipe["steps"] = 6
(root / "continued.json").write_text(json.dumps(recipe, indent=2))
PY
mica train --recipe "$MICA_RUN/continued.json" \
  --resume "$MICA_RUN/run" --output "$MICA_RUN/resumed"
```

若改变数据、world_size 或其他 recipe 字段，则不要用精确恢复。
可创建新 recipe，设置 `"initialize_from": "run"`（假设 recipe 与 run 位于同一目录），并使用新的输出目录。
这只迁移模型权重，优化器与训练步数重新开始；模型结构必须匹配。
