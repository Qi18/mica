<div align="center">

# 💎 Mica

### 从小模型出发，构建自己的语言模型

**模型结构 · 数据处理 · 预训练 · 指令微调 · 评测 · 推理**

[![Tests](https://github.com/Qi18/mica/actions/workflows/repository-check.yml/badge.svg?branch=main)](https://github.com/Qi18/mica/actions/workflows/repository-check.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)](pyproject.toml)
[![Version](https://img.shields.io/badge/Mica-v0.2.0-6366f1)](docs/mica-roadmap.md)
[![GitHub stars](https://img.shields.io/github/stars/Qi18/mica?style=social)](https://github.com/Qi18/mica/stargazers)

[快速开始](#-快速开始) · [模型结构](#-模型结构) · [数据与训练](#-数据与训练) · [推理与服务](#-推理与服务) · [效果与验证](#-效果与验证)

</div>

---

# 📌 项目介绍

Mica 是一个面向个人开发者的语言模型项目。从一个可以读懂、可以训练的小模型出发，把模型结构、数据处理、训练、评测和推理放在同一套工程中，让每次结构修改都能经过完整的训练与验证。

模型核心使用原生 PyTorch 实现。你可以直接查看 Attention、RoPE、RMSNorm、Dense/MoE FFN 和 KV Cache，也可以通过统一的 `mica` 命令准备数据、启动训练、恢复 checkpoint 和运行推理服务。

当前主线提供约 **64M 参数的 Dense 配置**，并支持 **约 198M 总参数、约 64M 激活参数的 MoE 配置**。小型 CPU 配置用于快速跑通流程，完整配置用于后续训练与架构演进。

初始架构派生自 [MiniMind](https://github.com/jingyaogong/minimind)。Mica 的工程入口、训练与恢复逻辑和后续结构演进独立维护；来源说明与原始许可证保留在 [NOTICE](NOTICE) 和 [LICENSE](LICENSE)。

### 🎉 本项目包含

- **模型实现**：Dense / MoE、GQA、RoPE、RMSNorm、SwiGLU 与 KV Cache。
- **数据准备**：文本预训练、单轮问答 SFT、prompt loss mask、JSONL 校验与按需读取。
- **训练入口**：配置驱动的 FP32 训练、DDP、梯度累积和全局有效 token loss。
- **断点恢复**：定期 checkpoint、SIGTERM/SIGINT 保存、模型与 optimizer 恢复、各 rank 随机状态恢复。
- **模型评测**：按有效 token 加权的 NLL / PPL，配套训练正确性与兼容性测试。
- **模型推理**：命令行文本生成与基础 `/v1/chat/completions` 服务。
- **架构演进**：独立 `MicaConfig` / `MicaForCausalLM`，保留旧权重导入和数值兼容测试。
- **技术资料**：数据工程、预训练、后训练与 KV Cache / GQA 的已有案例与源码阅读笔记。

> 当前 Mica CLI 已验证单进程和双进程 CPU 训练。CUDA 多卡尚待实机验收；LoRA、DPO、GRPO/CISPO、Agentic RL 和完整通用 benchmark 的专用训练脚本保留在 trainer/，尚未统一纳入 CLI；历史实验资料位于 archive/。

### 📦 模型与配置

| 配置 | 参数规模 | 用途 | 当前状态 |
|---|---:|---|---|
| CPU Smoke | 微型模型 | 验证安装、训练、保存、加载与评测 | [可直接运行](configs/smoke/cpu.json) |
| Mica Dense 64M | 63,912,192 | 预训练、SFT、结构改造 | [模型配置](configs/models/mica-64m.json) |
| Mica MoE 198M | 198,416,640；名义激活 63,936,768 | 稀疏 FFN 与路由改造 | 基于 Dense 配置设置 `use_moe=true` |
| S10 兼容权重 | 原 MiniMind 64M | 验证旧权重导入与推理 | strict-load 与生成已通过；未公开下载 |

模型配置可用不等于权重已经发布。目前没有 Mica 自有训练权重的公开下载地址。

### 📝 更新日志

<details open>
<summary><b>2026-09-15 · Mica v0.2</b></summary>

- 项目统一命名为 Mica，GitHub 仓库为 `Qi18/mica`。
- 建立独立模型包与 `mica` 命令行入口。
- 加入按需数据读取、DDP、梯度累积和 token 加权训练。
- 加入定期原子 checkpoint、中断保存和精确恢复。
- 完成 13 项测试，覆盖兼容性、Dense/MoE、DDP、恢复和保存失败处理。

</details>

---

# 🚀 快速开始

## Ⅰ 安装项目

需要 **Python 3.10+**。当前验证的依赖范围是 **PyTorch 2.6**、**Transformers 4.51**。

```bash
git clone https://github.com/Qi18/mica.git
cd mica

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

mica doctor
```

`mica doctor` 会显示项目版本、PyTorch / Transformers 版本和可见 GPU 数量。项目名与 CLI 为 `mica`，Python 导入名为 `mica`，安装分发名为 `mica-llm`。

## Ⅱ 在 CPU 上跑通训练

无需下载权重或数据，仓库自带微型模型配置和 token 示例：

```bash
mica train \
  --recipe configs/smoke/cpu.json \
  --output outputs/smoke

mica evaluate \
  --model outputs/smoke \
  --data examples/tokens.jsonl
```

这一步验证训练、保存、加载与评测链路。示例只有少量合成 token，不具备对话能力。

## Ⅲ 查看训练产物

```text
outputs/smoke/
├── latest.json                   # 最近一次完整 checkpoint
├── run.json                      # 状态、数据指纹、训练规模
├── metrics.jsonl                 # 每个 update 的 loss 与有效 token 数
└── checkpoints/
    └── step-00000004/
        ├── config.json
        ├── model.pt
        ├── training.pt           # optimizer、step、各 rank RNG
        └── recipe.json
```

训练、评测和推理均可使用运行根目录；加载时自动解析 `latest.json`。

---

# 🧠 模型结构

Mica 的模型实现位于 [modeling_mica.py](model/modeling_mica.py)。

```text
Token IDs
   │
Embedding
   │
8 × Transformer Block
   ├── RMSNorm → GQA Attention + RoPE → Residual
   └── RMSNorm → SwiGLU / MoE FFN    → Residual
   │
RMSNorm
   │
Shared LM Head
   │
Next-token logits
```

| 模块 | Dense 64M 默认配置 |
|---|---|
| 词表 | 6,400 |
| 层数 / hidden size | 8 / 768 |
| Query heads / KV heads | 8 / 4 |
| Head dimension | 96 |
| FFN intermediate size | 2,432 |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| 词嵌入 | Embedding 与 LM Head 权重共享 |
| MoE 可选项 | 4 experts、top-1 routing |

```python
from mica import MicaConfig, MicaForCausalLM

config = MicaConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
model = MicaForCausalLM(config)

# 切换 MoE
moe = MicaForCausalLM(MicaConfig(use_moe=True))
```

后续可从 Attention、FFN、路由策略等模块演进结构。改变结构时应保存对应配置，并重新验证权重加载、forward/backward 与 KV Cache。详细说明见 [架构文档](docs/mica-architecture.md)。

---

# 🛠️ 数据与训练

## Ⅰ 准备数据

**预训练**：每行一个文本对象。

```json
{"text": "语言模型通过上下文预测下一个 token。"}
```

**单轮 SFT**：每行一组 prompt 与 response。

```json
{"prompt": "什么是语言模型？", "response": "语言模型学习文本中的规律，并根据上下文预测或生成后续内容。"}
```

使用与模型匹配的 tokenizer 转换数据：

```bash
mica data \
  --source /path/to/source.jsonl \
  --tokenizer tokenizer \
  --output outputs/train.jsonl \
  --max-length 768
```

SFT 的 prompt label 设置为 `-100`，只监督回复。截断后没有有效监督 token 的样本会报错。输出逐行生成；出现错误时不会留下一个看似完成的数据文件。

初始 tokenizer 保留在 `tokenizer/`，以维持旧权重兼容。修改词表后，需要同步模型配置并重新训练相应权重。

## Ⅱ 配置训练

训练 recipe 使用 JSON，以下示例采用微型结构与真实 tokenizer 词表，用于验证文本训练：

```json
{
  "model": {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 6400,
    "max_position_embeddings": 768
  },
  "data": "../outputs/train.jsonl",
  "device": "cpu",
  "steps": 100,
  "batch_size": 2,
  "gradient_accumulation_steps": 2,
  "save_every": 25,
  "learning_rate": 0.001,
  "seed": 42
}
```

将其保存为 `configs/custom.json`，再执行：

```bash
mica train --recipe configs/custom.json --output outputs/custom
```

`data` 相对 recipe 文件解析。使用完整 64M 模型时，将 `model` 替换为 [64M 配置](configs/models/mica-64m.json) 的内容，并根据设备调整 batch、学习率和训练预算。

SFT 可在 recipe 中设置 `initialize_from`，指向已有 Mica 模型目录。该路径同样相对 recipe 解析。

## Ⅲ 多进程与梯度累积

先用双进程 CPU 示例验证 DDP：

```bash
torchrun --standalone --nproc_per_node=2 -m mica train \
  --recipe configs/smoke/cpu-ddp.json \
  --output outputs/ddp
```

CUDA 配置将 `device` 设为 `cuda`，每个进程使用 `LOCAL_RANK` 对应的可见 GPU。实际 GPU 数量由 `--nproc_per_node` 指定。

**全局 batch = 每进程 batch × 累积次数 × 进程数。**

不同长度、不同 prompt mask 的样本按全局有效 token 数归一化 CE，避免分组方式改变监督权重。

## Ⅳ 中断与恢复

`save_every` 控制定期保存。SIGTERM / SIGINT 在当前 optimizer update 完成后保存；CLI 返回 130，状态记为 `interrupted`。

增加 recipe 的总 `steps`，然后恢复到新输出目录：

```bash
mica train --recipe configs/custom.json \
  --resume outputs/custom --output outputs/custom-resumed
```

精确恢复要求相同的数据指纹、进程数和其余 recipe 配置。更多细节见 [训练与恢复指南](docs/mica-training.md)。

---

# 💬 推理与服务

## Ⅰ 导入已有权重

如果持有与配置匹配的原始 MiniMind state_dict：

```bash
mica import-legacy \
  --checkpoint /path/to/legacy.pth \
  --config configs/models/mica-64m.json \
  --output outputs/imported
```

导入使用严格参数检查，并记录源权重 SHA256。旧权重迁移保持其原始能力与来源。

## Ⅱ 命令行对话

```bash
mica generate \
  --model outputs/imported \
  --tokenizer tokenizer \
  --prompt "介绍一下你自己" \
  --max-new-tokens 64
```

当前为贪心解码，模型与 tokenizer 必须匹配。

## Ⅲ HTTP 服务

```bash
pip install -e ".[serve]"

mica serve \
  --model outputs/imported \
  --tokenizer tokenizer \
  --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mica","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

接口支持文本消息、`max_tokens` 和非流式返回。当前使用 Flask 开发服务器，流式、工具调用和采样参数尚未实现。

---

# 📊 效果与验证

### 工程验证

- Dense / MoE 与初始实现的 FP32 输出兼容。
- 单进程、双进程 DDP 的连续训练与恢复结果一致。
- 梯度累积与大 batch 的 token 加权训练结果在浮点容差内一致。
- SIGTERM 保存后可继续训练，结果与连续训练一致。
- 模拟保存失败后，仍能读取上一次完整 checkpoint。
- S10 原始权重导入、命令行生成和 HTTP 请求已通过验证。

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

### 已有模型案例

以下结果来自迁移前的 MiniMind 64M 与既有训练流程，作为项目技术积累保留，不代表 Mica 新架构的训练成绩。

| 方向 | 观察 | 资料 |
|---|---|---|
| 数据与预训练 | P03 共享 validation NLL 2.604，优于 P02 的 3.191；对比包含多项数据管线变化 | [项目案例](archive/docs/final_report.md) |
| 指令微调 | S10 在阶段冻结协议下 IFEval prompt strict 17.38%，较 S09 +6.29pp | [SFT 报告](archive/docs/phases/phase2-sft.md) |
| KV Cache / GQA | A10 固定轨迹测量中 cache decode 加速 0.82–6.23×，GQA KV 字节减半 | [推理性能](archive/experiments/08-inference/I01-I04-a10-20260914/report.md) |

更多源码分析见 [阅读索引](archive/docs/source_reading/README.md)。数据集、权重、optimizer 状态与完整日志单独保存，不进入 Git。

---

# 🗂️ 项目目录

```text
mica/
├── model/              # 唯一模型实现、LoRA 与 checkpoint 加载
├── dataset/            # 数据转换、Dataset 与按需索引
├── trainer/            # 统一训练引擎及各阶段训练脚本
├── evaluation/         # token 加权评测
├── inference/          # 生成、服务与交互入口
├── tokenizer/          # 独立维护的 Tokenizer
├── configs/            # 当前模型、smoke 与数据配置
├── scripts/            # 可复用数据、评测与工程工具
├── tests/
├── examples/
├── docs/
├── archive/            # 旧实验、阶段脚本、报告与博客
└── mica.py             # CLI 与 Python 入口
```

## 保留的训练脚本

预训练、SFT、LoRA、DPO、GRPO、PPO、蒸馏和 Agent RL 脚本位于 trainer/，均导入同一份 Mica 模型。
mica train 使用配置驱动训练引擎；其他专用脚本未全部迁入该入口，本轮不做全量训练验收。
额外依赖见 requirements-training.txt（历史依赖快照，安装前核对与主线依赖的兼容性）。

```bash
cd trainer
python train_pretrain.py --help
# 正式运行时显式传入 --data_path、--save_dir 等参数
```

历史重放见 [归档说明](archive/README.md)。L20 原 minimind/ 下忽略的权重和日志保持原位置，不是源码依赖；全新克隆不含该目录。

# 🧭 后续计划

- [x] 独立模型包、CLI、数据处理与训练闭环。
- [x] CPU DDP、梯度累积、定期 checkpoint 与恢复。
- [x] 基础评测、文本生成与 HTTP 服务。
- [ ] CUDA 多卡验收、混合精度与学习率调度。
- [ ] LoRA、DPO、GRPO/CISPO、Agentic RL 统一入口。
- [ ] 完整 benchmark、SwanLab 集成与训练可观测性。
- [ ] Attention / FFN / MoE 结构演进与模型权重发布。

进度与范围见 [Roadmap](docs/mica-roadmap.md)。

# 🤝 致谢与来源

感谢 [MiniMind](https://github.com/jingyaogong/minimind) 提供清晰的小模型实现。初始模型来源 commit 为 `393e387e9ad99f0f04c296e4c5e7353f4444629f`，原始 Apache-2.0 许可证与来源声明保留在仓库中。

欢迎通过 [Issues](https://github.com/Qi18/mica/issues) 讨论问题，或提交 [Pull Request](https://github.com/Qi18/mica/pulls) 改进模型与工程实现。