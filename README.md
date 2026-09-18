<div align="center">

# 💎 Mica

### 从小模型出发，构建自己的语言模型

**模型结构 · 数据处理 · 预训练 · 后训练 · 评测 · 推理**

[架构](docs/architecture.md) · [训练](docs/training.md) · [开发](docs/development.md)

</div>

## 项目介绍

Mica 是独立维护的小型语言模型项目，提供 Dense/MoE 模型、Tokenizer、数据转换、训练、评测与推理。
初始实现源自 MiniMind，后续直接在 Mica 内维护；来源与许可证见 [NOTICE](NOTICE)、[LICENSE](LICENSE)。

主线 Dense 默认配置为 8 层、隐藏维度 768、8 个 Query 头、4 个 KV 头、词表 6400；模型代码位于 [model/model_mica.py](model/model_mica.py)。
没有可据此宣称为新架构训练成果的公开权重。

## 快速开始

以下命令从仓库根目录执行，使用 Python 3.10+。主线依赖为 PyTorch 2.6 与 Transformers 4.51，见 [安装配置](pyproject.toml)。
本例创建自己的临时数据与配置，不需要仓库中的 configs/ 或 examples/。

### 1. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mica doctor
```

GPU 环境需使用与驱动兼容的 PyTorch 构建；doctor 显示 GPU 不等于多卡训练已验收。
安装分发名为 mica-llm，命令和 Python 导入名为 mica。

### 2. 创建示例

在同一个终端继续操作。每次用新目录，避免覆盖旧产物：

```bash
export MICA_RUN="$(mktemp -d /tmp/mica-quickstart.XXXXXX)"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["MICA_RUN"])
rows = [{"text": "你好，这是 Mica 的训练示例。"},
        {"text": "语言模型根据前面的文本预测下一个 token。"}]
(root / "raw.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
recipe = {
    "model": {"hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "intermediate_size": 64, "vocab_size": 6400,
              "max_position_embeddings": 256},
    "data": "tokens.jsonl", "device": "cpu", "steps": 4,
    "batch_size": 2, "gradient_accumulation_steps": 1,
    "save_every": 2, "cpu_threads": 2, "seed": 42,
    "learning_rate": 0.001
}
(root / "recipe.json").write_text(json.dumps(recipe, indent=2))
PY
mica data --source "$MICA_RUN/raw.jsonl" --tokenizer tokenizer \
  --output "$MICA_RUN/tokens.jsonl" --max-length 128
```

词表使用真实 Tokenizer 的 6400；不要把仅适合合成 token 的小词表配置用于该 Tokenizer。

### 3. 训练、评测、生成

```bash
mica train --recipe "$MICA_RUN/recipe.json" --output "$MICA_RUN/run"
mica evaluate --model "$MICA_RUN/run" --data "$MICA_RUN/tokens.jsonl"
mica generate --model "$MICA_RUN/run" --tokenizer tokenizer \
  --prompt "你好" --max-new-tokens 8
```

这里只验证工程流程。两条样本、四次更新不产生可用的对话模型；生成内容可能没有意义。
示例评测使用训练数据，不是泛化成绩。训练产物会保留在 MICA_RUN 指向的目录。

### 4. 双进程 CPU

使用不同的输出目录，不要覆盖上一步：

```bash
torchrun --standalone --nproc_per_node=2 -m mica train \
  --recipe "$MICA_RUN/recipe.json" --output "$MICA_RUN/ddp"
```

这与单进程的全局 batch 不同，不用于证明数值一致；自动测试覆盖等效 batch 对照。
恢复训练见 [训练与恢复](docs/training.md)，真实数据格式见 [训练指南](docs/training.md)。

## 本地 HTTP

```bash
pip install -e ".[serve]"
mica serve --model "$MICA_RUN/run" --tokenizer tokenizer --host 127.0.0.1 --port 8000
```

另一个终端：
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":8}'
```

支持文本 messages、max_tokens 和非流式返回。stream=true、tools、tool_choice、temperature、top_p、n 不受支持。
这是 Flask 开发服务器，没有生产鉴权或并发部署验收，不应直接暴露公网，也不是完整 OpenAI API 实现。

另有 [inference/chat.py](inference/chat.py) 的交互入口，使用原生 .pth/其他加载约定；不要与 CLI checkpoint 目录混用。

## 评测与验证

mica evaluate 返回按有效 label token 加权的 NLL、PPL、targets 和 rows；-100 不计入监督。
请使用独立验证集，并固定 Tokenizer、截断长度和 mask。快速开始中的训练集 loss 不是泛化成绩，也不是综合 benchmark 分数。

```bash
python -m unittest discover -s tests -v
```

CPU DDP、恢复等单元测试不代表所有阶段脚本的完整 GPU 验收。当前目录清理后的 CI 仍需更新，见 [开发指南](docs/development.md)。

## 项目结构

```text
mica/
├── model/              # 模型、LoRA 与加载
├── dataset/            # 数据转换、Dataset 与索引
├── trainer/            # 分阶段训练与 common 共享引擎
├── evaluation/         # NLL / PPL
├── inference/          # 生成、交互与基础服务
├── tokenizer/
├── tests/
├── docs/
├── mica.py
└── pyproject.toml
```

数据、权重和完整日志单独存储，不纳入 Git。历史资料可通过 Git 历史追溯，不依赖工作区中已移除的 archive/。

## 后续计划

- 更新 CI，移除旧目录依赖。
- 完成各阶段与 CUDA 多卡训练验收。
- 改进训练效率、评测、模型结构和推理能力。

## 来源

初始实现来自 [MiniMind](https://github.com/jingyaogong/minimind)，commit 为 393e387e9ad99f0f04c296e4c5e7353f4444629f。
Mica 独立维护，不再同步上游；来源声明和许可证保留于 [NOTICE](NOTICE)、[LICENSE](LICENSE)。
历史结果不作为新架构训练成绩；需要追溯时查看 Git 历史。
