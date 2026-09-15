# Mica

一个可持续演进的语言模型工程，覆盖模型实现、数据处理、训练、评测与推理。

Mica 从可理解的小模型起步，为修改 Attention、FFN、MoE 和训练方法提供独立代码与可执行入口。初始模型源自 MiniMind，来源与修改说明见 [NOTICE](NOTICE)；已有训练结果仍保留原始模型和协议名称。

## 快速开始

需要 Python 3.10+。首版验证依赖为 PyTorch 2.6 与 Transformers 4.51。

```bash
git clone https://github.com/Qi18/minimind-lab.git mica-llm
cd mica-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mica doctor
mica train --recipe recipes/smoke/cpu.json --output outputs/smoke
mica evaluate --model outputs/smoke --data examples/tokens.jsonl
```

当前 GitHub 地址保留为 minimind-lab；项目、Python 包和 CLI 分别为 Mica、mica_llm、mica。
CPU 示例使用微型模型与合成 token，只验证训练/保存/加载链路，不具备对话能力。

## 项目能力

| 能力 | 当前入口 |
|---|---|
| Dense / MoE 模型 | MicaConfig、MicaForCausalLM，独立实现 |
| 文本预训练与单轮 SFT 数据 | mica data，支持 prompt mask |
| FP32 单设备 / DDP 训练 | mica train，JSON recipe，梯度累积 |
| 定期保存与续训 | save_every、SIGTERM/SIGINT 保存，--resume 恢复各 rank RNG |
| 语言建模评测 | mica evaluate，按有效 token 加权 NLL/PPL |
| 旧权重迁移 | mica import-legacy，严格校验参数键与形状 |
| 文本生成 | mica generate，贪心解码 |
| 基础 HTTP 服务 | mica serve，非流式 /v1/chat/completions |

数据准备按行处理，训练使用紧凑偏移索引和按需解码；checkpoint 定期保存并在 optimizer update 边界处理中断。双进程 CPU DDP 已验证，CUDA 多卡尚待验收。用法见 [训练与恢复](docs/mica-training.md)。LoRA、DPO、GRPO/CISPO、Agentic RL、SwanLab 和完整 benchmark 尚待迁入，见 [Roadmap](docs/mica-roadmap.md)。

## 数据与训练

准备 JSONL：预训练每行为 `{"text":"..."}`，单轮 SFT 为 `{"prompt":"...","response":"..."}`。

```bash
mica data --source /path/to/source.jsonl --tokenizer minimind/model \
  --output outputs/train.jsonl --max-length 768
```

复制并调整 [CPU recipe](recipes/smoke/cpu.json)：data 相对 recipe 文件解析，
model 定义架构，device 选择 cpu/cuda，steps 表示总 optimizer updates。
64M 配置见 [模型配置](recipes/models/mica-64m.json)。
SFT 在 recipe 中设置 initialize_from 指向 Mica 模型目录；已有 model.pt
和 config.json 可作为初始化。单轮 SFT 只监督回复 token。

续训将 recipe 的 steps 增大，其余字段保持相同，输出到新目录：

```bash
mica train --recipe /path/to/continued.json \
  --resume outputs/smoke --output outputs/continued
```

## 导入已有权重与生成

```bash
mica import-legacy --checkpoint /path/to/legacy.pth \
  --config recipes/models/mica-64m.json --output outputs/imported
mica generate --model outputs/imported --tokenizer minimind/model \
  --prompt "介绍一下你自己" --max-new-tokens 32
```

只接受与配置匹配的原始 state_dict。迁移保持模型结构和权重，不代表新模型已训练。
Tokenizer 必须与权重匹配。实际权重和数据单独保存，仓库只包含代码与示例。

## 启动服务

```bash
pip install -e ".[serve]"
mica serve --model outputs/imported --tokenizer minimind/model --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mica","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

当前为 Flask 开发服务，支持文本消息与 max_tokens。流式、工具调用和采样参数尚未实现。

## 工程结构

```text
src/mica_llm/   模型、数据、训练、评测、推理与 CLI
recipes/       模型与训练配置
examples/      最小示例数据
tests/         数值兼容、checkpoint 与续训测试
docs/          架构、使用与迁移路线
minimind/      历史源码与 tokenizer，保留兼容性参照
scripts/       历史训练和评测实现，逐步迁入新入口
experiments/   已有结果与原始配置的可追溯档案
```

设计说明见 [架构文档](docs/mica-architecture.md)。
在原始 64M 模型上完成的数据工程、预训练、后训练与推理优化工作，
可从 [项目案例](docs/final_report.md)、[源码阅读](docs/source_reading/README.md)、
[KV Cache/GQA 性能报告](experiments/08-inference/I01-I04-a10-20260914/report.md) 进入。
历史报告记录的是原始实现的结果，不作为 Mica 新结构的成绩。

## 开发验证

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

未来模型结构变化需明确配置与版本，并验证 forward/backward、KV cache、保存与恢复。
