# 训练前准备实验

这里统一保存正式训练前的环境、数据和模型运行时验证。它们回答“能否可信地开始训练”，不作为模型能力结果。

| 顺序 | 实验 | 目的 | 完成门槛 |
|---|---|---|---|
| Stage0 | [E00 L20 baseline](E00-l20-baseline-20260823/report.md) | 验证 8×L20、BF16、NCCL、CPFS、`/dev/shm` 和 SwanLab | 环境 smoke test 通过 |
| Stage1 | [E01 Tokenizer/Dataset](E01-tokenizer-dataset-20260823/report.md) | 固定 Tokenizer，审计数据格式、规模、截断和 padding | 数据 manifest 与质量报告完成 |
| Stage2 | [E02 Model Probe](E02-model-probe-20260823/report.md) | 验证 Dense/MoE、DDP、BF16、SwanLab 和断点恢复 | 100-step 三 seed probe 与 resume 通过 |

数据工程实验（与上表同目录，但不产出模型能力结果，尚未进 `../registry.csv`，见 `docs/phases/phase0-preparation.md` §8）：

| 实验 | 目的 | 自有状态 |
|---|---|---|
| [D01 Pretrain v1 数据](D01-training-data-v1-20260828/report.md) | 构建并验收 P03 使用的 1.28B target Pretrain v1（`final-remix-v1`） | `accepted`（2026-08-31 11:16:03 UTC） |
| [D02 SFT v1 raw 物化](D02-sft-data-v1-20260901/README.md) | 固定来源元数据并物化 raw 对象 | `implementation_only`（final mix 仍 fail-closed） |
| [D03 SFT v1 容量剖析](D03-sft-capacity-20260901/README.md) | 测量真实 tokenizer / chat template 下的 assistant-target 容量 | `capacity_profiled_not_trainable` |
| [D04 SFT v1 扩源](D04-sft-source-expansion-20260901/README.md) | 冻结扩源输入与 v2 口径，补 D03 的 bucket 缺口 | `configured_not_materialized_not_profiled` |
| [D05 SFT v2 容量剖析](D05-sft-capacity-v2-20260901/README.md) | 按 gross / origin_exclusive / exact_unique 三层重算容量 | 仅容量评估，不写 `_SUCCESS` |

目录约定：

```text
00-preparation/
├── E00-l20-baseline-20260823/
├── E01-tokenizer-dataset-20260823/
├── E02-model-probe-20260823/
├── D01-training-data-v1-20260828/
├── D02-sft-data-v1-20260901/
├── D03-sft-capacity-20260901/
├── D04-sft-source-expansion-20260901/
└── D05-sft-capacity-v2-20260901/
```

准备阶段通过后，正式训练从 [`../../../experiments/01-pretrain`](../../../experiments/01-pretrain) 的 P01 开始。
