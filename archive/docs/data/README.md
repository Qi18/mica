# 数据工程文档索引

Pretrain 与 SFT 的样本结构、Token 口径、构建阶段和验收条件不同，因此分开维护协议；跨阶段的训练路线和统一评测仍由顶层文档管理。

| 阶段 | 当前状态 | 配置 | 正式脚本 | 协议 |
|---|---|---|---|---|
| Pretrain v1 | `accepted` | `configs/data/pretrain/` | `scripts/data/pretrain/` | [Pretrain 数据协议](pretrain/data_protocol.md) |
| SFT v1 | `design_pending_implementation`、`trainable: false` | `configs/data/sft/` | `scripts/data/sft/`（目前仅 raw 阶段） | [SFT 数据协议](sft/data_protocol.md) |

## 共享入口

- [完整实验与源码学习计划](../experiment_plan.md)
- [统一评测计划](../evaluation_protocol.md)
- [正式数据脚本说明](../../../scripts/data/README.md)
- Python 数据依赖：`configs/data/requirements.txt`

## 状态含义

- `raw_not_trainable`：来源已解析或下载，但未完成 canonical build、确定性切分和独立审计；
- `design_pending_implementation`：配置和门禁已经设计，但缺少正式 builder 或 auditor；
- `built_pending_external_audit`：builder 已产出完整候选数据，尚未通过独立验收；
- `accepted`：独立 auditor 全量回读通过，并写出绑定配置、代码、Tokenizer 和数据摘要的 `_SUCCESS`；
- `rejected`：任一 fatal gate 失败，不能启动正式训练。

## 共同原则

1. 原始数据与可训练数据分层存放，raw cache 不能直接冒充正式训练集。
2. builder 和独立 auditor 分离；builder 不能给自己的输出签发 accepted。
3. 数据切分在训练前固定，训练器不得再次随机切分。
4. 固定评测集污染检查属于发布门禁，不能在观察分数后调整。
5. Git 只保存配置、脚本、manifest、审计摘要和报告；数据正文留在 L20/外部数据仓库。
