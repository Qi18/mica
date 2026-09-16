# D02 SFT data v1 — 160M raw materialization

状态：`implementation_only`。正式全量未启动；raw materialization 当前无 blocker，final mix/build 仍为 fail-closed。

## 本阶段边界

该阶段只负责固定来源元数据并把原始对象顺序物化到：

`/data/datasets/minimind-lab/data-v1/sft-v1-160m/`

每个对象使用独立 `.tmp`、`.done.json`、SHA-256、rows、bytes 和字段清单。已完成且证据一致的对象会被强校验后跳过；输出或 done 单边存在、fingerprint 漂移、SHA/rows/bytes/fields 不一致时 fail-closed，不自动覆盖。

下载始终单源串行执行，并使用 `nice 10`、`ionice best-effort/7`，避免与正在运行的预训练竞争 CPU 和 CPFS IO。

## P0：raw 绝不是训练输入

本阶段保留各上游源的原始字段，所有 manifest/done 都必须记录：

- `schema_stage=raw_not_trainable`
- `trainable=false`

现有 D01 `prepared/sft_v1.*.jsonl` 顶层含 `source`、`source_record_hash`，message 字段也不统一，已触发 Hugging Face Features `CastError`。它只能作为 schema/pipeline fixture，绝不能作为正式训练输入。

后续 formal builder 必须生成以下唯一训练 schema：

1. JSONL 顶层只允许 `conversations`；
2. 每条 message 恰好包含 `role/content/reasoning_content/tools/tool_calls` 五个键；
3. 缺失的可选字段写显式 `null`；
4. source、revision、license、record hash、原始位置等 provenance 全部写 sidecar，不进入训练 JSONL。

## 分阶段门禁

Raw materialization 必须先完成，才能测量各来源的 assistant-token capacity，因此当前：

- `raw_materialization_blockers: []`，允许顺序下载/复用七个已固定来源；
- `final_build_blockers` 仍包含摘要源缺失和 capacity 未冻结；
- strict-format 和 translation 是后续 derived stage，不在 raw materializer 中伪造。

这些 final blocker 禁止生成正式 mix、训练 shards 或 `_SUCCESS`，但不阻断原料物化。不得通过重复样本、扩大模板合成量或静默改变 mix 来绕过 final blocker。

## 命令

静态检查与无网络极小 self-test：

```bash
RUN_MODE=self-test bash experiments/00-preparation/D02-sft-data-v1-20260901/command.sh
```

只回读固定 revision/license 元数据：

```bash
RUN_MODE=resolve bash experiments/00-preparation/D02-sft-data-v1-20260901/command.sh
```

顺序启动 raw materialization（不会生成训练 shards 或 `_SUCCESS`）：

```bash
RUN_MODE=materialize bash experiments/00-preparation/D02-sft-data-v1-20260901/command.sh
```

不要运行 D01 `command.sh` 来构建 full 数据；该入口只面向代理产物，并会写入 D01 的 `raw-samples/normalized/grouped/token-filtered/prepared` 路径。

## OASST1 复用

OASST1 不重复下载。materializer 会对现有完整文件重算 84,437 rows、137,489,537 bytes、SHA-256 `41d3aa...f84e` 和 required fields，全部一致后才在 D02 raw 目录创建原子 symlink，并把复用源写入 done sidecar。
