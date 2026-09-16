# D03 SFT v1 capacity profile

## 目的

A1 只测量正式 SFT 数据在 MiniMind 真实 tokenizer、chat template、768 token 上限和 shifted labels 口径下的 assistant-target-token 容量。验收预算为 train 160M、validation 0.8M、test 0.8M，因此 profiler 同时检查总 eligible capacity 相对 161.6M 的差额；它不生成正式训练候选、shard 或成功标记，也不启动训练。

## 输入

- D02 已物化并带有 done evidence 的基础 raw 源，以及 configs/data/sft/acceptance_v1.yaml 中固定的 split/bucket 预算。
- OASST1 直接从已校验 raw message tree 在本地重建，不再次下载。
- 摘要源固定为 databricks/databricks-dolly-15k 的 revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a，许可声明 CC-BY-SA-3.0。其 raw 文件为 data root 下的 raw/databricks_dolly_15k.jsonl。
- 翻译容量从固定 Alpaca GPT4 中英对齐字段单独派生。
- strict-format 只实测当前 500-row generator，并投影达到 24M assistant tokens 所需行数。

validate 模式允许 Dolly raw 尚未物化，并将其明确列为 missing；full profile 对任何 missing 或 one-sided evidence 都 fail closed。

## 口径和门禁

每条候选消息都规范为恰好五个键：role、content、reasoning_content、tools、tool_calls；可选值写 null。容量统计不会把 provenance 写进训练 payload。

长多轮会话只在完整 user turn 边界拆分。每个被统计 chunk 必须同时满足：

- chat template 渲染后不超过 768 tokens；
- assistant marker 数等于 assistant message 数；
- 每个 assistant span 都找到闭合 EOS；
- 按训练时 input/label shift 后至少有一个 valid target；
- tool call 与 tool response 保持在同一完整 turn 内。

单个完整 turn 本身超过 768 时整 turn 丢弃并计数，不做截断。全局 exact digest 用临时 SQLite 主键统计，避免把全部 digest 留在内存；成功后只保留 capacity report，不保留 SQLite 或 full candidate spool。

## 运行

```bash
cd /data/projects/minimind-lab-data-v1
RUN_MODE=self-test bash experiments/00-preparation/D03-sft-capacity-20260901/command.sh
RUN_MODE=validate bash experiments/00-preparation/D03-sft-capacity-20260901/command.sh
RUN_MODE=profile bash experiments/00-preparation/D03-sft-capacity-20260901/command.sh
```

当前只允许完成 self-test 和 validate。Dolly raw evidence 补齐后，才可启动 full profile。

## 输出边界

full profile 只写 capacity_report.json，并把状态标为 capacity_profiled_not_trainable。报告按源和验收 bucket 给出 raw rows、invalid、完整 turn 拆分、overlength drop、exact duplicates、unique chunks、assistant markers、闭合 spans、shifted assistant-target tokens，以及 summarization 和 translation 的独立子项；同时报告 161.6M total eligible capacity 的 shortfall/surplus。总量通过不能替代每个 split/bucket 的 quota，也不允许 record 跨 bucket 或 split 复用。

A1 完成后仍不允许 formal build。下一步必须依据实测容量冻结 mix/token quota，再实现 payload 与 provenance sidecar 锁步的正式 builder 和独立 auditor。
