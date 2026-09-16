# D05 SFT v2 capacity profile

## 目的

D05 修复 D03 会高估可用容量的几个口径问题。它仍然只做容量评估，不生成训练 shard、不会写 `_SUCCESS`，也不代表数据已达到正式验收标准。

容量报告固定分三层：

1. `gross`：通过 schema、完整 turn、768 tokens、assistant marker/EOS 和 shifted-target 门禁的候选；
2. `origin_exclusive`：同一个 donor 只允许一个稳定 variant 获得容量声明，避免原始样本、翻译或 strict 派生同时计数；
3. `exact_unique`：在 origin 互斥之后做全局 exact digest 去重，最终只用这一层判断 bucket shortfall。

origin variant 和 exact winner 都由显式 priority key 决定；结果不依赖 source 扫描顺序。

## 新增来源与门禁

- `tigerbot_alpaca_zh_0_5m`：显式读取 `instruction/input/output`，计入中文通用候选。
- `numinamath_cot`：对 `source` 做 strip + casefold exact 比较，排除 `gsm8k` 和 `math`；`messages` 必须与 `problem/solution` 对齐。只接受能高置信解析为 balanced terminal `\\boxed{...}` 或明确 terminal `Final Answer:/Answer:` 的记录；reasoning/final 不能 roundtrip 时拒绝。
- `magicoder_evol_instruct_110k`：显式读取 `instruction/response`，计入代码候选。
- `tulu_3_sft_personas_instruction_following`：只报告 constraint 的 `declared/supported/verified`。自然语言约束不是机械验证；当前 verifier 不完整，因此整源 fail closed，actual capacity 为 0。
- `cnn_dailymail_summary`：显式读取 `article/highlights/id`；固定 summary instruction + article 为 user，highlights 为 assistant。只统计完整 turn 渲染后不超过 768 tokens 的候选，并报告 eligible chunk/token 与 overlength turn。

UltraChat 使用 `sha256_utf8(prompt_id) % 5` 固定 3:2 分区：余数 0/1/2 只进入 `english_general`，3/4 只进入 `multiturn`。multiturn chunk 至少包含两个 assistant message；不满足的 chunk 不回流到另一个 bucket。报告要求 origin-exclusive cross-bucket group 为 0。

正式 strict 策略是 `strict_format_v2`：donor 固定为 Tulu、origin-exclusive、禁止复用且只接收有 verifier evidence 的 transform。当前 500 行 `strict_format_v1` generator 仅保留为 baseline，用于估算平均 token 和投影行数；它不代表 v2 actual capacity，也不写入 gross/origin/exact 任一 actual capacity 层。

## 离线运行

```bash
cd /data/projects/minimind-lab-data-v1
RUN_MODE=self-test bash experiments/00-preparation/D05-sft-capacity-v2-20260901/command.sh
RUN_MODE=validate bash experiments/00-preparation/D05-sft-capacity-v2-20260901/command.sh
RUN_MODE=profile bash experiments/00-preparation/D05-sft-capacity-v2-20260901/command.sh
```

`self-test` 和 `validate` 都强制 Hugging Face/Transformers offline。`validate` 允许 raw source 尚未物化，并明确列入 `missing_sources`；任何 one-sided evidence 或 pin/license/schema 冲突会失败。

本次只执行 self-test 和 validate。只有五个新增源都完成 pinned raw + `.done.json`，且 validate 回读 `ready_for_full_profile=true` 后，才允许单独启动 `RUN_MODE=profile`。full profile 可能扫描数百万行，不能和 validate 混为一项。

## 自测覆盖

- stable dedup 正反扫描顺序得到同一个 winner；
- donor 原始与 strict 派生不能双计，origin-exclusive cross-bucket group 为 0；
- UltraChat 3:2 residue 分区和 multiturn 至少两个 assistant；
- Numina boxed/terminal parser、gsm8k/math 排除、reasoning/final roundtrip；
- Numina chat template 中 reasoning/final 均保留，shifted target 与 MiniMind `generate_labels` sentinel 一致；
- Tulu constraint 只声明、不虚构 verified；
- CNN/DailyMail summary 合法样本可计，超长完整 turn 整体拒绝；
- strict projection 不增加 actual capacity。

## 后续验收边界

即使 D05 exact-unique 容量覆盖 161.6M total eligible budget，仍需后续正式 builder 和独立 auditor 完成：冻结 train/validation/test quota、prompt-group split、near dedup、七项 benchmark contamination、逐源 provenance、最终文件 SHA/rows/tokens，以及唯一的 `_SUCCESS.status=accepted`。D05 本身始终保持 `trainable=false`。
