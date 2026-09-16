# D04 SFT v1 source expansion

状态：`configured_not_materialized_not_profiled`。本阶段只冻结扩源输入和 v2 口径，不代表数据通过正式验收。

## 为什么需要扩源

D03 在 MiniMind tokenizer、chat template、768 tokens、完整 turn、闭合 EOS、shifted labels 和 exact dedup 口径下，虽然总 eligible capacity 为 238,791,045 assistant tokens，超过 161.6M 总预算，但六个 bucket 未达到各自配额：

| bucket | target | D03 available | exact shortfall |
|---|---:|---:|---:|
| chinese_general | 48,480,000 | 8,181,944 | 40,298,056 |
| multiturn | 16,160,000 | 1,430,203 | 14,729,797 |
| strict_format | 24,240,000 | 18,732 | 24,221,268 |
| math | 16,160,000 | 1,013,421 | 15,146,579 |
| summarization_translation | 8,080,000 | 5,922,924 | 2,157,076 |
| code | 8,080,000 | 1,710,629 | 6,369,371 |

基线绑定到 `D03-sft-capacity-20260901/capacity_report.json`，SHA-256 为 `6ce01f4960a125ed6a3013c2674a71402355663a7e8088f3e224aa04dd080ab2`。总量富余不能替代逐 bucket 验收。

## 固定补充源

- 中文：`TigerResearch/tigerbot-alpaca-zh-0.5m@14101774e35107ba5a18566da1e8e64d0b8f0174`，Apache-2.0。
- 数学：`AI-MO/NuminaMath-CoT@9d8d210c9f6a36c8f3cd84045668c9b7800ef517`，Apache-2.0。
- 代码：`ise-uiuc/Magicoder-Evol-Instruct-110K@b0079beaa0361d82412520b873715bee59cc7dd4`，Apache-2.0。
- 严格格式：`allenai/tulu-3-sft-personas-instruction-following@fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e`，ODC-BY。
- 摘要：`abisee/cnn_dailymail@96df5e686bee6baa90b8bee7c28b81fa3fa6223d` 的 `3.0.0` config，字段为 `article/highlights/id`；HF cardData 和 license tag 都机器可读为 Apache-2.0。

现有源保持优先顺序，五个补充源只追加在其后。D03 摘要/翻译短缺 2,157,076 tokens 的基线证据仍保留；CNN/DailyMail 摘要源已固定，但能否补足必须由 v2 profiler 实测，当前状态是 `source_selected_capacity_pending`。SmolTalk 的固定 README 虽说明 `smol-summarize` 为 Apache-2.0，但该 revision 的 HF 元数据没有机器可读 license，因此按 fail-closed 原则未纳入 v1。

## v2 归属与推理门禁

- UltraChat 按 `sha256_utf8(prompt_id) % 5` 分桶，余数 0/1/2 归 english_general，3/4 归 multiturn，比例 3:2。一个 `prompt_id` 的全部 chunks 只能进入同一 bucket，且不允许重复使用。该策略状态为 `pending_v2_profiler`；D03 结果未应用此策略。
- strict-format v2 的 donor origin 互斥：同一 `(source_id,id)` 只能贡献原始记录或一个通过验证的 transform，不能二者同时使用，也不能跨 bucket/split 复用。每次变换必须绑定 transform 版本、输入/输出 digest 和 verifier 状态。
- NuminaMath profile 必须先 strip 再用 case-fold exact 排除 `source in {gsm8k, math}`。reasoning 解析为 fail-closed；只有 messages 或 problem/solution 通过 parse、闭合 assistant span 和 shifted-target 门禁后才计容量。

## 命令边界

`command.sh` 只是 D02 raw materializer 的窄入口：

```bash
RUN_MODE=validate bash experiments/00-preparation/D04-sft-source-expansion-20260901/command.sh
RUN_MODE=materialize bash experiments/00-preparation/D04-sft-source-expansion-20260901/command.sh
```

默认模式为 `validate`；该模式先通过 D02 刷新固定 revision/license 元数据，再执行 raw 配置与 evidence 校验，但不下载数据。创建配置时不自动运行上述命令，也不启动 D03/D05 profile。

## 后续验收

扩源后仍必须：

1. 对每个新 raw 对象记录固定 revision、license、rows、bytes、SHA-256 和 required fields；
2. 使用 v2 profiler 按相同 tokenizer/768/完整 turn/闭合 EOS/shifted label 口径重新计算；
3. 六个 bucket 各自满足含 heldout 的 token quota，禁止跨 bucket/split 复用；
4. 确认 CNN/DailyMail 的实测容量足以关闭摘要/翻译缺口；
5. 之后才允许冻结 mix、构建训练 payload/sidecar 并由独立 auditor 写 accepted `_SUCCESS`。
