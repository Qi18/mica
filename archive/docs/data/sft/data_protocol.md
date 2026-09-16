# SFT v1 数据设计与验收协议

更新时间：2026-09-04
当前状态：`smoke_accepted_pilot_and_formal_pending`
训练资格：1M smoke 已验收；8M pilot 与 32M formal 暂不可训练

## 1. 为什么重新设计

旧方案直接把目标写成 160M assistant tokens，但当时只有 proxy 流水线、raw source 物化和容量分析，没有正式 builder、独立 auditor，也没有证据证明 160M 的混合比例和质量最优。因此不再把“凑满 160M”作为完成条件。

新方案先验证数据链路和训练收益，再逐级扩大：

| 阶段 | train assistant loss-target tokens | 目的 | 进入下一阶段的门 |
|---|---:|---|---|
| smoke | 1M | 验证 schema、loader、Tool 完整性和 auditor | 全部门禁通过，可重复构建 |
| pilot | 8M | 验证 mix 容量、训练信号和固定评测方向 | 相比 Pretrain 基线有目标能力增益且无异常退化 |
| formal v1 | 32M | 首个可复现、可对比的正式 SFT | 独立 audit accepted，完成固定评测 |
| optional scale | 64M | 只验证规模扩展是否继续有效 | 固定评测提升且 Base benchmark 无不可接受回归 |

64M 不是自动执行项；若 32M 已进入收益平台或出现遗忘，就停止扩量。160M 仅保留为旧目录名和历史方案，不再是当前预算。

## 2. 正式入口与缺口

- 来源配置：`configs/data/sft/sources_v1.yaml`
- 构建设计：`configs/data/sft/build_v1.yaml`
- 验收门禁：`configs/data/sft/acceptance_v1.yaml`
- 污染配置：`configs/data/sft/contamination_v1.yaml`
- raw resolver：`scripts/data/sft/resolve_sources.py`
- raw materializer：`scripts/data/sft/materialize_raw.py`
- formal builder：`scripts/data/sft/build_sft_v1.py`
- independent auditor：`scripts/data/sft/audit_sft_v1.py`

builder 只写 pending manifest；独立 auditor 复算训练标签、去重和来源约束，并且仅在污染报告也 accepted 时签发 `_SUCCESS`。

## 3. 数据构建流水线

```text
freeze source revisions + licenses
  → materialize/reuse raw objects with .tmp/.done evidence
  → normalize into canonical conversations
  → reject incomplete turns / partial tool exchanges / zero targets
  → compute shifted assistant loss-target tokens
  → group exact + near duplicates by origin
  → deterministic train/validation/test split
  → fixed-eval contamination filtering
  → bucket quota allocation without origin reuse
  → payload + provenance sidecars + manifest
  → independent full audit
  → _SUCCESS.status=accepted
```

训练器只读取已经冻结的 split 文件，不能在运行时再次随机切分。

## 4. Raw cache 边界

当前已物化原始源保留在：

`/data/datasets/minimind-lab/data-v1/sft-v1-160m/`

目录名来自旧目标，现仅表示已验证的 raw cache，避免重复下载和移动大文件；它不表示已有 160M 可训练 tokens。新正式工作区和输出分别使用：

- work：`/data/datasets/minimind-lab/data-v1/sft-v1-32m/work/`
- final：`/data/datasets/minimind-lab/data-v1/sft-v1-32m/final/`

对象级 `.done` 只证明原始文件物化完成，不能替代 canonical build、污染检查或独立验收。

## 5. Canonical 训练 schema

每一行顶层只允许：

```json
{"conversations": [{"role": "user", "content": "...", "reasoning_content": null, "tools": null, "tool_calls": null}]}
```

- 顶层 key 必须精确为 `conversations`；
- message key 必须精确为 `role/content/reasoning_content/tools/tool_calls`；
- 可选值显式写 `null`，避免来源 schema 漂移；
- role 仅允许 `system/user/assistant/tool`；
- 来源、revision、license、origin、transform、bucket 和 split 放 sidecar，不写进训练 payload；
- Tool call 与 Tool response 必须作为完整交换保留；
- 截断只能保留完整 turn，不能截断在消息或 Tool 交换中间。

## 6. 核心验收门禁

1. invalid JSON/conversation、zero shifted target、非 assistant 监督 Token 均为 0；
2. incomplete assistant turn、partial Tool exchange、assistant EOS 漏监督均为 0；
3. exact/near duplicate 先组成 origin component，再做确定性 split；
4. 同一 origin 不得跨 bucket、split 或 variant family 重复使用；
5. train/validation/test 的 exact 与 near overlap 均为 0；
6. 固定七项、GSM8K、MBPP 的 exact/containment/near overlap 均为 0；
7. 来源 revision、license、行数、字节和 SHA-256 可追溯；
8. builder 产出的 manifest 只能是 pending，只有独立 auditor 可以签发 accepted marker。

完整阈值以 `configs/data/sft/acceptance_v1.yaml` 为准。污染协议已冻结到 `lm-eval 0.4.12 / c1133b5` 和查询构建器 SHA-256；当前 blocker 是 strict-format 容量不足，以及 32M 最终 mix 与正式质量阈值尚未冻结。

## 7. 旧实验如何保留

- D01 的 5,907 条 proxy 数据只证明 schema、split、label mask 和污染流水线可运行，不是正式 SFT；
- D02 保存 raw source 解析与物化证据；
- D03-D05 保存容量、来源扩展、格式投影和 adapter 设计证据；
- 旧 profiler/生成脚本统一保存在 `experiments/00-preparation/D05-sft-capacity-v2-20260901/tools/`，不得作为正式入口；
- 历史 JSON、日志和命令保持原样，避免事后改写证据。

## 8. 2026-09-04 smoke 验收结果

验收产物：`/data/datasets/minimind-lab/data-v1/sft-v1-32m/final/smoke/`

| split | rows | shifted assistant loss-target tokens |
|---|---:|---:|
| train | 6,801 | 1,001,279 |
| validation | 144 | 21,213 |
| test | 135 | 21,707 |

- 八桶训练 token 配比达到 25/15/10/20/10/10/5/5 的目标，并仅允许单条完整记录造成的最小超配；
- strict-format 共有 13 个机械验证通过的约束族，train 为 200,086 targets；
- exact prompt、near prompt、exact conversation 分别拒绝 245、178、20 个候选；
- 对 31,186 条固定评测查询的 exact、containment、near overlap 均为 0；
- 独立 auditor failures 为 0，`SFTDataset(max_length=768, augment=False)` 加载抽检通过，`_SUCCESS.status=accepted`。

修正 manifest 路径并冻结状态配置后共完成三次独立重建；四个 payload/sidecar 数据文件的 SHA-256 三次完全一致，证明样本选择未漂移。中间结果保存在 `work/smoke-pre-manifest-fix-20260904/` 和 `work/smoke-pre-config-freeze-20260904/`。

## 9. 下一步

1. 扩展原生 strict-format 约束及机械 verifier；当前采样容量只有 279,871 targets，无法满足 8M pilot 的 1.6M 配额；
2. 对齐 formal acceptance 中的 reasoning 下限与 10% math 配额，补齐 auditor 尚未覆盖的正式质量门禁；
3. 容量通过后构建 8M pilot，做训练收敛、Chat/Tool/格式/数学和七项 Base benchmark A/B；
4. 依据 pilot 收益冻结 32M mix；只有 32M 评测有效后才决定是否扩到 64M。
