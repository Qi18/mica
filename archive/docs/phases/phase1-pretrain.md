# Phase 1 阶段报告：Pretrain 基线（P01 / P02 / P03）

- 阶段范围：`experiment_plan.md` 第 6 节
- 报告状态：accepted
- 出具日期：2026-09-03
- 数据/权重口径：MiniMind upstream `393e387e9ad99f0f04c296e4c5e7353f4444629f`，模型固定 63,912,192 参数（hidden 768、8 layers、vocab 6400），硬件固定 8×NVIDIA L20 + BF16 + seq 768 + 全局 sequence batch 256 + lr 5e-4 + seed 42 + 1 epoch

## 1. 阶段目标与研究问题

Phase 1 的目标是产出一个可以支撑后续 SFT 的 Base checkpoint，并把 Pretrain 阶段的证据收口。实际执行过程不是一次训练，而是三次：

1. **P01**：用官方 `pretrain_t2t_mini` 建立第一个可复现的 64M Base 基线；
2. **P02**：只把数据换成官方全量 `pretrain_t2t`（行数 6.67×），回答“扩大官方预训练数据能否改善 Base 能力”；
3. **P03**：P02 结果不满意后重做数据管线（自建 `pretrain-v1-1b28`：多源 remix、tokenizer 对齐切块、独立 validation、去重与污染门禁），回答“换数据质量与利用率能否改善 Base”。

本阶段回答：

- Q1：在固定模型/超参数下，单独扩大官方数据规模是否提升 Base 能力？
- Q2：重做数据管线（组成 + 切块 + 去重 + 污染门禁 + 独立 validation）是否提升 Base 能力与成本效率？
- Q3：三个 checkpoint 中哪一个应作为 SFT 的唯一正式 Base？

本阶段不回答：收益具体来自哪个数据源、哪种切块方式或哪个超参数（未做单变量消融）；也不回答生成/对话能力（Pretrain 不训练指令遵循）。

## 2. 实验清单

| experiment_id | status | 初始权重 | 数据 | 实验报告 |
|---|---|---|---|---|
| P01-dense-pretrain-mini-20260824 | completed（已在 main） | random-init | `pretrain_t2t_mini@312afb4f` | `experiments/01-pretrain/P01-dense-pretrain-mini-20260824/report.md` |
| P02-dense-pretrain-full-20260824 | completed（2026-09-03 已合入 main，含原 `stash@{0}` 的数据审计） | random-init | `pretrain_t2t@312afb4f` | `experiments/01-pretrain/P02-dense-pretrain-full-20260824/report.md` |
| P03-dense-pretrain-v1-1b28-20260901 | completed / accepted（非 release candidate） | random-init | `pretrain-v1-1b28/final-remix-v1` | `experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/report.md` |

### 2.1 SwanLab run

Workspace `@richliu0153`；完整索引见 [`swanlab-runs.md`](swanlab-runs.md)。

| experiment_id | run 角色 | project | URL |
|---|---|---|---|
| P01 | formal（`P01-Pretrain-Mini-64M-Seq768`） | MiniMind-Lab | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nfax3tyg0j217j1cz8y0b |
| P02 | formal（attempt 2） | MiniMind-Lab | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3i1muwq039fpfv89fq4ru |
| P02 | 失败 attempt 1（cache 写满） | MiniMind-Lab | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/q2lnh08i1dkvwtgey5d7z |
| P03 | probe（100 step） | MiniMind-Lab | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/d9is4iayxaw41ba95u92s |
| P03 | formal（1 epoch） | MiniMind-Lab | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/qdpjh47fjt98184oos4bl |
| P03 | eval logging | MiniMind-Lab | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/k9st16wqu3i7ijy2d7q9h |

P03 的 probe、formal、evaluation 三条 URL 已回填到实验目录的 `swanlab-url.txt` 和 `run.json`。P01/P02 的 run 已于 2026-09-01（stage5 收口 commit `5761979`）从 `MiniMind-Lab-Stage3/7iochx9kfe75qa2pt6d1u` 与 `MiniMind-Lab-Stage5/bs7n0qfcxykk13fammxis` 同步到统一 project `MiniMind-Lab`，因此三个实验现在同 project，旧 run id 以 `source_swanlab_run_id` 保留在 P01 的 `run.json`。但三者的训练数据、有效 target 预算与训练入口不同，**同 project 不等于可按 step 叠图比较**，跨实验结论仍只以本报告的固定评测数字为准。

## 3. 三份预训练数据对比

三份数据在同一口径下重新计量：tokenizer 为仓库自带 `minimind/model`，seq 768，`loss_target_tokens = len(截断后 raw tokens) + 1`（EOS），`padded_compute_slots = 行数 × 768`，chunk digest 为 `SHA256(BOS/raw/EOS 的 uint32-LE)`，与 `minimind/dataset/lm_dataset.py` 的 `PretrainDataset` 语义一致。

| 维度 | mini（P01） | full（P02） | pretrain-v1-1b28（P03） |
|---|---:|---:|---:|
| 文件字节 | 1,241,043,656 | 8,275,074,893 | 3,804,930,498（41 shards） |
| train 行数 | 1,270,238 | 8,468,827 | 2,313,483 |
| raw text tokens | 329,954,848 | 2,200,383,770 | 1,277,686,517 |
| shifted loss targets | 303,018,028 | 2,020,766,347 | 1,280,000,000 |
| padded compute slots | 975,542,784 | 6,504,059,136 | 1,776,754,944 |
| **target 利用率** | **31.06%** | **31.07%** | **72.04%** |
| 平均 targets/行 | 238.55 | 238.61 | 553.28 |
| 截断行数 | 49,829（3.92%） | 331,996（3.92%） | 0 |
| 丢弃 tail tokens | 28,207,058 | 188,086,250 | 0 |
| exact duplicate chunks | 207（0.0163%） | 1,433（0.0169%） | 0 |
| benchmark exact overlap | 未审计 | 0 | 0 |
| benchmark query-in-document containment | 未审计 | 115 | 0 |
| near-duplicate 抽样 | 未审计 | 20,480 行中 0 | 20,992 行中 0 |
| 训练期独立 validation | 无 | 无 | 11,525 行 / 6,400,000 targets |
| 行级来源 provenance | 无 | 无 | 完整 sidecar |
| `_SUCCESS` + 独立 auditor | 无 | 无 | accepted，14 项 check 全通过 |

三点结论：

1. **mini 与 full 是同分布的规模差异，不是质量差异**：平均 targets/行 238.55 vs 238.61、利用率 31.06% vs 31.07%、截断率 3.92% vs 3.92%、duplicate 率 0.0163% vs 0.0169%，几乎完全一致。因此 P01→P02 是干净的“数据规模”变量。
2. **v1 的核心差异是有效 token 利用率**：同样 768 的计算槽位，v1 每行承载 553.28 个 target，是官方数据的 2.32×，利用率从 31% 提升到 72%；`loss_target = text_tokens + 行数` 的恒等式成立，说明 v1 用切块替代截断，没有 tail token 丢弃。
3. **v1 的门禁更严**：exact duplicate 0、benchmark exact/containment 全为 0（全量扫描 2,325,008 行），并有独立 validation 与行级 provenance；full 的 115 个 containment 命中中既有明确题干重合也有通用短语碰撞，不能全部断言为泄漏，但它确实不满足 v1 的零污染门槛。

### 3.1 v1 的 source mix（train）

| Source | rows | loss target tokens | 占比 |
|---|---:|---:|---:|
| ChineseWebText2 | 1,139,577 | 576,000,000 | 45% |
| FineWeb-Edu | 619,195 | 384,000,000 | 30% |
| FineMath | 198,182 | 128,000,000 | 10% |
| Wikipedia zh | 137,999 | 64,000,000 | 5% |
| Wikipedia en | 119,777 | 64,000,000 | 5% |
| Stack v3（permissive code） | 98,753 | 64,000,000 | 5% |
| 合计 | 2,313,483 | 1,280,000,000 | 100% |

token 预算按 source 精确对齐目标值（`budgets.train.delta = 0`），因此“比例”是构建时的硬约束，不是事后统计。

### 3.2 数据身份

- mini：SHA-256 `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`；
- full：SHA-256 `31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d`，与官方 LFS oid 一致；
- v1：dataset fingerprint `cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd`，`_SUCCESS` SHA-256 `8c9c728ed063b214ebb74e905e0f6ccc4a510c7078408b87cda5d0c990639c7c`，共享 validation SHA-256 `0a7e8503f01bc185740b3e26e26326c43ca00309452ae1eec081d2ac2d9105cb`。

### 3.3 口径与缺口

- mini 与 full 的 targets/截断/duplicate 数字由本阶段用统一定义重算，脚本与结果位于 `/data/artifacts/minimind-lab/phase1-aux/`（`audit_official_pretrain.py`、`pretrain_official_target_audit.json`，transformers 4.51.1）。其中 full 的重算结果与 2026-09-02 的 P02 补审（`stash@{0}`）逐项一致：2,020,766,347 targets、331,996 截断行、188,086,250 丢弃 tail tokens、1,433 exact duplicate chunks，因此 mini 的 207 与 full 的 1,433 是同定义可比数字；
- mini 从未做 benchmark 污染扫描。它与 full 同分布，但“同分布”不等于“同样命中数”，不得把 full 的 115 个 containment 直接搬到 mini，也不得假设 mini 为 0；
- near-duplicate 三份都不是全量扫描，抽样为 0 不能外推为全量 0。

## 4. 三次训练结果对比

### 4.1 共享 validation（唯一可直接三方比较的 loss 口径）

三个 checkpoint 在 v1 固定的 11,525 行 / 6,400,000 target validation 上，用相同的 MiniMind shifted next-token CE 回溯评测（单卡 L20、BF16、batch 8、seq 768）：

| Checkpoint | validation NLL | PPL | 相对 P01 |
|---|---:|---:|---:|
| P01 | 3.56414 | 35.3092 | 基线 |
| P02 | 3.19096 | 24.3117 | NLL -0.37318 |
| P03 | **2.60432** | **13.5221** | NLL -0.95982，PPL -61.7% |

三次训练在 token-level 泛化上单调改善。该 validation 在 P01/P02 训练之后才建立，所以这是回溯式 checkpoint 对比，不能补写成 P01/P02 当时的训练期 validation。

training loss 不可跨实验比较（分布不同、横轴承载的有效 target 不同）：P01 末 100 点均值 2.0440、P02 为 1.7157、P03 为 2.6080。P02 的 training loss 最低，但共享 validation 明显差于 P03——这是本阶段最直接的“不能用 training loss 判泛化”的证据。

### 4.2 七项 Base 0-shot benchmark

统一协议：`lm-evaluation-harness 0.4.12`、commit `6d642546f4688648fced259eb3302efd36ece5af`、0-shot、无 chat template、batch 16、单卡 L20、seed 42；任务提供 `acc_norm` 时取 `acc_norm`，否则 `acc`；29,638 samples / 112,919 requests。

| 任务 | P01 | P02 | P03 | P02-P01 | P03-P02 | P03-P01 |
|---|---:|---:|---:|---:|---:|---:|
| C-Eval | 23.40 | 23.03 | 23.18 | -0.37 | +0.15 | -0.22 |
| CMMLU | 25.07 | 25.13 | 25.63 | +0.05 | +0.51 | +0.56 |
| ARC-Easy | 28.03 | 28.91 | 31.65 | +0.88 | +2.74 | +3.62 |
| PIQA | 51.85 | 51.47 | 51.90 | -0.38 | +0.44 | +0.05 |
| OpenBookQA | 29.00 | 26.60 | 26.00 | -2.40 | -0.60 | -3.00 |
| HellaSwag | 28.88 | 28.40 | 28.00 | -0.48 | -0.40 | -0.88 |
| Social IQA | 33.88 | 32.80 | 34.29 | -1.07 | +1.48 | +0.41 |
| **七项宏平均** | **31.44** | **30.91** | **31.52** | **-0.54** | **+0.62** | **+0.08** |

必须同时读三件事：

- P02 用 6.67× 数据、6.24× GPU-hours，宏平均反而低 0.54pp——**扩大同分布官方数据在当前配置下没有收益**；
- P03 相对 P02 恢复并超过（+0.62pp），单项最大增益是 ARC-Easy +2.74pp、Social IQA +1.48pp；
- **P03 相对 P01 的宏平均只有 +0.08pp**。也就是说，v1 数据把 validation NLL 降了 0.96、把有效 token 利用率翻了一倍多，但七项 benchmark 宏平均与最初的 mini 基线基本持平，OpenBookQA 还低 3.00pp。单 seed、64M 规模下 0.08pp 属于噪声量级，不能宣称“P03 的 Base 知识能力明显强于 P01”。

### 4.3 固定 greedy 续写（5 条固定提示，max_new_tokens 64）

| Checkpoint | 立即 EOS | 非空 | 实际质量 |
|---|---:|---:|---|
| P01 | 3 | 2 | 非空两条为 `1111…` 和 `""""…` 纯字符重复 |
| P02 | 5 | 0 | 五条 continuation 全空 |
| P03 | 0 | 4 | 四条全为 token 重复（`都是`×N、` is`×N、` =`×N、`标准大气压下`×N），第五条为 64 个换行 |

P03 只消除了“立即 EOS”退化，没有产生任何可用续写。三个 checkpoint 在自由生成上都不可用，这与 Pretrain 阶段定位一致：Base 只作为 SFT 的初始权重，不作为可发布生成模型。

### 4.4 成本与系统指标

| 指标 | P01 | P02 | P03 |
|---|---:|---:|---:|
| 每卡 batch × accumulation | 4 × 8 | 4 × 8 | 32 × 1 |
| micro-steps | 39,695 | 264,651 | 9,038 |
| optimizer updates | 4,962（估算） | 33,082（估算） | 9,038（实测） |
| 有效 targets / update | 61.1K | 61.1K | 141.6K |
| wall-clock | 45.83 min | 285.78 min | 46.37 min |
| 8 卡 GPU-hours | 6.11 | 38.10 | 6.18 |
| **有效 targets / GPU-hour** | **49.6M** | **53.0M** | **207.1M** |
| token-slot 吞吐上界（含 padding） | 354,794/s | 379,311/s | 638,745/s |
| GPU 利用率均值 / 中位数 | 98.74% / 99% | 98.70% / 99% | 99.64% / 100% |
| 单卡显存均值 / 峰值 | 3,505 / 3,612 MiB | 3,490 / 3,750 MiB | 15,606 / 16,676 MiB |
| checkpoint 大小 | 137,684,380 B | — | 137,685,299 B |

P03 用与 P01 相当的 6.18 GPU-hours 完成了 4.22× 的有效 targets；单位成本的有效 target 产出是 P01 的 4.18×、P02 的 3.90×。这里同时改变了 micro-batch 分解（4×8 → 32×1，显存 3.5GB → 15.6GB）和训练入口，因此吞吐提升不是纯数据效果；P01/P02 使用端到端 wall，P03 区分 active/训练循环/launcher wall，`638,745/s` 只能作为系统效率信号。P01/P02 的 optimizer updates 由 micro-steps ÷ 8 得到（39,695/8、264,651/8 均不整除，末个累积窗口不满），只有 P03 的 9,038 是 launcher 实测值。

## 5. 门控判定

P03 训练前预注册的验收门（见 P03 README）逐条判定：

| 门槛 | 实测 | 判定 |
|---|---|---|
| 技术完成：exit 0、9,038 updates、无 NaN/Inf/OOM/NCCL、strict export + SHA、validation 11,525 行/6.4M targets、七项 29,638 samples / 112,919 requests | 全部满足 | pass |
| 实验接受：共享 validation NLL ≤ P02 | 2.60432 vs 3.19096 | pass |
| 实验接受：七项 macro ≥ 30.91% | 31.52% | pass |
| 实验接受：任一任务相对 P02 回退 ≤ 2pp | 最大回退 OpenBookQA -0.60pp | pass |
| Release candidate：macro ≥ 31.44%（P01 水平） | 31.52% | pass（仅 +0.08pp，处于噪声量级） |
| Release candidate：5 条 greedy 至少 3 条非空且无明显重复退化 | 4 条非空，但 5 条全为重复退化 | **fail（非空数达标，无重复退化不达标）** |
| Release candidate：median GPU util ≥ 95% | 100% | pass |
| 参考项：token-slot 吞吐同口径对比 P02 | 口径不同，未按同一 wall 边界重算 | 未判定 |

结论：P03 通过“技术完成”和“实验接受”，**未通过 release candidate 的生成质量门**。因此 P03 可以作为下一阶段 SFT 的唯一正式 Base，但不得作为可发布的生成模型对外发布。

## 6. 失败与异常

- **P02 是本阶段的负实验**：6.67× 数据换来 -0.54pp 宏平均和 5/5 空续写。它没有技术故障，结论是“在 64M、1 epoch、seq 768 下单独扩大同分布官方数据无收益”，保留为历史负实验，不作废。
- **P02 attempt 1 失败**：8 个 rank 并发在默认 `/root/.cache/huggingface/datasets` 生成 Arrow cache，rank 7 报 `No space left on device`，未进入训练 step、无 checkpoint。修复为强制 cache 落 CPFS 并单进程预构建 cache；失败 run 作为异常证据保留在 SwanLab。该修复此后成为 P03 launcher 的标准前置步骤。
- **P01 的 OpenBookQA 高值未复现**：P01 29.00（比官方表高 5.40pp）→ P02 26.60 → P03 26.00。单 seed 下该任务波动明显，不应把 P01 的 OBQA 当作能力基准。
- **三次训练的生成退化**：见 §4.3，没有一次通过“无重复退化”。
- 评测环境侧异常（`typing_extensions` 版本、CMMLU loader 直连超时与镜像 429、`requests` 告警）均已在 P01/P02 报告中记录，不影响最终评分。

## 7. 结论边界

1. P01→P02 是干净的数据规模单变量（§3 证明两份数据同分布），可以说“扩大规模无收益”；
2. P01/P02→P03 **不是**单变量对比：数据组成、切块方式、去重与污染门禁、独立 validation、训练入口、micro-batch 分解同时改变，只能称为“新数据管线相对旧基线的整体收益”。要归因到 source mix、packing 或去重，需固定训练入口与 token budget 分别消融；
3. 全部结论基于单 seed（42），七项 benchmark 的 ±1pp 级差异不具备统计显著性；
4. 共享 validation 对 P01/P02 是回溯评测，且该 validation 与 v1 train 同源（同一 remix 管线的 held-out split），对 P03 是同分布验证、对 P01/P02 是跨分布验证——这一非对称性天然有利于 P03，NLL -0.96 不能全部归因为“P03 泛化更强”；
5. 吞吐与显存数字跨实验口径不同（wall 边界、micro-batch 分解），只作系统效率参考；
6. mini 未做污染扫描，full 的 115 个 containment 未逐条人工确认，near-duplicate 三份均为抽样。

## 8. 下一阶段前置条件

进入 Phase 2（SFT，先用官方数据）前必须完成：

1. ~~把 P02（含 `stash@{0}` 的数据审计）与 P03 的全部资产合入 main~~：已完成。2026-09-03 以两次 `--no-ff` 合并把 `stage5/p02-dense-pretrain-full` 与 `data/v1` 并入 main，`stash@{0}` 的 P02 数据审计（`audit_p02_data.py` + `eval/data_audit_{full.json,summary.md}` + `eval/validation_shared_p03.json`）单独落库，registry 两行的 `report_path` 现已可在 main 直接打开；
2. ~~补齐 P03 实验目录的标准追溯文件~~：已完成。顶层 `report.md`、`run.json`、`metrics.csv`、`swanlab-url.txt`（probe / formal / eval 三条）、`eval.json` 与 `checkpoint-manifest.txt` 均已生成，registry 状态已改为 `completed`；
3. ~~确认 P03 checkpoint 与 exported-base 的 SHA-256 记录在 checkpoint manifest 中~~：已完成。正式 checkpoint SHA-256 为 `0cfb7fc8fd9b3111f30b5528a1c8aacf8d6f633c8cde13c707c7cb44c83fd4fd`，HF export `model.safetensors` SHA-256 为 `fe6df50779ad9622bae2079dfeb3e52e0e2d74881edc7824903a754a8f8ddf07`；
4. 明确 SFT 从 P03 出发、数据先用官方 `sft_t2t_mini` / `sft_t2t`；如资源允许，用 8M targets 的 SFT smoke 同时从 P01 和 P03 初始化，验证“Base 选择”是否影响 SFT 结果——因为七项 macro 上 P03 仅比 P01 高 0.08pp，这个前提值得一次低成本核对（对应计划 7.3 的 S04B）；
5. 按计划 7.4 先审计官方 SFT 数据的 assistant targets、mask、重复与污染，不先构建自建数据；只有官方数据未过门槛且失败可归因到数据时，才在 Phase 2 内构建 SFT-v1（计划 7.6–7.8）。

遗留问题：

- v1 的收益无法归因到具体维度（待消融）；
- OpenBookQA 相对 P01 -3.00pp 未解释；
- Base 生成全部退化，是否会限制 SFT 后的生成质量，需要在 Phase 2 用同一评测口径回答。

## 9. 证据索引

| 类别 | 位置 |
|---|---|
| P01 全部资产 | `experiments/01-pretrain/P01-dense-pretrain-mini-20260824/`（main） |
| P02 报告与评测 | `experiments/01-pretrain/P02-dense-pretrain-full-20260824/`（main，来自合并 `stage5/p02-dense-pretrain-full`） |
| P02 数据审计补充 | 同目录 `audit_p02_data.py`、`eval/data_audit_full.json`、`eval/data_audit_summary.md`、`eval/validation_shared_p03.json`（原 `stash@{0}`，2026-09-03 落库） |
| P03 完整追溯包 | `experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/`（配置、命令、run、metrics、eval、报告、checkpoint manifest、SwanLab URL） |
| v1 数据审计 | `experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_remix_audit.json` |
| 共享 validation 三方回溯结果 | `.../P03-.../eval/validation-p0{1,2,3}-exact.json` |
| P03 系统指标 | `.../P03-.../eval/system-p03.json` |
| 训练权重与完整日志 | `/data/artifacts/minimind-lab/P0{1,2,3}-*/` |
| 本阶段 mini/full 同定义重算 | `/data/artifacts/minimind-lab/phase1-aux/` |
| SwanLab | 见 §2.1 与 [`swanlab-runs.md`](swanlab-runs.md) |

## 修订记录

| 日期 | 修改内容 | 原因 |
|---|---|---|
| 2026-09-03 | 首版：三次 pretrain 与三份数据的横向对比 | Phase 1 收口 |
| 2026-09-03 | §2.1 修正 P01 的 run（`MiniMind-Lab/nfax3tyg0j217j1cz8y0b`，旧 URL 已作废）；§8 第 1、2 条改为反映 registry 已补 P02/P03 行、P03 仍缺四份追溯文件 | main 之前未同步 stage5 收口的 SwanLab project 合并，且 registry 缺 P02/P03 行 |
| 2026-09-03 | P02/P03 资产合入 main 后同步状态行、§2.1 表、§8 第 1/2 条与 §9 证据索引，把「Git ref / stash」改为 main 内路径 | `stage5/p02-dense-pretrain-full`、`data/v1` 与 `stash@{0}` 已合入 main（合并 commit `6e1c67e`、`90a5c12`，审计落库 `c58a50f`） |
| 2026-09-03 | P03 追溯文件、三条 SwanLab URL、评测索引及 checkpoint/export SHA 全部补齐；统一校验通过后完成 Phase 1 内容验收 | P03 已从 `awaiting-report` 收口为 `completed / accepted`，但固定续写重复门仍失败 |
