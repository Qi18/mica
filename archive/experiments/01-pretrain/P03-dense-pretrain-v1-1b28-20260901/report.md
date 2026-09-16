# P03-dense-pretrain-v1-1b28-20260901

## 1. 目标与结论边界

P03 使用通过独立验收的 `pretrain-v1-1b28/final-remix-v1`，在 8×NVIDIA L20 上从随机权重训练 63,912,192 参数 Dense Base 模型，建立新的质量—成本基线。

P03 不是 P02 的严格单变量 A/B：数据来源与配比、tokenizer 对齐切块、独立 validation、训练入口和 micro-batch 分解均发生变化。因此本报告只能把收益归因于“P03 整体数据与训练管线”，不能归因到某一个数据源、packing 或超参数。

## 2. 冻结输入

| 项目 | P03 配置 |
|---|---|
| 模型 | Dense，63,912,192 参数，hidden 768，8 layers，vocab 6400 |
| 初始化 | random |
| 数据 | `pretrain-v1-1b28/final-remix-v1` |
| dataset fingerprint | `cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd` |
| 数据 manifest SHA-256 | `1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6` |
| train | 40 shards，2,313,483 rows，1,280,000,000 loss targets |
| validation | 1 shard，11,525 rows，6,400,000 loss targets |
| 训练 | 1 epoch，seq 768，global sequence batch 256 |
| 并行 | 8 ranks，每卡 batch 32，accumulation 1 |
| 优化 | AdamW，cosine LR `5e-4 → 5e-5`，grad clip 1.0 |
| dtype / seed | bfloat16 / 42 |
| 训练期 Lab commit | `222e39c934d7bb4d96472ceb40d22f8aa7fc578b` |
| MiniMind commit | `393e387e9ad99f0f04c296e4c5e7353f4444629f` |

数据来源配比为 ChineseWebText2 45%、FineWeb-Edu 30%、FineMath 10%、Wikipedia zh/en 各 5%、Stack v3 permissive code 5%。P03 使用 tokenizer 对齐切块，1.280B loss targets 对应约 1.777B padded slots，target 利用率 72.04%；training-visible exact duplicate、与 validation exact overlap、benchmark exact/containment 均为 0。near duplicate 只在 20,480 行确定性抽样中为 0，不能外推为全量 0。

## 3. 执行过程

100-step probe 使用首选 `B32×8/A1` profile，正常退出并留下独立 checkpoint、resume state、metrics 和 attempt 证据。正式训练随后复用相同 profile，从随机权重完成完整 1 epoch：

- attempt：`20260901T080746Z-542788`；
- 开始：2026-09-01 08:07:46 UTC；
- 结束：2026-09-01 08:54:08 UTC；
- exit code：0；
- optimizer steps：9,038 / 9,038；
- 无 NaN、Inf、OOM 或 NCCL failure；
- final checkpoint、best-validation checkpoint、resume state 和 HF export 均存在。

训练启动时仓库工作区包含未提交的数据管线、trainer 和评测文件，因此训练期 `lab_commit` 不能单独复现本次执行。runtime manifest 额外冻结了 `trainer_sha256=b88fb545...e6fc`、`config/protocol_sha256=b0c80378...579`、`command_sha256=51608dd4...1219`，这是本次运行的重要复现边界。相关代码后来已进入仓库，但报告不把历史 dirty worktree 描述成 clean-commit 运行。

## 4. 训练与系统结果

| 指标 | 结果 |
|---|---:|
| final train loss | 2.5997 |
| 最后 100 个日志点 loss 均值 | 2.6080 |
| final validation loss（训练循环） | 2.60431 |
| final validation PPL（训练循环） | 13.5219 |
| active training | 2,613.17 s |
| training-loop wall | 2,750.29 s |
| launcher wall | 2,787.12 s（46.45 min） |
| 8 卡合计 | 6.19 GPU-hours |
| active padded throughput | 679,925 slots/s |
| training-loop padded throughput | 646,027 slots/s |
| GPU 利用率 | mean 99.64%，median 100% |
| 单卡显存 | mean 15,605.9 MiB，peak 16,676 MiB |
| checkpoint | 137,685,299 bytes |

吞吐的分母是包含 padding 的固定 token slots，不是有效 targets/s。与 P02 的吞吐数字采用不同 wall-clock 边界，只能作为系统效率参考，不能宣称严格同协议提速。

## 5. 统一评测

### 5.1 共享 validation

独立复评使用同一 11,525 rows / 6.4M targets validation、MiniMind shifted next-token CE、BF16、单张 L20：

| Checkpoint | NLL | PPL |
|---|---:|---:|
| P01 | 3.56414 | 35.3092 |
| P02 | 3.19096 | 24.3117 |
| P03 | **2.60432** | **13.5221** |

P03 相对 P02 的 NLL 下降 0.58664（18.4%），PPL 下降 44.4%。P03 只使用 P02 的 63.3% 有效 train targets，却取得更低共享 validation NLL，支持 P03 整体数据管线具有更高的有效数据利用率和 token-level 泛化能力。

### 5.2 七项 Base 0-shot

协议：`lm-evaluation-harness 0.4.12`、commit `6d642546f4688648fced259eb3302efd36ece5af`、0-shot、无 chat template、batch 16、seed 42、单张 L20；共 29,638 samples / 112,919 requests。

| 任务 | P02 | P03 | P03-P02（pp） |
|---|---:|---:|---:|
| C-Eval | 23.03 | 23.18 | +0.15 |
| CMMLU | 25.13 | 25.63 | +0.51 |
| ARC-Easy | 28.91 | 31.65 | +2.74 |
| PIQA | 51.47 | 51.90 | +0.44 |
| OpenBookQA | 26.60 | 26.00 | -0.60 |
| HellaSwag | 28.40 | 28.00 | -0.40 |
| Social IQA | 32.80 | 34.29 | +1.48 |
| 七项宏平均 | 30.91 | **31.52** | **+0.62** |

P03 在 5/7 个任务上提升，并满足预注册的 macro ≥30.91% 与任一任务相对 P02 回退不超过 2pp。OpenBookQA 和 HellaSwag 仍分别回退 0.60/0.40pp，不能只展示平均值而隐藏单项退化。

### 5.3 固定续写与推理

P02 的 5 个 greedy prompt 全部首 token EOS；P03 为 4/5 非空、0 个立即 EOS，改善了可生成性。但中文、英文、算术、科学和代码样例仍出现词语、符号或空白重复，因此没有通过“无明显重复退化”的 release-candidate 门。

单卡固定性能测试中，32-token generation median 为 128.65ms、248.73 token/s；TTFT median 为 4.81ms。该测试只描述固定输入下的系统表现，不代表真实服务容量。

## 6. 验收结论

| 层级 | 判定 | 依据 |
|---|---|---|
| 技术完成 | **通过** | exit 0、9,038 steps、checkpoint/export/validation/七项完整 |
| 实验接受 | **通过** | validation 优于 P02；macro 31.52%；最大单项回退 <2pp |
| Release candidate | **未通过** | 固定续写虽不再立即 EOS，但仍有明显重复 |

P03 的正式状态是 `completed` 且 `accepted`，可作为下一阶段 SFT 的首选 Base checkpoint；它不是可直接发布为可用生成模型的 release candidate。

## 7. 产物与证据

- SwanLab：[P03-Pretrain-V1-1B28-Full-64M-Seq768-B32x8-A1](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/qdpjh47fjt98184oos4bl)；
- 正式 checkpoint：`/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/formal-b32-a1-epoch1/checkpoints/p03_formal_b32_a1_768.pth`；
- checkpoint SHA-256：`0cfb7fc8fd9b3111f30b5528a1c8aacf8d6f633c8cde13c707c7cb44c83fd4fd`；
- HF export SHA-256：`fe6df50779ad9622bae2079dfeb3e52e0e2d74881edc7824903a754a8f8ddf07`；
- 机器可读运行摘要：[`run.json`](run.json)；
- 紧凑指标：[`metrics.csv`](metrics.csv)；
- checkpoint 清单：[`checkpoint-manifest.txt`](checkpoint-manifest.txt)；
- 完整产物清单：[`artifacts_manifest.json`](artifacts_manifest.json)；
- 评测原始结果：[`../../../../experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/eval`](../../../../experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/eval)。

权重和完整日志只保存在 CPFS，不提交 Git；Git 保存路径、大小、SHA、配置、紧凑指标和评测证据。checkpoint 尚未发布到 ModelScope/Hugging Face，因此 registry 只登记 CPFS 位置，不伪造公开下载地址。

## 8. 最终结论

P03 用更少的有效 targets 和显著更少的 optimizer updates，同时改善了共享 validation、七项宏平均和立即 EOS 问题，证明它比 P02 更适合作为后续 SFT Base。但 P03 同时改变了多项数据和训练管线因素，当前不能给出单因素因果归因；固定续写仍严重复读，也不能把 Base benchmark 提升解释为 Chat 能力已经可用。
