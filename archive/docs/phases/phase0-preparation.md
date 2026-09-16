# Phase 0 阶段报告：环境、数据与代码探针（E00 / E01 / E02）

- 阶段范围：`experiment_plan.md` 第 5 节
- 报告状态：draft（证据缺口已修；本报告与修复改动已于 2026-09-03 合入 main，计划 16.4 的硬门已过，见 §8）
- 出具日期：2026-09-03
- Lab commit：`90a5c12`（main，本报告描述的 E00–E02 资产已全部落库）；MiniMind upstream：`393e387e9ad99f0f04c296e4c5e7353f4444629f`
- 数据 revision：`jingyaogong/minimind_dataset@312afb4f76391145c6902f765bb51691c09a12f5`

## 1. 阶段目标与研究问题

Phase 0 只回答一个问题：**能否可信地开始训练**。它拆成三条互相独立的验收：

1. **运行时可信**（E00）：8×L20、BF16、NCCL、存储、SwanLab 与启动保护是否可用且可复现；
2. **数据可解释**（E01）：官方数据的身份（版本/SHA/行数）是否固定，Tokenizer 与各 Dataset 从原始 JSON 到 `input_ids`/`labels`/mask 的语义是否能被证明；
3. **模型与训练路径正确**（E02）：Dense/MoE 能否从 token 解释到 logits/loss，8 卡 BF16 能否连续跑完 100 optimizer steps，断点恢复后 step/optimizer/LR/SwanLab 是否连续。

本阶段明确不回答：模型质量、收敛性、数据质量优劣、Dense 与 MoE 的效率结论。所有 loss 与吞吐数字只用于证明"跑得对"，不用于证明"训得好"。

## 2. 实验清单

三个实验在 `experiments/registry.csv` 中均已登记为 `completed`，资产已在 main。

| experiment_id | status | registry 行 | Lab commit | 报告 |
|---|---|---|---|---|
| E00-l20-baseline-20260823 | completed | 第 2 行 | `9ca9fe21899fc5e0a4b4b425cfecba03eca4161b` | `experiments/00-preparation/E00-l20-baseline-20260823/report.md` |
| E01-tokenizer-dataset-20260823 | completed | 第 3 行 | `080179c9d1ec2a0752e2a21155af899e7509c5b9` | `experiments/00-preparation/E01-tokenizer-dataset-20260823/report.md` |
| E02-model-probe-20260823 | completed | 第 4 行 | `5ccb66702c2f1f1b969c79240fd5fe5e247131cd` | `experiments/00-preparation/E02-model-probe-20260823/report.md` |

时间与成本：E00 `04:10:00Z → 05:18:03Z`（1h08m，含人工核查）；E01 `08:36:21Z → 13:40:24Z`（5h04m，含数据下载与脚本两次修复复跑）；E02 `16:20:07Z → 16:22:52Z`（2m45s，8 卡 ≈ 0.37 GPU-hours）。均为 2026-08-23，`exit_code` 全为 0。

### 2.1 SwanLab run

Workspace `@richliu0153`；跨阶段索引见 [`swanlab-runs.md`](swanlab-runs.md)。

| experiment_id | run 角色 | project | run id | URL |
|---|---|---|---|---|
| E00 | 无云端 run | — | — | 本阶段只验环境不训练，不产生 run；无本地 `swanlog/`。`swanlab-url.txt` 现记录登录时间线（04:22:35Z blocked → 05:18:03Z pass）与首批云端 run 位置 |
| E01 | 不适用 | — | — | 本阶段不训练模型，不产生 run |
| E02 | seed 42（含 step50→100 resume，同一 run） | MiniMind-Lab | `iq14wfm1nc1ca8iigdbop` | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/iq14wfm1nc1ca8iigdbop |
| E02 | seed 43 | MiniMind-Lab | `is8yx09hw3341ar8qvvfa` | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/is8yx09hw3341ar8qvvfa |
| E02 | seed 44 | MiniMind-Lab | `kngkosspce6sfuzkoztzm` | https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/kngkosspce6sfuzkoztzm |

seed42 的 resume 没有产生新 run：step1–50 与 step51–100 写入同一个 `iq14wfm1nc1ca8iigdbop`，这正是本阶段要验证的"曲线可拼接"。registry 中 E00/E01 的 `swanlab_url` 为空，按计划 16.5 应显式写"无云端 run"或"不适用"。

## 3. 关键配置与数据 fingerprint

### 3.1 运行时快照（E00 `environment.json`，记录于 2026-08-23T04:23:00Z）

| 项 | 值 |
|---|---|
| GPU | 8×NVIDIA L20，单卡 46,068 MiB，driver 535.161.08，核查时 8 卡全空闲 |
| Python / PyTorch / CUDA | 3.10.12 / 2.6.0+cu124 / 12.4（cuDNN 90100，NCCL 2.21.5） |
| 库 | transformers 4.57.6、datasets 3.6.0、tokenizers 0.22.2、swanlab 0.7.11 |
| 存储 | `/data` NFS 读写，容量 3.6T（已用 9.5G）；`/dev/shm` 858,993,459,200 B（800 GiB） |
| 环境路径 | `/data/venvs/minimind-lab`（MiniMind 官方 requirements pin） |
| attention backend | `flash_sdpa_enabled: true`（E02 `results/architecture.json`） |
| 已知告警 | 系统 `litellm` 要求 `openai>=1.68.2`，MiniMind pin `openai==1.59.6`；训练入口不导入 litellm，保留告警不偏离官方 pin |

### 3.2 数据身份（E01 `data_manifest.json`，6 个文件全部匹配固定 revision 的大小与 SHA-256）

| 文件 | 角色 | 字节 | 行数 | 坏 JSON | SHA-256 前 12 位 |
|---|---|---:|---:|---:|---|
| `pretrain_t2t_mini.jsonl` | mini pretrain | 1,241,043,656 | 1,270,238 | 0 | `6dd6716c84ab` |
| `sft_t2t_mini.jsonl` | mini SFT | 1,739,201,170 | 905,718 | 0 | `abb1e76b2056` |
| `dpo.jsonl` | preference | 53,653,322 | 17,166 | 0 | `ee934a8a455c` |
| `rlaif.jsonl` | RLAIF | 23,754,740 | 19,502 | 0 | `8c6634db971f` |
| `agent_rl.jsonl` | agentic RL | 82,036,930 | 39,988 | 0 | `cb96bcc8096a` |
| `agent_rl_math.jsonl` | agentic RL math | 18,372,683 | 20,000 | 0 | `5f2f6f8e90d8` |

Tokenizer 固定为仓库自带 `minimind/model`：vocab 6400，BOS `<|im_start|>`、EOS `<|im_end|>`、PAD `<|endoftext|>`，本阶段不训练也不替换。官方数据存放 `/data/datasets`，不进 Git。

E02 探针数据：源自上表 `pretrain_t2t_mini.jsonl` 前 8,192 条，`max_length` 128，产物 `/data/artifacts/minimind-lab/E02-model-probe-20260823/probe-data.pt`（8,390,460 B，SHA-256 `287f5352ff98…`），有效 target tokens 952,428。

### 3.3 探针训练口径（E02 `config.json`）

Dense、hidden 768、8 层、BF16、world size 8、每卡 batch 4、seq 128、lr 5e-4、100 optimizer steps、seeds 42/43/44；seed42 额外做 step1–50 / step51–100 两段。该口径刻意用预分词数据隔离 dataloader 开销，与 Phase 1 正式训练（seq 768、全局 sequence batch 256）不可比。

## 4. 结果

### 4.1 环境与运行时门禁（E00）

11 项检查全部通过：ACK `/readyz` HTTP 200、Pod 1/1 Running 0 restart、真实 OpenSSH 握手、8 卡空闲、CUDA matmul、`is_bf16_supported()`、8 rank all-reduce 结果 36（期望值 36）、CPFS 读写、`/dev/shm` 800 GiB、SwanLab（见 §5 的记录冲突）。

随机初始化前向 smoke（BF16 CUDA，`[1,16]` 输入）：

| 模型 | 参数量 | logits | finite | peak allocated |
|---|---:|---|---|---:|
| Dense | 63,912,192 | [1, 16, 6400] | 是 | 149.35 MiB |
| MoE | 198,416,640 | [1, 16, 6400] | 是 | 406.41 MiB |

新增启动基础设施：`preflight_l20.py`（GPU 空闲/CUDA/BF16/CPFS/`/dev/shm`/SwanLab 登录）、`run_guarded.py`（单实例文件锁、命令回显、GPU 二次门禁、信号转发）、`nccl_smoke.py`、`init_experiment.py`、`validate_experiment.py`。这五个脚本仍在 `scripts/` 中，是后续所有阶段 `command.sh` 的第一行。

### 4.2 模型结构（E02 `results/architecture.json`）

| 分组 | Dense | MoE |
|---|---:|---:|
| embedding + lm_head | 4,915,200 | 4,915,200 |
| attention | 14,157,312 | 14,157,312 |
| norm | 13,056 | 13,056 |
| mlp | 44,826,624 | 179,331,072 |
| **总参数** | **63,912,192** | **198,416,640** |
| 每 token 激活参数（估算） | 63,912,192 | 63,936,768 |

- `embedding + lm_head` 恰为 6400 × 768 = 4,915,200，说明输入 embedding 与 LM head 共享权重；
- MoE 的 mlp 满足 4 × 44,826,624 + 24,576 = 179,331,072，即 4 个 expert 加 24,576 个路由/门控参数；总参数是 Dense 的 3.1045×，而 top-1 每 token 激活参数只多出同样的 24,576（1.00038×）；
- 前向形状链路完整记录：`input_ids [2,16]` → 各层 `[2,16,768]` → `lm_head [2,16,6400]`；
- KV cache 从 prefix `[2,8,4,96]` 增长到 `[2,9,4,96]`（8 层、4 个 KV head、head_dim 96），continued logits `[2,1,6400]`，证明 GQA 与增量解码路径正确。

### 4.3 三 seed 100-step 探针（E02 `results/summary*.json`，数字对应 §2.1 的三个 run）

| seed | 初始 loss | 最后一步 loss | 全程最低 loss | 有效 tokens/s | samples/s | 平均 GPU util | peak allocated (rank0) | nvidia-smi peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8.8794 | 6.6754 | 6.4980 | 50,641 | 437.2 | 89.0% | 1,392.96 MiB | 1,944 MiB |
| 43 | 8.8939 | 6.5579 | 6.4978 | 50,647 | 435.6 | 90.3% | 1,392.96 MiB | 1,956 MiB |
| 44 | 8.9409 | 6.5112 | 6.5112 | 50,521 | 433.7 | 90.1% | 1,392.96 MiB | 1,956 MiB |

派生结论：

- 初始 loss 均值 8.9047，ln(6400) = 8.7641，高出 0.14 nats——随机初始化接近但不等于均匀分布，符合预期，没有 mask 或 label 错位的迹象；
- **不能用最后一步 loss 比较 seed**：按最后一步，三 seed 极差 0.164 nats（均值 6.5815、标准差 0.0846）；按全程最低 loss，极差只有 0.0134（6.4978–6.5112）。seed42 的 step99→100 单步就从 6.5229 跳到 6.6754，说明 0.16 的极差主要来自单步 batch 噪声而不是 seed 差异，100-step 级别的 loss 差异无论如何都不能用于质量结论；
- LR 调度可解析校对：step1 4.9989e-4、step50 正好 2.75e-4、step100 5e-5，与 cosine 从 5e-4 降到 floor `lr/10 = 5e-5` 的解析值一致（5e-5 + 0.5×4.5e-4×(1+cosπ·step/100)）；
- 梯度全程 finite，seed42/43 的 grad norm 最大 4.15/4.34；**seed44 出现单点 13.77 的尖峰**（其余步均 ≤ ~4），未造成 NaN 或中断，但说明短探针也不保证梯度平稳；
- 吞吐 seed 间标准差 71 tokens/s，CV 0.14%，说明运行时本身稳定；
- `tokens/s ÷ samples/s = 116.19`，与探针数据的 952,428 ÷ 8,192 = 116.26 targets/样本一致，确认 tokens/s 统计的是有效 target 而非 padded slot（逐步日志的 `valid_tokens` 也仅 3,555–3,741 / step，即全局 32 条序列 × ≈116）；对应 slot 利用率 116.26/128 = 90.8%；
- 100 步 × 8 卡 × 4 = 3,200 条序列，约 372K target tokens，仅覆盖 8,192 条探针数据的 39%，是一次"不到一个 epoch"的运行时验证；
- 该吞吐不可外推：Phase 1 的 P01 在 seq 768 正式口径下实测 303,018,028 targets / 45.83 min ≈ 110,196 有效 targets/s，是探针的 2.18×。

Resume 证据（seed42）：从 step50 checkpoint 恢复到 step100，optimizer state 含 90 个参数条目，step50→51 的 LR `2.75e-4 → 2.679e-4`、loss `6.9928 → 6.8876`，`metrics-seed42.jsonl` 中后 50 条均带 `phase_resumed` 标记，run id 全程 `iq14wfm1nc1ca8iigdbop`。保留的 checkpoint：`/data/artifacts/minimind-lab/E02-model-probe-20260823/checkpoints/seed42-resume.pt`，SHA-256 `9036139bc52f…`，767,063,342 B。seed43/44 不保存冗余 checkpoint。

### 4.4 数据观察（E01）与 Phase 1 全量精算的交叉验证

E01 对每个文件固定 seed 42 抽样 2,000 条，统一按 768 观察：

| 数据 | 平均 tokens | P50 | P90 | P99 | max | 超 768 比例 | 平均 padding |
|---|---:|---:|---:|---:|---:|---:|---:|
| pretrain mini | 261.9 | 213 | 408 | 1,420 | 2,053 | 4.20% | 529.7 |
| SFT mini | 493.3 | 491 | 719 | 933 | 1,292 | 6.55% | 280.7 |
| DPO | 472.5 | 449 | 837 | 1,069 | 1,299 | 15.75% | 314.6 |
| RLAIF | 317.3 | 350 | 557 | 731 | 771 | 0.05% | 450.7 |
| Agent RL | 611.0 | 545 | 1,177 | 1,600 | 1,969 | 29.30% | 255.6 |
| Agent RL Math | 408.0 | 412 | 443 | 455 | 464 | 0% | 360.0 |

长样本风险集中在 Agent RL（29.3% 超 768）与 DPO（15.75%），是 Phase 6/7 必须先处理的截断风险。

把 pretrain mini 的抽样结果与 Phase 1 对同一文件的**全量**精算（`/data/artifacts/minimind-lab/phase1-aux/pretrain_mini_target_audit.json`，1,270,238 行逐行计量）对齐：

| 指标 | E01 抽样 2,000 条 | Phase 1 全量精算 | 相对偏差 |
|---|---:|---:|---:|
| 平均 raw tokens/行 | 261.888 | 259.758 | +0.82% |
| 超 768 比例 | 4.20% | 3.9228% | +7.07% |
| 非 pad slot/行 | 238.33（768 − 529.666） | 239.552 | −0.51% |

结论：2,000 条抽样对**均值类**指标可信到 1% 以内，但对**尾部比率**有 7% 量级的相对偏差。因此规模与预算类结论（有效 target 总量、截断损失、去重率）必须像 Phase 1 那样做全量精算，抽样只用于早期风险识别。

Tokenizer 压缩率（六类文本各 20 条，启发式分类）：中文 1.389、英文 2.898、代码 1.922、数学 1.569、新闻 1.432、领域 1.627 字符/token。

Dataset 语义（`results/sample_transformations.json` + 合成 fixture）：Pretrain 为 BOS/text/EOS 后 padding、全部非 pad token 参与 loss；SFT 仅 assistant span 参与 loss（fixture 33 valid / 95 ignored）；DPO 对 chosen/rejected 分别 shift，assistant mask 20/15；RLAIF 把最后答案移出 prompt 返回待 rollout 的 generation prompt；AgentRL 返回 `messages[:-1]`、tools 与 gt，不在 Dataset 内 tokenize。

## 5. 门控判定

对计划 5.1–5.3 的要求逐条判定：

| 预注册要求（计划位置） | 实测 | 判定 |
|---|---|---|
| 8 卡 L20 / CUDA / BF16（5.1） | 8×46,068 MiB、torch 2.6.0+cu124、`is_bf16_supported()` true | pass |
| NCCL all-reduce（5.1） | 8 rank 全部得到 36 | pass |
| CPFS / `/dev/shm` / 网络（5.1） | NFS 读写 3.6T、800 GiB、ACK `/readyz` 200、真实 SSH | pass |
| SwanLab 登录且凭据不入库（5.1） | E00 登录时间线已回填（04:22:35Z blocked → 05:18:03Z pass），E02 三个云端 run 落地，凭据未进仓库/日志/CPFS | pass |
| launcher 具备 GPU 空闲门、单实例锁、resume 验证（5.1） | `preflight_l20.py` + `run_guarded.py` + seed42 resume 实测 | pass |
| launcher SIGTERM 安全退出（5.1） | 仅保证信号转发到子进程组、不留孤儿；MiniMind 源码无 SIGTERM 即时落盘，恢复点仍是最近一次 `save_interval` | partial（已知限制，未改上游） |
| 记录 PyTorch/CUDA/Transformers/attention backend/commit（5.1） | `environment.json` + `architecture.json`（`flash_sdpa: true`）+ registry 三列 commit | pass |
| 原始 JSON → template → input_ids → labels 可解释（5.2） | 五类 Dataset 全部给出实测 mask 与形状 | pass |
| 按中英/代码/数学统计压缩率、截断、padding、有效 target（5.2） | 压缩率与截断/padding 完成；**有效 target 仅到 fixture 与抽样级**，官方 mini 的全量有效 target 直到 Phase 1 才精算 | partial |
| 固定官方 tokenizer，不更换（5.2） | vocab 6400 全程未变，`train_tokenizer.py` 只作阅读 | pass |
| 参数量拆解与 Tensor shape（5.3） | 四组参数拆解 + 逐层形状 + KV cache 增长 | pass |
| 单卡 BF16 forward/backward（5.3） | Dense/MoE loss 与梯度均 finite | pass |
| 8 卡 100-step probe 无数值/分布式错误（5.3） | 三 seed 均 100/100 步完成，loss 与 grad norm 全程 finite，无 NaN/Inf/OOM/NCCL 故障（seed44 有单点 grad norm 13.77 尖峰，不影响完成） | pass |
| 一次断点恢复且 step/optimizer/loss/SwanLab 连续（5.3） | 90 条 optimizer state、LR 与 loss 连续、run id 不变 | pass |
| 100-step probe 尽量 3 seed（4.2） | seeds 42/43/44 | pass |

阶段结论：Phase 0 通过，两项 partial 均已记录且不阻塞 Phase 1（SIGTERM 即时落盘是上游限制，有效 target 精算已在 Phase 1 补齐）。

## 6. 失败、异常与作废实验

本阶段没有 invalidated 实验，三个实验均为 completed。过程中的异常与处理：

| 现象 | 处理 | 证据 |
|---|---|---|
| 本地 `kubectl v1.35.3` 间歇 Go socket `bad file descriptor` | 直连 `/readyz` 始终 200；编排临时改用官方 SHA-256 校验过的 `v1.34.11`，未改 ACL | E02 report「异常与边界」 |
| SSH ProxyCommand 的 socket stdin 触发误导性 6443 报错 | 在 Skill 中把 stdin 规范化为管道 | E01 report「异常与修复」1–2 |
| L20 无法直连 HuggingFace | 改用 `hf-mirror`，最终以官方 size + SHA-256 校验文件身份 | `command.sh` 的 `HF_ENDPOINT` |
| 第一版观察器漏了字符串形式 `tool_calls` 的反序列化，SFT 只统计到 1,823/2,000 条 | 按官方 `SFTDataset` 语义修正后 2,000 条全部入统计 | E01 report 3 |
| 新增文件补丁行数错误导致脚本被截断 | 增加 `summary.json` 非空验收后完整复跑 | E01 report 4 |
| NCCL 未找到外部 tuner plugin | 明确回退 internal tuner，8 rank 正常完成 | E02 report |
| SwanLab 提示新版本 | 保持已验收的 0.7.11，不在实验中升级依赖 | E02 report |
| 历史 Official Zero 资产盘点 | 在权威仓库、`/data/cache`、`/data/projects`、CPFS `.snapshots` 中均未找到数据、权重、日志、run id 或 snapshot；结论只能写"曾运行并恢复"，不能写已完成 | E00 report「Official Zero 历史资产」 |

**记录冲突（2026-09-03 已修正）**：E00 原先的 `metrics.csv`（`swanlab_login,0,bool,blocked`）、`environment.json`（`"swanlab_login": "blocked"`）与 `swanlab-url.txt`（`# Pending: ... not logged in`）时间戳均在 04:22–04:23，而 `report.md` 验收表写“SwanLab 通过”并称 05:18:03Z 复跑 preflight 为 pass，两者互相矛盾。修正方式是把两个时间点同时记下而不重写历史：`metrics.csv` 拆为 `swanlab_login_first_check,0,…,blocked,04:22:35Z` 与 `swanlab_login,1,…,pass,05:18:03Z`；`environment.json` 保留 04:23 快照的 `blocked` 并新增 `smoke_tests.swanlab_login_recheck`；`swanlab-url.txt` 改为“本阶段无云端 run + 登录时间线 + 首批云端 run（E02 三 seed）”；`report.md` 补一段时间线说明。registry 的 E00/E01 `swanlab_url` 从空值改为 `n/a-no-cloud-run` / `n/a-no-training`。三个实验目录改后 `validate_experiment.py` 均 `validation=pass`。

## 7. 结论边界

- Phase 0 的全部结论都是运行时正确性，**没有任何模型质量含义**。100 步、seq 128、预分词、3,200 条序列的 loss 与吞吐都不可外推（对照：P01 正式口径的有效吞吐是探针的 2.18×）。
- 三 seed 在 100 步内的最后一步 loss 极差达 0.164 nats，而全程最低 loss 极差只有 0.0134；两者差近一个数量级，说明短探针的 loss 数字对“取哪一步”极度敏感，本身就不能做质量比较。
- E01 的长度/截断/padding 统计基于每文件 2,000 条抽样；文本分类为启发式规则，不是官方标签。与 Phase 1 全量精算比较后已知：均值类偏差 <1%，尾部比率偏差可达 7%。
- E01 的 `data_manifest.json` 只覆盖 6 个文件。官方全量 `pretrain_t2t.jsonl`（8,275,074,893 B，2026-08-24 下载，P02 使用）不在其中，其身份记录在 P02 资产而非 Phase 0，因此"官方数据身份已固定"这句话严格只对上述 6 个文件成立。
- MoE 在本阶段只做了结构拆解与单卡 forward/backward，没有训练、吞吐或质量结论；上游"原生 PyTorch MoE 约慢 50%"未验证，留给 Phase 7。
- 环境记录是 2026-08-23 的快照。到 2026-09-03，`/data/venvs/minimind-lab` 在当前会话环境中已不存在，Phase 1 的补充审计改用系统 `python3` + transformers 4.51.1，与本阶段记录的 4.57.6 不同版本——跨阶段复算必须各自标注版本，不能默认"同一环境"。
- Phase 0 没有固定 SwanLab project 命名口径，导致 P01 最初落在 `MiniMind-Lab-Stage3`、P02 落在 `MiniMind-Lab-Stage5`，直到 2026-09-01 才被手工同步到统一 project `MiniMind-Lab`（stage5 收口 commit `5761979`）。代价是旧 run id 作废、早期文档与 registry 全部需返工回填（见 Phase 1 报告 §2.1）；阶段开始前就定下 tracking 命名口径是一条应当写进 Phase 0 的验收项。
- 公开记录不保存公网 IP、ACL、集群 ID、账号或凭据；E00 中与网络白名单相关的细节被刻意省略，这部分不可复现由安全约束造成，不是证据缺失。

## 8. 下一阶段前置条件与未解决问题

Phase 0 报告转 accepted 前必须完成（前四项已于 2026-09-03 完成）：

1. ~~**回填 E00 的 SwanLab 证据**~~：已完成。`metrics.csv` / `environment.json` / `swanlab-url.txt` / `report.md` 现均记录“04:22:35Z 未登录 → 05:18:03Z 复检 pass，首批云端 run 在 E02”；registry 的 E00/E01 `swanlab_url` 已按计划 16.5 填写而非留空（§6）。
2. ~~**提交已修改但未落库的文件**~~：已完成。`experiments/00-preparation/E01-…/eval.json`、`E02-…/eval.json` 与 `scripts/eval/summarize_model_probe.py` 已把 `protocol` 从 `docs/training_plan.md#…` 改指 `docs/experiment_plan.md`，连同 E00 artifact 与 registry 的修正一起落库（commit `dea2824`）。
3. ~~**修复源码阅读笔记的引用断裂**~~：已完成。`docs/source_reading/`（README + 00/01/02）与 `docs/upstream-minimind.md` 已从 HEAD 恢复，因为它们是计划 §17 要求的阶段产物且被 README、`repository-management.md`、E01/E02 报告引用；只保留删除 `docs/training_plan.md`（已被 `experiment_plan.md` 取代，全仓无剩余引用）。同时把 `docs/source_reading/README.md` 的序列补上遗漏的 `00-training-runtime.md`，并在仓库 README 加上阶段报告索引链接。
4. ~~**合入 main**~~：已完成。`docs/experiment_plan.md` 与整个 `docs/phases/`（含本报告）已于 commit `2584c5c` 落库并推送 origin/main，满足计划 16.4。

留给后续阶段的未解决问题：

- **D01–D05 数据准备实验未登记**（2026-09-03 新暴露）：合并 `data/v1` 后，`experiments/00-preparation/` 多出 D01（Pretrain v1 数据，`accepted`）与 D02–D05（SFT v1 物化与容量剖析，均为 not-trainable 中间态）五个目录，但 `experiments/registry.csv` 和本目录 README 的表都只有 E00–E02。这五个目录的自有状态词（`accepted`、`implementation_only`、`capacity_profiled_not_trainable`、`configured_not_materialized_not_profiled`）与 `experiments/README.md` 的 status 词表不同体系，补登记前需先定下“数据工程实验是否进 registry、进则用哪套状态词”，不宜先写行造成 schema 漂移。
- **Phase 2 需要的官方 SFT 事实 Phase 0 没有给**：assistant target 总量、mask 正确率、跨 split 重复与固定 benchmark 污染都没做全量审计，只有 fixture 级的 33 valid / 95 ignored 与 2,000 条长度抽样。计划 7.4 的审计必须补做。
- **官方全量 SFT 数据当前不可得**：`/data/datasets/minimind/312afb4f…/` 只有 `sft_t2t_mini.jsonl`（905,718 行），没有 `sft_t2t.jsonl`。计划 7.3 的 S04「官方全量 32M assistant targets」要先确认该文件能否下载并校验；否则 32M 预算只能从 mini 中抽取，且必须在 Phase 2 报告里改写口径。
- **Agent RL / DPO 的长样本截断风险**已量化（29.3% / 15.75% 超 768）但未处理，Phase 5/6 启动前需要决定截断策略或提高 `max_seq_len`。
- **SIGTERM 即时 checkpoint 缺失**：目前最坏情况会丢失一个 `save_interval` 区间的进度，长训练前可考虑在 launcher 侧补 handler，属上游改动需单独评审。

## 9. 证据索引

- Git：Lab commit `90a5c12`（main，已合入 `stage5/p02-dense-pretrain-full` 与 `data/v1`）；本报告与计划拆分落库于 `2584c5c`，E00 证据修复于 `dea2824`；实验 commit `9ca9fe2`（E00）、`080179c`（E01）、`5ccb667`（E02）；MiniMind upstream `393e387e9ad99f0f04c296e4c5e7353f4444629f`。
- 实验目录：`experiments/00-preparation/{E00-l20-baseline-20260823,E01-tokenizer-dataset-20260823,E02-model-probe-20260823}/`；索引 `experiments/00-preparation/README.md`；registry `experiments/registry.csv` 第 2–4 行。
- eval / 结果 manifest：E00 `eval.json`（`status: not_applicable`）、`environment.json`、`metrics.csv`；E01 `data_manifest.json`、`results/{dataset_audit,compression_metrics,sample_transformations,summary}.json`、`data_quality_report.md`；E02 `results/{architecture,probe-data-manifest,summary,summary-seed42/43/44,metrics-seed42/43/44}`。
- artifacts：`/data/artifacts/minimind-lab/E02-model-probe-20260823/`（`probe-data.pt` SHA-256 `287f5352ff98…`；`checkpoints/seed42-resume.pt` SHA-256 `9036139bc52f…`，767,063,342 B）。
- 交叉验证数据：`/data/artifacts/minimind-lab/phase1-aux/pretrain_mini_target_audit.json`（Phase 1 全量精算）。
- 源码阅读笔记（2026-09-03 已恢复并落库于 `2584c5c`）：`docs/source_reading/{README,00-training-runtime,01-data-tokenizer,02-model-architecture}.md`。
- SwanLab：见 §2.1 与 [`swanlab-runs.md`](swanlab-runs.md)。

## 修订记录

| 日期 | 修改内容 | 原因 |
|---|---|---|
| 2026-09-03 | 首次出具（draft） | 补齐 Phase 0 阶段报告，并记录 3 处证据缺口与 2 项 partial 门 |
| 2026-09-03 | 修复§8 第 1、3 条：回填 E00 的 SwanLab 时间线证据（artifact + report + registry），恢复 `docs/source_reading/` 与 `docs/upstream-minimind.md`；同步更新 §2.1、§5、§6、§9 | 消除“报告与 artifact 矛盾”和“产物引用断裂”两处证据缺口 |
| 2026-09-03 | §7 修正 SwanLab project 结论：P01/P02 的 run 已同步到统一 project，原写“三者无法叠图”不准确 | 排查 registry 缺行时发现 stage5 收口 commit `5761979` 已做过 project 合并 |
| 2026-09-03 | §8 第 2、4 条标为已完成，§9 更新 Git 坐标；新增未解决问题“D01–D05 数据准备实验未登记” | 阶段报告与修复已合入 main；合并 `data/v1` 后新增五个未入 registry 的数据实验目录 |
