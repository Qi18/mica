# MiniMind Lab 统一实验与学习计划

更新时间：2026-09-03  
权威工作区：`/data/projects/minimind-lab`

## 0. 文档定位

本文是 MiniMind Lab 唯一的计划性文档，统一回答以下问题：

1. 每个阶段为什么做、具体训练什么；
2. 需要阅读哪些源码、通过什么观察实验理解实现；
3. 每个阶段预期获得什么能力，以及如何证明能力发生变化；
4. 何时允许进入下一阶段，失败实验如何保留；
5. 实验、SwanLab、checkpoint、GitHub、博客和简历如何形成证据链。

固定且可执行的评测命令、参数和文件格式见 [`evaluation_protocol.md`](evaluation_protocol.md)；每个 Phase 完成后的横向收口结论写入 `docs/phases/` 下的阶段报告（规范见第 16 节）；跨阶段的最终结论汇总在 [`final_report.md`](final_report.md)。这些文档不再重复规划内容。

## 1. 项目目标与研究问题

### 1.1 主线目标

- 从随机初始化完成 64M Dense 的 Pretrain → SFT → Preference/RL → Agentic RL 闭环；
- 比较官方数据与自建数据的质量、有效 token 利用率、能力收益和训练成本；
- 理解 SFT、DPO、GRPO/CISPO、Agentic RL 各自优化什么，以及能力增益来自哪里；
- 对比 Full FT 与 LoRA、Dense 与 MoE、CE-only 与 off-policy/on-policy 蒸馏；
- 将训练实验和源码阅读沉淀为可复现报告、博客与简历项目。

### 1.2 核心研究问题

1. 数据质量、去重、污染控制与 tokenizer 对齐切块如何影响 Base 泛化？
2. 官方 SFT 数据能把 Base 训成多可用的对话模型，SFT 带来多少指令遵循、对话、格式和 Tool Call 能力、又遗忘多少 Base 能力，自建数据在什么条件下才值得做？
3. DPO 是否在 chosen-only SFT 之外带来独立偏好收益，是否产生长度投机或过度拒答？
4. GRPO/CISPO 是否提高 held-out 可验证任务正确率，而不只是提高训练 reward？
5. Agent SFT 与 Agentic RL 分别做了什么，RL 是否真正提升工具选择、错误恢复和端到端任务成功率？
6. LoRA、MoE 与蒸馏在效果、显存、吞吐与训练成本上是否值得，on-policy 蒸馏是否比 off-policy 更有效？

### 1.3 项目边界

- 64M Dense 是本项目的完整主线，不扩展到 1B；算力投入全部用于数据质量、后训练对照和评测可信度，不以参数量替代实验质量；
- 不为了覆盖功能而把所有后训练权重无控制地串联；
- loss、reward、SwanLab 曲线和主观样例只用于诊断，不能单独证明能力提升；
- 同时改变数据、训练入口和超参数时，只能称为“新基线比较”，不能声称严格单变量收益；
- Agent SFT 只是模仿工具轨迹，只有模型在环境反馈下探索并优化任务结果时才称为 Agentic RL；
- MiniMind 上游升级需单独评审，不静默改变已有实验基线。

## 2. 当前事实与实验状态

- 实验仓库：`Qi18/minimind-lab`；MiniMind 源码位于普通目录 `minimind/`；
- 初始上游 commit：`393e387e9ad99f0f04c296e4c5e7353f4444629f`；
- 训练脚本参数虽沿用 `--use_wandb` / `--wandb_project`，实际后端是 SwanLab；
- 配置必须区分源码默认值、历史运行值和本 Lab 主动选择的实验值；
- `experiments/registry.csv` 是实验状态索引，单个实验目录保存完整证据。

| 实验 | 当前状态 | 定位 |
|---|---|---|
| P01 Dense Pretrain Mini | 已完成并合入 main | Mini Pretrain 历史基线 |
| P02 Dense Pretrain Full | 训练、评测和补充审计完成，待收口 | 旧 full 数据负实验 |
| P03 Dense Pretrain V1 1B28 | 训练与完整评测完成，待收口 | 后续正式 Base |
| Phase 2 SFT | accepted with limitations | S10 release；IFEval prompt strict 17.38%，七项 chat-template macro 33.00% |

P03 当前结果：63,912,192 参数、9,038 optimizer steps、46.37 分钟、6.18 GPU-hours、validation NLL 2.6043、PPL 13.52、七项 Base macro 31.52%。在正式收口前，这些数字仍需由 main 中的 registry、报告和 manifest 可追溯。

## 3. 总体能力链与实验依赖

```text
数据审计 / Tokenizer / Model Probe
                 ↓
              P03 Base
                 ↓
     Phase 2：官方 SFT 数据 + LR probe
                 ↓
   未过门槛且归因到数据 → 同阶段构建 SFT-v1
                 ↓
        最佳 Full SFT checkpoint S★
          ├── Full FT vs LoRA（Phase 3）
          ├── chosen-only vs DPO（Phase 4）
          ├── SFT continuation vs GRPO/CISPO（Phase 5）
          └── Agent SFT vs Agentic RL（Phase 6）

独立扩展（Phase 7）：64M Dense vs 198M-A64M MoE，不扩展到 1B
最后执行（Phase 8）：CE-only vs off-policy 蒸馏 vs on-policy 蒸馏（OPD），需要 Phase 7 或上游提供的 MoE Full SFT teacher
```

| 阶段 | 做了什么 | 预期能力变化 | 主要证明方式 |
|---|---|---|---|
| Pretrain | 对通用文本做 next-token prediction | 语言建模、知识和基础续写 | validation NLL/PPL、七项 Base、固定续写 |
| SFT | 用指令—回答、多轮、格式和工具轨迹做监督训练 | 指令遵循、对话格式、基础推理与 Tool Call | 指令/格式/Tool 通过率及 Base 遗忘 |
| DPO | 用 chosen/rejected 对优化 policy 相对 preference | 回答偏好、帮助性、表达与拒答边界 | held-out preference、盲测 win rate、长度控制 |
| GRPO/CISPO | 对多条 completion 按可验证 reward 更新策略 | 数学/代码等可验证推理正确率 | held-out pass@1/pass@k、reward、KL、稳定性 |
| Agent SFT | 模仿已有多轮工具调用轨迹 | 工具格式、工具选择与参数生成的初始能力 | Tool schema、选择和参数准确率 |
| Agentic RL | 在真实工具环境中交互，用最终任务结果训练 | 多步规划、观察利用、错误恢复与任务闭环 | 端到端成功率、调用正确率、步骤与成本 |

后训练实验默认从同一个最佳 SFT checkpoint 独立分叉；除非实验专门研究串联收益，否则不使用 DPO → GRPO → Agent RL 的串联权重来替代直接基线。

## 4. 全局实验规范

### 4.1 可归因原则

- 每个实验只回答一个主要问题；
- 对照组固定初始化权重、数据预算、seed、optimizer、scheduler、模板和评测协议；
- 使用实际参与 loss 的 target tokens，不用样本行数或 padded slots 代替训练量；
- P02/P03 因数据组成、切块、validation 和训练入口均不同，只作为整体管线比较；
- GRPO/CISPO 只改变 loss；Full FT/LoRA 只改变参数更新方式；DPO 必须有 chosen-only 控制组。

### 4.2 Batch 与 seed 口径

```text
global_sequence_batch = per_gpu_batch × world_size × accumulation_steps
targets_per_optimizer_step ≈ global_sequence_batch × 平均有效 target tokens
```

- 全量 Pretrain 固定 seed 42；100-step probe 尽量使用 3 个 seed；
- DPO、GRPO/CISPO、Agentic RL 的关键结论尽量运行 seed 42/43/44；
- 资源不足只能单 seed 时必须标明，不能给出稳定性结论。

### 4.3 三层完成状态

1. **技术完成**：进程 exit 0、无 NaN/Inf/OOM/NCCL 故障、checkpoint 可 strict load、指标齐全；
2. **实验有效**：目标指标按预注册门槛改善，通用能力回退可接受，数据与评测无污染；
3. **发布候选**：多项证据一致、结果可复现、成本明确、资产和报告完整。

## 5. Phase 0：环境、数据与代码探针

### 5.1 环境与运行时

- 核验 8 张 L20、CUDA、BF16、NCCL all-reduce、CPFS、`/dev/shm` 和网络；
- 核验 SwanLab 登录，不把密钥放入命令、日志、CPFS 或 Git；
- 正式 launcher 必须具有 GPU 空闲门、单实例锁、SIGTERM 安全退出和 resume 验证；
- 记录 PyTorch、CUDA、Transformers、attention backend 和 Git commit。

### 5.2 Tokenizer 与 Dataset 源码

阅读：

- `minimind/trainer/train_tokenizer.py`；
- `minimind/model/tokenizer.json`、`tokenizer_config.json`；
- `minimind/dataset/lm_dataset.py` 中 Pretrain/SFT/DPO/RLAIF/AgentRL Dataset；
- chat template、`<think>`、`<tool_call>` 与 assistant-only label mask。

观察：原始 JSON → template → input_ids → labels；按中文、英文、代码、数学统计压缩率、截断、padding 和有效 target。主线固定官方 tokenizer；更换 tokenizer 会破坏 embedding、LM head 和历史 checkpoint 可比性，只允许隔离实验。

### 5.3 模型与训练运行时源码

阅读：

- `minimind/model/model_minimind.py` 的 Embedding、RMSNorm、RoPE、GQA、MLP、Residual、LM Head；
- Dense/MoE、Router、top-k、aux loss、KV Cache；
- 各 `train_*.py` 的 DDP、optimizer、scheduler、checkpoint、resume 和 SwanLab；
- `trainer_utils.py` 的权重保存和 run 恢复逻辑。

观察：参数量拆解、Tensor shape、单卡 forward/backward、8 卡 100-step probe、一次断点恢复。阶段门是能从输入解释到 logits/loss，probe 无数值或分布式错误，恢复后的 step、optimizer、loss 和 SwanLab 连续。

## 6. Phase 1：Pretrain 基线与数据重做

### 6.1 实际路径

Phase 1 不是一次训练，而是三次预训练加一次数据重做：先用官方数据把管线跑通，效果不满意后才重做数据。

| 实验 | 数据 | 有效 targets | 共享 validation NLL | 七项 macro | 定位 |
|---|---|---:|---:|---:|---|
| P01 | 官方 `pretrain_t2t_mini` | 303M | 3.5641 | 31.44 | 第一条可复现 mini 基线 |
| P02 | 官方 `pretrain_t2t` 全量 | 2,021M | 3.1910 | 30.91 | 数据放大 6.7 倍，loss 降但 benchmark 回退，负实验 |
| P03 | 自建 `pretrain-v1-1b28` | 1,280M | 2.6043 | 31.52 | 重做数据后的正式 Base |

本阶段回答三个问题：扩大旧数据是否有收益（否）；重做数据管线是否有收益（validation 有明显收益，七项 macro 相对 P01 仅 +0.08pp 属噪声量级）；谁作正式 Base（P03）。三次训练与三份数据的逐项对比、门控判定和结论边界见 [`phases/phase1-pretrain.md`](phases/phase1-pretrain.md)。

### 6.2 P02/P03 资产收口

- P02：保存配置、命令、曲线、成本、七项 benchmark、共享 validation、数据审计、checkpoint manifest 和 SwanLab URL；
- P03：状态改为 completed，新增正式报告、registry、checkpoint/export SHA、训练摘要、评测与 artifact manifest，并补齐 `swanlab-url.txt`；
- Phase 1 不重新训练；main 必须能从 registry 定位上述证据；
- P03 验收后成为后续唯一正式 Base，P02 作为“扩大旧数据没有按比例获益”的历史负实验。

### 6.3 Pretrain 评测

- 固定 validation NLL/PPL；
- C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag、Social IQA；
- 固定 greedy 续写、重复和立即 EOS；
- wall、GPU-hours、有效/padded throughput、利用率、显存、checkpoint 大小；
- 数据有效 targets、截断、padding、重复、跨 split overlap 与 benchmark contamination。

### 6.4 遗留问题

- P03 相对 P01 的七项 macro 只有 +0.08pp，只能声明“validation loss 明显更低”，不能声明“能力明显更强”；
- P03 的 5 条固定 greedy 续写全部为 token 重复，未通过其预注册的生成质量门，Phase 1 报告状态为 draft；
- v1 的收益无法归因到 source mix、去重或 tokenizer 对齐切块中的具体一项，消融待有算力预算时补。

## 7. Phase 2：SFT（官方数据优先，必要时自建数据）

Phase 2 回答一个完整问题：把 P03 变成可交互、可遵循的模型需要什么数据。执行顺序固定为先官方数据、后自建数据——与 Phase 1 一致，先用现成数据把 mask、训练、评测和判定门跑通并拿到可比基线，只有官方数据未过门槛且失败可归因到数据时才构建 SFT-v1。自建数据是本阶段的条件分支，不是默认动作。

### 7.1 起点

- 初始权重固定为 P03；

### 7.2 阶段路线

```text
S01-smoke（官方 mini，约 1M）
      ↓
S02 官方完整 SFT（LR probe 属于 S02 attempts）
      ↓ 未过行为门
S03 官方 pilot / S04 自建 pilot（等预算 A/B）
      ↓ 均未过门
S05 官方 pilot 低 LR 重放 / S06 官方 Base 对照（归因实验）
      ↓
S07 低唯一性修复 / S08 多样化修复
      ↓
S09 定向 Chat 课程（step 400 通过行为门）
      ↓
S10 verifier-backed IFEval 课程（Phase 2 release）
```

### 7.3 官方数据实验序列

| 实验 | 初始权重 | 数据 | assistant targets | 目的 |
|---|---|---|---:|---|
| S01-smoke | P03 | 官方 mini | 约 1M | smoke：mask、loss、数值稳定、strict load、resume、生成、SwanLab |
| S02 attempt LR-A/B/C | P03 | 官方 mini | 各 8M | LR probe：`1e-5` / `3e-5` / `5e-5` |
| S02 formal | P03 | 官方全量 | 实际约 2.4007B | 官方数据的正式 SFT；原计划 32M，实际完整跑 1 epoch |
| S06 | 官方预训练权重 | 官方 pilot | 8M | 核对 Base 来源对行为失败的影响 |

S01-smoke 不用于宣称任何能力。S02 的 LR attempts 按 validation NLL、指令/格式通过率、Tool 合法率和 Base 保留选配置，不按最低 train loss；S02 formal 的最终执行口径以完整 1 epoch 为准。

### 7.4 官方 SFT 数据必须先量化的事实

在 S02-formal 之前用与 Phase 1 相同的定义审计官方 SFT 数据，否则无法区分“数据不好”和“训练不好”：

- shifted assistant target tokens 总量、每行 targets 与有效利用率；
- 非 assistant token 被监督的数量（必须为 0）；
- 截断行数与丢弃 tail tokens；
- exact duplicate、跨 split 重复与固定 benchmark 污染；
- 非法 JSON/conversation、零 assistant target、不完整 tool exchange 计数。

### 7.5 官方数据验收门与自建数据触发条件

预注册门槛（S02-formal 相对 P03，均用本 Lab 固定行为集与评测口径）：Tool schema 合法率 ≥80%；七项 macro ≥30.0%，相对 P03 回退不超过约 1.5pp，任一单项不回退超过 4pp；固定生成无空回答、无明显复读、无长度膨胀；Chat 通过 ≥7/10、格式约束 ≥4/6、重复异常 ≤2/10、Tool 端到端成功 ≥60%。

- 全部通过：S02-formal 即最佳 SFT checkpoint（下称 S★），Phase 2 直接收口，7.6–7.8 不执行，阶段报告需写明“官方数据已达门槛，未构建自建数据”；
- 未通过且 7.4 的审计把失败归因到数据（污染、重复、mask 错误、targets 结构缺失、bucket 覆盖不足）：进入 7.6，启动理由与要修的具体缺陷写在 Phase 2 报告；
- 未通过但可归因到 LR、模板、checkpoint 选择或评测口径：先修正重跑官方数据实验，不启动数据自建。

不得以“自建数据更好”为默认前提跳过官方数据基线。

### 7.6 SFT-v1 分级预算（条件触发）

| 数据版本 | shifted assistant target tokens | 用途 |
|---|---:|---|
| SFT-v1-smoke | 1M | schema、mask、loader、auditor |
| SFT-v1-pilot | 8M | 与官方数据等预算 A/B |
| SFT-v1-formal | 32M | 自建数据的正式 SFT |
| SFT-v1-scale | 64M | 仅在 32M 仍有明确收益时启用 |

### 7.7 SFT-v1 初始构成与数据硬门

| Bucket | 初始比例 |
|---|---:|
| 中文通用指令 | 30% |
| 英文通用指令 | 15% |
| 多轮对话 | 10% |
| 严格格式 | 15% |
| 数学与推理 | 10% |
| Tool Calling | 10% |
| 总结与翻译 | 5% |
| 代码 | 5% |

比例只用于容量规划，不能在 pilot 和独立评测前宣称最优；实际比例必须针对 7.5 暴露的缺陷调整，不照抄本表。

数据硬门：

- 非法 JSON/conversation、零 assistant target、不完整 tool exchange 均为 0；
- 非 assistant token 被监督为 0；
- train/validation/test 精确与近似跨 split 重复为 0；
- 固定 benchmark 污染为 0；同一 origin 不跨 bucket 或 split 重用；
- 改变输入扫描顺序不改变输出；
- 独立 auditor 通过后才写入可训练 `_SUCCESS`。

### 7.8 等预算对照与正式训练

| 实验 | 初始权重 | 数据 | assistant targets |
|---|---|---|---:|
| S03 | P03 | 官方 SFT 数据 | 8M |
| S04 | P03 | SFT-v1-pilot | 8M |

S03 只有在数据快照、审计状态和训练配置完全一致时才能复用 7.3 的胜出 LR run；只要正式对照使用了新的 accepted 数据快照，就必须重跑。固定模型、seed、LR、scheduler、seq、chat template、checkpoint 选择、总 assistant targets 和评测集，只回答“自建 SFT 数据是否优于官方数据”。

公平 batch 按“每次 optimizer update 的 shifted assistant targets”定义，而不是按样本条数定义：当两套数据的平均监督回答长度不同时，允许调整每卡 sequence batch，使 assistant targets/update 与 optimizer steps 的差异均不超过 2%，并在配置和报告中披露；否则较短回答的数据会因为多出约数倍更新次数而获得不公平优势。

- 只有 S04 胜出才构建 SFT-v1-formal；实际 S04 未胜出，因此没有用扩大预算掩盖失败；
- S05、S06 是失败归因对照，不是新的 release 候选；
- S07–S09 逐步提高有效监督密度，S10 再使用独立 verifier-backed 课程完成 IFEval 验收。

预注册验收门与 7.5 一致，并追加：S04 相对 S03 在指令/格式/Tool 主指标上提升，且七项 macro 不额外回退超过 1pp。

### 7.9 SFT 能力判定

比较 P03 → S★：指令遵循、多轮对话、严格格式、数学/代码、Tool Call 得到多少提升；同时报告七项 Base、validation 与固定知识任务回退。SFT 的目标是“可交互和可遵循”，不是证明事实知识或推理能力必然全面上升。Phase 1 已知 P03 的固定续写全部退化，S★ 必须用同一口径回答“SFT 后生成是否恢复正常”。

Phase 2 阶段报告必须同时收口两件事：官方数据能达到的能力上限，以及自建数据是否被触发、触发后是否带来独立收益。

### 7.10 实际执行与收口（2026-09-07）

名义上的 32M 计划没有被补写成实际执行：官方完整数据实际约含 2.4007B assistant targets。官方完整 SFT 虽然 loss 收敛，但仅达到 Chat 2/10、格式 0/6、重复异常 7/10、Tool E2E 5/8，因此触发数据与能力修复分支。

最终 release 为 S10：以 S09 step 400 为起点，在 41,720 条、4.163M assistant targets 的 verifier-backed 课程上训练；与冻结 IFEval 541 prompts 的 exact/normalized overlap 为 0。统一评测得到 prompt strict 17.38%、instruction strict 29.98%，分别较 S09 提升 6.29pp 和 7.68pp；七项 chat-template macro 33.0024%，Chat 8/10、格式 5/6、重复异常 1/10、Tool E2E 7/8。

因此 Phase 2 状态为 `accepted with limitations`，S★ 固定为 S10。限制是绝对 IFEval 仍低，language 无提升，keywords 与 length constraints 略有回退；这不阻塞后续 Phase，但后续结论必须保留该边界。权威阶段报告见 `docs/phases/phase2-sft.md`。
## 8. Phase 3：Full FT vs LoRA

LoRA 是明确领域上的效率实验，不替代主线 Full SFT。从 S★ 出发，在完全相同的领域 train/validation/test、targets 和评测上比较：

| 实验 | 方法 | 主要变量 |
|---|---|---|
| L01 | Full parameter continuation | 更新全部参数 |
| L02 | LoRA | 只更新 adapter |

报告领域指标、七项回归、可训练参数、峰值显存、wall、GPU-hours、adapter 大小，以及 merge 前后输出一致性。没有独立领域验收集时只允许 smoke。

### 8.1 实际收口（2026-09-08）

L00/S10、L01 Full FT、L02 LoRA r16 的训练与独立评测完成，阶段为 accepted with limitations。两组 MBPP 均0/500（S10为2/500），未证明代码能力提升；LoRA本次训练wall/显存分别下降约33%/36%。保留S10为后续基线。实际训练seed42、数据划分seed20260907、两组LR不同及其他限制见 [Phase3报告](phases/phase3-lora.md)。

## 9. Phase 4：DPO 偏好优化

阅读 DPODataset、policy/ref log-prob、DPO loss、beta 和长度偏差，理解 DPO 是静态 off-policy preference optimization，不是在线探索。

| 实验 | 方法 | 目的 |
|---|---|---|
| D01 | S★ 不继续训练 | 生成与偏好基线 |
| D02 | chosen-only SFT | 控制只看优质答案的收益 |
| D03 | DPO chosen/rejected | 验证 preference objective 独立收益 |

评测 held-out preference accuracy、chosen/rejected margin、盲测 win/tie/loss、长度控制胜率、输出长度、拒答率、KL、指令/Tool 和七项回归。DPO 必须同时优于 D01 和 D02，且收益不能主要来自答案变长、模板化或过度拒答。

### 9.1 Phase4 实际收口（2026-09-08）

官方 DPO 数据冻结为 train 14,194 / validation 1,000 / test 1,000，独立审计通过。D01/D02/D03 从 S10 独立启动；LR 4e-8、beta 0.15、seed 42、global batch 32。D03 test preference credit 71.8%，相对 D01/D02 的 paired bootstrap 差值区间均大于 0；七项、IFEval、Chat/Tool 回归门通过。

但 200 条 Qwen3-8B 盲评中，D03 vs D01 score 0.515（95% CI [0.4875, 0.5425]），vs D02 0.480（[0.4525, 0.5075]），未满足“同时优于 D01/D02”的预注册质量门。阶段状态 completed-not-promoted，保留 S10；Phase5 仍从 S10 独立分叉。另发现 FP16 保存会抹除小幅 DPO 更新，已改为 float32 checkpoint，并将旧 D02/D03 FP16 inference 权重作废。完整边界见 [Phase4报告](phases/phase4-dpo.md)。

## 10. Phase 5：GRPO/CISPO 可验证强化学习

阅读 reward、reference、group-relative advantage、importance ratio、KL，以及 `train_grpo.py` 中 GRPO/CISPO 的 loss 分支。PPO 作为理解 Actor/Critic、GAE 和 clipped objective 的学习支线，不作为当前主实验的前置条件。

| 实验 | 方法 |
|---|---|
| R02A | S★ 原始基线 |
| R02B | 等 prompt batch / update 数的 SFT continuation |
| R02C | GRPO |
| R02D | CISPO |

使用 exact match 或程序 verifier 的无污染数学/代码任务；固定初始化、数据、seed、rollout 数、reward 和生成参数，GRPO/CISPO 只改变 loss。报告 held-out pass@1/pass@k、reward、KL、entropy、completion length、group reward std、degenerate group rate 和 reward-hacking 抽查。reward 上升但 held-out 正确率不升，实验失败。

### 10.1 Phase5 启动（2026-09-08）

首轮 R01/v1 因 template/answer 联合混杂在 validation 被判无效，test 未打开。修复后的 verifiable-math-v2 冻结为 train 1,000 / validation 200 / test 400，程序 verifier 与跨 split 审计通过。正式 R02A/B/C/D 从 S10 独立分叉；正式 test 在三组训练完成前保持封闭。固定超参数、可学习性探针、SFT 与 RL 的预算边界和晋级门见 [Phase5报告](phases/phase5-verifiable-rl.md)。

### 10.2 Phase5 实际收口（2026-09-08）

R02B/C/D 均完成 250 outer steps、500 optimizer updates。R02C GRPO 与 R02D CISPO 的 test pass@1 都是 25.0%，但 R02C 400/400 恒答 A，R02D 的选择分布为 B 300、A 100；对应 pass@4 仅 25.0% / 25.25%，较 S10 的 41.0% 分别下降 16.0 / 15.75pp。R02C 相对 S10 的 pass@1 差值为 +1.75pp（paired bootstrap 95% CI [+0.25,+3.50]），完全由答案位置策略造成；R02D 同为 +1.75pp，但区间 [-4.25,+8.00] 跨 0。

两组的七项、IFEval 与固定 Chat/Tool 回归门基本守住，但均未通过“无答案位置塌缩”的主门；R02B SFT control 还出现 IFEval 16.82%→7.39%、Chat 8/10→0/10、Tool E2E 7/8→1/8 的灾难性遗忘。阶段状态 `completed-not-promoted`，保留 S10。单 seed 与窄域四选一算术只允许得出“当前 verifier-RL 配置失败并暴露稀疏 reward/退化 group 问题”，不能比较 GRPO/CISPO 的一般优劣。

## 11. Phase 6：Agent SFT 与 Agentic RL

### 11.1 Agent SFT 基线

先构建 train-disjoint 的工具任务和轨迹，固定 tools schema、超时、最大轮数、重试与错误处理。Agent SFT 使用已有轨迹监督模型学习：

```text
用户请求 → 工具选择 → 参数 JSON → tool observation → 最终答案
```

它的验收是格式合法、工具选择和参数准确，但不能用它证明模型具备环境探索与错误恢复能力。

### 11.2 Agentic RL

只有基础 Tool Call 稳定后才启动。模型在环境中完成：

```text
观察 → 规划 → 工具调用 → 环境反馈 → 修正/继续 → 最终答案
```

| 实验 | 方法 | 回答的问题 |
|---|---|---|
| A00 | S★ 通用 SFT | 未做 Agent 训练的基线 |
| A01 | Agent SFT | 模仿轨迹能获得多少工具能力 |
| A02 | Agent SFT + Agentic GRPO | RL 是否提升端到端成功率 |
| A03 | Agent SFT + Agentic CISPO | 与 GRPO 的效果/稳定性差异 |
| A04 可选 | Torch vs SGLang rollout | rollout 系统效率 |

正式评测至少准备约 200 条 train-disjoint 任务，报告 tool selection accuracy、schema validity、argument semantic accuracy、execution success、observation usage、final answer accuracy、end-to-end success、平均调用/轮数、未完成率、无效调用率、延迟和 token 成本。Agentic RL 的主要结论来自端到端成功率，不来自训练 reward。

## 12. Phase 7：Dense vs MoE

### 12.1 对照设计

对比约 64M Dense 与约 198M-A64M、4 experts、top-1 MoE。阅读 Router、dispatch、aux load-balancing loss、总参数/激活参数和 expert 负载；固定数据、有效 targets、seq、硬件和评测，报告质量、吞吐、显存、负载不均与稳定性。上游“原生 PyTorch MoE 约慢 50%”只作参考，必须在 L20 重测。

### 12.2 范围限定

本 Phase 只回答“相同数据预算下，稀疏扩容是否比 Dense 更划算”，不引入 1B：本项目不做 1B 扩展，也不做为 1B 服务的 scaling probe。MoE 对照完成后，剩余算力优先回到数据质量与后训练对照。

## 13. Phase 8：蒸馏（off-policy 与 on-policy）

蒸馏放在最后，因为它是全部阶段中唯一依赖外部模型的实验：需要一个同 tokenizer、且已做过 SFT 的 MoE teacher，而这个 teacher 要么来自 Phase 7 自己训练并 SFT 的 MoE，要么来自上游发布的 MiniMind MoE Full SFT。

启动前提（前三条均不满足则本 Phase 出具“已跳过/推迟”报告）：

- Phase 2 已确定 S★作为 Student；
- 存在可用 teacher：tokenizer、vocab 和 chat template 与 S★ 完全一致，权重可 strict load，并在本 Lab 固定评测上明确强于 S★；teacher 不强于 student 时蒸馏无意义；
- teacher 来自 Phase 7 时，以 Phase 7 阶段报告为门；来自上游时记录权重来源、revision 和 SHA-256；
- K04（OPD）额外要求可用的 rollout 路径：以 Phase 5 的 GRPO/CISPO 采样实现为基础，只把 reward 替换为 teacher 的 per-token 分布；rollout 不可用时只执行 K01–K03，并在报告中说明 K04 推迟原因。

| 实验 | Teacher | 训练序列来源 | Loss | 目的 |
|---|---|---|---|---|
| K01 | 无 | 固定数据集 | CE-only | 普通续训控制组 |
| K02 | MoE Full SFT | 固定数据集 | 0.5 CE + 0.5 KL | off-policy 蒸馏的独立收益 |
| K03 可选 | MoE Full SFT | 固定数据集 | alpha/temperature 消融 | 理解超参数 |
| K04 OPD | MoE Full SFT | student 自采样 | per-token reverse KL | on-policy 蒸馏是否优于 off-policy |

### 13.1 off-policy 对照（K01–K03）

固定数据、初始权重、batch、steps、seed 和 LR。记录 CE、KL、总 loss、教师—学生一致率、成本和统一评测；只有 K02 在 held-out 上优于 K01 才能称为蒸馏有效。K02 优于 K01 但不及 teacher 时，只能称为“部分能力迁移”。

### 13.2 on-policy 蒸馏 K04

K04 相对 K02 只改一处：训练序列由 student 自己采样，而不是取自固定数据集。单步流程为 student 对固定 prompt 集生成 completion → teacher 对同一 token 序列做一次 no-grad forward → 按 per-token reverse KL（student 相对 teacher）更新 student。它在 student 自己会走到的状态上给出逐 token 稠密监督，不需要 verifier 或 reward model，因此比 Phase 5 的可验证 RL 便宜，但比 K02 贵：每步多一次采样和一次 teacher forward。

固定项：prompt 集与 K02 数据同源且 train-disjoint、初始权重 S★、seed、生成参数（temperature、top-p、max_new_tokens、每 prompt rollout 数）、student 实际更新的 target token 预算与 K02 相同。teacher 全程冻结在 eval + BF16 + no-grad，不参与梯度。

必须记录：per-token reverse KL、teacher—student 一致率、平均 completion 长度与长度分布、重复率、rollout 与 teacher forward 各自的时间占比、峰值显存（student 训练态与 teacher 推理态同驻）、有效 target tokens/GPU-hour。

判定：K04 只有在等 target 预算下于 held-out 优于 K02 才能称 on-policy 有收益。reverse KL 是 mode-seeking，若 K04 出现输出显著变短、复读上升或生成多样性塌缩，即使 held-out 指标上升也只记为“以多样性换取分布贴合”，并在阶段报告写明代价。KL 下降但 held-out 不升时，与“reward 上升但正确率不升”同等处理，判为失败。

## 14. 统一评测矩阵

具体命令、chat template、manifest 字段和结果文件格式以 [`evaluation_protocol.md`](evaluation_protocol.md) 为准。

| Checkpoint | Base 七项/NLL | Instruction | Preference | Math/Verifier | Tool/Agent | 系统指标 |
|---|---|---|---|---|---|---|
| P03 Base | 必测 | 固定样例 | - | 基线 | - | 必测 |
| Full SFT/LoRA | 回归 | 必测 | 基线 | 必测 | Tool 基线 | 必测 |
| DPO | 回归 | 必测 | 必测 | 回归 | 回归 | 必测 |
| GRPO/CISPO | 回归 | 回归 | - | 必测 | - | 必测 |
| Agent SFT/RL | 回归 | 回归 | - | 可选 | 必测 | 必测 |
| MoE | 必测 | 按阶段 | 按阶段 | 按阶段 | 按阶段 | 必测 |
| Distill | 回归 | 必测 | - | 必测 | 回归 | 必测 |

每个阶段都要回答：目标指标提升多少；哪些通用指标回退；成本变化多少；多 seed/失败样例是否支持结论；是否存在污染、长度偏差、reward hacking 或格式投机。

## 15. 实验目录与证据链

```text
experiments/<stage>/<experiment-id>/
├── README.md
├── config.json
├── command.sh
├── data_manifest.json
├── run.json
├── metrics.jsonl 或 metrics.csv
├── eval/
│   ├── eval_manifest.json
│   ├── base_benchmarks.json
│   ├── task_eval.json
│   ├── generation_samples.jsonl
│   └── system_metrics.json
├── report.md
├── checkpoint-manifest.txt
└── swanlab-url.txt
```

- GitHub：配置、命令、报告、紧凑指标和 SHA；
- SwanLab：训练曲线、系统指标与评测指标；
- `/data/artifacts/minimind-lab`：checkpoint、resume state 和完整日志；
- ModelScope/Hugging Face：只发布 accepted checkpoint；
- `experiments/registry.csv`：实验完成当天登记。

只有输入 checkpoint、数据 fingerprint、Lab/MiniMind commit、配置、SwanLab、checkpoint SHA、目标评测、通用回归和报告全部齐全，实验才可标记 completed。

## 16. 阶段报告

### 16.1 三层文档职责

每个 Phase 完成后必须在 `docs/phases/` 下产出一份阶段报告，作为该 Phase 的收口凭证。

| 文档 | 位置 | 职责 |
|---|---|---|
| 实验报告 | `experiments/<stage>/<experiment-id>/report.md` | 一次运行的配置、指标、异常和单实验结论 |
| 阶段报告 | `docs/phases/phase<N>-<slug>.md` | 该 Phase 内所有实验的横向对比、能力增量与回退、成本、失败与门控判定 |
| 最终报告 | `docs/final_report.md` | 只汇总已收口阶段报告的核心结论，不重复过程细节 |

### 16.2 固定文件名

| Phase | 阶段报告 |
|---|---|
| Phase 0 环境、数据与代码探针 | `docs/phases/phase0-preparation.md` |
| Phase 1 Pretrain 基线与数据重做 | `docs/phases/phase1-pretrain.md` |
| Phase 2 SFT（官方数据优先，必要时自建） | `docs/phases/phase2-sft.md` |
| Phase 3 Full FT vs LoRA | `docs/phases/phase3-lora.md` |
| Phase 4 DPO 偏好优化 | `docs/phases/phase4-dpo.md` |
| Phase 5 GRPO/CISPO | `docs/phases/phase5-verifiable-rl.md` |
| Phase 6 Agent SFT 与 Agentic RL | `docs/phases/phase6-agentic.md` |
| Phase 7 Dense vs MoE | `docs/phases/phase7-moe.md` |
| Phase 8 蒸馏（off-policy 与 on-policy） | `docs/phases/phase8-distill.md` |

`docs/phases/README.md` 维护阶段索引与状态；新报告从 `docs/phases/_template.md` 复制。

### 16.3 必需小节

1. 阶段目标与研究问题（引用本文对应 Phase）；
2. 实验清单：experiment_id、status、registry 行、SwanLab run 完整 URL、checkpoint/export SHA；
3. 关键配置、数据 fingerprint 与 Lab/MiniMind commit；
4. 结果横向对比表：目标指标、通用回归、系统与成本指标；
5. 门控判定：对预注册验收门槛逐条给出 pass/fail；
6. 失败与 invalidated 实验及其证据位置；
7. 结论边界：不可归因项、单 seed、污染、长度偏差、reward hacking 风险声明；
8. 下一阶段前置条件与未解决问题；
9. 证据索引：Git ref/commit、`/data/artifacts/minimind-lab` 路径、eval manifest。

### 16.4 出具规则

- 阶段内所有实验的 status 在 `registry.csv` 落定为 completed 或 invalidated 后才写阶段报告；
- 阶段报告只使用能从 registry、eval manifest、SwanLab、checkpoint manifest 追溯的数字，不写未产出的指标；
- 阶段报告与对应实验资产必须在同一次或相邻 PR 合入 main，未合入 main 不得声明该 Phase 完成；
- 阶段报告合入 main 是进入下一 Phase 的硬门；Phase 3/4/5/6/8 这类从同一 SFT checkpoint（S★）并行分叉的实验，以 Phase 2 报告为门；Phase 8 蒸馏额外以 Phase 7 报告（或已记录来源的上游 MoE teacher）为门；
- Phase 被跳过、推迟或取消时同样出具一页报告，说明原因与重新启动的门控条件（例如 Phase 8 的 teacher 可用性）；
- 结论发生变化时更新原报告并在文末追加“修订记录”，不删除或改写旧结论；
- 阶段报告确认的结论才可上升到 `final_report.md`、博客和简历。

### 16.5 SwanLab 链接口径

阶段报告必须能从每个结论反向定位到具体的训练 run：

- 实验清单给出可点击的完整 URL `https://swanlab.cn/@<workspace>/<project>/runs/<run_id>`，不只写 run id 或 project 名；
- 一个实验存在多个 run（probe、formal、resume attempt、eval logging、多 seed）时逐条列出并标明角色与 run name，不用一个 URL 代表全部；
- 每张指标表注明数字来自哪个 run 或哪份 eval manifest；涉及 loss/reward/KL/系统曲线的描述必须附对应 run；
- URL 的唯一录入处是 `experiments/<stage>/<id>/swanlab-url.txt` 和 `registry.csv` 的 `swanlab_url`，两者必须一致，并在 run 结束当天回填；阶段报告只引用，不另存一份事实；
- 跨阶段汇总索引维护在 `docs/phases/swanlab-runs.md`，内容必须与上述两个来源一致；
- 未登录、未同步或仅有本地 `swanlog/` 的 run，在报告中写明“无云端 run”及本地 logdir 路径，不得留空或写不可访问的链接；
- 断点续训产生新 run 时全部保留，并说明 step 区间如何拼接。

## 17. 源码学习、博客与简历交付

每个阶段同步完成四类输出：

1. **源码地图**：入口、Dataset、forward/loss、optimizer、checkpoint、eval 的调用链；
2. **最小观察实验**：用可打印的样本、shape、mask、log-prob、reward 或轨迹证明理解；
3. **实验报告**：配置、曲线、能力增量、回退、成本、失败和边界；
4. **阶段报告**：按第 16 节把该 Phase 的实验横向收口到 `docs/phases/`；
5. **博客草稿**：解释“这一阶段做了什么、源码如何实现、能力为何变化、如何评测”。

简历只使用能链接到 GitHub 报告、SwanLab run 和固定评测的数字。推荐最终概括：构建 64M LLM 全流程实验平台；优化数据有效 token 利用率；完成 SFT、偏好优化、可验证 RL 与 Agentic RL 对照；建立质量、能力、成本和复现证据链。未完成阶段不得写成已实现成果。

## 18. 停止条件

以下任一情况立即停止或作废：

- 数据 `_SUCCESS` 不存在或 independent auditor 未通过；
- loss/reward 出现 NaN/Inf、有效 targets 为零或 mask 错误；
- OOM、NCCL rank 退出、checkpoint 无法 strict load/resume；
- reward 上升但 held-out success/accuracy 下降；
- KL、长度、复读、拒答或无效工具调用持续失控；
- train/eval 污染；
- 配置、数据、代码 SHA 与冻结协议不一致；
- 上一 Phase 的阶段报告未合入 main 就开始下一 Phase 的正式训练。

失败实验保留最小证据并登记为 invalidated，不通过删曲线或改名掩盖失败。

## 19. 推荐执行顺序

1. 将 P02 审计、报告与证据收口到 main；
2. 修正 P03 状态，补齐 report、registry、checkpoint/export manifest 并合入 main；
3. 审计官方 SFT 数据（见 7.4）并执行 S01-smoke；
4. 完成 S02-LR probe（官方 mini，8M targets）；
5. 训练 S02-formal（官方全量 32M）并按 7.5 逐条判定；
6. 仅当 7.5 把失败归因到数据时，构建 SFT-v1 smoke/pilot，执行 S03/S04 等预算 A/B；实际修复链继续执行 S05–S10，随后出具 Phase 2 报告并确定 S★；
7. 建立 Agent SFT 工具基线和独立 Agent eval；
8. 执行 Full FT vs LoRA（Phase 3）；
9. 执行 SFT vs chosen-only vs DPO（Phase 4）；
10. 执行等预算 SFT vs GRPO vs CISPO（Phase 5）；
11. 执行 Agent SFT vs Agentic GRPO/CISPO（Phase 6）；
12. 执行 64M Dense vs 198M-A64M MoE 对照（Phase 7）；
13. 最后，在存在可用 MoE Full SFT teacher 时执行 CE-only vs off-policy 蒸馏 vs on-policy 蒸馏（Phase 8，K04 需要 Phase 5 的 rollout 路径）；否则出具“已跳过”报告。

每完成一个 Phase，先把对应的 `docs/phases/` 阶段报告与实验资产合入 main，再执行下一条。

优先形成五条可量化结论：P03 数据管线相对 P02 的收益；官方 SFT 数据能达到的能力上限以及自建数据是否必要；SFT 带来的能力与遗忘；DPO/GRPO 的独立增益；Agentic RL 相对 Agent SFT 的端到端任务收益。项目成果不是“跑过多少脚本”，而是每条结论能否从代码、数据、SwanLab、checkpoint 和固定评测相互追溯。

## Phase8 执行修订（2026-09-09）

用户授权使用 Qwen3-8B 教师实施蒸馏。当前 K01/K02 采用序列答案 CE 对照，
替代本轮原定同词表 KL 路径；token-KL / OPD 尚未实施，不宣称完成。
详见 [冻结的执行预注册](phases/phase8-sequence-v1-preregister.md) 和
[Phase8 当前状态](phases/phase8-distill.md)。验收前 S10 仍是 release。

## Phase8 收口决定（2026-09-09）

用户决定停止扩展本阶段：序列蒸馏K01/K02已完成并按负结果归档，S10保留。
历史方案中的白盒KL、K03/K04与OPD暂缓且未实施；不再自动启动。
项目可进入结果整理与简历展示，简历只列完成内容，不声称所有后训练都带来能力提升。
