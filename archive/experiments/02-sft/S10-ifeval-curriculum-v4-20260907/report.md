# S10：IFEval 可验证指令课程

## 结论

S10 达到 Phase 2 阶段验收目标，并被选为 release checkpoint，状态为 `accepted with limitations`。相对直接基线 S09，IFEval prompt-level strict 从 11.09% 提升到 17.38%（+6.29pp），instruction-level strict 从 22.30% 提升到 29.98%（+7.68pp）。同 chat template 的七项通用评测 macro 为 33.00%，与 S09 的 32.99% 基本持平。

这证明 verifier-backed 的英文可验证指令课程有效，但提升主要集中在格式、可检测内容和无逗号约束，尚未解决语言切换、关键词和长度约束。

## 数据

- 数据集：`/data/datasets/minimind-lab/data-v1/sft-ifeval-curriculum-v4/`
- train/validation/test：41,720 / 1,539 / 1,535 rows。
- train assistant targets：4,163,325 tokens；zero target=0；`max_seq_len=768` 截断=0。
- 覆盖 IFEval 全部 25 类指令及 14 类兼容组合。
- 生成答案全部通过 lm-eval IFEval strict checker。
- train/validation/test 的 exact 和 normalized prompt overlap 均为 0。
- 三个 split 与冻结的 541 条 IFEval prompt 的 exact 和 normalized overlap 均为 0。
- 数据边界：这是 benchmark-family-targeted curriculum，不等价于开放域对话质量数据。

## 训练

- 初始化：S09 step400。
- 7×L20（物理 GPU 1–7，避开 GPU0 的既有任务），bf16。
- 3 epochs，1,116 optimizer steps，micro batch 16/GPU，global batch 112。
- LR `1e-5` cosine decay，`max_seq_len=768`，关闭随机 system prompt augmentation。
- 训练时间：2026-09-07T10:02:52Z 至 10:06:36Z，约 3 分 44 秒。
- validation loss：1.865881 → 0.289649；最佳点为最终 step 1116。
- 峰值显存：6,937 MiB/GPU。
- 训练 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/bxdob3rh>

## IFEval 对比

统一协议：lm-eval 0.4.12，IFEval v4，541 prompts，0-shot，seed 42，greedy，`--apply_chat_template`。

| 模型 | Prompt strict | Prompt loose | Instruction strict | Instruction loose |
|---|---:|---:|---:|---:|
| P03 | 10.54% | 11.28% | 23.02% | 23.98% |
| S02 official full | 10.72% | 12.20% | 22.30% | 23.86% |
| S09 | 11.09% | 11.83% | 22.30% | 23.98% |
| **S10** | **17.38%** | **19.78%** | **29.98%** | **32.97%** |

S10 strict prompt 通过数为 94/541，S09 为 60/541，即新增 34 条严格通过；loose prompt 从 64/541 增至 107/541。

### Instruction category（S09 → S10）

| 类别 | S09 | S10 | 差值 |
|---|---:|---:|---:|
| detectable_content | 15.1% | 45.3% | +30.2pp |
| detectable_format | 19.1% | 43.9% | +24.8pp |
| punctuation | 25.8% | 37.9% | +12.1pp |
| change_case | 12.4% | 20.2% | +7.9pp |
| startend | 0.0% | 4.5% | +4.5pp |
| combination | 1.5% | 3.1% | +1.5pp |
| language | 6.5% | 6.5% | 0.0pp |
| keywords | 41.1% | 38.0% | -3.1pp |
| length_constraints | 35.0% | 31.5% | -3.5pp |

平均输出字符数从 S09 的 3,698.8 降到 S10 的 3,490.0；提升不是单纯依赖输出变长。

## 七项通用能力回归

| 任务 | S09 | S10 | 差值 |
|---|---:|---:|---:|
| C-Eval | 23.4770 | 23.2541 | -0.2229 |
| CMMLU | 25.2892 | 25.2461 | -0.0431 |
| ARC-Easy | 34.7222 | 34.3434 | -0.3788 |
| PIQA | 56.0936 | 56.1480 | +0.0544 |
| OpenBookQA | 28.0000 | 28.0000 | 0.0000 |
| HellaSwag | 28.3609 | 28.2016 | -0.1593 |
| Social IQA | 35.0051 | 35.8240 | +0.8189 |
| macro | 32.9926 | **33.0024** | **+0.0098** |

## 原有行为回归

- Chat：8/10（验收门槛 ≥7/10）。
- 格式：5/6（门槛 ≥4/6）。
- 重复异常：1/10（门槛 ≤2/10）。
- Tool schema/argument/E2E：7/8（87.5%）。

相对 S09 的 Chat 10/10、格式 6/6、重复 0/10 有轻微回退，但仍通过预注册门槛；Tool E2E 保持 7/8。

## 产物

- checkpoint：`/data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/checkpoints/s08_best_val_768.pth`
- checkpoint SHA-256：`46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131`
- 数据 manifest：`/data/datasets/minimind-lab/data-v1/sft-ifeval-curriculum-v4/manifest.json`
- 原始训练、逐题 IFEval、七项评测、行为评测：`/data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/`
- 评测 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/encs5zuk>

## 下一步

Phase 2 已收口，后续 Full FT vs LoRA、DPO、GRPO/CISPO 和 Agentic RL 默认从 S10 release 独立分叉。若后续专门追求更高 IFEval，可追加 S10R1，集中修复 `language`、`keywords`、`length_constraints` 和 `combination`；该实验不阻塞进入下一阶段，且 IFEval 仍只做最终验收，checkpoint 选择使用独立内部验证集。
