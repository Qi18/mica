# MiniMind Lab 统一评测计划

更新时间：2026-08-21

## 1. 目标与比较边界

本项目的评测只服务于两类结论：

1. **自身实验纵向对比**：判断数据规模、训练阶段、算法、模型结构和训练成本变化带来的真实收益。
2. **MiniMind 官方结果横向对比**：使用官方 README 给出的七项任务和调用口径，判断自训练模型与官方同规模模型的差距。

训练 loss、reward、主观样例和 SwanLab 曲线只能解释训练过程，不能单独作为模型效果结论。所有正式结论必须来自固定评测集，并同时报告绝对分数、相对变化和成本。

训练/验证/测试切分和固定评测污染门禁分别由 [Pretrain 数据协议](data/pretrain/data_protocol.md) 与 [SFT 数据协议](data/sft/data_protocol.md) 管理；本文只定义跨训练阶段保持一致的评分与对比口径。

当前主协议不把 DCLM CORE、IFEval、HumanEval 或其他外部模型排行榜作为完成门槛。它们可作为后续扩展，不能与 MiniMind 官方七项分数混成一个总分。

## 2. 两条评测轨道

### 2.1 轨道 A：自身实验纵向对比

自身对比优先回答：新实验相对直接基线提升了什么、退化了什么、付出了多少成本。

| 对比组 | 实验 | 直接基线 | 主要变量 | 主要结论 |
|---|---|---|---|---|
| Pretrain 数据 | P02 full | P01 mini | Pretrain 数据规模 | Base 能力与单位成本变化 |
| 官方 SFT | S02 official full | P03 Base | 官方完整 SFT | Chat/Tool 能力与遗忘 |
| SFT 数据 A/B | S04 custom pilot | S03 official pilot | 等 assistant-target 的数据差异 | Chat/Tool 能力与遗忘 |
| 领域训练 | Full FT | S10 release | 全参数领域训练 | 领域收益与通用回归 |
| 参数高效训练 | LoRA | S10 release | LoRA 适配 | 领域收益、显存和耗时 |
| 偏好优化 | DPO | S10 release | Preference optimization | 目标任务收益与通用回归 |
| PPO | PPO | S10 release | Actor/Critic RL | held-out reward、准确率与成本 |
| Policy optimization | GRPO | S10 release | GRPO objective | held-out 任务收益与稳定性 |
| Loss 对照 | CISPO | GRPO | `loss_type=cispo` | 与 GRPO 的效果/稳定性差异 |
| 模型结构 | MoE | 同预算 Dense | Dense/MoE | 质量、吞吐、显存和激活参数 |
| 蒸馏 | Distill | 同结构未蒸馏学生 | CE + KL | 学生质量与训练成本变化 |
| Agentic RL | Agent RL | S10 Tool 或对应 RL 基线 | 多轮工具环境训练 | 端到端任务成功率变化 |

每个对比只能主动改变一个主要变量。若同时更换数据、初始化权重、训练 Token、评测模板或推理后端，则只能记录为新基线，不能声称单变量收益。

### 2.2 轨道 B：MiniMind 官方结果横向对比

官方 README 使用 `lm-evaluation-harness` 和以下七项任务：

- `ceval-valid`；
- `cmmlu`；
- `arc_easy`；
- `piqa`；
- `openbookqa`；
- `hellaswag`；
- `social_iqa`。

官方 64M/198M 参考分数如下。报告必须逐项展示，不只报告平均值。

| 官方模型 | 参数规模 | C-Eval | CMMLU | ARC-Easy | PIQA | OpenBookQA | HellaSwag | Social IQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| minimind-3 | 64M | 24.89 | 25.38 | 28.49 | 50.65 | 23.60 | 28.28 | 34.19 |
| minimind-3-moe | 198M | 25.48 | 24.32 | 27.74 | 50.71 | 26.20 | 27.43 | 34.03 |
| minimind-3-exam | 64M | 30.98 | 26.12 | 35.61 | 56.26 | 24.20 | 28.40 | 34.19 |

本表来源于本仓库 `minimind/README.md`，对应当前引入的 MiniMind 源码 commit `393e387e9ad99f0f04c296e4c5e7353f4444629f`。升级源码后若官方表发生变化，必须保留旧表并新增协议版本，不能覆盖历史参照。

`minimind-3-exam` 使用过选择题格式对齐数据，不能当作普通 Pretrain/SFT 基线。与它比较时必须标注“exam-format-aligned”，不得据此宣称通用知识超过普通官方模型。

官方 README 没有固定 `lm-evaluation-harness` commit、数据集 revision 和完整运行环境。因此本项目结果应表述为“官方任务与调用口径对齐”，不能声称 bitwise reproduction。首次正式评测时必须固定本项目自己的 harness commit，后续所有自身对比保持不变。

## 3. Checkpoint 与必测矩阵

| Checkpoint | 官方七项 | Chat/Tool 固定集 | RL held-out | Agent held-out | 系统指标 |
|---|---|---|---|---|---|
| P01/P02 Base | 必测，不加 chat template | 不测 | 不测 | 不测 | 必测 |
| S01–S10 | 必测，加 chat template | 必测 | S10 作为后训练基线 | S10 Tool 作为基线 | 必测 |
| Full FT/LoRA | 必测通用回归 | 必测领域集 | 不测 | 不测 | 必测 |
| DPO | 必测通用回归 | 必测目标集 | 可选 | 不测 | 必测 |
| PPO | 必测通用回归 | 必测目标集 | 必测 | 不测 | 必测 |
| GRPO/CISPO | 必测通用回归 | 必测目标集 | 必测 | 不测 | 必测 |
| MoE | 必测 | 按对应训练阶段执行 | 按对应训练阶段执行 | 不测 | 必测 |
| Distill | 必测 | 按 teacher/student 任务执行 | 不测 | 不测 | 必测 |
| Agentic RL | 必测通用回归 | 必测 Tool 基线 | 若复用 RL 任务则必测 | 必测 | 必测 |

P01/P02/P03、S10 是主线结果；后训练 checkpoint 必须与其**直接初始化基线**比较，不能只和更早或结构不同的模型比较。

## 4. 官方七项运行口径

### 4.1 模型准备

- 使用 `minimind/scripts/convert_model.py` 导出可被 Hugging Face loader 读取的目录。
- 记录原始 checkpoint 路径和 SHA-256、导出目录、MiniMind commit、Lab commit、Tokenizer 版本与配置。
- Base 与 Chat checkpoint 分开导出，不覆盖已有结果。

### 4.2 Base 模型

Base 模型不使用 `--apply_chat_template`：

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/exported-base,dtype=auto \
  --tasks ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa \
  --batch_size 16 \
  --device cuda:0 \
  --trust_remote_code \
  --output_path /path/to/result-dir
```

### 4.3 SFT/Chat 模型

SFT、DPO、PPO、GRPO、CISPO 和 Agent checkpoint 使用相同任务，但增加 `--apply_chat_template`：

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/exported-chat,dtype=auto \
  --tasks ceval-valid,cmmlu,arc_easy,piqa,openbookqa,hellaswag,social_iqa \
  --batch_size 16 \
  --device cuda:0 \
  --trust_remote_code \
  --apply_chat_template \
  --output_path /path/to/result-dir
```

正式执行前以当前固定的 `lm_eval --help` 校验参数。命令模板若因 harness 版本变化而调整，必须新建评测协议版本，不得静默修改历史结果。

官方七项准确率使用 README 示例的 `dtype=auto`。CPU/GPU 设备可以根据执行成本选择，但必须记录并在自身对比中保持一致；吞吐、延迟和显存等系统指标另用第 7 节固定的 L20 BF16 条件测试，不能和官方准确率命令混为一次性能对比。

### 4.4 选择题评分说明

七项任务以 harness 的条件概率选择题评分为准，不用自由生成答案后再做字符串匹配。报告同时保留 harness 原始 metric 名称及 normalized/非 normalized 结果；主表只选择一个事先固定的 metric，不能在看到结果后择优展示。

## 5. 自身专项评测

### 5.1 Chat、Tool 与领域任务

- Chat：固定、可判分的指令集合，报告成功率、格式遵循率、重复率和拒答异常。
- Tool：基于 `minimind/scripts/eval_toolcall.py` 固化测试样本，报告工具选择、参数合法性和最终答案正确率。
- 领域任务：从未进入训练数据的固定 held-out 集评测，报告任务准确率和通用七项回归。
- 人工抽查只能作为补充；必须固定抽样 seed、样本数和评分 rubric。

### 5.2 PPO、GRPO 与 CISPO

训练指标记录：

- reward；
- KL to reference；
- policy loss；
- PPO 的 critic loss、approx KL 和 clip fraction；
- GRPO/CISPO 的 group reward std 与 advantage mean/std；
- response length；
- NaN、OOM、异常梯度和退化样本数量。

效果结论只使用与训练集隔离的 held-out 任务：

- task accuracy / pass rate；
- held-out average reward；
- 格式成功率；
- 相对 S02 或直接 RL 基线的变化；
- 官方七项通用能力回归。

reward 上升但 held-out 准确率不升，不能判定 RL 有效。GRPO 与 CISPO 必须固定初始化、数据、seed、rollout 数、reward 函数和生成参数，只改变 loss。

### 5.3 Agentic RL

Agentic RL 使用与训练轨迹隔离的固定工具任务，至少报告：

- tool selection accuracy；
- argument validity；
- execution success；
- final answer accuracy；
- end-to-end task success；
- average tool calls / turns；
- unfinished rate；
- invalid tool-call tag rate；
- 推理时间与 Token 成本。

环境版本、工具 schema、超时、最大轮数、失败重试和执行错误处理必须固定。Agentic RL 的主要结论来自端到端任务成功率，不来自训练 reward。

## 6. 固定条件与数据隔离

每次正式评测必须在 `eval_manifest.json` 中记录：

- experiment ID、checkpoint ID、直接基线 ID；
- Lab commit、MiniMind commit、`lm-evaluation-harness` commit；
- checkpoint SHA-256、模型配置和 Tokenizer 版本；
- benchmark 名称、dataset revision、split 和样本数量；
- few-shot 数、chat template、system prompt；
- batch size、dtype、device、seed；
- decoding 参数、最大输入/输出长度；
- CUDA、PyTorch、Transformers 和推理后端版本；
- GPU 型号、数量及评测开始/结束时间。

训练、格式对齐和评测数据必须隔离。若训练数据或 LoRA 数据与 benchmark 有样本交集，相关分数标记为 `contaminated`，不得进入官方对比主表。

## 7. 系统性能对比

系统指标只在相同硬件和相同输入分布下比较：

- wall-clock time；
- tokens/s、samples/s；
- GPU utilization；
- peak allocated/reserved VRAM；
- checkpoint size；
- 推理首 Token 延迟和生成吞吐。

每次性能测试固定 batch size、输入长度、输出长度、dtype、attention backend 和 GPU 数量。至少预热 5 次、正式重复 20 次，报告 median 和 P95；出现 OOM 的配置直接记录为 OOM，不通过缩短输入掩盖。

## 8. 结果目录与文件格式

每个 checkpoint 的评测结果放在对应实验目录：

```text
experiments/<stage>/<experiment-id>/eval/
├── eval_manifest.json
├── official_benchmarks.json
├── task_eval.json
├── system_metrics.json
├── samples.jsonl
└── summary.md
```

- `official_benchmarks.json`：保留 `lm-evaluation-harness` 原始输出或其不可变副本。
- `task_eval.json`：Chat、Tool、领域、RL 或 Agent held-out 指标。
- `system_metrics.json`：训练/推理系统指标和测试条件。
- `samples.jsonl`：失败样本、预测、标签和错误类型；不得包含密钥或敏感数据。
- `summary.md`：官方差距、自身基线变化、成本变化和异常解释。

大体积完整日志和权重放入 `artifacts/` 或外部存储；Git 只保存 manifest、聚合结果、必要样例和报告。

## 9. 结果表与判定规则

### 9.1 官方七项主表

| checkpoint | template | C-Eval | CMMLU | ARC-Easy | PIQA | OpenBookQA | HellaSwag | Social IQA | 七项宏平均 | 与官方同类差值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

宏平均只用于快速概览，不能替代逐项分数。官方同类差值按同结构、同训练阶段优先选择：64M Dense 对 minimind-3，MoE 对 minimind-3-moe；`minimind-3-exam` 单列。

### 9.2 自身对比主表

| experiment | direct baseline | target metric | baseline | result | absolute delta | relative delta | 七项平均回归 | train GPU-hours | inference cost |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

每个实验至少回答：

1. 相对直接基线，目标指标的绝对和相对变化是多少？
2. 七项通用能力中哪些提升、哪些下降？
3. 训练和推理成本变化多少？
4. 多个 seed 或固定失败样本是否支持同一结论？
5. 是否存在污染、reward hacking、长度偏差或格式投机？

任何通过阈值必须在训练前写入实验 config；不得观察结果后倒推阈值。若尚无合理阈值，则如实报告差值和不确定性，不使用“达到官方水平”“显著提升”等结论。

## 10. 完成条件

一次正式评测只有同时满足以下条件才算完成：

1. 直接基线和实验模型使用同一协议版本；
2. 官方七项或本阶段必测任务完整，无静默跳过；
3. manifest、原始结果、聚合表和失败样例能够互相追溯；
4. Git commit、checkpoint、SwanLab run 和报告路径已经登记；
5. 数据污染、OOM、异常退出和不兼容项均已显式标记；
6. 结果已写入实验报告和 `experiments/registry.csv` 对应记录。
