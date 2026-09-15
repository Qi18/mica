# MiniMind Lab

> 基于开源 [MiniMind](https://github.com/jingyaogong/minimind) 的个人大模型训练与评测实验室。不是 MiniMind 官方仓库。

MiniMind Lab 是一个围绕 64M 级小语言模型展开的可复现实验项目，覆盖数据与 Tokenizer、模型结构、预训练、SFT/LoRA、DPO、GRPO/CISPO、Agentic RL、统一评测和推理优化。项目重点不是复刻一次训练，而是用统一协议回答：一次看似有效的训练改动，是否真的提升了目标能力，又是否损伤了通用能力。

本仓库保存实验配置、结果、源码阅读笔记和报告；模型源码通过 Git Subtree 导入 [`minimind/`](minimind/)，当前基线固定到上游 commit `393e387`。

## 我完成的工作

- 建立从环境门禁、数据审计、训练、统一评测到归档的实验工程，并为每次运行记录源码、数据、命令、硬件、权重摘要和 SwanLab 链接。
- 在 8×NVIDIA L20 上完成 64M Dense 模型的 Pretrain、Full SFT、LoRA、DPO、GRPO/CISPO、Agentic RL 与 Dense/MoE 对照。
- 用 held-out、配对 bootstrap、行为门和通用回归识别“训练指标上涨但能力没有提升”的负结果，包括答案位置 reward hacking、灾难性遗忘和离线偏好目标错位。
- 实现并测量 KV Cache / GQA 四臂推理实验，保留 FP32/BF16 数值一致性边界、吞吐、显存和原始测量证据。
- 将上游 MiniMind 作为固定来源快照维护；个人实验逻辑、配置、评测脚本与报告均在本仓库演进。

## 代表性结论

| 主题 | 结果 | 结论边界 |
|---|---|---|
| Pretrain 数据工程 | P03 仅用 P02 约 63% 的有效 targets，将共享 validation NLL 从 3.191 降至 2.604，训练 wall 缩短 83.8% | 是整套数据管线收益，不能归因于单一数据源 |
| SFT | S10 的 IFEval prompt strict 为 17.38%，相对 S09 +6.29pp，七项 macro 基本稳定 | 提升集中在可验证指令，不能外推为开放域 Chat 全面提升 |
| DPO / RL | DPO 离线 preference credit 提升，但盲评未胜出；GRPO/CISPO 暴露答案位置策略与 pass@4 回退 | 均按 completed-not-promoted 收口，S10 保持 release |
| Dense / MoE | MoE 七项 macro +0.56pp，但配对区间跨 0，训练循环慢约 40% | 单 seed 下没有明确综合优势 |
| KV Cache / GQA | A10 上 cache decode 为 0.82–6.23×；GQA 将 KV 字节精确减半，但 cache-on 吞吐略低于等价 MHA | 仅适用于当前小模型和原生实现，不代表融合推理框架 |

详细数字、协议与失败分析见[最终实验报告](docs/final_report.md)和[阶段报告索引](docs/phases/README.md)。

## 快速开始

```bash
git clone https://github.com/Qi18/minimind-lab.git
cd minimind-lab

# 不需要 GPU：检查公开仓边界、Markdown 本地链接和 Python 语法
python3 scripts/check_repository.py
python3 -m compileall -q scripts minimind/model minimind/trainer
```

训练与评测依赖沿用 [`minimind/requirements.txt`](minimind/requirements.txt)。正式复现实验前先阅读[实验计划](docs/experiment_plan.md)、[统一评测协议](docs/evaluation_protocol.md)和[仓库管理方式](docs/repository-management.md)；命令、配置与硬件口径以对应实验目录中的 `command.sh`、`config.json`、`run.json` 为准。

## 项目目标

- 在 NVIDIA L20 环境完成 MiniMind 64M Dense 的端到端训练闭环。
- 用统一协议比较 Pretrain、SFT、LoRA、DPO、GRPO/CISPO 与 Agentic RL。
- 将每项结论绑定到源码 commit、实验配置、SwanLab run 和评测结果。
- 形成可用于博客与简历的工程化报告，而不是只保留训练日志。

## 仓库导航

- [统一实验与学习计划](docs/experiment_plan.md)
- [阶段报告索引](docs/phases/README.md)
- [仓库管理方式](docs/repository-management.md)
- [统一评测协议](docs/evaluation_protocol.md)
- [数据工程文档索引](docs/data/README.md)
  - [Pretrain v1 数据协议](docs/data/pretrain/data_protocol.md)
  - [SFT v1 数据协议](docs/data/sft/data_protocol.md)
- [A10 KV Cache / GQA 推理实验](experiments/08-inference/I01-I04-a10-20260914/report.md)
- [最终实验报告](docs/final_report.md)
- [简历项目介绍](docs/resume_project.md)
- [Phase8 蒸馏负结果与收口报告](experiments/07-distill/sequence-v1-20260909/report.md)
- [训练前准备实验](experiments/00-preparation/README.md)
- [实验登记规范](experiments/README.md)
- [源码阅读索引](docs/source_reading/README.md)
- [MiniMind 上游来源与同步](docs/upstream-minimind.md)

## 实验主线

```text
Preparation
   ├─ Environment / L20 Baseline
   ├─ Data / Tokenizer Audit
   └─ Model / DDP / Resume Probe
                  ↓
Pretrain ───────────────→ Base Evaluation
        ↓
Full SFT
   ├────┼───────────┬─────────────┐
   ↓    ↓           ↓             ↓
 LoRA  DPO     GRPO / CISPO   Agentic RL
   └────┴───────────┴─────────────┘
                  ↓
            Unified Evaluation
                  ↓
       Service / Report / Resume
```

DPO、GRPO/CISPO 和 Agentic RL 默认从同一个 Full SFT 基线分支，避免把多个阶段串联后无法归因。

## 结果总表

| 阶段 | 初始化权重 | 数据 | 训练成本 | 目标指标 | 通用能力回归 | SwanLab | 权重 |
|---|---|---|---:|---:|---:|---|---|
| Pretrain mini（P01） | Random | `pretrain_t2t_mini` 1,270,238 行 | 45.83 min / 6.11 GPU-hours | 七项宏平均 31.44 | 无自身基线；官方口径参考 +0.66 pp | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nfax3tyg0j217j1cz8y0b) | L20 保留，SHA `71efd40d` |
| Pretrain full（P02） | Random | `pretrain_t2t` 8,468,827 行 | 285.78 min / 38.10 GPU-hours | 七项宏平均 30.91 | 相对 P01 -0.54 pp；官方口径参考 +0.12 pp | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3i1muwq039fpfv89fq4ru) | L20 保留，SHA `7065a461` |
| Pretrain v1（P03） | Random | `pretrain-v1-1b28/final-remix-v1` 2,313,483 行 / 1.280B targets | 46.37 min / 6.18 GPU-hours | 七项宏平均 31.52；共享 validation NLL 2.60432 | 相对 P02 +0.62 pp，相对 P01 +0.08 pp | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/qdpjh47fjt98184oos4bl) | L20 保留，SHA `0cfb7fc8` |
| Full SFT（S10 release） | S09 step 400 / P03 64M | `ifeval-curriculum-v4` 41,720 行 / 4.163M assistant targets | 7×L20，3 epochs / 1,116 steps / 3.73 min | IFEval prompt strict 17.38%，较 S09 +6.29 pp；Chat 8/10；Tool E2E 7/8 | 七项 chat-template macro 33.00，与 S09 持平；语言/关键词/长度仍有限 | [train](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/bxdob3rh) / [eval](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/encs5zuk) | L20 保留 release，SHA `46aeab66` |
| Full FT（L01） | S10 | CodeAlpaca 19,015行 / 4.820M实际targets | 94.65s / 0.15775 GPU-h | MBPP 0/500；val NLL 0.7577 | 七项32.48%；Chat7/10；格式3/6 | [train](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/pz1x0ux9) | CPFS；不替换S10 |
| LoRA r16（L02） | S10 | 同L01 | 63.46s / 0.10576 GPU-h；峰值4,464MiB | MBPP 0/500；adapter 0.393M参数 | 七项32.67%；Chat8/10；格式5/6 | [train](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s0l5ishg) | adapter约779KiB；不替换S10 |
| DPO（D03） | S10 | official-dpo-v1 14,194 train pairs | 1×L20 / 376.21 s | preference credit 71.8%；对 D02 +16.6pp | 七项 33.032%；盲评 vs D02 score 0.480，未晋级 | [train](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3w1au7dk) / [eval](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/csvnmhkj) | FP32 恢复权重仅 L20 保留；S10 仍为 release |
| GRPO / CISPO（R02C/R02D） | S10 | verifiable-math-v2：1,000 train / 200 val / 400 test | 1×L20；44.56 s / 43.64 s | pass@1 均 25.0%，但 GRPO 100% 答 A、CISPO 75% 答 B；不晋级 | 七项 33.06% / 33.13%；IFEval 15.34% / 17.19%；Chat 9/10 / 7/10 | [GRPO](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/6t4l6v1o) / [CISPO](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/k6ez8f9j) / [eval](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3sfq26tj) | FP32 仅 L20 保留；S10 仍为 release |
| Agentic RL（A02/A03） | A01-v3 Agent SFT | 混合graph/tools；128 prompts ×4 rollout | 1×L20共享；136.84s / 134.88s | A01/GRPO 451/480；CISPO461/480（+2.08pp） | CISPO七项33.22%、IFEval15.34%；旧Tool5/8，不晋级 | [GRPO](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1pxh36vn) / [CISPO](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/vdwjd3sg) | 仅CPFS，S10保留 |

表格只填写已经完成并能追溯到实验目录的结果。

[Phase 3 已收尾](docs/phases/phase3-lora.md)：LoRA本次更省资源，但两组代码正确率均无提升；这是负结果，不作为代码能力提升的简历结论。

[Phase 4 已收尾](docs/phases/phase4-dpo.md)：D03 的 held-out preference credit 达 71.8%，但 200 条盲评没有胜过 D02（score 0.480，95% CI [0.4525, 0.5075]），因此完成但不晋级，继续保留 S10。

[Phase 5 已收尾](docs/phases/phase5-verifiable-rl.md)：修复 v1 的模板—答案位置混杂后，R02 的 GRPO/CISPO 在平衡 test 上都只有 25% pass@1，并塌缩为答案位置策略；pass@4 还比 S10 低约 16pp，因此完成但不晋级。

当前结论：P03 是 Pretrain 主线，S10 是后训练共同 release。Phase3–5 均已完成受控对照，但没有新的候选权重通过目标能力门，均不替换 S10；DPO 的离线 preference 改善和 RL 的训练 reward 都不能单独外推为真实能力提升。

`minimind/` 已作为普通源码目录进入仓库，不需要初始化 Submodule。

## 产物边界

- GitHub：配置、命令、指标、评测结果、报告和外部链接。
- SwanLab：训练曲线、系统指标、样本输出与实验对比。
- Hugging Face：最终选中的少量模型权重与 Model Card。
- L20：数据集、活动 checkpoint、优化器状态、缓存和完整原始日志。

未经评测或无法复现的数字不进入 README 和简历。

[Phase6已按completed-not-promoted收尾](docs/phases/phase6-agentic-rl.md)：CISPO在单seed合成任务中有专项收益，但旧Tool未达7/8门槛；通用回归已完整执行，不替换S10。

## Phase7 最新结果

Dense/MoE同数据预算原生对照已收尾：七项macro 31.47% / 32.03%，差+0.56pp且配对95%CI跨0；MoE验证NLL略差、训练循环耗时多约40%。结论为completed-not-promoted，S10保持release。[完整报告](docs/phases/phase7-moe.md)。
