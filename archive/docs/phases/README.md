# 阶段报告索引

每个 Phase 完成后在本目录出具一份阶段报告，规范见 [`../experiment_plan.md`](../experiment_plan.md) 第 16 节。新报告从 [`_template.md`](_template.md) 复制，文件名固定。

阶段报告合入 main 是进入下一 Phase 的硬门；报告只允许写能从 `experiments/registry.csv`、eval manifest、SwanLab 和 checkpoint manifest 追溯的数字，并必须给出对应训练 run 的完整 URL（参见 [`swanlab-runs.md`](swanlab-runs.md)）。

| Phase | 阶段 | 报告 | 状态 |
|---|---|---|---|
| 0 | 环境、数据与代码探针 | [`phase0-preparation.md`](phase0-preparation.md) | draft |
| 1 | Pretrain 基线与数据重做 | [`phase1-pretrain.md`](phase1-pretrain.md) | accepted |
| 2 | SFT（官方数据基线、定向课程与 IFEval） | [`phase2-sft.md`](phase2-sft.md) | accepted with limitations |
| 3 | Full FT vs LoRA | [`phase3-lora.md`](phase3-lora.md) | accepted with limitations（负结果收尾） |
| 4 | DPO 偏好优化 | [phase4-dpo.md](phase4-dpo.md) | completed-not-promoted |
| 5 | GRPO/CISPO 可验证强化学习 | [`phase5-verifiable-rl.md`](phase5-verifiable-rl.md) | completed-not-promoted |
| 6 | Agent SFT 与 Agentic RL | [phase6-agentic-rl.md](phase6-agentic-rl.md) | completed-not-promoted：实验验收收尾，S10保留 |
| 7 | Dense vs MoE | [phase7-moe.md](phase7-moe.md) | completed-not-promoted：原生对照实验收尾 |
| 8 | 蒸馏：off-policy 与 on-policy（依赖 MoE teacher） | [phase8-distill.md](phase8-distill.md) | 进行中：K00 teacher资格检查；未启动正式训练 |

状态取值：未开始 / 进行中 / 待出具 / draft / accepted / accepted with limitations / 已跳过。报告出具或状态变化的当天更新本表。

Phase 2 内含一个条件分支：只有官方 SFT 数据未过门槛且失败可归因到数据时才构建 SFT-v1；无论是否触发，结论都写在同一份 `phase2-sft.md`（见 [`../experiment_plan.md`](../experiment_plan.md) 7.5）。

Phase 8 蒸馏排在最后：它需要同 tokenizer 的 MoE Full SFT teacher，依赖 Phase 7 或上游模型；其中 K04 on-policy 蒸馏（OPD）还要复用 Phase 5 的 rollout 路径。无可用 teacher 时以“已跳过”出具一页报告；teacher 可用但 rollout 不可用时只报 K01–K03 并说明 K04 推迟原因。本项目不扩展到 1B，因此没有对应的 Phase。
