# L01-code-full-ft-20260907

状态：completed；本实验结果不替换 S10 release。

- MBPP pass@1：0.00%（0/500）。
- 七项 chat-template macro：32.4759%。
- IFEval：{"prompt_level_strict_acc": 17.929759704251385, "inst_level_strict_acc": 30.215827338129497, "prompt_level_loose_acc": 20.33271719038817, "inst_level_loose_acc": 32.49400479616307}。
- Chat：7/10；格式：3/6；Tool E2E：75.0%。
- 训练实测：{"event": "completed", "global_step": 594, "baseline_validation_loss": 0.9745552012169754, "best_validation_loss": 0.7577329559380626, "wall_seconds": 94.64988684654236, "peak_memory_mib": 6936.5078125, "total_parameters": 63912192, "trainable_parameters": 63912192, "trainable_ratio": 1.0, "timestamp": "2026-09-07T14:48:36Z", "gpu_hours": 0.15774981141090394, "overall_assistant_targets_per_second": 50926.959984803056, "training_seed": 42, "dataset_split_seed": 20260907}。
- 训练 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/pz1x0ux9；评测 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/gtetdf4e。

数据、成本对照、限制及阶段判定见 [Phase 3 报告](../../../docs/phases/phase3-lora.md)。
逐项指标及原始结果路径/SHA 见 eval.json；训练过程见 metrics.jsonl（训练组）；checkpoint-manifest.txt 保留权重指纹和存储路径。
MBPP 的空提取 424/500、提取后语法错误 5/500，仅用于诊断；不得把 loss 下降解释为代码正确率提升。
