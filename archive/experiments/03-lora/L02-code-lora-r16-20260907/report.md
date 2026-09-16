# L02-code-lora-r16-20260907

状态：completed；本实验结果不替换 S10 release。

- MBPP pass@1：0.00%（0/500）。
- 七项 chat-template macro：32.6742%。
- IFEval：{"prompt_level_strict_acc": 19.038817005545287, "inst_level_strict_acc": 31.414868105515588, "prompt_level_loose_acc": 22.36598890942699, "inst_level_loose_acc": 34.2925659472422}。
- Chat：8/10；格式：5/6；Tool E2E：75.0%。
- 训练实测：{"event": "completed", "global_step": 594, "baseline_validation_loss": 0.9745552012169754, "best_validation_loss": 0.8376368255766593, "wall_seconds": 63.45794105529785, "peak_memory_mib": 4464.2412109375, "total_parameters": 64305408, "trainable_parameters": 393216, "trainable_ratio": 0.006114820078584993, "timestamp": "2026-09-07T14:51:02Z", "gpu_hours": 0.10576323509216308, "overall_assistant_targets_per_second": 75959.46102001017, "training_seed": 42, "dataset_split_seed": 20260907}。
- 训练 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s0l5ishg；评测 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/8z9jka3t。

数据、成本对照、限制及阶段判定见 [Phase 3 报告](../../../docs/phases/phase3-lora.md)。
逐项指标及原始结果路径/SHA 见 eval.json；训练过程见 metrics.jsonl（训练组）；checkpoint-manifest.txt 保留权重指纹和存储路径。
MBPP 的空提取 394/500、提取后语法错误 5/500，仅用于诊断；不得把 loss 下降解释为代码正确率提升。
