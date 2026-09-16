# L00-code-baseline-20260907

状态：completed；本实验结果不替换 S10 release。

- MBPP pass@1：0.40%（2/500）。
- 七项 chat-template macro：33.0024%。
- IFEval：{"prompt_level_strict_acc": 17.375231053604438, "inst_level_strict_acc": 29.97601918465228, "prompt_level_loose_acc": 19.77818853974122, "inst_level_loose_acc": 32.97362110311751}。
- Chat：8/10；格式：5/6；Tool E2E：87.5%。
- 训练实测：{}。
- 训练 run：n/a-no-training；评测 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/980w5n9d。

数据、成本对照、限制及阶段判定见 [Phase 3 报告](../../../docs/phases/phase3-lora.md)。
逐项指标及原始结果路径/SHA 见 eval.json；训练过程见 metrics.jsonl（训练组）；checkpoint-manifest.txt 保留权重指纹和存储路径。
MBPP 的空提取 336/500、提取后语法错误 21/500，仅用于诊断；不得把 loss 下降解释为代码正确率提升。
