# 分阶段训练

| 目录 | 内容 |
|---|---|
| tokenizer/ | Tokenizer 训练 |
| pretrain/ | 统一预训练入口，可选精确 validation |
| sft/ | 全参 SFT、validation 与 batch 探针 |
| lora/ | LoRA 微调 |
| dpo/ | 偏好优化 |
| grpo/ | GRPO |
| ppo/ | PPO |
| agent/ | Agent RL |
| moe/ | Dense/MoE 对照训练（原 train_phase7.py） |
| distill/ | 蒸馏 |
| common/ | 配置驱动引擎 engine.py、共享工具 utils.py、rollout.py |

启动目录、数据、权重和恢复约束统一见 [训练指南](../docs/training.md)。
