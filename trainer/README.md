# 分阶段训练

所有阶段共用 model/model_mica.py，不复制模型实现。

| 目录 | 内容 |
|---|---|
| tokenizer/ | Tokenizer 训练 |
| pretrain/ | 预训练及 validation 版本 |
| sft/ | 全参 SFT、validation 与 batch 探针 |
| lora/ | LoRA 微调 |
| dpo/ | 偏好优化 |
| grpo/ | GRPO |
| ppo/ | PPO |
| agent/ | Agent RL |
| moe/ | Dense/MoE 对照训练（原 train_phase7.py） |
| distill/ | 蒸馏 |
| common/ | 配置驱动引擎 engine.py、共享工具 utils.py、rollout.py |

## 启动方式

配置驱动的统一入口不变，在仓库根目录执行 mica train --recipe configs/smoke/cpu.json --output outputs/smoke。

保留的阶段脚本使用原有相对路径约定：**从 trainer/ 目录执行，不要 cd 到阶段子目录**：

```bash
cd trainer
python pretrain/train_pretrain.py --help
python sft/train_full_sft.py --help
python lora/train_lora.py --help
python dpo/train_dpo.py --help
python moe/train_moe.py --help
```

默认 ../dataset、../tokenizer、../out 和 ../checkpoints 仍相对 trainer/ 解析。
正式运行请显式指定数据与产物路径。目录整理不改变训练算法，亦不代表已完成各阶段全量 GPU 验收。
历史 archive/ 中的原始命令不改写，重放使用归档对应提交。
