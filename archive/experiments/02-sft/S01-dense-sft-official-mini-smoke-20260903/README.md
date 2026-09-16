# S01 Dense SFT Official Mini Smoke（8×L20）

状态：工程 smoke 已完成，能力门禁未通过。

## 目的

从 P03 Base 出发，用官方 `sft_t2t_mini` 的确定性小样本验证 chat template、assistant mask、分布式训练、validation、checkpoint 和 SwanLab 链路。

S01 只做工程 smoke，不用于宣称 Chat、Tool 或 benchmark 能力提升。正式能力结论从后续 8M LR probe 和 32M 正式训练产生。

## 冻结配置

- 3,000 行，seed 42；256 行固定 validation；训练约 1M shifted assistant targets；
- P03 checkpoint 初始化；8×L20；BF16；每卡 batch 8；1 epoch；lr `1e-5`；
- 关闭训练期增强，确保 manifest 的 target 统计与 trainer 一致；
- SwanLab：project `MiniMind-Lab`，run `S01-SFT-OfficialMini-Smoke-P03-64M-B8x8`。

## 训练结果

- 42/42 step，退出码 0，SwanLab 完成上传 154 条记录；
- validation loss：`2.951357 -> 2.681668`；
- 最后训练窗口 loss：`2.667477`；
- best checkpoint SHA-256：`c3cb6200ea516d6c5776ab687588832eb253a7f9a0b6fd3ac7fe746b839483b9`；
- SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/e4php2e6>。

## 行为验收

| 指标 | P03 Base | S01 smoke |
|---|---:|---:|
| Chat 通过 | 0/10 | 0/10 |
| 格式通过 | 0/6 | 1/6 |
| Chat 重复异常 | 9/10 | 10/10 |
| Tool 选择准确率 | 0% | 0% |
| Tool 端到端成功率 | 0% | 0% |

结论：数据、assistant mask、DDP、validation、checkpoint、导出和 SwanLab 链路可运行；但 3,000 行、约 1.16M 训练 targets 不足以产生可用指令能力。本实验不进入能力结论，下一步使用固定 8M targets 做学习率对比。
