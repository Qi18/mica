# Phase7 Dense vs MoE

当前状态：completed-not-promoted，2026-09-09用户验收收尾。S10保留为release。
M01/M02正式训练及各29638题评测已完成；宏平均31.4703%/32.0317%，差+0.5613pp，配对95%CI[-0.1647,+1.2866]。
MoE验证NLL略差、训练循环耗时增加约40%；未证明综合优势。
完整报告见[Phase7](../../docs/phases/phase7-moe.md)，comparison.json、training-summary.json和swanlab-cloud-verification.json给出证据。
M00六探针、M00b r1/r2失败与r3通过记录全部保留；这些历史目录描述的是当时快照，不代表当前阶段状态。
正式实验：[M01](M01-dense-pretrain-1b28-20260908/report.md)、[M02](M02-moe-pretrain-1b28-20260908/report.md)。
原始权重/数据/题目/完整日志只在CPFS，未删除；无文本配对分数压缩归档以便复算。
