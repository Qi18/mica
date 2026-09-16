# Phase6 RL v2 FP32复验（探索性）

在v2训练及新test之前登记，沿用v1预注册的三臂、数据、奖励、64更新预算、LR与验收标准。
v1 A02完成38更新、A03完成22更新后，被generation/teacher-forced logprob最大差>0.20门禁中止；
未保存最终模型、未打开test，不能比较这两个不等预算中止点。保留日志、源码和FAILURE.json。

v2唯一系统修复：生成、old/reference/new前向统一FP32，并关闭TF32；不放宽审计阈值。
两组重新从A01-v3 LR3e-6 checkpoint-570开始，不恢复中止模型。
先重复同规模32prompt×4rollout探针，符合奖励方差门槛才训练。
数据、seed42、T1/top_p1/top_k0、32outer×2inner、LR1e-6、KL0.01和两个surrogate保持不变。
实际token数按各模型rollout记录，不能声称严格同token消耗。
最终三臂graph/tools评测也统一FP32；旧Chat/Tool三臂统一现有FP16脚本。
v3 frozen test此前仍未打开；三臂权重SHA冻结后才用于一次性RL比较。
本轮仍非正式promote，A01旧Tool退化问题与IFEval/七项门禁保持不变。
