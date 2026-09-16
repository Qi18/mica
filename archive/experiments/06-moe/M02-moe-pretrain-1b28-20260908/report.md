# M02-moe-pretrain-1b28-20260908

状态：completed；2026-09-09 Phase7实验验收收口，模型未晋级。
随机初始化、同数据顺序、1.28B唯一有效targets、9038updates，训练正常完成。
最终NLL=2.62454679，PPL=13.79831930。
七项宏平均=32.03167444%，29638题，固定final checkpoint，无chat template。
训练循环耗时=8938.690s；4张L20顺序运行两臂。
训练：[SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/uvoc97pz)；评测：[SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/dxfvjdxy)。
配置见config.json，训练过程见metrics.jsonl，评测与权重SHA见eval/manifest.json，完整结果见eval/results.json。
逐题无文本配对记录保留为eval/paired-scores.jsonl.gz；原始题目/输出、权重、数据与完整日志只留CPFS。
完整结论与限制见[阶段报告](../../../docs/phases/phase7-moe.md)。
