# R02A-s10-math-baseline-20260908

状态：completed；S10 不训练控制组。

- test：pass@1 23.25%，sample accuracy 20.94%，empirical pass@4 41.00%；choice-valid 97.75%，strict format 0%。
- greedy 分布：A 373 / C 18 / invalid 9，已经存在明显 A 偏置，所以结论必须看分桶和 pass@4。
- 通用回归复用同一 S10 FP32 权重的 Phase4 结果：七项 33.0146%，IFEval prompt strict 16.8207%，Chat 8/10，Tool E2E 7/8。
- checkpoint SHA-256：`851ad62d58986bc72b6d8b0e17ebd06e6635f8f5c7161aff8e50ab1104569bce`。
- 统一评测：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3sfq26tj。
