# R02B-math-sft-control-20260908

状态：completed-control-failed；不晋级。

- 250 outer steps / 500 updates，wall 15.16 s，4,000 completion targets；logged loss 2.2454→1.1699。
- test：pass@1 5.50%，sample accuracy 14.75%，empirical pass@4 41.25%；302/400 greedy 输出无有效选项。
- 通用回归出现灾难性遗忘：IFEval 7.3937%，Chat 0/10、格式 0/6、Tool E2E 1/8；七项 macro 32.9025%。
- 训练：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/wqhvn9wd；统一评测：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3sfq26tj。
- checkpoint SHA-256：`37ec69a7f1a35e043898b46f0acb9ac7528f9c1a124c46c827b63484767e684a`；仅作失败证据。
