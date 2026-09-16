# R02C-math-grpo-20260908

状态：completed-not-promoted。

- 250 outer steps / 500 updates，16 rollouts/prompt，wall 44.56 s，85,646 completion tokens，mean training reward 24.21%，峰值显存 3,325.80 MiB。
- test pass@1 25.00%，相对 R02A +1.75pp，paired bootstrap 95% CI [+0.25,+3.50]；但 400/400 greedy 全答 A，empirical pass@4 从 R02A 的 41.00% 降到 25.00%。
- 通用回归：七项 33.0599%，IFEval 15.3420%，Chat 9/10、格式 6/6、Tool E2E 7/8，回归门通过但不能抵消目标任务的位置塌缩。
- 训练：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/6t4l6v1o；统一评测：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3sfq26tj。
- checkpoint SHA-256：`9bf2c4634cc9c194d7a5afa2f7abfa7eccef6bc173aa0291dc6378894edaa1c1`；不替换 S10。
