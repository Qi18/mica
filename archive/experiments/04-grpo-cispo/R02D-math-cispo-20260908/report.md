# R02D-math-cispo-20260908

状态：completed-not-promoted。

- 250 outer steps / 500 updates，16 rollouts/prompt，wall 43.64 s，85,267 completion tokens，mean training reward 23.38%，峰值显存 3,325.80 MiB。
- test pass@1 25.00%，相对 R02A +1.75pp，paired bootstrap 95% CI [-4.25,+8.00]；greedy 分布 B 300 / A 100，empirical pass@4 25.25%。
- 通用回归：七项 33.1256%，IFEval 17.1904%，Chat 7/10、格式 4/6、Tool E2E 7/8；回归门通过，但目标任务显著位置塌缩且主区间跨 0。
- 训练：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/k6ez8f9j；统一评测：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3sfq26tj。
- checkpoint SHA-256：`e906c5114075c54e74071c724aa906f0dbac6229292bee5d5182c0e221f90a19`；不替换 S10。
