# D01-s10-preference-baseline-20260908

状态：completed；不训练，作为 Phase4 的 S10 共同基线。

- 数据：冻结 official-dpo-v1 test 1,000 对。
- reference-relative preference credit：50.0%。这是 policy=reference 时奖励差恒为 0 的定义值，不是 S10 天然选对 50%。
- test chosen/rejected NLL：2.28272 / 2.28669；raw length-normalized credit：52.3%。
- 七项 chat-template macro：33.0146%；IFEval prompt/instruction strict：16.8207% / 29.7362%。
- 固定行为：Chat 8/10、格式 5/6、重复异常 1/10、Tool E2E 7/8。
- 200 条 Qwen3-8B 盲评中，D03 相对本基线 18 胜 / 170 平 / 12 负，score 0.515，bootstrap 95% CI [0.4875, 0.5425]，未证明回答质量显著更好。
- 训练 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/os17qekn；统一评测 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/csvnmhkj。

逐项指标、权重 SHA 与原始 artifact 路径见 eval.json、checkpoint-manifest.txt；阶段判定见 [Phase4 报告](../../../docs/phases/phase4-dpo.md)。
