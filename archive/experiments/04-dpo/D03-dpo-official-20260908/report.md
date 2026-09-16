# D03-dpo-official-20260908

状态：completed-not-promoted；偏好目标有效，但未通过预注册的回答质量门槛，保留 S10 release。

- 训练：1 epoch、444 updates、4,765,435 chosen / 4,375,948 rejected targets；wall 376.21 s，峰值显存 9,084.02 MiB。
- validation DPO loss 从 0.693147 降至 0.605810，preference credit 达 69.1%；test DPO loss 0.594240，preference credit 71.8%。
- 相对 D01 的 held-out preference credit +21.8 pp，bootstrap 95% CI [19.0, 24.6]；相对 D02 +16.6 pp，[12.8, 20.5]，偏好目标通过。
- 长度检查：chosen 更长/不更长 credit 分别 69.01% / 74.89%，margin 与长度差 Pearson r=0.0219，未见 D02 那样的强长度依赖。
- 七项 macro：33.0320%；IFEval prompt/instruction strict：16.8207% / 29.7362%；固定 Chat 8/10、格式 5/6、Tool E2E 7/8，通用回归门通过。
- 200 条 Qwen3-8B 盲评：相对 D01 为 18/170/12，score 0.515，[0.4875, 0.5425]；相对 D02 为 13/166/21，score 0.480，[0.4525, 0.5075]。未证明回答质量优于控制组。
- 训练 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3w1au7dk；统一评测 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/csvnmhkj。

原始 FP16 inference checkpoint 将 test preference credit 从 71.8% 压到 53.6%，已作废。训练入口已改为 float32 保存，正式结果使用从 resume.pt 恢复并复核的 float32 checkpoint。
