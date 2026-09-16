# D02-chosen-only-sft-20260908

状态：completed；作为“只看 chosen 优质回答”的 DPO 控制组，不替换 S10 release。

- 训练：1 epoch、444 updates、4,765,435 chosen targets；wall 182.35 s，峰值显存 5,249.71 MiB。
- 最终 validation chosen NLL 从 S10 的 2.26642 降至 2.13479（-5.8%）。末个训练 batch 的 CE 为 1.95096；训练日志的局部升降不是“未收敛”的依据。
- test preference credit：55.2%；但 chosen 更长时 75.86%，chosen 不更长时仅 32.28%，reward margin 与长度差 Pearson r=0.627，存在明显长度偏差。
- 七项 macro：33.0636%；IFEval prompt/instruction strict：16.4510% / 29.8561%；固定 Chat/格式/Tool 与 D01 相同。
- 预注册的“最小 validation DPO loss”不适用于 chosen-only 训练，否则会错误选择初始 S10；本次透明修正为固定一轮训练完成后的 last checkpoint，不据此调参。
- 200 条盲评中，D03 相对 D02 为 13 胜 / 166 平 / 21 负，score 0.480，95% CI [0.4525, 0.5075]。
- 训练 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/51vdt97x；统一评测 run：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/csvnmhkj。

原始 best.pth/last.pth 曾因保存前 model.half() 丢失微小更新，已作废；正式结果使用从 resume.pt 恢复的 float32 checkpoint，见 checkpoint-manifest.txt。
