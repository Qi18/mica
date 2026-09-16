# Phase 4：DPO 偏好优化

状态：completed-not-promoted；完成日期 2026-09-08。D03 通过离线偏好目标，但没有通过相对 D02 的盲评质量门，因此不替换 S10，也不把本轮 DPO 写成回答质量提升。

## 1. 问题与对照

三组共同从 S10（SHA-256 46aeab66...）出发：

| 实验 | 方法 | 训练目标 | 作用 |
|---|---|---|---|
| D01 | S10 不训练 | 无 | 生成、回归和 reference-relative 偏好基线 |
| D02 | chosen-only SFT | chosen assistant CE | 区分“多看优质答案”本身的收益 |
| D03 | reference-relative DPO | chosen/rejected pair | 检验 rejected 信号的独立贡献 |

D02/D03 使用同一 train 顺序、seed 42、1 epoch、global batch 32、LR 4e-8；D03 beta=0.15。D02 和 D03 的 chosen 曝光及 updates 相同，但 D03 额外前向 frozen reference 与 rejected，因此计算量不等价。

## 2. 数据与隔离

官方 dpo.jsonl 固定 revision 312afb4f76391145c6902f765bb51691c09a12f5，原始 SHA-256 ee934a8a...，上游说明抽样自 llamafactory/DPO-En-Zh-20k。原始 17,166 对经过质量和长度过滤后保留 16,194 对，按 prompt hash/seed 42 划分 train 14,194、validation 1,000、test 1,000。

独立审计确认跨 split exact/near duplicate 为 0，并对固定 IFEval、Chat、Tool prompt 进行近重复过滤。该检查不能证明 S10 历史训练语料与本测试集全量无语义重合。正式训练只读取 train/validation；test 在 checkpoint 序列化异常排查时被打开，之后没有据此修改超参数。

## 3. 训练结果

| 指标 | D01 | D02 chosen-only | D03 DPO |
|---|---:|---:|---:|
| updates | 0 | 444 | 444 |
| chosen / rejected targets | — | 4.765M / 0 | 4.765M / 4.376M |
| wall | — | 182.35 s | 376.21 s |
| 峰值显存 | — | 5,249.71 MiB | 9,084.02 MiB |
| final validation DPO loss | 0.693147 | 0.739003 | 0.605810 |
| final validation preference credit | 50.0% | 53.4% | 69.1% |
| final validation chosen NLL | 2.26642 | 2.13479 | 2.25197 |

D02 的训练 loss 是 chosen CE，不能用 DPO loss 选模；其 validation chosen NLL 下降 5.8%，说明训练确实学习了 chosen 数据。末段 batch loss 波动是 batch 难度差异，不等于整体 loss 发散。原预注册把两组都写成“最小 validation DPO loss”是协议错误；D02 透明修正为固定一轮的 last checkpoint。

## 4. Held-out preference

| 指标 | D01 | D02 | D03 |
|---|---:|---:|---:|
| test DPO loss | 0.693147 | 0.695743 | 0.594240 |
| preference credit | 50.0% | 55.2% | 71.8% |
| reward margin | 0.000 | 0.169 | 0.295 |
| raw length-normalized credit | 52.3% | 51.9% | 53.0% |
| chosen 更长 / 不更长 credit | 50.0 / 50.0% | 75.86 / 32.28% | 69.01 / 74.89% |
| margin 与长度差 Pearson r | — | 0.627 | 0.022 |

D03-D01 preference credit 为 +21.8 pp，paired bootstrap 95% CI [19.0, 24.6]；D03-D02 为 +16.6 pp，[12.8, 20.5]。所以 DPO 在自身离线目标上有稳定独立增益，且不像 D02 那样强依赖 chosen 更长。但 raw length-normalized credit 仅 53.0%，说明收益主要来自相对 reference 的序列 log-ratio，不应等价解释为模型绝对判断正确率。

## 5. 统一回归

| 指标 | D01 | D02 | D03 |
|---|---:|---:|---:|
| 七项 macro | 33.0146% | 33.0636% | 33.0320% |
| IFEval prompt strict | 16.8207% | 16.4510% | 16.8207% |
| IFEval instruction strict | 29.7362% | 29.8561% | 29.7362% |
| Chat / 格式 | 8/10 / 5/6 | 8/10 / 5/6 | 8/10 / 5/6 |
| Tool E2E | 7/8 | 7/8 | 7/8 |
| 重复异常 | 1/10 | 1/10 | 1/10 |

三组使用同一 float32 导出与同一评测协议，D03 通过预注册的通用回归门，但没有可辨识的通用能力提升。

## 6. 200 条生成盲评

候选均用 greedy、max_new_tokens=192、关闭 thinking。Qwen3-8B judge 看不到模型身份、reference chosen/rejected，以正确性、相关性、帮助性和表达为准；win=1、tie=0.5、loss=0。

| 对比 | D03 胜/平/负 | D03 score | bootstrap 95% CI |
|---|---:|---:|---:|
| D03 vs D01 | 18 / 170 / 12 | 0.515 | [0.4875, 0.5425] |
| D03 vs D02 | 13 / 166 / 21 | 0.480 | [0.4525, 0.5075] |

D03 只有 2/200 条比 D01 更长、16/200 条比 D02 更长，盲评结果不是靠普遍拉长输出获得。另一方面，D01/D03 有 177/200 个回答完全相同，D02/D03 有 85/200 个完全相同；高 tie 和单 judge/single decoding 限制了敏感度，不能把结果当作人工真值。

## 7. 序列化事故

原训练入口在保存 inference checkpoint 前调用 model.half()。DPO 的更新很小，D03 FP16 checkpoint 的 test preference credit 仅 53.6%，而从 resume.pt 恢复的 float32 状态为 71.8%；validation 也精确复现训练期的 0.605810 / 69.1%。因此 D02/D03 原 best.pth、last.pth 作废，正式结果使用 last_fp32_recovered.pth。训练入口已修为保存 float32 state dict。该事故也说明小步后训练不能把 checkpoint dtype 当作无关实现细节。

## 8. 判定与下一步

- 通过：数据审计、公式/梯度 smoke、held-out preference、长度检查、通用回归、可追溯训练/评测。
- 未通过：D03 没有在盲评中同时优于 D01 和 D02；尤其相对 D02 score 0.480。
- 结论：Phase4 完成但不晋级，继续保留 S10。不能把“preference credit 71.8%”写成“对话质量提升 71.8%”。
- 后续若重做 DPO，应先提高偏好数据与目标对实际回答质量的对齐度，并增加人工抽样或第二 judge；不得继续复用本 test 调 beta/LR。Phase5 仍从 S10 独立分叉。

训练 run：D01 <https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/os17qekn>；D02 <https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/51vdt97x>；D03 <https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3w1au7dk>；统一评测 <https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/csvnmhkj>。

机器可读证据位于 experiments/04-dpo/comparison.json、preference-comparison.json、blind-judge-summary.json 和各实验 eval.json；完整逐样本结果保留在 CPFS /data/artifacts/minimind-lab/。

实现与收口证据 commit：235de0fe0877896043caec4f326792e48948a442。
