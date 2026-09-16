# Phase 5：GRPO/CISPO 可验证强化学习

状态：completed-not-promoted；启动并收口于 2026-09-08。Phase4 的 D03 未晋级，本阶段按总计划从 S10 独立分叉；没有候选替换 S10。

## 预注册问题

1. 在程序可验证的 train-disjoint 算术选择任务上，GRPO/CISPO 能否提升 held-out pass@1，而不只是提高训练 reward？
2. 与相同 prompt batch 和 optimizer update 数的 SFT continuation 相比，RL 是否有独立收益？
3. GRPO 和 CISPO 在复用同一 rollout 做第二次 policy update、importance ratio 不再恒为 1 时，稳定性和泛化是否不同？

## 数据与 verifier

verifiable-math-v2 为确定性生成的四选一加减乘算术：train 1,000、validation 200、test 400；task key 和 normalized prompt 跨 split 均为 0 重合。四个答案位置在 train/validation/test 分别严格均衡为每类 250/50/100，防止恒答某个位置获得超过 25% 的捷径；操作和四种中英文模板也分层平衡。

正确性 reward 只接受一个明确且不歧义的首选项标签，可兼容 A、A. 14、The correct answer is A.14；多个选项标签或无明确标签均为 0。严格“只输出字母”另记 format，不混入 correctness reward。程序重新计算运算结果并核对 option/label。审计只证明 Phase5 内部分割隔离，不能证明全部历史 S10 语料从未出现同类表达式。

这是一项窄域多选算术实验，不是 GSM8K/自由文本数学推理。若通过，只能声明 verifier RL 管线和该任务上的泛化结果。

## 对照与固定项

| 实验 | 方法 | 目的 |
|---|---|---|
| R02A | S10 不训练 | 初始策略基线 |
| R02B | SFT continuation | 直接监督正确选项的控制组 |
| R02C | GRPO | 对称 PPO-style ratio clipping |
| R02D | CISPO | detached upper-clipped importance weight |

R02B/C/D 固定初始权重 S10、train 顺序、seed 42、LR 3e-6、1,000 prompts、batch prompts 4、每外层 batch 2 次 policy update，共 500 optimizer updates。R02C/D 固定 16 rollouts/prompt、temperature 0.8、top-p 0.95、max new tokens 8、beta 0.02。GRPO epsilon 0.2；CISPO upper clip 2.0。

SFT 与 RL 只配平 prompt batch 和 optimizer update 数：SFT 使用一个真值 completion，RL 每 prompt 采样 16 条且还有 frozen reference forward，因此不宣称 FLOPs、target tokens 或 wall-clock 等预算。GRPO/CISPO 复用每批 rollout 做 2 次 policy update；第一次 ratio=1，第二次才实际检验 clip 分支差异。若每 rollout 只更新一次，二者在本实现中梯度等价，不能做方法对比。

参数以 FP32 保存和更新、BF16 autocast 前向；不再使用会抹除小幅后训练更新的 FP16 checkpoint 保存。

## v1 无效实验与协议修复

首轮 R01 使用的 v1 只检查了答案位置、模板的各自边际均衡，却用同一个递增 index 分配二者，导致 template_id 能确定性预测 answer_position。R01C/D 训练 reward 接近 1，但 validation 出现答案整体错位一格、pass@1=0%；这是明确的 reward hacking。

test 当时仍封闭。R01 run 和失败报告原样保留，不参与方法结论。v2 将答案位置按 index%4、模板按 (index//4)%4 构成交叉联合均衡；auditor 新增 16 个 answer×template 单元的全支持和计数差检查。R02 除数据联合分配外沿用相同超参数与门槛。

## 训练前探针

- 数据审计通过：train/validation/test task 和 prompt overlap 均为 0，答案位置完全平衡。
- 数值 smoke：SFT/GRPO/CISPO 均完成 2 outer steps / 4 updates，finite loss/grad，保存权重可加载。
- R02A validation：pass@1 24.0%，choice-valid 95.5%，strict format 0%；sample accuracy 20.5%，empirical pass@4 35.5%。
- R02A 几乎恒选 A：validation 200 条中 A 185、C 6、invalid 9；所以必须报告按 answer position 分桶，不能只看总体 reward。

正式训练前不打开 test。

## 指标与验收

训练记录 reward、group reward std、degenerate/all-zero group rate、pass@group、approx KL、entropy、completion length、clip fraction、gradient norm、rollout wall、总 wall、显存和 token 数。test 固定 greedy pass@1 为主，另报 fixed-sampling accuracy/pass@4、choice-valid、strict format，以及 operation/template/answer-position 分桶。

配对 bootstrap 95% CI 用同一 400 test task。R02C/R02D 只有同时满足以下条件才晋级：

- 相对 R02A 与 R02B 的 pass@1 差值区间下界均大于 0；
- 收益不是恒答某个位置：按 answer position 报告，choice distribution 不塌缩；
- training reward 上升同时 held-out pass@1 上升；
- 七项 macro 相对 S10 不退超过 1pp，IFEval prompt strict 不退超过 2pp；Chat 至少 7/10、格式至少 4/6、Tool E2E 至少 6/8、重复异常最多 2/10。

若 reward 升而 held-out 不升，或只靠答案位置/格式捷径，按负结果收口并保留 S10。test 只在三组训练全部完成后一次性打开，不用来调 LR、beta、clip 或步数。

## 正式结果与收口

R02B/C/D 均正常完成 250 outer steps、500 optimizer updates，FP32 checkpoint 可加载；test 在三组训练结束后只打开一次，未按 test 调参。统一评测 run：[Phase5-Evaluation-R02A-R02B-R02C-R02D](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3sfq26tj)。

| 候选 | 方法 | 训练 wall | train reward | test pass@1 | sampled accuracy | empirical pass@4 | greedy 选择分布 |
|---|---|---:|---:|---:|---:|---:|---|
| R02A | S10 不训练 | — | — | 23.25% | 20.94% | 41.00% | A 373 / C 18 / invalid 9 |
| R02B | SFT control | 15.16 s | — | 5.50% | 14.75% | 41.25% | invalid 302 / D 71 / B 17 / C 10 |
| R02C | GRPO | 44.56 s | 24.21% | 25.00% | 24.88% | 25.00% | A 400 |
| R02D | CISPO | 43.64 s | 23.38% | 25.00% | 24.81% | 25.25% | B 300 / A 100 |

成对 bootstrap 使用相同 400 个 task、seed 42、10,000 次重采样。R02C−R02A 的 greedy pass@1 为 +1.75pp，95% CI [+0.25,+3.50]，但其 400/400 恒答 A；R02D−R02A 也是 +1.75pp，CI [-4.25,+8.00]。两组的 pass@4 相对 R02A 分别下降 16.00pp（[-19.75,-12.25]）和 15.75pp（[-22.00,-9.25]）。因此数值上的 25% 是平衡四选一任务的答案位置策略，不是算术泛化。

| 候选 | 七项 macro | IFEval prompt strict | Chat | 格式 | Tool E2E | 主门 |
|---|---:|---:|---:|---:|---:|---|
| R02A / S10 | 33.0146% | 16.8207% | 8/10 | 5/6 | 7/8 | baseline |
| R02B / SFT | 32.9025% | 7.3937% | 0/10 | 0/6 | 1/8 | 失败：灾难性遗忘 |
| R02C / GRPO | 33.0599% | 15.3420% | 9/10 | 6/6 | 7/8 | 失败：100% 答 A |
| R02D / CISPO | 33.1256% | 17.1904% | 7/10 | 4/6 | 7/8 | 失败：75% 答 B，CI 跨 0 |

R02C 日志点的 degenerate-group rate 均值为 93%，R02D 为 85%；大量 all-zero/all-one group 使 group-relative advantage 失去学习信号。CISPO 后半程 logged reward 高于前半程（32.09% vs 19.01%），但 held-out 仍只是位置塌缩，直接说明训练 reward 不能作为收口指标。

### 判定

R02C/R02D 都未通过预注册的“相对 A/B 显著提升且不发生答案位置塌缩”条件，不晋级；S10 继续作为 release。R01 的模板—答案位置联合混杂与 R02 的答案位置塌缩都作为 reward-hacking 失败证据保留。完整机器可读证据见 `experiments/04-grpo-cispo/comparison.json`、`paired-bootstrap.json`、各实验的 `eval.json` 与 `metrics.csv`。

本实验只有一个训练 seed，任务是合成四选一加减乘，不是 GSM8K 或自由文本数学推理；不能据此声称 GRPO/CISPO 一般无效，也不能据此比较两种算法的一般优劣。下一轮若重开，应先提高非退化 group 比例、采用需要生成数值/过程的 verifier 任务，并增加 seed 43/44，而不是在本 test 上继续调参。

## R03 重训预注册（2026-09-08）

R02 失败后重新开放 Phase5，但不复用已经打开的 R02 test。R03 改为自由数值 `mod 10` verifier：train 4,000、validation 500、test 1,000，add/sub 与答案 0–9 严格均衡，answer×template 联合计数差不超过 1，跨 split task overlap 为 0。reward 只在整个 completion 可解析为正确整数时为 1，不再接受选项字母或忽略错误数值。

共同 warm-start R03W 从 S10 出发，使用 4,000 条数学监督和 1,000 条 S10 IFEval 课程 replay。只在 validation 比较 25/50/100/200 outer-step probe，选择最早满足 pass@1≥10%、sample accuracy 15%–50%、sample nondegenerate group≥50%、Chat≥7/10、Tool E2E≥6/8 的 checkpoint。正式 R03A（不继续训练）、R03B（SFT control）、R03C（GRPO）、R03D（CISPO）从同一 R03W 分叉。

R03B/C/D 固定 LR 5e-7、batch prompts 10（每批答案0–9各一条）、2 inner updates、800 optimizer updates；RL 固定 8 rollouts/prompt、temperature/top-p 1.0、max_new_tokens 4、beta 0.05、entropy coefficient 0.002。C/D 只有在相对 A/B 的 test pass@1 paired-bootstrap 95% CI 下界均大于0、sample nondegenerate 提升、最大单数字输出占比≤30%，且通用回归门通过时才晋级。test 在 B/C/D 全部完成前封闭，R03 test 不用于超参数调整。
