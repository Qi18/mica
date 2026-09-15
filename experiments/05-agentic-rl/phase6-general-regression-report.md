# Phase6 通用能力回归：S10 / A01 / GRPO / CISPO

日期2026-09-08；四臂本轮完整重跑，只评测不训练，结果仅在L20。

## 协议
现有lm-eval 0.4.12、IFEval v4完整541题、0-shot、seed42、FP16、batch16、chat template。
IFEval沿用max_gen_toks1280和greedy。七项使用全部样本；优先acc_norm，否则acc，按七项等权macro。
四臂task hashes、样本数、逐题doc/prompt hashes和chat template均一致；权重评测前后SHA不变。
S10为实际训练起点的FP32导出，评测时同样转换FP16；不混用历史评测结果。

## 核心结果（%）

| 模型 | IFEval prompt strict | prompt loose | instruction strict | instruction loose | 七项macro |
|---|---:|---:|---:|---:|---:|
| S10 | 17.3752 | 19.7782 | 29.9760 | 32.9736 | 33.0024 |
| A01 | 15.8965 | 19.5933 | 29.7362 | 33.2134 | 33.2222 |
| A02 | 15.8965 | 19.9630 | 29.3765 | 33.2134 | 33.1674 |
| A03 | 15.3420 | 18.6691 | 28.6571 | 31.6547 | 33.2176 |

## 七项明细（%）

| 任务 | S10 | A01 | A02 GRPO | A03 CISPO |
|---|---:|---:|---:|---:|
| ceval-valid | 23.2541 | 23.3284 | 23.3284 | 23.4027 |
| cmmlu | 25.2461 | 25.4360 | 25.4188 | 25.4792 |
| arc_easy | 34.3434 | 34.3434 | 34.5539 | 34.5539 |
| piqa | 56.1480 | 56.6921 | 56.5288 | 56.5288 |
| openbookqa | 28.0000 | 28.4000 | 28.2000 | 28.6000 |
| hellaswag | 28.2016 | 28.1219 | 28.0621 | 28.0323 |
| social_iqa | 35.8240 | 36.2334 | 36.0798 | 35.9263 |

## 分离Agent SFT与RL影响

| 对比 | IFEval strict差值pp [95%CI] | 改善/退化题数 | 七项macro差值pp |
|---|---:|---:|---:|
| A01-minus-S10 | -1.4787 [-3.8817, +0.9242] | 17/25 | +0.2197 |
| A02-minus-A01 | +0.0000 [-1.6636, +1.6636] | 10/10 | -0.0548 |
| A03-minus-A01 | -0.5545 [-2.0333, +0.9242] | 7/10 | -0.0046 |
| A02-minus-S10 | -1.4787 [-3.6969, +0.7394] | 15/23 | +0.1650 |
| A03-minus-S10 | -2.0333 [-4.2514, +0.1848] | 14/25 | +0.2152 |

区间为10000次配对bootstrap，seed42。instruction-level按prompt整组重采样，完整区间见summary.json。
区间跨0表示当前题集不足以证明稳定变化，不等于能力严格不变；不根据结果临时设容差。
A01-S10反映Agent SFT及混合回放的整体变化；A02/A03-A01才是本轮RL增量。

## 验收边界
本轮补齐IFEval及七项通用回归证据，不自动晋级或替换S10。旧Tool仍5/8，未达7/8门槛。
任务熟悉度和S10课程的benchmark-family定向性限制外推，不能把这些分数等同开放域聊天能力。
本次不新增训练数据污染审计，也不使用回归结果继续训练或挑选checkpoint。
各模型的完整逐题文本、harness配置和日志在CPFS；仓库只保留轻量指标/哈希和IFEval逐题布尔结果。
没有SwanLab等外部上传，没有commit/push。

## 复现入口
脚本：scripts/eval/run_phase6_general.py；汇总：scripts/eval/summarize_phase6_general.py。
协议：docs/phases/phase6-general-regression-protocol.md。
- S10: /data/artifacts/minimind-lab/phase6-general-S10-20260908（耗时442.4秒）。
- A01: /data/artifacts/minimind-lab/phase6-general-A01-20260908（耗时444.7秒）。
- A02: /data/artifacts/minimind-lab/phase6-general-A02-20260908（耗时450.5秒）。
- A03: /data/artifacts/minimind-lab/phase6-general-A03-20260908（耗时438.7秒）。

## Phase6 SwanLab 补传完成（2026-09-08）

6个历史指标记录已获用户授权补传至MiniMind-Lab；云端均为FINISHED，282项指标回读一致。
仅上传指标/元信息，不上传权重、数据集或完整逐题文本；不改变completed-not-promoted结论。

- A02-A03-rl-v2-summary-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/563ab2ac)
- A00-A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/79f294dc)
- A00-s10-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0cb93996)
- A01-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/da935c86)
- A02-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/c74379f7)
- A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/a739ed89)
