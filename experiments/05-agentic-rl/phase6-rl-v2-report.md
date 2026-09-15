# Phase6 Agentic RL v2：A01 / GRPO / CISPO固定对照

状态：exploratory-completed-not-promoted；日期2026-09-08。
用户明确要求在A01旧Tool未通过时开始探索性RL。该例外不代表A01被接受，S10仍为通用基线。

## 训练与实现
三臂同起点A01-v3 LR3e-6 checkpoint-570；A01不更新，A02 GRPO、A03 CISPO。
两组同seed42、LR1e-6、32outer×2inner=64updates、每outer4prompt×4rollout，共512episodes。
训练仅用v3-r1 train/task pools，不用val/test。graph/tools各半；顺序和生成上限相同，实际token消耗分别记录。
reward=0.8严格E2E+0.2正确有序工具前缀；工具观察只进上下文，只有生成assistant action/EOS参与loss。
两组共同token-mean、beta0.01 KL，GRPO ratio clip[0.8,1.2]，CISPO upper IS cap2并stop-gradient。
这是控制surrogate差异的实现：GRPO使用token-normalized变体，CISPO加入共同KL，不是原论文全配方复现。
生成与teacher-forced old/ref/new统一FP32，关闭TF32和dropout；每批核验新旧ratio、掩码、真实采样概率差。
T1/top_p1/top_k0；对微小生成/回算差做detached IS修正。固定末步，不按val/test选择checkpoint。

## v1失败与v2修复
v1 BF16两组分别在38/22更新后触发generation/teacher logprob最大差>0.20门禁，未保存最终模型、未评测test。
保留中止日志、配置、源码和FAILURE.json，不能把不等预算中止点用于算法优劣比较。
v2不放宽阈值，改为统一FP32并从同一A01重新训练。FP32 probe128条成功121条，32group中7组有方差。
probe初始最大概率差约2.72e-5。正式训练审计值见各metrics.jsonl。

## 训练结果
| 臂 | 更新 | 采样E2E | 实际assistant tokens | 训练计时秒 |
|---|---:|---:|---:|---:|
| A02 | 64 | 91.41% | 47243 | 136.8 |
| A03 | 64 | 91.60% | 47250 | 134.9 |

训练reward/loss不是SFT loss或评测分数；两组on-policy样本会随模型更新分叉。训练计时不包含全部后续评测，GPU与驻留任务共享。

## 冻结测试与旧能力回归

| 臂 | graph test | tools test | 合计E2E | 旧Chat | 旧Tool |
|---|---:|---:|---:|---:|---:|
| A01 | 237/240 (98.75%) | 214/240 (89.17%) | 451/480 (93.96%) | 8/10 | 5/8 |
| A02 | 239/240 (99.58%) | 212/240 (88.33%) | 451/480 (93.96%) | 8/10 | 5/8 |
| A03 | 239/240 (99.58%) | 222/240 (92.50%) | 461/480 (96.04%) | 8/10 | 5/8 |

Agent三臂统一FP32、greedy、同任务预算；旧Chat/Tool统一既有FP16脚本/max_new_tokens128。
测试此前未被模型评测，三臂SHA冻结后一次性打开；此后不再称为未见测试，也不能继续用它调参。
全部逐题工具/参数/执行/最终答案/成本见各test目录predictions.jsonl与summary.json。

## 配对差异（percentage points，10000次task-level bootstrap95%区间）

| 对比 | graph差异 | tools差异 | 合计差异 [95%CI] |
|---|---:|---:|---:|
| A02-minus-A01 | +0.83 | -0.83 | +0.00 [-1.46, +1.46] |
| A03-minus-A01 | +0.83 | +3.33 | +2.08 [+0.83, +3.54] |
| A03-minus-A02 | +0.00 | +4.17 | +2.08 [+0.83, +3.33] |

区间包含0时，本轮不足以证明稳定提升；即便未包含0，也只适用于当前合成任务与预算。
单seed、共享模板会限制统计外推，不能据此宣称算法普遍优劣。

## 保持与收口边界
A01起点旧Tool为5/8，本轮比较RL相对A01的变化，但正式保持门槛仍是S10的7/8，不能降低。
IFEval、七项基准未在这轮运行。探索性结果归档不等于Phase6正式验收，默认模型不替换。

## 证据入口
- 预注册：docs/phases/phase6-rl-v1-preregister.md、phase6-rl-v2-preregister.md。
- 冻结：experiments/05-agentic-rl/rl-v2-test-freeze.json（models/data SHA）。
- 训练：scripts/launch/train_phase6_rl_fp32.py；评测：scripts/eval/eval_phase6_rl_v2.py。
- 权重仅在CPFS；训练/评测provenance含实际源码和数据指纹。未经请求未commit/push。
- 汇总SwanLab：未上传，等待用户确认；仅本地汇总
- A01 swanlab_val: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/60s7cgkf
- A01 swanlab_test: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/37rec37k
- A02 swanlab_train: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1pxh36vn
- A02 swanlab_val: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0l8s0wh8
- A02 swanlab_test: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/l6kyh7yq
- A03 swanlab_train: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/vdwjd3sg
- A03 swanlab_val: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/yognri61
- A03 swanlab_test: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/op09vul1

## 附加审计与收益解释

CISPO相对A01改善的10题包括graph chain1题/redirect1题，以及tool random5题/multistep2题/time1题，未发现原先通过题退化。
随机数子集主要是数值参数抽取改善；其中一个graph样本是将观察结果358正确读出，另一个修复了错误lookup参数。不能把这些结果写成开放域规划涌现。
旧Tool的E2E三臂均5/8，但tool-selection/argument-validity/execution子指标从A01的75%降到两组RL的62.5%；因此不能写成全部旧工具指标保持。
GRPO记录的最大ratio超界token比例约0.355%，19/32个outer的末次inner非零；CISPO在cap2下没有触发上界裁剪。
这不是CISPO大规模高off-policy收益的验证；单seed、少量更新与当前任务足以限制算法优劣外推。
汇总上传曾被自动安全审核拦截；当前使用local-only模式生成本报告，未绕过拦截。训练及各臂评测已有SwanLab记录，汇总另待授权。

## 通用回归补充（2026-09-08，随后评测，保留上文历史状态）

本轮S10/A01/GRPO/CISPO全部完成IFEval541题与七项全量评测，协议与样本哈希一致。

| 模型 | IFEval strict | 七项macro |
|---|---:|---:|
| S10 | 94/541 | 33.0024% |
| A01 | 86/541 | 33.2222% |
| A02 | 86/541 | 33.1674% |
| A03 | 83/541 | 33.2176% |

完整逐项变化与配对区间见[通用回归报告](phase6-general-regression-report.md)。
不临时添加下降容差；旧Tool仍5/8，模型不晋级。完整原始证据保存在CPFS，仅轻量结果入仓库，未上传、未commit/push。

## Phase6 SwanLab 补传完成（2026-09-08）

6个历史指标记录已获用户授权补传至MiniMind-Lab；云端均为FINISHED，282项指标回读一致。
仅上传指标/元信息，不上传权重、数据集或完整逐题文本；不改变completed-not-promoted结论。

- A02-A03-rl-v2-summary-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/563ab2ac)
- A00-A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/79f294dc)
- A00-s10-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0cb93996)
- A01-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/da935c86)
- A02-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/c74379f7)
- A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/a739ed89)
