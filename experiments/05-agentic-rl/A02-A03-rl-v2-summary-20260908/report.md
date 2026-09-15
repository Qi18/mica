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

## SwanLab 补传（2026-09-08）

用户已授权补传历史指标与实验元信息；未重新训练/评测，不上传权重、数据或完整逐题文本。
云端已完成且54项指标回读一致：[实验记录](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/563ab2ac)。上文仅本地状态为补传前历史。
