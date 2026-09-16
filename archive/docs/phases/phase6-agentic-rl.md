# Phase6 阶段验收：Agent SFT 与 Agentic RL

- 报告状态：accepted with limitations（用户同意按有局限的实验结果收尾）。
- 阶段结论：**completed-not-promoted**；不代表所有模型门槛通过，S10继续作为release。
- 收口日期：2026-09-08；范围：统一计划第11、16、17节。
- Lab训练基线：234797d加Phase6工作区源码，实际版本见各source-snapshot/provenance；交付commit见本文件Git历史。
- MiniMind基线：393e387e9ad99f0f04c296e4c5e7353f4444629f。

## 1. 目标与结论

验证轨迹SFT能否学会观察—工具调用—最终回答，以及GRPO/CISPO是否进一步提升完整任务成功率。
构建的是有限、可重复的合成工具环境，不是BFCL或开放域Agent基准。

固定480题test中，A01和GRPO均451/480，CISPO461/480，较A01提升2.0833pp，
配对10000次bootstrap95%区间[+0.8333,+3.5417]。单seed且同模板族，不能推出算法普遍优劣。
旧Tool三臂均5/8，低于S10的7/8，故不晋级。通用回归已完成，不再列为待评测。

## 2. 实验清单与SwanLab

下表直接来自registry；completed表示实验执行完成，不表示该模型通过晋级。
首轮BF16审计中止记invalidated，原始attempt状态保存在phase6-acceptance.json及各FAILURE.json。
汇总/通用回归最初为local-only；用户授权后已补传并回读核验，CPFS路径见各archive-manifest。

| experiment_id | 执行状态 | 实验报告 | SwanLab |
|---|---|---|---|
| A00-s10-agent-baseline-20260908 | completed | [报告](../../experiments/05-agentic-rl/A00-s10-agent-baseline-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/p3yd932a) |
| A01-agent-sft-pilot-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-agent-sft-pilot-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/jkra8o25) |
| A01-v2-transfer-baseline-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v2-transfer-baseline-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1t7ndogk) |
| A01-agent-sft-v2-pilot-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-agent-sft-v2-pilot-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/mabi301l) |
| A01-v2-counterfactual-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v2-counterfactual-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/miy5qid6) |
| A01-v2-rl-eligibility-probe-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v2-rl-eligibility-probe-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/hsn1xolt) |
| A00-v2-frozen-test-20260908 | completed | [报告](../../experiments/05-agentic-rl/A00-v2-frozen-test-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/j3b8qcu0) |
| A01-v2-frozen-test-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v2-frozen-test-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/gfltoxnx) |
| A01-v2-evaluation-summary-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v2-evaluation-summary-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/8yso53w9) |
| A01-v3-mixed-lr3e6-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v3-mixed-lr3e6-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/52u6fhj4) |
| A01-v3-mixed-lr1e6-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v3-mixed-lr1e6-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/5h3qys7d) |
| A00-v3-s10-graph-validation-20260908 | completed | [报告](../../experiments/05-agentic-rl/A00-v3-s10-graph-validation-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1a6xkwzx) |
| A00-v3-s10-tools-validation-20260908 | completed | [报告](../../experiments/05-agentic-rl/A00-v3-s10-tools-validation-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/wtets9k0) |
| A01-v3-validation-summary-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v3-validation-summary-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/p6sbux9p) |
| A01-v3-agent-rl-probe-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v3-agent-rl-probe-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0a9nthb8) |
| A01-v3-agent-rl-fp32-probe-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-v3-agent-rl-fp32-probe-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/494uwnlh) |
| A02-agentic-grpo-v1-20260908 | invalidated | [报告](../../experiments/05-agentic-rl/A02-agentic-grpo-v1-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1hza7eyl) |
| A03-agentic-cispo-v1-20260908 | invalidated | [报告](../../experiments/05-agentic-rl/A03-agentic-cispo-v1-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/fgvlab1b) |
| A02-agentic-grpo-v2-20260908 | completed | [报告](../../experiments/05-agentic-rl/A02-agentic-grpo-v2-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1pxh36vn) |
| A03-agentic-cispo-v2-20260908 | completed | [报告](../../experiments/05-agentic-rl/A03-agentic-cispo-v2-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/vdwjd3sg) |
| A01-agentic-rl-v2-val-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-agentic-rl-v2-val-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/60s7cgkf) |
| A01-agentic-rl-v2-test-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-agentic-rl-v2-test-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/37rec37k) |
| A02-agentic-rl-v2-val-20260908 | completed | [报告](../../experiments/05-agentic-rl/A02-agentic-rl-v2-val-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0l8s0wh8) |
| A02-agentic-rl-v2-test-20260908 | completed | [报告](../../experiments/05-agentic-rl/A02-agentic-rl-v2-test-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/l6kyh7yq) |
| A03-agentic-rl-v2-val-20260908 | completed | [报告](../../experiments/05-agentic-rl/A03-agentic-rl-v2-val-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/yognri61) |
| A03-agentic-rl-v2-test-20260908 | completed | [报告](../../experiments/05-agentic-rl/A03-agentic-rl-v2-test-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/op09vul1) |
| A02-A03-rl-v2-summary-20260908 | completed | [报告](../../experiments/05-agentic-rl/A02-A03-rl-v2-summary-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/563ab2ac) |
| A00-s10-general-regression-20260908 | completed | [报告](../../experiments/05-agentic-rl/A00-s10-general-regression-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0cb93996) |
| A01-general-regression-20260908 | completed | [报告](../../experiments/05-agentic-rl/A01-general-regression-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/da935c86) |
| A02-general-regression-20260908 | completed | [报告](../../experiments/05-agentic-rl/A02-general-regression-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/c74379f7) |
| A03-general-regression-20260908 | completed | [报告](../../experiments/05-agentic-rl/A03-general-regression-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/a739ed89) |
| A00-A03-general-regression-20260908 | completed | [报告](../../experiments/05-agentic-rl/A00-A03-general-regression-20260908/report.md) | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/79f294dc) |

各run的角色、名称和输入checkpoint可查对应报告/provenance；训练与评测不共用一个含糊的run链接。

## 3. 数据、训练和复现

- v1简单记录环境，v2加入branch/retry/redirect；v3混合graph与多schema工具、历史chat/native-tool回放。
- v3实际数据mixed-agent-v3-r1：graph train800/val160/test240，工具800/160/240。
  训练9115条assistant decision、406611 targets；manifest见data-manifest-v3-r1.json。
- A01-v3两组同S10起点、同数据/seed42、1epoch、570updates，只比较LR3e-6和1e-6；
  均固定末步，旧Tool门禁失败。后续用户明确要求探索性RL，作为不晋级的实验例外。
- A02/A03同A01 LR3e-6 checkpoint-570，seed42、LR1e-6、32outer×2inner、
  4prompt×4rollout/outer，64updates、512episodes。实际assistant tokens分别47243/47250。
- reward=0.8严格E2E+0.2正确有序工具前缀；只有assistant action/EOS进loss，observation只作为条件。
- GRPO ratio clip[0.8,1.2]；CISPO detached IS cap2；共同token-mean与KL beta0.01，
  是受控surrogate变体，不是原论文完整训练配方复现。
- v2正式RL统一FP32、关闭TF32/dropout；toy梯度方向、mask、old/new ratio及采样/回算一致性检查通过。
- 两组训练计时136.84/134.88秒，峰值allocated2744/2730MiB；共享GPU0/1剩余显存，
  约0.0380/0.0375占用GPU-hours只是单卡wall换算，不是专卡效率或账单成本。
- 数据没有统一_SUCCESS文件；实际证据是构建manifest、oracle执行/长度/去重检查及本次SHA审计，
  不能声称额外独立auditor流程已经完成。代码在tools schema之外不执行任意代码或网络请求。
- test在三臂权重SHA冻结后才打开；现在是已打开测试，不得再用其调参后称作独立测试。

指纹：[数据](../../experiments/05-agentic-rl/data-manifest-v3-r1.json)；
[三臂权重与test冻结](../../experiments/05-agentic-rl/rl-v2-test-freeze.json)。
原始权重、完整数据、日志在/data/artifacts/minimind-lab与/data/datasets/minimind-lab，不入Git。

## 4. 横向结果

| 模型 | graph test | tools test | 总E2E | IFEval strict | 七项macro | 旧Chat/Tool |
|---|---:|---:|---:|---:|---:|---:|
| S10 | 本轮RL对照不重复此项 | — | — | 94/541 (17.3752%) | 33.0024% | 8/10、7/8 |
| A01 | 237/240 | 214/240 | 451/480 (93.9583%) | 86/541 (15.8965%) | 33.2222% | 8/10、5/8 |
| GRPO | 239/240 | 212/240 | 451/480 (93.9583%) | 86/541 (15.8965%) | 33.1674% | 8/10、5/8 |
| CISPO | 239/240 | 222/240 | 461/480 (96.0417%) | 83/541 (15.3420%) | 33.2176% | 8/10、5/8 |

来源：[RL固定对照](../../experiments/05-agentic-rl/phase6-rl-v2-report.md)、
[四臂通用回归](../../experiments/05-agentic-rl/phase6-general-regression-report.md)；
各表可反查逐题predictions、results及eval manifest。S10旧Chat/Tool沿用同模型同协议的已归档结果。

CISPO相对A01 IFEval strict为-0.5545pp，配对区间[-2.0333,+0.9242]；七项macro为-0.0046pp。
不能将“区间跨0”写成“证明完全不退化”。旧工具选择/参数有效/执行子指标由A01的75%降至两组RL的62.5%，
虽然E2E都为5/8，仍不能声称旧工具各项能力保持。

## 5. 门控判定

| 门槛/交付 | 结果 | 判定 |
|---|---|---|
| graph验证E2E≥90%、新工具≥85% | CISPO97.5%、96.25% | pass |
| schema≥95%、语义参数/执行≥90%、retry≥60% | CISPO99.64%、98.04%/98.75%、97.5% | pass |
| 旧Chat≥8/10、格式≥5/6、复读异常≤1 | 8/10、5/6、1 | pass |
| 旧Tool≥7/8 | 5/8 | **fail，不晋级** |
| IFEval541和七项全量回归 | 四臂同协议完成，task/template/sample hashes一致 | 证据完整；未临时增设下降容差 |
| 固定checkpoint、完整轨迹、数据/代码指纹 | 已归档 | pass |
| 阶段报告与资产提交main | 本次交付，提交后核对远端SHA | 发布核验后完成 |

用户批准的是带有负结果的阶段收尾，不是降低7/8门槛，也不是修改S10 release。

## 6. 失败、审计与限制

- v2专项SFT test217/240，但旧Tool7/8→1/8；v3回放仅恢复至5/8。全部保留。
- v2旧rollout probe全成功、组内方差0，不能伪称RL有效更新。
- RL v1 BF16分别在38/22updates因采样/回算概率max差>0.20中止，两个不等预算结果不作算法对比。
- FP32复验不放宽阈值，最大差降至约7.38e-5/5.15e-5；两组重新同起点完成64updates。
- CISPO的10题收益集中于随机数参数5、多步2、时间1，以及graph chain/redirect各1；
  有结果358误读352得到修复的样本，不代表开放域规划能力涌现。
- GRPO少量token触发裁剪，CISPO cap2本轮未触发；不能声称验证了大规模高off-policy优势。
- 单seed、同模板族、部分历史replay来源已用于S10，exact/normalized去重不是语义零污染；
  IFEval是已用于课程研究的基准，不是整个项目从未见过的能力测试。
- 累积轨迹执行与精确答案作为observation使用证据；v2有反事实检查，
  v3/RL没有新增完整反事实复验，不把v2证据自动迁移为RL的因果证明。

## 7. 源码地图与最小观察实验

| 环节 | 源码/证据 | 学到的边界 |
|---|---|---|
| graph与工具环境 | scripts/phase6_agent_v2.py、phase6_retention_env.py | 工具执行成功不等于任务成功，参数要查语义 |
| 数据与SFT | scripts/data/agentic/build_phase6_v3.py、scripts/launch/train_phase6_sft.py | per-row tools、assistant-only labels、右padding与EOS |
| rollout与RL | scripts/launch/train_phase6_rl_fp32.py | action token保留，observation mask，组内advantage，old/reference分离 |
| 梯度最小实验 | 各unit-checks.json、mask-ratio-check.json | GRPO裁剪梯度与CISPO detached权重不同；零方差advantage为0 |
| 固定评测 | scripts/eval/eval_phase6_rl_v2.py、run_phase6_general.py | 同checkpoint、同任务/decoder与严格E2E |
| 统计与归档 | scripts/eval/summarize_phase6_rl_v2.py、summarize_phase6_general.py；scripts/sync/archive_phase6_* | reward不是评测，Git只存轻量可追溯证据 |

## 8. 下一阶段与交付边界

本报告及资产合入main后可按统一计划进入Phase7；仍以S10作为共同release，不把未晋级CISPO当默认模型。
A04 rollout系统对比为可选项，本轮未做。旧Tool修复、新seed及新的冻结任务集作为后续研究，不阻塞负结果阶段归档。
简历可写“完成64M Agent SFT与两种Agentic RL对照，在480条受控测试上CISPO +2.08pp并完成通用回归”；
必须保留单seed、合成环境和未晋级边界，不能写“全面提升对话/通用Agent能力”。

## 9. 证据与修订

完整证据索引见experiments/05-agentic-rl、registry、docs/phases/swanlab-runs.md。
汇总及通用回归已获用户授权补传SwanLab，云端指标回读一致；现有训练/单臂评测链接均保留。
博客为本仓库草稿，不发布网站。

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-09-08 | 汇总SFT、RL失败/复验及通用回归，按completed-not-promoted收尾 | 用户明确同意提交验收，保留模型门槛失败 |

## Phase6 SwanLab 补传完成（2026-09-08）

6个历史指标记录已获用户授权补传至MiniMind-Lab；云端均为FINISHED，282项指标回读一致。
仅上传指标/元信息，不上传权重、数据集或完整逐题文本；不改变completed-not-promoted结论。

- A02-A03-rl-v2-summary-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/563ab2ac)
- A00-A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/79f294dc)
- A00-s10-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0cb93996)
- A01-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/da935c86)
- A02-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/c74379f7)
- A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/a739ed89)
