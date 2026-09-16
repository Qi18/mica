# Phase7 阶段报告：Dense vs MoE

- 阶段范围：统一计划第12节。
- 状态：completed-not-promoted；2026-09-09 用户批准实验收尾。
- Lab训练基线：01a9910 + 已归档Phase7源码；精确文件SHA见formal-evidence/source-manifest.json。
- MiniMind来源：393e387e9ad99f0f04c296e4c5e7353f4444629f。
- 发布模型：S10保持不变；本阶段不声称MoE全面或稳定更优。

## 1. 目标与实验清单

比较相同数据预算、名义激活参数相近时，原生Dense/MoE的语言建模质量、七项能力和成本。
M00六个三seed短探针仅排查实现；M00b r1/r2作废，r3恢复/DDP/完整validation门禁通过。
M01 Dense与M02 MoE从随机初始化独立训练并完成评测；正式训练各一个seed。
实验和所有probe/attempt链接见[实验目录](../../experiments/06-moe/README.md)及[SwanLab索引](swanlab-runs.md)。

### SwanLab正式run

- M01-dense-pretrain-1b28-20260908：训练 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nyy90ef5；评测 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/opixiuyd
- M02-moe-pretrain-1b28-20260908：训练 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/uvoc97pz；评测 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/dxfvjdxy

四个正式训练/评测run均FINISHED，宏平均云端回读一致，见swanlab-cloud-verification.json。

## 2. 配置、数据与源码

4×L20（GPU4–7），两臂顺序运行；每卡batch32、accum2、global batch256、seq768、1epoch、9038updates、seed42。
AdamW默认wd0.01，cosine LR5e-4到5e-5，clip1；FP32主权重+BF16 autocast，TF32关闭。
确定性算法、CUBLAS=:4096:8、固定DDP bucket1024MiB/find_unused_parameters=True、NCCL Ring/Simple。
Dense参数63,912,192；MoE总参数198,416,640，名义激活63,936,768，不等于实测FLOPs。
原生4experts/top1、norm_topk_prob=True、aux系数5e-4/层；CE与aux分开记录。

数据：/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1。
41分片SHA核验通过；fingerprint=cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd。
唯一有效targets=1,280,000,000；DDP补齐重复1行，两臂相同，实际日志有效曝光Dense=1280000754、MoE=1280000754。
原生PretrainDataset截text至seq-2再包BOS/EOS；validation=11525行/6400000targets。
梯度目标仍是原生rank/microbatch等权，token加权日志只是诊断。
训练源码minimind/trainer/train_phase7.py；配置configs/moe/phase7-formal.json；启动scripts/launch/start_phase7_formal.py。

## 3. 质量与成本结果

| 指标 | Dense M01 | MoE M02 |
|---|---:|---:|
| validation NLL（低优） | 2.60688777 | 2.62454679 |
| validation PPL（低优） | 13.55679330 | 13.79831930 |
| 七项macro % | 31.470333 | 32.031674 |
| 训练循环wall秒 | 6366.266 | 8938.690 |
| 训练循环GPU-hours（4卡） | 7.0736 | 9.9319 |
| 最大观测峰值显存MiB | 12959.07 | 16741.83 |
| active padded tokens/s | 293792.9 | 208376.0 |

吞吐含padding，不冒称有效targets/s；循环wall不含下载/前置门禁/评测。训练循环wall增加40.41%。
源数据见experiments/06-moe/training-summary.json及各臂metrics.jsonl。

| benchmark | Dense % | MoE % | MoE−Dense pp |
|---|---:|---:|---:|
| ceval-valid | 23.4770 | 22.9569 | -0.5201 |
| cmmlu | 25.2115 | 25.1425 | -0.0691 |
| arc_easy | 28.6195 | 32.7020 | +4.0825 |
| piqa | 53.0468 | 54.1893 | +1.1425 |
| openbookqa | 27.2000 | 26.6000 | -0.6000 |
| hellaswag | 28.1418 | 28.2912 | +0.1494 |
| social_iqa | 34.5957 | 34.3398 | -0.2559 |

## 4. 评测协议与统计

两臂原生MiniMindForCausalLM strict load，固定epoch-end FP16权重，不替换best-val；
同HFLM adapter、lm_eval0.4.12、FP16、0-shot、无chat template、batch16、max_length32768、seed42、add_bos_token=False、无limit。
每臂29638题；逐题doc_hash/prompt_hash、task配置/版本/样本数完全一致。
仅忽略序列化Python function repr中的进程内地址，真实任务内容和提示仍逐题校验。
选择题按任务定义acc/acc_norm计分，再等权平均七个benchmark，不是RL rollout。
配对分层bootstrap10000次、seed42：macro差+0.561342pp，探索性95%CI [-0.164671,+1.286634]pp，跨0。
CI只描述题目抽样，不包含训练seed变异；未做多重比较校正。ARC-Easy局部提升不等于整体稳定优势。
历史P03/S10使用过不同模型adapter/chat协议，不直接把这些小数点差异解释为阶段收益。
复现：scripts/eval/compare_phase7_base.py；无文本配对记录与comparison.json已归档。

## 5. 验收门控

| 门控 | 实测 | 判定 |
|---|---|---|
| 四卡训练与恢复一致性 | r3 Dense361/MoE681项模型与optimizer tensor最大差0 | pass |
| 同数据与完整validation | SHA通过；两臂11525行/6.4M targets | pass |
| 训练与固定final评测完成 | 两臂9038updates；各29638题；权重未变 | pass |
| SwanLab可追溯 | 正式四run FINISHED，评测macro回读一致 | pass |
| 质量/成本综合优势 | macro小幅升、CI跨0、NLL差、wall多约40% | 未证明；不晋级 |

用户已允许原生对照实验收尾，不把模型必须胜出作为验收条件。

## 6. 失败、修复与未证实解释

r1恢复比较失败，重复连续训练也不一致；r2仍差1.054e-6，均保留invalidated，不降低原门槛。
r3固定DDP归约路径后恢复检查精确一致，重新从随机初始化正式训练。
评测推理全部完成后，最初summary错误读取C-Eval/CMMLU父组n-samples而KeyError。
原始FAILURE和运行源码snapshot保留；finalize_phase7_base.py递归汇总叶任务、验证全部样本与权重SHA、续写原SwanLab run完成，不重跑推理、不隐藏失败。
配对工具最初假定有task_hashes字段、再遇到function repr地址差异；最终使用真实字段和逐题hash核验，不删除真实协议差异。

源码MOEFeedForward的top1概率归一化约等于p/p，抵消gate的直接任务梯度。
CPU FP32 toy：主输出替代损失gate梯度max约2.328e-10，aux约1.635e-5；不是正式全训练梯度审计。
gate主要靠均衡aux更新，专家本身仍学习。是否导致当前差距尚未做路由消融，不能写成已证实根因。
训练末期负载采样约22%–28%，含padding，不能据此证明有效token的专业分工；aux也包含padding。
同token预算分散到四个专家，可能训练曝光不足；MoE NLL追近Dense但不能保证延长训练会反超。
以上改进作为后续独立实验，不修改本轮权重与原生基线。

## 7. 下一阶段

Phase8先进行teacher资格检查。M02仅Base、未经SFT，也未证明强于S10，不直接作为teacher。
允许按统一计划检查上游MoE Full SFT，冻结来源/revision/SHA、tokenizer/vocab/chat template、strict load及固定评测。
未通过teacher门控不得启动K01–K04正式蒸馏；不自动改变release，不扩大到1B。

## 8. 证据与保留策略

Git目录：experiments/06-moe，包含配置、命令、训练指标、source/data/恢复门禁、评测manifest/results、配对分数及云端回读。
CPFS训练：/data/artifacts/minimind-lab/phase7-formal-r3-20260908。
CPFS评测：/data/artifacts/minimind-lab/phase7-{dense,moe}-base-eval-20260909。
权重SHA：各臂eval/manifest.json；完整原始题目输出、数据、权重、resume和日志留CPFS，不入Git。
本次收尾不删除任何checkpoint，不操作GPU0–3其他项目。

## 修订记录

2026-09-09：正式训练/七项评测/配对统计/云端回读完成，按completed-not-promoted收尾。
