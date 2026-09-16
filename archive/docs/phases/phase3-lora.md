# Phase 3：Full FT vs LoRA

状态：accepted with limitations（对照实验已收尾；领域能力提升未达成）。
证据与训练实现 commit：0d5125e307463976ed795ab735fc67f4e46f9150。
日期：训练 2026-09-07；最终评测及报告 2026-09-08。
本阶段完成实验与证据验收，不代表 L01/L02 通过代码模型发布验收。主线 S★ 继续固定为 S10；Phase 4 仍从 S10 启动。

## 目标与结论

从同一个 S10 出发，以相同代码数据和训练预算比较 Full FT 与 LoRA。LoRA 本次 wall 降低 32.96%，峰值 allocated 显存降低 35.64%；adapter 797,594 bytes，合并权重 137,685,109 bytes。领域 validation loss 均下降，但 MBPP 均为 0/500，未证明代码任务能力提升。不能把这组结果描述为“同等代码效果下 LoRA 更省”，因为正确率处在地板区间。

| 指标 | L00：S10 基线 | L01：Full FT | L02：LoRA r16 |
|---|---:|---:|---:|
| MBPP pass@1 | 0.40% (2/500) | 0.00% (0/500) | 0.00% (0/500) |
| 七项 macro | 33.0024% | 32.4759% | 32.6742% |
| IFEval prompt strict | 17.3752% | 17.9298% | 19.0388% |
| IFEval instruction strict | 29.9760% | 30.2158% | 31.4149% |
| Chat | 8/10 | 7/10 | 8/10 |
| 格式 | 5/6 | 3/6 | 5/6 |
| Tool E2E | 87.5% | 75.0% | 75.0% |
| 代码提取为空 | 336/500 | 424/500 | 394/500 |
| 领域 validation NLL（训练 DDP 口径） | 0.974555 | 0.757733 | 0.837637 |
| 可训练参数 | 0 | 63,912,192 | 393,216（含 adapter 总参数的 0.6115%） |
| 训练 wall | — | 94.65 s | 63.46 s |
| 训练 GPU-hours（6 卡） | — | 0.15775 | 0.10576 |
| 峰值 allocated 显存 | — | 6,936.51 MiB | 4,464.24 MiB |
| 实际 assistant targets | — | 4,820,231 | 4,820,231 |


IFEval 的小幅变化仅作本次单种子观察，不宣称显著提升。Full FT 的 Chat/格式退至 7/10、3/6；LoRA 保持 8/10、5/6。两者 Tool E2E 都从 7/8 降为 6/8。七项 macro 分别回退约 0.53pp、0.33pp，逐项结果见 comparison.json。

## 数据与执行协议

- 原始 CodeAlpaca 20,022 条，revision 152bb5e9a29651266b018106053980070a0521a1，CC-BY-4.0。
- 去掉 6 条空样本和 1 条与 MBPP test 提示存在包含关系的样本；冻结 train 19,015、validation 1,000。
- 数据目录：/data/datasets/minimind-lab/phase3/code-domain-v1r1；每轮 train 1,607,230 shifted assistant targets，validation 86,381。
- MBPP full/test 500 条，revision 4bb6404fdc6cacfda99d4ac4205087b89d32030c，CC-BY-4.0；仅用于最终测试，不用于训练或选 checkpoint。
- 初版 v1 含 loader 不接受的顶层 origin_id，probe 后标为 _INVALIDATED；v1r1 把来源移入 provenance.jsonl。正式两组都用 v1r1。
- 构建器执行时包含审计；独立 auditor 在本次收尾补跑，因此不能声称“独立审计先于训练”。文件 SHA 全部一致；train/val 与 MBPP 的 normalized exact/5-gram Jaccard ≥0.8 重复为 0，零 targets、非 assistant mask 错误为 0。
- 此污染结论只覆盖 Phase3 新增数据和 MBPP 提示，不等同于审计 S10 全部历史预训练数据，也不覆盖任意语义改写。
- 两组均 6×L20（GPU2–7），global sequence batch 96、3 epochs、594 optimizer steps、bf16、seq 768、augmentation off。训练 seed 实际为 42（trainer 默认），数据划分 seed 为 20260907；早期 config 混用二者，现已纠正。实际 targets 从窗口日志相加，受 drop_last 影响，不直接以整库 targets×epochs 替代。
- L01 LR=1e-5；L02 LR=1e-4。因此是两种训练配置的对照，存在 LR 混杂，不能归因于只切换 LoRA。
- 两组按最低 DDP validation NLL 选模，均为 step594。DistributedSampler 把 validation 1,000 补至 1,002 条，日志计数为 86,604 targets；表中 NLL 为该相同但轻微重采样的训练口径。
- 单次 wall 包含初始/周期 validation 和 checkpoint 保存，不含进程启动、数据构建或最终评测；GPU-hours=wall×6/3600。共享节点仍有 GPU0/1 服务，结果不是独占节点多次测速。启动使用目标卡空闲检查替代全节点 guard。
- 保存 best/last 推理权重及 adapter；没有保存 optimizer/scaler/RNG resume state，不能声称支持从中断点精确续训。所有权重留在 CPFS，Git 只保存 SHA/大小/路径。

## 评测与失败分析

MBPP 采用 lm-eval 0.4.12 的 mbpp_instruct：0-shot、chat template、greedy、256 new tokens、seed42、extract_code 与官方任务内置测试。七项和 IFEval 均 0-shot、chat template、FP16；行为集 greedy/192 tokens/open_thinking=false。L00 复用同权重 S10 的七项、IFEval、行为结果；MBPP 单独跑，来源与 SHA 在各 eval.json。

MBPP 空提取分别为 336、424、394/500。固定 extractor 对输出格式敏感，故总分同时反映代码内容和格式遵循。抽样中既有未形成可提取代码块、重复尾部，也有函数签名/算法不符合测试要求；不能把 0 分完全归因于提取器。原始及提取后样例见各实验 failure_examples.json，完整 500 条响应留在 CPFS。

CodeAlpaca 是广义代码指令，未筛为纯 Python 或可执行验证通过的数据；它与 MBPP 的函数签名/单元测试要求存在任务差异。这是后续数据改进假设，本阶段没有做能确证因果的消融。不要根据 MBPP test 定向构建答案或继续选模。

## LoRA 源码与合并验证

上游 apply_lora 仅注入方阵 Linear：此模型中是 8 层 q_proj/o_proj，共16处；r16，A 高斯初始化、B 零初始化、无 alpha 缩放/dropout。冻结原模型，只优化 A/B，共393,216参数；训练中关闭 compile。共用 trainer 新增 rank、参数清单、adapter-only 保存和系统指标。

首次 bf16 单提示检查在 atol=0.02 时失败（最大差0.25），后放宽至0.3的结果保留为历史观察，不单独作为强验收证据。补充 verify_lora_merge_precision.py 禁用 TF32，在 FP32 下验证 B@A 合并，6条提示均通过 atol/rtol=1e-4，最大差1.77e-5。FP16部署检查6/6生成 token一致、提示位置argmax一致；不保证所有输入逐token等价。

## 交付与复现

- [比较指标](../../experiments/03-lora/comparison.json)、[数据清单](../../experiments/03-lora/data-manifest.json)、[独立审计](../../experiments/03-lora/independent-audit.json)。
- 每个 L00/L01/L02 实验包含 config、command、run、metrics、eval、report、checkpoint manifest、SwanLab链接；训练组额外保存原始小型训练 metrics.jsonl。
- scripts/data/lora/build_code_domain_v1.py 构建；audit_code_domain_v1.py 独立审计；scripts/eval/run_phase3_eval.sh 在新输出目录复跑最终评测。
- scripts/eval/summarize_phase3.py 汇总；--publish 上传到既有 MiniMind-Lab 项目，以 receipt 避免重复创建。
- command.sh 明示实际训练 seed，复跑需给新的 EXPERIMENT_ID，且要求 minimind/out/s10_best_val_768.pth 的 SHA 与 S10 一致。

- L00-code-baseline-20260907: n/a-no-training / https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/980w5n9d
- L01-code-full-ft-20260907: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/pz1x0ux9 / https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/gtetdf4e
- L02-code-lora-r16-20260907: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s0l5ishg / https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/8z9jka3t

## 阶段判定与后续边界

实验协议、实际训练、独立测试、失败样例、资源成本与合并正确性证据已齐。以 accepted with limitations 收尾，保留负结果；L01/L02 不升级为主线模型，也不声称领域模型验收成功。没有预注册数值提升阈值，不在看过结果后倒推门槛。只有单训练种子、不同 LR、val padding、事后独立审计和无精确 resume 是本轮限制。

后续 Phase 4 的直接基线仍为 S10；本次请求只完成 Phase 3，不启动 Phase 4。若另行研究代码适配，可固定 LR/增加种子、构建训练侧可执行验证与函数签名约束数据，并用新的 held-out 集检验泛化。
