# Phase8 阶段报告：蒸馏（off-policy / on-policy）

- 状态：进行中，仅K00 teacher资格检查；正式K01–K04尚未启动。
- 启动日期：2026-09-09；Phase7报告已在main，commit 29ad8e1493bab0549e3bb8ed23e4aa2c22fbd59d。
- Student固定为Phase2 release S10，不使用Phase7随机初始化Base当student。
- Phase7原生MoE实验已收尾且未晋级，不能直接作为Full SFT teacher。

## 1. 本阶段目标

在相同student初始权重、数据/有效targets预算与评测协议下，比较CE-only、off-policy蒸馏与on-policy蒸馏。
先证明teacher可用且强于student，再训练；下载成功或strict load成功均不等于质量验收。

## 2. K00上游teacher资格检查

候选：ModelScope gongjy/minimind-3-pytorch 的 full_sft_768_moe.pth。
锁定revision=24a40130901f5229e841b41f0c070ebd95f3631d。
发布SHA256=a050020ea6d1b9e824693d0db525b1a0a8b40f36a934ea8d89da161368f20cc1；406719725 bytes。
来源：https://modelscope.cn/models/gongjy/minimind-3-pytorch。
不使用agent/grpo/ppo模型冒充Full SFT，不使用含exam LoRA的权重。
官方原生pth不含tokenizer；使用固定官方源码393e387的tokenizer，核对与S10的vocab映射、完整分词流水线、special tokens及chat template语义一致，保存各文件SHA。
这说明兼容性与解释约定，不单独证明官方权重历史使用了哪个tokenizer revision。

Student checkpoint：
/data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/checkpoints/s08_best_val_768.pth；
SHA256=46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131。

## 3. 质量门控预注册（看到候选评测结果之前冻结）

1. 两臂CPU strict load、有限logits、同vocab/tokenizer/chat template；固定权重SHA，推理后再校验。
2. 同一原生MiniMind/HFLM适配路径重新评测S10与teacher，不把历史Qwen3导出结果直接作精确基线。
3. 第一门：完整IFEval541题、0-shot、chat template、FP16、greedy、max_new_tokens1280、batch1、seed42。
   原生generate会直接除temperature，因此适配器使用temperature=1、do_sample=False、top_p=1、top_k=0实现无采样argmax；不把temperature=0传入原生实现。
   原生generator不支持HF stopping criteria；IFEval until=[]，用EOS及1280上限结束；不推广本适配器到其他生成任务。
4. 在全部541题prompt hash配对一致后，teacher的prompt strict差值须大于0且配对bootstrap95%区间下界大于0，作为本阶段“明确强于”的操作化判据。
5. 第一门通过后才继续七项同chat协议与固定行为门禁：七项macro相对同协议S10不得下降超过1pp；Chat>=8/10、格式>=5/6、重复<=1/10、Tool>=7/8。
6. 任一必要门不通过，记录candidate-not-qualified，推迟正式蒸馏；不通过缩短输出、调整模板或反复选seed来挽救test分数。
7. K00是teacher选择/资格评估集，不是最后蒸馏收益的独立盲测；正式训练前另冻结train-disjoint held-out用于K01/K02/K04比较。

## 4. 后续实验及归因限制

- K01：从S10开始CE-only续训控制组。
- K02：同固定数据，0.5 CE + 0.5 forward KL。
- K03：必要的KL方向/CE混合权重匹配控制；也可做temperature/alpha消融。
- K04：student自采样、teacher冻结、per-token reverse KL；rollout与更新token对齐、mask/shift、无梯度泄漏必须先过单测。
- 统一计划将K02/K04写为“只改序列来源”不够准确：当前定义还同时改变KL方向和CE混合项。
  因此没有匹配loss的off-policy控制时，只能比较两套训练配方，不能把差值单独归因到on-policy采样。
  该问题必须在正式配置冻结时解决；当前不修改旧实验、不贸然启动K04。
- 记录CE/KL/总loss、teacher-student一致率、长度/重复、吞吐/显存/成本、IFEval和通用回归；KL下降不等于能力提升。

## 5. 证据与运行边界

准备脚本：scripts/eval/prepare_phase8_teacher.py。
原始目录：/data/artifacts/minimind-lab/phase8-k00-teacher-gate-20260909。
归档目录：experiments/07-distill/K00-teacher-qualification-20260909。
只使用检查为空闲的GPU4/5做资格评测，GPU0–3其他项目不操作。
SwanLab统一MiniMind-Lab，资格评测标记K00，不能命名为K01/K02训练。
不上传权重或原始数据、不删除checkpoint、不改变S10 release。

## 启动记录

兼容性已通过：官方teacher文件哈希/大小匹配，S10原生pth哈希与Phase2 release清单一致，两个模型strict load与CPU有限logits通过。
tokenizer从本地Git对象中的固定上游393e387读取，与当前源码/S10导出tokenizer语义完全一致，不同步或改变上游源码。
准备过程保留两类失败：GitHub raw网络超时；初版误将S10 safetensors导出文件SHA用于原生pth。
后者已对照Phase2报告和实际两文件哈希纠正，未放宽校验、未更换student权重。
K00执行scripts/eval/run_phase8_qualification.py，student GPU5、teacher GPU4；目前尚无质量结论。

## 2026-09-09：当前执行的 Qwen3-8B 序列蒸馏分支

本节修订 Phase8 的实际执行路线；早期同词表 KL/OPD 方案保留为历史设计，尚未实施。
Qwen3-8B 完整 IFEval prompt strict 为 80.96%，原生 S10 为 18.48%，官方 MoE 为 12.01%。
这仅证明指令遵循差距，不代表教师已通过通用能力或事实正确性验收。

当前采用 Qwen3-8B → S10 的 response/sequence-level CE 蒸馏，不跨不同 tokenizer 直接计算 token KL。
K01 原答案 CE 续训、K02 教师答案 CE 蒸馏：同一组验收通过的 4,096 个 prompt，
各自从 S10 独立开始，4,194,304 个有效 assistant target tokens，512 次更新，LR 3e-6。
有效 token 预算相同不表示行曝光次数或 epoch 相同。
8,192 条候选正在生成教师答案；必须先通过成对过滤与 smoke，才开始正式训练。
验证/测试各 512 条已冻结。无精确归一化重合不等于语义独立，指令模板家族仍有复用。

验收、限制和精确配置见 [序列蒸馏预注册](phase8-sequence-v1-preregister.md)；
实验记录见 [K01/K02 执行目录](../../experiments/07-distill/sequence-v1-20260909/report.md)。
所有结果留在 CPFS、训练与评测记录到 MiniMind-Lab SwanLab；当前不自动晋升模型、不提交推送。

## Phase8 最终归档结论（2026-09-09；覆盖前文启动快照）

训练与评测全部完成，实验按负结果归档（completed-negative-not-promoted）。
用户决定暂停后续白盒蒸馏与 OPD；两者未实施，不列为已完成成果。
K02 数值验收不通过，保持 S10 release。广义对话人工复核未完成，不据此认证教师数据质量。
这不是模型能力验收通过，也不是整个蒸馏方法被否定。

| 指标 | S10 | K01 原答案 CE | K02 教师答案 CE |
|---|---:|---:|---:|
| 新约束集 strict（256题） | 93.36% | 93.75% | 44.92% |
| IFEval prompt strict（541题） | 18.48% | 18.85% | 14.60% |
| 七项 macro（29,638题） | 33.0028% | 32.9498% | 32.9280% |
| Chat | 8/10 | 9/10 | 8/10 |
| 格式 | 5/6 | 5/6 | 5/6 |
| 重复 | 1/10 | 0/10 | 1/10 |
| Tool | 7/8 | 6/8 | 6/8 |

K02−K01 新约束集差值 -48.8281pp，逐题配对 bootstrap 95% CI [-55.0781,-42.5781]。
IFEval 相对 S10 下降 3.8817pp；七项 macro 基本持平，Tool 6/8 低于7/8门槛。
K01/K02 各4,194,304有效targets、512updates，从相同S10初始化；
训练分别337.85/192.13秒，数据各4096条（2048约束+2048通用问答）。
每遍有效targets为154835/670669，累计row_draws为110979/25602。
row_draws 是重复读取条目的计数，不是独立样本数；等token不等于等epoch或等行曝光。
答案长度/样本曝光差异是潜在混杂，未完成因果诊断，不能写成已确认退步原因。

新约束集与训练复用模板/约束家族，精确归一化去重不能证明语义独立；
IFEval曾用于teacher资格选择，非完全独立盲测；单seed不证明跨训练随机性的稳健性。
S10七项使用单卡，K01/K02使用8卡请求分片；精度与每卡batch协议保持，
batch组成变化可能带来数值差异，不对极小macro差值作因果或显著性结论。
本轮仅黑盒序列级CE蒸馏，不是token-KL、白盒分布蒸馏或OPD。
JSON序列化故障与GPU绑定故障、修复记录保留，不能把失败尝试抹去。

详细证据：[实验报告](../../experiments/07-distill/sequence-v1-20260909/report.md)。
