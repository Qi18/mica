# Phase8 序列级蒸馏 v1：预注册

2026-09-09 用户授权Qwen3-8B蒸馏实验；状态：数据准备/受控pilot，未训练完成、未晋级。
原计划同词表白盒KL/OPD在本分支不执行：Qwen词表与MiniMind不同。这里只比较同prompt、两套答案的CE训练；不得冒称token KL或OPD收益。

## 已知teacher证据

Qwen3-8B完整IFEval541题非思考：prompt strict80.9612%、instruction strict87.0504%；S10本轮原生重测18.4843%、31.1751%；官方MoE为12.0148%、23.6211%。
教师只在指令维度取得候选资格，并未宣称全部事实答案正确。Qwen资格评测模型assets SHA清单固定实际权重；生成使用vLLM0.8.5.post1 FP16而资格评测使用Transformers FP16，记录引擎差异，不宣称bitwise一致。

## 数据与对照

- 从已有v4构建器生成4096条新编号的约束候选，附真实verifier参数；从broad-repair-v2训练split取4096条单轮候选。
- validation/test各512条，分别256新编号约束+256原broad对应split；在teacher生成之前冻结，不用于训练、不用于过滤teacher训练答案。
- 排除冻结IFEval541、固定Chat/Tool及历史S10 held-out的原题/归一化匹配；跨split相同扫描为0。非语义近重复证明；同verifier家族和模板仍复用，不能称全新开放域泛化。
- broad replay可能已被S10训练见过；新课程不复制IFEval题目。来源、SHA、拒绝原因保留。
- Teacher只收到原始用户messages，不接触original答案、验证器参数或任何held-out答案；非思考greedy，max_new_tokens512。
- 拒绝空答、thinking块、生成触顶、MiniMind编码失败、assistant>512tokens或完整序列>768；不静默截断。
- 约束答案必须通过strict checker；broad只做结构与严重重复过滤，未证明事实全对。人工质量抽查是最终推广门，不自动宣称数据高质量。
- 对称保留4096对：2048约束+2048broad；某类不足2048则阻止训练，不自动降低过滤阈值。
- K01学习这些prompt的原始答案；K02学习同prompt的teacher答案。两臂按相同ID顺序进入数据流。

## 训练预算与数值门

固定S10原生pth SHA46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131；均独立加载，不串联K01→K02。
有效assistant targets各4194304，8192targets/optimizer update，共512updates；microbatch8，最大seq768。
单卡FP32主权重+BF16 autocast，AdamW wd0.01，LR3e-6、5% warmup后cosine到10%，clip1，seed20260909。
按有效token归一化梯度；长样本跨token window时仅拆labels监督区间，保留完整上下文，防止丢target或双计。
同token预算/同prompt集合不等于相同样本曝光次数：分别记录每轮targets、epoch、row draws、实际forward数。
固定final checkpoint，不按teacher测试分数选模型；共用原始答案validation NLL仅作诊断，不能据此认定teacher风格错误。
正式前做CPU编码/shift/EOS/右padding检查、真实模型padding不变性和短训练finite/保存strict-load检查。
保存model/optimizer与采样位置；尚无独立进程resume等价性认证，不自动断点续训。
数据门和smoke任何失败均停止controller，不启动后续训练。

## 评测和验收

S10/K01/K02同原生FP16接口：独立冻结test约束strict、完整IFEval541、七项同chat协议、固定Chat10/格式6/Tool8与重复。
所有最终模型仅评测一次，不据test继续调参。IFEval已用于teacher选择，因此不当作完全独立的最终盲测。
主证据：新冻结约束test上K02相对K01的paired bootstrap95%CI下界>0；IFEval相对S10不降；七项macro回退<=1pp；Chat>=8/10、格式>=5/6、重复<=1/10、Tool>=7/8。
同时报告相对S10及K01的差值、效应区间和样本量。KL下降不在本实验指标中。
只有程序指标、人工broad质量抽查、完整回归及用户验收都通过才能推广；否则按负结果收口，S10保持release。
K02/K01对照测的是teacher答案替换配方收益，包含长度、风格与每轮曝光变化，不能声称纯知识注入因果效应。

## 执行边界

/data/projects/minimind-lab为唯一源码工作区；数据/原始输出/权重只留CPFS，不上传Git。
GPU6生成teacher答案，GPU4/5训练/评测，GPU7可运行S10基线评测；每次启动重新检查空闲，不操作GPU0–3其他项目。
SwanLab统一MiniMind-Lab；vLLM既有环境不增改包，生成指标由独立minimind环境回填，训练/评测在线记录。
代码、配置、数据manifest、验收和错误证据保存在实验目录；本轮未获额外提交推送指令，不自动push。

## 用户要求的评测调度变更：8 卡单模型并行（2026-09-09）

按用户要求，将模型间并行改为模型串行；每个待评模型使用 GPU0–7 八个独立副本，
由主进程将 harness requests 按原始索引 round-robin 分配，收集后恢复原始顺序，
统一交给原 harness 计算任务/组指标，不平均各分片均值。
单卡生成 batch1、likelihood batch16，权重、FP16、模板、seed、生成上限及题集不变。
likelihood 分片会改变 batch 组成，不宣称与原单卡逐位数值等价。
分片覆盖测试包含 0/1/7/8/9/541/112919 请求；重复或遗漏会触发断言。
已完成 S10 保留，K01→K02 串行运行；旧未完成评测目录保存在 before-eight-gpu，
不删除任何训练权重。已有约束/行为结果仍按协议校验复用。

入口：scripts/launch/start_phase8_eval_8gpu.py。
评测：scripts/eval/eval_phase8_sequence_8gpu.py。
并行适配器：scripts/eval/phase8_eight_gpu.py。
原始目录下 eight-gpu-manifest.json / eight-gpu-source-snapshot 固定执行记录，
eightgpu-eval-<arm>-driver.log 记录运行日志，
eval-<arm>/worker-<gpu>-ready.json 与 worker-<gpu>-progress.json 记录每卡进程和生成进度。
