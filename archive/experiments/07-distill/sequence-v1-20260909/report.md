# Qwen3-8B → S10：K01/K02 序列蒸馏试验

## 当前状态

2026-09-09 已启动教师答案生成与带门禁的后台控制器；正式训练尚需数据过滤通过。
K01 训练实现 smoke 已通过（2 updates / 16,384 targets），不是正式结果。
源代码与预注册快照已保存在原始目录 source-snapshot，implementation-manifest.json 固定 SHA。
本报告是启动快照；运行时以 pipeline-status.json、各阶段 DONE/FAILURE 文件为准。

## 对照设计

| 项目 | K01 | K02 |
|---|---|---|
| 初始化 | S10 固定 SHA | 同一 S10 |
| 训练答案 | 原始参考答案 | Qwen3-8B 生成答案 |
| prompt | 同一组 4096 个 | 与 K01 成对 |
| 优化目标 | assistant CE | assistant CE |
| 预算 | 4,194,304 targets / 512 updates | 相同 |
| 学习率 | 3e-6 | 3e-6 |
| GPU | 4 | 5 |

新冻结的约束测试 256 题为主要对照指标，另测完整 IFEval 541、
七项 benchmark 29,638 题、Chat/格式/重复和 mock Tool。
新约束集仍与训练共享指令家族，不能称作全新能力盲测。
广义对话样本需人工质量复核；仅规则通过不证明答案事实正确。
无论数值如何均不自动替换 S10；单 seed 试验不证明跨训练随机性的稳健收益。

## 执行入口与证据

配置：configs/distill/phase8-sequence-v1.json。
脚本：scripts/data/{prepare,generate,finalize}_phase8_sequence.py；
scripts/launch/{train,start}_phase8_sequence.py；
scripts/eval/eval_phase8_sequence.py；公共标签/token窗口逻辑 scripts/phase8_common.py。

CPFS 原始目录：/data/artifacts/minimind-lab/phase8-sequence-v1-20260909。
数据目录：/data/datasets/minimind-lab/phase8-sequence-v1-20260909。
控制器按生成→过滤→smoke→两组训练→三组评测→比较推进，失败后不放宽阈值、不自动重跑。
训练及评测使用 MiniMind-Lab SwanLab；教师生成指标先本地记录，完成后回填同项目。
最终 comparison.json 保存 paired bootstrap、所有回归门禁与不自动晋升结论。

## 已有证据

- Qwen3-8B IFEval：[SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/fjfxv8yz)。
- K01 实现 smoke：[SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/vvlunoys)。
- 正式训练收益：待完成，不能以 smoke loss 推断。

## 评测序列化故障与恢复（2026-09-09）

训练已完成，K01/K02 均为 512 updates / 4,194,304 targets，权重未改变。
首轮评测 S10 保存七项结果时遇到 function、K01 保存 IFEval 结果时遇到 torch.dtype 无法 JSON 序列化。
总控制器等待其他子进程结束，故总 FAILURE 文件滞后于子进程故障；不能仅以总状态判断成功。

修复 phase8_common.py 的显式 JSON 编码：数值与数组保持可分析类型，dtype/device 转描述，
函数配置转 module/qualname 描述；未知类型仍报错。嵌套数值、dtype、函数、NumPy、Tensor
以及未知类型拒绝测试通过，证据 serialization-fix-test.json。IFEval 改为优先保存逐题样本。
旧目录整体移入 failed-attempt-1，旧驱动日志、SwanLab runs 和训练权重保留。
停止了核验过 PID/命令的旧评测控制器与 K02 评测进程，不停止其他项目。

恢复入口 scripts/launch/recover_phase8_sequence_eval.py 只执行评测与比较，不重训。
对已落盘约束/行为结果，验证 checkpoint、协议、配置哈希和测试 ID 后复用；
未保存的 IFEval / 七项结果重算。新日志 recovery-eval-{S10,K01,K02}-driver.log。
恢复代码快照、哈希和启动凭证见 recovery-source-snapshot / recovery-manifest.json /
recovery-controller-launch.json。当前恢复评测已启动，尚未宣称完整通过。

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

八卡首次运行核验发现进程启动后修改 CUDA_VISIBLE_DEVICES 未生效，计算集中于 GPU0；
已停止本实验控制器、主评测进程与从 ready 记录核验身份的 8 个 worker。
现场保留在 eightgpu-binding-attempt，修复为显式 torch.cuda.set_device(gpu) 和 model.to(cuda:gpu)，
ready 文件同时记录 actual_cuda_device 并断言与分配一致，之后重启评测。

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

机器结果见 comparison.json；train-*/eval-* 保留紧凑指标、manifest、权重SHA和SwanLab链接。
完整日志/逐题输出/权重仍在CPFS，位置见 raw-artifact-inventory.json。
