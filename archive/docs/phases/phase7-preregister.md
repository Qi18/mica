# Phase7 Dense vs MoE：启动预注册

统一计划第12节；2026-09-08；状态：进行中，先执行M00架构探针，不是正式全量训练。
Phase6报告已在main；Phase6 SwanLab补传工作区改动保持，不混入本阶段或自动推送。

## 固定对照
- 64M Dense与4experts/top1 MoE均随机初始化，seed42/43/44；参数量由实例实测。
- 相同seed与数据顺序，不宣称不同架构拥有完全相同初始参数（RNG消耗不同）。
- GPU7顺序运行，两臂seed43反转顺序；GPU0/1驻留任务不操作。
- 100updates、batch4、seq768、AdamW默认weight_decay0.01、LR5e-4、clip1。
- scheduler分母578371=ceil(2313483/4)，probe不压缩为100步cosine；正式batch改变后需独立重新注册。
- 数据沿用P03验收目录，每40个train shard取前10行，validation取前64行；记录tokenized SHA。
- 主损失CE与aux分开；NLL/PPL只由CE计算。Router负载含padding，不能当作仅有效targets的负载。
- 前10步warmup和每10步路由观测步不进入稳态吞吐，训练预算仍计入全部100步。
- FP32主权重+BF16 autocast；相同硬件、同tokenizer、同输入样本与targets；脚本不修改上游模型。
- checkpoint/optimizer落盘回读只验证roundtrip，不冒称完成独立进程SIGTERM/resume一致性验收。

## 前置与门控
技术探针要求所有loss/grad有限、无OOM、strict load、targets一致、输出完整、SwanLab记录齐全。
正式训练前仍须验证独立进程resume和DDP，并按实测资源冻结batch、全球targets、schedule与GPU分配。
正式质量对照：同1.28B有效targets、完整6.4M validation NLL及七项Base（不加chat template），同时比较成本；
不以100步loss、短validation或激活参数相同宣称MoE更优，也不假定MoE会成为强teacher。
Phase8仅在同tokenizer MoE Full SFT teacher真实强于S10后进入。

## 源码阅读与观察
model_minimind.py：MiniMindConfig、MOEFeedForward、aux_loss汇总；训练入口CE+aux。
重点检查top1归一化后的主损失对gate梯度，以及aux如何驱动路由；保留上游行为，不静默改算法。
记录总参数/名义每token激活参数、专家负载、aux、有效targets吞吐与显存。
证据：configs/moe/phase7-probe.json；scripts/launch/run_phase7_probe.py；
/data/artifacts/minimind-lab/phase7-probe-20260908。完整数据与权重不入Git。

## 启动实测

M00六个100步探针已完成。结果见../../experiments/06-moe/README.md。
正式全量训练未启动；仍需正式batch、DDP与独立resume门控。三seed短探针不代表能力验收。

## 正式启动预注册（2026-09-08）

M01 Dense与M02 MoE独立从随机初始化开始，先后使用GPU4/5/6/7，四卡DDP、每卡batch32、累积2、global batch256、seq768、seed42。
固定9038步cosine、AdamW默认wd0.01、LR5e-4、clip1、BF16 autocast/FP32主权重、TF32关闭。
正式数据复用P03原生PretrainDataset：先截text到seq-2，再包BOS/EOS，按pad token屏蔽labels。
M00手工编码路径不作为正式数据证据；本轮门禁与正式训练均走原生dataset，完整validation必须精确6400000 targets/11525 rows。
数据唯一targets为1.28B；world4 DistributedSampler会重复1条以等分（共2313484 exposures），两臂同顺序，记录实际targets，不冒称无重复精确1.28B曝光。

进入正式训练前：
1. 两臂四卡连续12步，对比真实SIGTERM第6步+独立进程resume至12步；全部模型/optimizer tensor须allclose(atol=rtol=1e-6)，数值记录，不放宽失败门槛。
2. 两臂正式batch各100步full-data profile，schedule仍9038；full validation计数必须匹配。
3. GPU4-7每次启动前空闲检查，阶段控制器文件锁；GPU0/1不操作，不停止其他任务。
4. fresh formal目录，profile模型不继续训练；最终固定epoch-end checkpoint用于质量对照，best-val单独保留。
每250步原子保存resume及推理权重，每1000步完整validation；SIGTERM在optimizer边界同步停下并保存。
每100步采样一次router，含padding；CE/aux分开。梯度更新仍沿用rank/microbatch等权的原生目标，token加权仅用于诊断日志。
两臂先后运行会存在硬件时序差异，单seed正式结果不宣称稳定算法优势。

启动控制器：scripts/launch/start_phase7_formal.py
训练入口：minimind/trainer/train_phase7.py（从已有P03入口独立扩展，不改旧实验脚本）。
固定配置：configs/moe/phase7-formal.json
证据根目录：/data/artifacts/minimind-lab/phase7-formal-20260908
正式训练完成后仍须七项Base评测与阶段报告，控制器不自动声明Phase7收尾、不晋级、不commit/push。

### r2确定性修订

r1连续/恢复比较失败；再跑连续训练也有约0.002最大权重差，因此不能归因为恢复损坏。证据保留phase7-formal-20260908/determinism-diagnosis.json。
不放宽1e-6容差，r2统一torch deterministic algorithms及CUBLAS_WORKSPACE_CONFIG=:4096:8，重新随机初始化两臂执行全部门禁。
r2产物根目录：/data/artifacts/minimind-lab/phase7-formal-r2-20260908。

### r3固定归约路径

r2确定性复验仍有1.054e-6差异，未过原阈值。当前PyTorch reducer.hpp明确find_unused_parameters=True且非static graph会关闭bucket rebuild；r3使用此设置及bucket_cap_mb1024、NCCL Ring/Simple，保留原数值阈值再验证。
该设置有额外遍历/归约成本，两臂和正式profile一致使用；不把吞吐与旧r1直接等同。
当前产物根目录：/data/artifacts/minimind-lab/phase7-formal-r3-20260908。

r3恢复门禁实测：Dense361/MoE681项tensor，连续与恢复最大绝对误差均0；41个正式分片SHA一致。证据见experiments/06-moe/M00b-r3-gates-20260908。

## 正式训练启动

r3所有门禁通过，M01 Dense已实际开始，9038updates；SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nyy90ef5。
M02 MoE在同组GPU4-7排队自动接续，独立随机初始化，不从M01权重继续。完整证据根目录/data/artifacts/minimind-lab/phase7-formal-r3-20260908。
七项Base待训练后进行，当前未晋级；未commit/push。

## 最终状态（2026-09-09）

以上内容保留为启动时预注册与历史快照。正式两臂训练及评测已完成，阶段按completed-not-promoted收尾，最终证据见[Phase7报告](phase7-moe.md)。
