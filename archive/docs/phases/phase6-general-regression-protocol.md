# Phase6通用能力回归协议

2026-09-08，按用户要求执行，不进行训练或模型晋级。
模型：S10实际训练起点的FP32导出、A01-v3 LR3e-6 checkpoint-570、
A02 GRPO v2与A03 CISPO v2固定末步；四臂全部重跑。

使用现有lm-eval 0.4.12及本地缓存，0-shot、seed42、apply_chat_template、FP16、batch16，
每进程GPU显存上限15%，不改变训练和其他进程。
IFEval v4完整541 prompts，任务既有greedy/max_gen_toks1280配置，不缩短输出或抽样。
七项完整集合：ceval-valid、cmmlu、arc_easy、piqa、openbookqa、hellaswag、social_iqa。
沿用历史汇总方式：各任务优先acc_norm，否则acc，七项等权macro。
这不是官方榜单的新提交，也不等同于官方模型分数。

分别报告A01相对S10、A02/A03相对A01以及相对S10的百分点变化。
IFEval保存逐题prompt/instruction strict/loose结果，并计算配对区间；不能仅看均值。
七项保留逐题结果和原始配置，防止宏平均掩盖某项下降。
现有文档尚未预注册统一下降容差，本轮不临时补一个通过阈值，只报告实际变化与限制。
旧Tool7/8门槛不变；本次评测不自动导致Phase6收口。
这些基准已用于历史课程研究，不声称对整个项目完全未见；本轮不使用它们继续训练或选checkpoint。

全部使用本地离线缓存，不启用SwanLab或其他上传。保存权重/配置/源码指纹、完整日志及逐题结果到CPFS。
仓库归档轻量汇总及可追溯清单，不将权重、数据集或庞大全量日志加入Git；未经请求不commit/push。
