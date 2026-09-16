# A00：S10 Agent 基线（validation）

状态：completed；角色：Phase6 v1 pilot baseline；不是正式 test 结果。
输入：S10 导出 FP32 模型；GPU1 空闲显存；greedy，batch8，最多4轮，每轮96tokens。
数据：record-agent-v1 validation 240 条；train/test 未用于此评测。
SwanLab URL 见本目录 swanlab-url.txt，指标见 metrics.json。

- E2E：1/240 = 0.4167%；schema validity：74.2547%。
- 工具选择：54%；参数语义：49.25%；execution success：82.8431%。
- lookup / chain / retry E2E：1.25% / 0% / 0%。
- 共20.588秒，平均78.363生成tokens/任务；批量吞吐，不是单请求延迟。

E2E 要求有序正确工具链 + 严格仅数字最终回复。基线有部分已正确查询，但最终输出说明性语句而失分；
因此低分不能解释为“完全不会使用工具”。逐题轨迹保留便于区分格式错误与执行错误。
原始产物：`/data/artifacts/minimind-lab/A00-s10-agent-baseline-20260908/validation-agent/`。
