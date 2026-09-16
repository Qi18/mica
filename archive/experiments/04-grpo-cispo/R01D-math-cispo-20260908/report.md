# R01D-math-cispo-20260908

状态：invalidated-data-confound；不得用于能力结论。

v1 虽然答案位置和模板各自边际均衡，但两者由同一 index 线性分配，template_id 可确定性预测 answer_position。R01C/D 的 train reward 接近 1 后，validation pass@1 均为 0%，并呈现 A→B、B→C、C→D、D→A 的系统错位。这是 reward hacking，不是算术能力。

问题在 validation 阶段发现，正式 test 未打开。原始训练 run 保留：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/ecv3zn1h。修复只改变数据联合分配和 auditor，正式对照改名 R02A/B/C/D。
