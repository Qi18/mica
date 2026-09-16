# 原方案修订：固定轨迹测量
修订发生于正式运行前，保留 initial-config.json、correctness.json 和 precision-diagnosis.json，不调整原容差。

原方案中 FP32 logits 与 greedy 比较通过。BF16 未满足固定的相对 RMSE 门槛，并在部分真实提示词上出现 greedy 分歧。关闭 SDPA 的复核仍存在差异；独立 K 投影测试也观察到矩阵形状变化带来的 BF16 舍入差异。

因此本次仍使用原生 BF16 实现测试效率，但四个实验臂统一输入 FP32 GQA 生成的 continuation。每步仍执行 argmax，不过下一步输入采用固定 reference trace，避免不同生成结果造成输入内容混杂。原生模型代码、权重与固定阈值均保持不变。

允许给出：固定工作负载下的 decode 吞吐、TTFT、显存、缓存占用和当前实现的加速比。
不允许给出：BF16 输出完全等价、无质量损失、GQA 提升模型能力、任意推理后端均有相同收益。

原定 BF16 一致性验收为失败；效率测量可以完整完成并报告，不能将整体标为无条件通过。
