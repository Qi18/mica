# 源码阅读索引

建议顺序：

1. [训练运行时结构](00-training-runtime.md)；
2. [数据与 Tokenizer](01-data-tokenizer.md)；
3. [模型结构与前向传播](02-model-architecture.md)；
4. Pretrain；
5. Full SFT 与 LoRA；
6. DPO；
7. PPO 原理；
8. GRPO 与 CISPO；
9. Rollout Engine；
10. Agentic RL；
11. 评测、模型转换与服务。

每篇笔记应包含：模块职责、入口与调用链、关键 Tensor shape、训练/推理差异、观察实验、尚未验证的问题，以及对应源码 commit。
