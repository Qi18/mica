# Mica 架构

Mica 的运行代码位于 `model/、dataset/、trainer/、inference/`，安装后通过 `mica` 命令使用。
初始 Dense/MoE 实现从本仓库已有 MiniMind 模型派生，保持 state_dict 键名一致。
未来结构演进直接在 Mica 实现中进行，使用迁移前冻结的 Dense/MoE logits 检查兼容，不保留第二份模型实现。

| 模块 | 职责 |
|---|---|
| model/model_mica.py | 唯一 Dense/MoE 模型 |
| model/model_lora.py | LoRA |
| model/runtime.py | checkpoint 加载、导入与批处理 |
| dataset/prepare.py | 数据转换 |
| dataset/indexed.py | 按需索引 |
| dataset/lm_dataset.py | 专用训练 Dataset |
| trainer/common/engine.py | DDP、累积、checkpoint 与恢复 |
| trainer/<stage>/train_*.py | 保留的各阶段训练脚本 |
| evaluation/loss.py | NLL / PPL |
| inference/generation.py | 生成与服务 |
| mica.py | CLI 与公开导入 |

## 模型结构

默认 Dense 为 Decoder-only Transformer：Embedding → 8 个 Block → RMSNorm → LM Head。
每个 Block 为 RMSNorm → GQA/RoPE 注意力 → 残差 → RMSNorm → SwiGLU FFN → 残差。
隐藏维度 768，Query/KV 头数 8/4，词表 6400，输入输出 embedding 共享权重；MoE 替换 FFN。
max_position_embeddings 是位置表范围，不等于已经验证的长上下文能力。

训练入口、数据格式和恢复约束见 [训练指南](training.md)，运行示例见 [README](../README.md)。

## 结构升级规则

1. 保存架构配置和版本；新结构不得冒充旧结构 strict-load 成功。
2. 检查 forward/backward、cache 与保存/恢复。
3. 对改变的层新增有意义的数值测试。
4. 旧报告保留其原始模型名、设备和协议。
