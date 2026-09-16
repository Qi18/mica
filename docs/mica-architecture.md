# Mica 架构

Mica 的运行代码位于 `model/、dataset/、trainer/、inference/`，安装后通过 `mica` 命令使用。
初始 Dense/MoE 实现从本仓库已有 MiniMind 模型派生，保持 state_dict 键名一致。
未来结构演进直接在 Mica 实现中进行，使用迁移前冻结的 Dense/MoE logits 检查兼容，不保留第二份模型实现。

| 模块 | 职责 |
|---|---|
| model/modeling_mica.py | 唯一 Dense/MoE 模型 |
| model/model_lora.py | LoRA |
| model/runtime.py | checkpoint 加载、导入与批处理 |
| dataset/prepare.py | 数据转换 |
| dataset/indexed.py | 按需索引 |
| dataset/lm_dataset.py | 专用训练 Dataset |
| trainer/training.py | DDP、累积、checkpoint 与恢复 |
| trainer/train_*.py | 保留的各阶段训练脚本 |
| evaluation/loss.py | NLL / PPL |
| inference/generation.py | 生成与服务 |
| mica.py | CLI 与公开导入 |

## 数据与训练

预训练输入为 `{"text": "..."}`，单轮 SFT 为 `{"prompt": "...", "response": "..."}`。
转换后每行保存 input_ids 和 labels，prompt/padding 的 labels 为 -100。
截断后没有有效 next-token target 会报错。数据转换按行执行；训练逐行建立偏移索引，按需解析 token。
LoRA、DPO、GRPO、Agentic RL 的历史脚本尚待迁入新 CLI。

训练用 FP32 AdamW、梯度裁剪及固定顺序 batch。Dense/MoE 均可配置。
续训允许增加总 steps，其余 recipe 与数据 SHA256 必须一致；
恢复 optimizer、各 rank CPU/CUDA RNG 和 step，要求相同 world_size。
checkpoint 定期保存，SIGTERM/SIGINT 在 update 边界触发保存；详见 [训练与恢复](mica-training.md)。

## 产物

每个 step 目录包含 config.json、model.pt、training.pt 和 recipe.json。
运行根目录包含 latest.json、run.json、metrics.jsonl，加载时自动解析最后一个完整 checkpoint。
模型只依赖 Mica 包即可加载；旧权重导入需要提供匹配的架构配置并 strict load。
导入表示格式迁移，不能将原始 MiniMind 训练成绩归属于新设计的模型。

## 推理

serve 提供 /health 与 /v1/chat/completions 的基础字段。
当前支持文本消息、max_tokens、非流式贪心生成，拒绝 streaming、
tools 和采样参数。默认绑定 127.0.0.1，使用 Flask 开发服务器。
服务并非完整 OpenAI API 实现，也未进行生产部署验收。

## 结构升级规则

1. 保存架构配置和版本；新结构不得冒充旧结构 strict-load 成功。
2. 检查 forward/backward、cache 与保存/恢复。
3. 对改变的层新增有意义的数值测试。
4. 旧报告保留其原始模型名、设备和协议。