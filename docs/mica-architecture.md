# Mica 架构

Mica 的运行代码位于 `src/mica_llm/`，安装后通过 `mica` 命令使用。
初始 Dense/MoE 实现从本仓库已有 MiniMind 模型派生，保持 state_dict 键名一致。
未来结构演进直接在 Mica 实现中进行，历史 MiniMind 模型保留为兼容性测试参照。

| 模块 | 职责 |
|---|---|
| modeling_mica.py | MicaConfig、Attention、Dense/MoE FFN、因果语言模型 |
| data.py | 文本/单轮问答转 token JSONL，SFT prompt mask |
| runtime.py | recipe、训练、恢复、checkpoint、token 加权 NLL |
| inference.py | 贪心文本生成、非流式 chat completions HTTP 接口 |
| cli.py | 统一命令入口 |

## 数据与训练

预训练输入为 `{"text": "..."}`，单轮 SFT 为 `{"prompt": "...", "response": "..."}`。
转换后每行保存 input_ids 和 labels，prompt/padding 的 labels 为 -100。
截断后没有有效 next-token target 会报错。v0.1 将数据载入内存，适用于小数据和功能验证；
历史大规模数据处理、DDP、LoRA、DPO、GRPO、Agentic RL 脚本尚未迁入新 CLI。

训练用 FP32 AdamW、梯度裁剪及固定顺序 batch。Dense/MoE 均可配置。
续训允许增加总 steps，其余 recipe 与数据 SHA256 必须一致；
恢复 optimizer、CPU/CUDA RNG 和 step。checkpoint 保存于命令完成时，
尚不支持定时保存或进程中断自动保存。

## 产物

模型目录包含 config.json 和 model.pt；训练目录另外包含 training.pt、
recipe.json、run.json 和 metrics.json。
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
