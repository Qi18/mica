# Mica 交付路线

## v0.1：项目入口与兼容基线

- 独立模型代码和可安装 Python 包。
- 文本/单轮 SFT 数据准备、FP32 单设备训练和恢复。
- token 加权 NLL、旧 state_dict 导入、文本生成。
- 基础非流式 chat completions 服务。
- Dense/MoE 旧实现等价、缓存、续训一致性自动检查。

## 后续版本

- 训练：大数据流式读取、DDP、定期 checkpoint、SwanLab 集成。
- 后训练：逐项迁入 LoRA、DPO、GRPO/CISPO、Agentic RL，保留历史控制组。
- 评测：整合七项 benchmark、IFEval 和 Chat/Tool suite。
- 模型：按独立设计引入 Attention/FFN/MoE 结构变化并记录配置版本。
- 推理：流式返回、批处理和部署方案。

历史脚本仍位于 scripts/ 和 minimind/trainer/；其存在不代表上述功能
已经通过 Mica CLI 的验证。GitHub 仓库改名在迁移入口验证完成后单独执行，
不通过重命名历史目录破坏原始命令与证据链接。
