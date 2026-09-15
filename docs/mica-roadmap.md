# Mica 交付路线

## v0.1：项目入口与兼容基线

- 独立模型代码和可安装 Python 包。
- 文本/单轮 SFT 数据准备、FP32 单设备训练和恢复。
- token 加权 NLL、旧 state_dict 导入、文本生成。
- 基础非流式 chat completions 服务。
- Dense/MoE 旧实现等价、缓存、续训一致性自动检查。

## v0.2：训练与恢复

- 按行数据转换、紧凑 JSONL 偏移索引和按需 token 解码。
- DDP、梯度累积与全局有效 token loss。
- 定期原子 checkpoint、SIGTERM/SIGINT 保存及各 rank RNG 恢复。
- 双进程 CPU 对照与精确恢复验收，CUDA 多卡验收待完成。

## 后续版本

- 训练：CUDA 多卡验收、混合精度、调度器、索引缓存、SwanLab 集成。
- 后训练：逐项迁入 LoRA、DPO、GRPO/CISPO、Agentic RL，保留历史控制组。
- 评测：整合七项 benchmark、IFEval 和 Chat/Tool suite。
- 模型：按独立设计引入 Attention/FFN/MoE 结构变化并记录配置版本。
- 推理：流式返回、批处理和部署方案。

历史脚本仍位于 scripts/ 和 minimind/trainer/；其存在不代表上述功能
已经通过 Mica CLI 的验证。GitHub 仓库已统一为 Qi18/mica。
历史目录和记录中的原始路径继续保留，以便追溯原始命令与证据。
