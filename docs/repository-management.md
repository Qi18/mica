# 仓库维护

- model/、dataset/、trainer/、evaluation/、inference/：直接维护的主线代码。
- mica.py：CLI 和公开模型导入入口；安装分发名仍为 mica-llm，避免与其他分发混淆。
- configs/：当前配置；archive/：历史资料和重放索引。
- 主线模型只有 model/model_mica.py 一份；架构修改必须更新数值、cache、训练和恢复测试。
- L20 /data/projects/mica 为权威 checkout；不移动数据与 checkpoint。
- 上游来源与许可证必须保留，不再进行 subtree 同步。
- 新训练使用 mica train；保留的专用 trainer 脚本须按其依赖和参数单独验证。
