# 项目工具

data/：可复用的数据构建与审计；eval/：通用评测、导出和 LoRA 验证；
launch/：环境检查与启动保护；sync/：数据下载。
根目录 trainer/ 包含保留的训练入口；历史阶段专用脚本在 archive/scripts/。
工具的保留不代表全部经过本轮 GPU、外部服务或全量数据验收。
