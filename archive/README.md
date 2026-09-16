# 历史资料归档

整理前的完整可运行代码版本：[625fbe7](https://github.com/Qi18/mica/tree/625fbe74692201969bcd9fa32dd4e72213b1d730)。

- experiments/：实验编号、配置、指标和报告保持原有内部结构。
- configs/、scripts/：历史阶段专用配置、启动、评测和汇总脚本。
- docs/、blog/：历史协议、阶段报告、阅读笔记和草稿。
- upstream/：初始上游说明与图片；仅供来源追溯，不参与构建。
- migration-map.json：本次文件移动清单。

归档脚本不属于当前受支持的运行入口。原始命令和绝对数据路径不批量改写；重放时在独立 checkout 使用上述历史提交。Markdown 导航链接随文件移动修复。
当前配置位于根目录 configs/，训练代码位于 trainer/。
数据、checkpoint 和日志不进入 Git。L20 原 minimind/ 下的忽略产物保持原物理位置，因此本机可能仍存在该目录；全新克隆没有该源码目录。
