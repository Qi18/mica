# 开发与验证

## 代码位置

模型位于 model/model_mica.py；数据位于 dataset/；阶段训练位于 trainer/，共同引擎位于 trainer/common/engine.py。
mica.py 为 CLI 与公开导入入口。来源和许可证见 [NOTICE](../NOTICE) 和 [LICENSE](../LICENSE)。

从仓库根目录：
```bash
python -m unittest discover -s tests -v
python -m compileall -q mica.py model dataset trainer evaluation inference tests
git diff --check
```

测试覆盖 Dense/MoE 冻结数值基准、cache、数据校验、加权累积、CPU DDP、保存失败、中断与恢复；它们不代表各专用阶段脚本的全量 GPU 验收。
架构变化需要明确更新配置、测试和权重兼容说明，而不是静默修改数值基准。

## 工作区边界

代码在 L20 /data/projects/mica 维护；数据、checkpoint 与日志不因文档整理移动。
当前 configs/、examples/、scripts/、archive/ 已在工作区被移除，本文不要求恢复它们。
.git 中的历史记录仍可用于追溯。docs 中的 .orig 文件是本地残留，不作为文档入口。

## CI 状态

现有 [.github 工作流](../.github/workflows/repository-check.yml)仍引用已经删除的 scripts/check_repository.py、configs/smoke/ 与 examples/tokens.jsonl。
因此本地单元测试通过不等于当前清理后的 CI 可用。文档修复不重建这些目录，也不修改 CI；后续需要单独更新工作流。
