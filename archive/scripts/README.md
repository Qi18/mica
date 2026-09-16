# 实验脚本

- `launch/`：调用 `minimind/trainer/` 的 L20 启动脚本；
- `sync/`：整理远端日志、SwanLab 链接和实验摘要；
- `eval/`：统一评测入口；
- `cleanup/`：按 checkpoint manifest 清理冗余产物。

脚本必须支持 dry-run 或输出最终命令，并禁止把凭据写进参数、日志和仓库。

## Stage 0 启动工具

```bash
# 正式训练前门禁；默认要求 SwanLab 已登录
scripts/launch/preflight_l20.py

# 8 卡 NCCL all-reduce
/data/venvs/minimind-lab/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/launch/nccl_smoke.py

# 受保护的训练启动；命令前先检查 GPU、SwanLab 和单实例锁
scripts/launch/run_guarded.py --dry-run -- torchrun --standalone \
  --nproc_per_node=8 minimind/trainer/train_pretrain.py --help

# 创建并验证实验目录
scripts/launch/init_experiment.py 01-pretrain P01-dense-mini
scripts/sync/validate_experiment.py experiments/01-pretrain/P01-dense-mini
```

`--skip-preflight` 只用于记录清楚的诊断实验，不能用于正式训练。
