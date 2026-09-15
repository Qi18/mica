# Mica 仓库管理方式

## 1. 仓库职责

`mica` 是语言模型工程项目，GitHub 仓库为 [Qi18/mica](https://github.com/Qi18/mica)，当前主线包括：

- `src/mica_llm/`：模型、数据处理、训练、评测与推理；
- `recipes/`、`examples/`：可复现的运行配置与最小示例；
- `tests/`：模型兼容性、训练、DDP 和断点恢复测试；
- `docs/`：架构、使用说明与开发路线。

`minimind/` 保留通过 Git Subtree 导入的上游源码、Tokenizer 与许可证，不代表项目仍叫 MiniMind。
`experiments/`、历史脚本和报告保留原始名称、路径及元数据，以便追溯；历史能力不等同于已迁入 Mica CLI。

## 2. 权威工作区

L20 的 `/data/projects/mica` 是权威 checkout。代码修改、验证、commit 和 push 都在 L20 完成，不使用本地仓库中转。
旧路径 `/data/projects/minimind-lab` 保留为兼容链接，已有数据、权重和历史运行记录不迁移。

Mica 的训练产物格式与恢复约束见 [训练说明](mica-training.md)；历史实验保留原有 `lab_commit`、`minimind_source_commit` 等字段，不批量改写历史证据。

## 3. 分支策略

- `main`：Mica 主线，保存经过验证的模型、CLI、训练与推理代码、测试和文档。
- `feature/<name>`：训练逻辑、统一评测、奖励函数等代码能力改造。
- 同一份代码下仅超参数不同的运行，不创建新分支，用实验 ID 和 SwanLab run 区分。

建议实验 ID：

```text
<stage>-<model>-<dataset>-<variant>-<date>
```

示例：

```text
pretrain-dense64m-mini-baseline-20260821
agent-dense64m-agentrl-cispo-20260821
```

## 4. 历史实验目录契约

`experiments/00-preparation/` 是训练前准备的统一入口，包含 E00 环境基线、E01 Tokenizer/数据审计和 E02 模型/DDP/resume 探针。它们不是正式模型训练结果；正式训练从 `experiments/01-pretrain/` 的 P01 开始。

准备实验在 `config.json` 和 `run.json` 中的 `stage` 统一记录为 `00-preparation`，具体顺序由 `preparation_step` 和实验 ID 区分；`registry.csv` 保留 stage0/1/2 以便按原执行阶段筛选。

每个正式实验目录至少包含：

```text
config.json
command.sh
run.json
metrics.csv
eval.json
report.md
checkpoint-manifest.txt
swanlab-url.txt
```

`run.json` 应包含：

```json
{
  "experiment_id": "agent-dense64m-agentrl-cispo-20260821",
  "stage": "agentic_rl",
  "lab_commit": "<sha>",
  "minimind_source_commit": "393e387e9ad99f0f04c296e4c5e7353f4444629f",
  "entrypoint": "minimind/trainer/train_agent.py",
  "base_weight": "full_sft",
  "dataset": "agent_rl.jsonl",
  "hardware": "8x NVIDIA L20",
  "dtype": "bfloat16",
  "seed": 42,
  "status": "planned"
}
```

## 5. 结果存放位置

| 内容 | GitHub | SwanLab | Hugging Face | L20 |
|---|---:|---:|---:|---:|
| 配置与命令 | 是 | 可选 | 否 | 是 |
| 标量和曲线 | 摘要 | 是 | 否 | 是 |
| 评测 JSON/CSV | 是 | 可选 | 可选 | 是 |
| 完整日志 | 否 | 否 | 否 | 是 |
| 中间 checkpoint | 否 | 否 | 否 | 是 |
| 最终选中权重 | 链接 | 链接 | 是 | 是 |
| 数据集 | 仅 manifest | 否 | 外部来源 | 是 |

## 6. Checkpoint 策略

每个阶段最多长期保留：

- `last`：用于中断恢复；
- `best_target`：目标任务最优；
- `best_retention`：兼顾通用能力回归的候选；
- `release`：最终公开权重。

删除 checkpoint 前，先生成 `checkpoint-manifest.txt`，记录路径、step、指标、大小、保留或删除理由，以及是否已上传 Hugging Face。

## 7. 上游源码工作流

保留的 MiniMind 上游源码作为普通目录参与项目 commit；Mica 当前模型入口位于 `src/mica_llm/`。来源、初始 commit 和审计结论记录在 [`upstream-minimind.md`](upstream-minimind.md)。

查看源码改动：

```bash
git status --short -- minimind
git diff -- minimind
```

更新官方源码必须在独立分支显式进行：

```bash
git switch -c feature/sync-minimind-upstream
git subtree pull \
  --prefix=minimind \
  https://github.com/jingyaogong/minimind.git \
  master --squash
```

同步后必须检查 `minimind/` 的 diff、运行 smoke test，并更新实验元数据中的 `minimind_source_commit`。正式实验开始后不得自动漂移源码版本。

## 8. 发布门槛

只有满足以下条件的实验才能进入 README：

1. 配置、命令、commit 和数据版本齐全；
2. SwanLab run 可访问；
3. 目标指标和通用能力回归都完成；
4. 报告解释收益、代价、失败与适用边界；
5. 结果可以由保存的命令重新运行。
