# 实验目录规范

Stage0–2 都是正式训练前的准备验证，统一放在 `00-preparation/`；P01 开始才进入正式训练目录。准备实验仍用 E00/E01/E02 保留执行顺序和职责边界。

```text
experiments/
├── 00-preparation/   # E00 环境、E01 数据、E02 模型探针
├── 01-pretrain/
├── 02-sft/
├── 03-lora/         # Phase3 Full FT vs LoRA
├── 03-dpo/          # 历史空占位
├── 04-dpo/          # Phase4 D01/D02/D03
├── 04-grpo-cispo/
└── 05-agentic-rl/
```

每个实验在对应目录下建立独立目录，名称使用稳定实验 ID，不只使用日期。

```text
experiments/<stage>/<experiment-id>/
├── config.json
├── command.sh
├── run.json
├── metrics.csv
├── eval.json
├── report.md
├── checkpoint-manifest.txt
└── swanlab-url.txt
```

`registry.csv` 是全局索引。实验开始时登记为 `planned/running`，完成统一评测和报告后才能改为 `completed`。

`status` 取值：`planned`、`running`、`awaiting-report`（训练与评测已完成，但实验目录的 `report.md` / `run.json` / `metrics.csv` 尚未回填）、`completed`、`invalidated`。

`swanlab_url` 不留空：无云端 run 写 `n/a-no-cloud-run`，不训练模型的准备实验写 `n/a-no-training`（对应计划 16.5）。`lab_commit` 是定位证据的唯一可靠指针：当某个实验的资产尚未合入 `main`（例如 P02 在 `stage5/p02-dense-pretrain-full`、P03 在 `data/v1`），`report_path` 指的是该 ref 上的路径，用 `git show <ref>:<path>` 回读，不能假设 `main` 已收录。

使用模板创建实验，避免遗漏追溯文件：

```bash
scripts/launch/init_experiment.py 01-pretrain P01-dense-mini
scripts/sync/validate_experiment.py experiments/01-pretrain/P01-dense-mini
```

模板的 `command.sh` 默认拒绝执行。只有填写并 review 最终命令后才能移除占位退出逻辑。
正式训练必须通过 `scripts/launch/run_guarded.py`，并在 `run.json` 中记录直接基线、源码版本和 SwanLab run id。
权重和完整日志不进入 Git，只在 manifest 中保存 SHA-256、大小、路径和用途。
