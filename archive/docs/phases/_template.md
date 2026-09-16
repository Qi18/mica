# Phase <N> 阶段报告：<阶段名称>

- 阶段范围：`experiment_plan.md` 第 <X> 节
- 报告状态：draft / accepted
- 收口日期：YYYY-MM-DD
- Lab commit：
- MiniMind commit：

## 1. 阶段目标与研究问题

本阶段回答的问题、对应的能力假设，以及本阶段明确不回答的问题。

## 2. 实验清单

| experiment_id | status | 输入 checkpoint | 数据 | 实验报告 |
|---|---|---|---|---|
|  |  |  |  |  |

### 2.1 SwanLab run

URL 从 `experiments/<stage>/<id>/swanlab-url.txt` 与 `registry.csv` 引用，逐个 run 列出角色（probe / formal / resume attempt / eval / seed）。无云端 run 时写“无云端 run”并给出本地 `swanlog/` logdir。

| experiment_id | run 角色 | project | run name | URL |
|---|---|---|---|---|
|  |  |  |  |  |

## 3. 关键配置与数据 fingerprint

硬件、dtype、seq、global sequence batch、accumulation、LR/scheduler、seed、有效 target tokens；数据 fingerprint 与 `_SUCCESS` 位置。

## 4. 结果横向对比

| 指标 | 基线 | 对照 A | 对照 B | 差值 |
|---|---:|---:|---:|---:|
|  |  |  |  |  |

目标指标、通用能力回归（七项 macro / validation NLL）、系统与成本指标（wall、GPU-hours、峰值显存、吞吐）分别列出。每张表注明数字来自哪个 SwanLab run 或哪份 eval manifest。

## 5. 门控判定

| 预注册门槛 | 实测 | pass/fail |
|---|---|---|
|  |  |  |

## 6. 失败与作废实验

现象、根因、证据位置、是否登记为 invalidated。

## 7. 结论边界

不可归因项、seed 数量、污染扫描范围、长度偏差与 reward hacking 抽查结论；哪些数字只能作为参考信号。

## 8. 下一阶段前置条件

进入下一 Phase 必须先满足的条件，以及本阶段留下的未解决问题。

## 9. 证据索引

- Git ref / commit：
- artifacts：`/data/artifacts/minimind-lab/...`
- eval manifest：
- checkpoint manifest / SHA-256：
- SwanLab：

## 修订记录

| 日期 | 修改内容 | 原因 |
|---|---|---|
|  |  |  |
