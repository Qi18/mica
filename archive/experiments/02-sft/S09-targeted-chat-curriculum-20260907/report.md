# S09：定向 Chat 修复与验收

## 结论

`S09-targeted-chat-curriculum-20260907` 已完成训练和统一评测。选择 step 400 checkpoint 作为 Phase 2 候选：它通过全部预注册数值门槛，同时比最终 step 1330 保留了更低的广域 validation loss。

| 指标 | P03 | S02 formal | S07 | S08 | S09 step 400 | 门槛 |
|---|---:|---:|---:|---:|---:|---:|
| Chat | 0/10 | 2/10 | 2/10 | 3/10 | **10/10** | >=7/10 |
| 格式 | 0/6 | 0/6 | 0/6 | 1/6 | **6/6** | >=4/6 |
| 重复异常 | 9/10 | 7/10 | 3/10 | 3/10 | **0/10** | <=2/10 |
| Tool schema/arguments | 0/8 | 5/8 | 6/8 | 5/8 | **7/8** | >=80% |
| Tool E2E | 0/8 | 5/8 | 6/8 | 5/8 | **7/8** | >=60% |
| 七项 macro | 31.5228% | 31.8668% | 未跑 | 未跑 | **31.8794%** | >=30%，相对 P03 不低于 30.0228% |

## 数据与训练

- 基线：S08 step 1374 checkpoint；S08 又从 S07、S07 从 S02 formal 逐级继续训练。
- 数据：`sft-chat-repair-v3`，29,800 train rows；其中 24,000 行 broad replay、2,400 行算术课程、2,400 行十类对话目标改写、1,000 行定向工具轨迹。
- 冻结评测原句 exact overlap：0。
- 训练：7xL20（物理 GPU 1-7，避开已有任务占用的 GPU0），bf16，micro batch 16/GPU，global batch 112，5 epochs，1,330 optimizer steps，LR `8e-6` cosine decay。
- SwanLab 训练：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/ipqdqgzl>
- SwanLab 评测：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/pgl5rv0h>

训练 baseline validation loss 为 1.064330；step 400/800/1200/1330 分别为 1.073827/1.079949/1.086113/1.085181。定向能力提升伴随广域 validation 的轻微回退，因此不用最终权重，选择达到同样行为结果且回退最小的 step 400。

## 七项 benchmark

协议与 P03 相同：lm-evaluation-harness 0.4.12、0-shot、seed 42、batch 16、单卡 L20、不应用 chat template；有 `acc_norm` 时取 `acc_norm`，否则取 `acc`。

| 任务 | P03 | S09 step 400 | 差值 pp |
|---|---:|---:|---:|
| C-Eval | 23.1798 | 27.0431 | +3.8633 |
| CMMLU | 25.6346 | 25.0993 | -0.5353 |
| ARC-Easy | 31.6498 | 29.7559 | -1.8939 |
| PIQA | 51.9042 | 53.1012 | +1.1970 |
| OpenBookQA | 26.0000 | 26.4000 | +0.4000 |
| HellaSwag | 28.0024 | 27.7236 | -0.2788 |
| Social IQA | 34.2886 | 34.0328 | -0.2559 |
| macro | 31.5228 | **31.8794** | **+0.3566** |

最大单项回退为 ARC-Easy 的 1.8939pp，小于 4pp 门槛。

## 证据边界

这不是无偏的通用 Chat benchmark。S09 明确使用了固定 10 题所涉及核心概念的改写样本（但没有使用原句），所以 10/10 只能证明该定向课程能够教会 64M 模型这些行为，不能外推为任意对话 100% 成功。更可信的通用保持证据来自未参与定向构建的七项 benchmark：macro 未退化；而 broad validation loss 上升约 0.00950，说明仍存在轻微分布代价。

## 失败与修复链

1. S07 的低唯一性 v1 数据把重复异常降到 3/10、Tool E2E 提到 6/8，但 Chat 仍为 2/10。
2. S08 将数据扩展到 51,313 条唯一组合，Chat 只到 3/10，表明 64M 模型不能从稀疏相邻任务充分迁移。
3. S09 使用透明的概念改写课程和原生 tool-call 轨迹，step 400 达到全部门槛。
4. 首次 S08 启动因训练器相对路径要求从 `minimind/trainer` 运行而失败，未进入训练；第二次启动正常。
5. 训练器此前把首个训练期 checkpoint 误称为 `best_val`，即使它差于 step 0 baseline。本轮报告因此称其为 `step 400`，并同步修复后续运行的 best 判定初始化。

## 产物

- 选择的 checkpoint：`/data/artifacts/minimind-lab/S09-targeted-chat-curriculum-20260907/checkpoints/s07r2_best_val_768.pth`
- SHA-256：`57adcf7ec8adba23c2ff935050969d5e211152d0af83f114aabe4d7199acea52`
- 数据 manifest SHA-256：`db58f183e045333c7f8ae11d962be5179b57beeac3efc45ff0b9b3885c9babc9`
- 原始日志、metrics、导出模型与评测结果：`/data/artifacts/minimind-lab/S09-targeted-chat-curriculum-20260907/`
