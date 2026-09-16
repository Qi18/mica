# S01 实验报告

## 结论

S01 完成了官方 mini SFT 数据的工程 smoke，但没有通过能力门禁。该 run 只证明训练、验证、checkpoint、HF 导出、固定行为评测和 SwanLab 记录链路可用。

## 训练

- 初始权重：P03（63,912,192 parameters）。
- 数据：3,000 行，训练 2,744 行，validation 256 行，训练 assistant targets 1,156,290。
- 配置：8×L20，BF16，batch 8/GPU，1 epoch，lr `1e-5`，无增强。
- 结果：42/42 steps，validation loss `2.951357 -> 2.681668`，最后训练窗口 loss `2.667477`。

## 评测

同一套固定题分别评测 P03 Base 和 S01 best checkpoint。P03 为 Chat 0/10、格式 0/6、重复异常 9/10、Tool E2E 0/8；S01 为 Chat 0/10、格式 1/6、重复异常 10/10、Tool E2E 0/8。

因此 loss 下降不能等价为指令能力提升。下一轮必须用足够的固定 target 预算，并同时依据 validation NLL、Chat/格式、Tool 和重复异常选择配置。

## 产物

- best checkpoint：`/data/artifacts/minimind-lab/S01-dense-sft-official-mini-smoke-20260903/checkpoints/s01_best_val_768.pth`
- 行为评测：`/data/artifacts/minimind-lab/S01-dense-sft-official-mini-smoke-20260903/eval/behavior/`
- P03 对照：`/data/artifacts/minimind-lab/S01-dense-sft-official-mini-smoke-20260903/eval/p03-baseline-behavior/`
- SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/e4php2e6>
