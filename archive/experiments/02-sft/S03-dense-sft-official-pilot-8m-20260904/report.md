# S03 官方 SFT 8M 实验报告

## 结论

训练和共同评测均完成，但没有通过 SFT 能力门禁。S03 是本轮官方数据对照，不是可接受的最终 Chat checkpoint。

## 数据与训练

- 初始权重：P03 64M，SHA-256 `0cfb7fc8fd9b3111f30b5528a1c8aacf8d6f633c8cde13c707c7cb44c83fd4fd`。
- 数据：accepted official-pilot；18,218 train rows，8,000,634 shifted assistant targets；污染审计 exact/containment/near 均为 0。
- 配置：8 x NVIDIA L20，bf16，max length 768，LR `5e-5`，batch 8/GPU，1 epoch，seed 42，无增强。
- 实际训练：284 optimizer steps，日志覆盖 7,982,796 assistant targets，41 秒。
- 收敛：独立 validation loss `2.749087 -> 2.179558`，峰值显存 4,044 MiB；无 NaN、OOM 或梯度异常。
- best checkpoint：`s05a_best_val_768.pth`，SHA-256 `61d5231697746d1b987cc81c6814cfdddd2371cab8bc7fc3ce7deb596b72ba65`。

## 共同评测

| 指标 | S03 |
|---|---:|
| Chat | 0/10 |
| 严格格式 | 0/6 |
| 复读异常 | 10/10 |
| Tool E2E | 0/8 |
| 七项 macro | 31.7673% |

七项明细：C-Eval 23.4027%、CMMLU 25.4878%、ARC-Easy 31.6919%、PIQA 52.2307%、OpenBookQA 27.0000%、HellaSwag 27.9626%、SocialIQA 34.5957%。

## 追溯

- 训练 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/7duhsb00>
- 评测 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/50vjh86o>
- 完整日志、指标、导出模型和评测：`/data/artifacts/minimind-lab/S03-dense-sft-official-pilot-8m-20260904/`
