# S04 自建 SFT-v1 8M 实验报告

## 结论

训练和共同评测均完成，但没有通过预注册晋级门。相对 S03，自建数据改善了复读异常和少量 Tool 行为，却没有改善 Chat 或严格格式，七项 macro 略低；因此当前 SFT-v1 不构建 32M，也不进入正式扩量。

## 公平对照

| 控制项 | S03 官方 | S04 自建 |
|---|---:|---:|
| shifted assistant targets | 8,000,634 | 8,001,601 |
| 日志实际 targets | 7,982,796 | 7,977,587 |
| optimizer steps | 284 | 289 |
| batch/GPU | 8 | 24 |
| 估算 targets/update | 28,171 | 27,687 |
| LR / scheduler / seed | `5e-5` / same / 42 | `5e-5` / same / 42 |

两套数据的平均回答长度相差约 3 倍，因此 sequence batch 按监督 target 密度缩放；optimizer steps 和 targets/update 的差异均小于 2%。模型、初始化权重、chat template、max length、评测集和解码参数完全一致。

## 数据与训练

- 数据：accepted SFT-v1-pilot；55,661 train rows，8,001,601 shifted assistant targets；污染审计 exact/containment/near 均为 0。
- 配置：8 x NVIDIA L20，bf16，max length 768，LR `5e-5`，batch 24/GPU，1 epoch，seed 42，无增强。
- 实际训练：289 optimizer steps，64 秒。
- 收敛：独立 validation loss `2.463244 -> 1.765359`，峰值显存 9,806 MiB；无 NaN、OOM 或梯度异常。
- best checkpoint：`s05b_best_val_768.pth`，SHA-256 `eaec972098a4ee383f6f754933cf6f77cc27cb3f6e3ef8865981153f79f8789a`。

validation 来自各自数据分布，不能用 S04 的较低 NLL 直接证明数据优于 S03；决策只使用同一套冻结行为集和七项 benchmark。

## 共同评测与决策

| 指标 | S03 | S04 | B - A |
|---|---:|---:|---:|
| Chat | 0/10 | 0/10 | 0 |
| 严格格式 | 0/6 | 0/6 | 0 |
| 复读异常（越低越好） | 10/10 | 7/10 | -3 |
| Tool E2E | 0/8 | 1/8 | +1 |
| 七项 macro | 31.7673% | 31.6290% | -0.1383pp |

S04 七项明细：C-Eval 23.1798%、CMMLU 25.6173%、ARC-Easy 31.4815%、PIQA 52.7203%、OpenBookQA 26.2000%、HellaSwag 27.8132%、SocialIQA 34.3910%。

相对 P03，S04 七项 macro 仅提升 0.1062pp；相对当前官方全量 S02-formal，低 0.2378pp。S02-formal 仍是当前较强参考（Chat 2/10、Tool E2E 5/8、macro 31.8668%），但它本身也未通过完整能力门。

## 失败归因边界

已确认：数据结构、assistant mask、来源哈希、跨 split 去重、benchmark 污染、训练数值稳定性和评测版本均通过。尚不能确认单一根因。

当前证据更支持以下组合问题：

1. 8M 总预算中严格格式约 8%、Tool 约 10%，对 64M 弱基座不足以稳定形成短答案、EOS 和工具协议行为。
2. 训练按 token 平均，长通用回答容易压过短格式答案；结构验收通过不代表语义质量和能力密度合格。
3. 样例输出仍有明显复读、答非所问和错误工具参数，说明不能只靠更低 validation loss 选数据。

## 下一轮入口

不扩大到 32M。先执行 SFT-v2 数据修复：分桶做人工语义抽检，提高短答案/EOS、严格格式和可执行 Tool 轨迹的有效监督密度；在同一 8M 预算内做 S04-R1，每 2M targets 跑一次冻结行为评测。只有 Chat、格式和 Tool 同时相对 S03 有实质提升，且七项额外回退不超过 1pp，才允许进入正式扩量；该条件最终未满足。

## 追溯

- 训练 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/h0k5g99o>
- 评测 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3jawb0uu>
- 完整日志、指标、导出模型和评测：`/data/artifacts/minimind-lab/S04-dense-sft-custom-pilot-8m-20260904/`
