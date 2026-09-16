# P01 Base 评测摘要

## 官方七项对比

主指标固定为任务提供 `acc_norm` 时使用 `acc_norm`，否则使用 `acc`；全部为 0-shot、Base 模型、不加 chat template。

| 任务 | P01 | 官方 minimind-3 64M | 差值（百分点） |
|---|---:|---:|---:|
| C-Eval | 23.40 | 24.89 | -1.49 |
| CMMLU | 25.07 | 25.38 | -0.31 |
| ARC-Easy | 28.03 | 28.49 | -0.46 |
| PIQA | 51.85 | 50.65 | +1.20 |
| OpenBookQA | 29.00 | 23.60 | +5.40 |
| HellaSwag | 28.88 | 28.28 | +0.60 |
| Social IQA | 33.88 | 34.19 | -0.31 |
| 七项宏平均 | 31.44 | 30.78 | +0.66 |

P01 的七项宏平均比 README 中的官方表高 0.66 个百分点，但不能据此声称已严格超过官方模型：官方没有固定 harness commit、dataset revision 和完整环境，本实验只能表述为“任务与调用口径对齐”。逐项看，C-Eval、CMMLU、ARC-Easy 和 Social IQA 仍低于官方，OpenBookQA 的较大正差值还需要在后续 P02 用同一协议复核。

## 固定续写

五个 greedy Base 续写样例未表现出稳定生成能力：三个提示立即输出 EOS，算术提示重复 `1`，代码提示重复引号。它们不是可判分 benchmark，但说明当前 mini 数据 1 epoch 的 Base checkpoint 只适合作为训练基线，不能作为可用生成模型发布。

## 训练与系统指标

- 训练完成 `39,695/39,695` micro-steps，最终单点 training loss 1.8419；最后 100/1000 个日志点均值分别为 2.0440/2.0817。
- wall-clock 45.83 分钟，8 卡合计 6.11 GPU-hours。
- 活跃采样行平均 GPU 利用率 98.74%，峰值显存 3,612 MiB/卡。
- 最大 token-slot 吞吐上界 354,794/s；该值包含 padding，不能当作有效非 padding token 吞吐。

## 可追溯性与异常

- checkpoint SHA-256：`71efd40d9fcd494bc5472891b66ea7f17167ae27ac341968bcd258a5a24b94e9`。
- harness：0.4.12，commit `6d642546f4688648fced259eb3302efd36ece5af`。
- 正式评分共执行 112,919 个 log-likelihood 请求，无 limit。
- 首次在线准备受 Hugging Face 直连超时和镜像 429 影响；最终评分使用完整本地缓存与离线模式。CMMLU 只改写下载 URL，数据 archive SHA-256 已记录。
- 原始 harness 结果见 `official_benchmarks.json`，完整执行条件见 `eval_manifest.json`。
