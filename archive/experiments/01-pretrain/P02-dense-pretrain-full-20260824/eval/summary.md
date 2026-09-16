# P02 Base 评测摘要

## P01 与官方七项对比

主指标固定为任务提供 `acc_norm` 时使用 `acc_norm`，否则使用 `acc`；全部为 0-shot、Base 模型、不加 chat template。P01、P02 使用同一 harness、环境和离线数据缓存。

| 任务 | P01 | P02 | P02-P01 | 官方 minimind-3 | P02-官方 |
|---|---:|---:|---:|---:|---:|
| C-Eval | 23.40 | 23.03 | -0.37 | 24.89 | -1.86 |
| CMMLU | 25.07 | 25.13 | +0.05 | 25.38 | -0.25 |
| ARC-Easy | 28.03 | 28.91 | +0.88 | 28.49 | +0.42 |
| PIQA | 51.85 | 51.47 | -0.38 | 50.65 | +0.82 |
| OpenBookQA | 29.00 | 26.60 | -2.40 | 23.60 | +3.00 |
| HellaSwag | 28.88 | 28.40 | -0.48 | 28.28 | +0.12 |
| Social IQA | 33.88 | 32.80 | -1.07 | 34.19 | -1.39 |
| 七项宏平均 | 31.44 | 30.91 | -0.54 | 30.78 | +0.12 |

数据从 mini 扩大到 full 后，P02 只在 CMMLU 和 ARC-Easy 上高于 P01，七项宏平均下降 0.54 个百分点。P02 对官方表的宏平均高 0.12 个百分点，但官方没有固定 harness commit、dataset revision 和完整环境，不能据此声称严格超过官方。

## 固定续写

五个 greedy Base 续写样例全部在第一个生成 token 输出 EOS，continuation 均为空。相比 P01 的三个立即 EOS、两个重复输出，P02 的自由续写没有因数据规模扩大而改善。

## 训练与系统指标

- 训练完成 `264,651/264,651` micro-steps，最终单点 training loss 1.8701；最后 100/1000 个日志点均值为 1.7157/1.7213。
- wall-clock 285.78 分钟，8 卡合计 38.10 GPU-hours，均为 P01 的 6.24 倍。
- 活跃采样行平均 GPU 利用率 98.70%，峰值显存 3,750 MiB/卡。
- token-slot 吞吐上界 379,311/s，比 P01 高 6.91%；该值包含 padding，不能当作有效非 padding token 吞吐。

## 结论

在当前 64M、1 epoch、seq 768 配置下，full 数据让训练末段平均 loss 更低，但没有改善七项宏平均和固定续写。training loss 不能替代下游评测；下一阶段应固定 SFT 数据与超参数，分别从 P01 和 P02 初始化，才能判断哪个 Base checkpoint 更适合作为后训练起点。

## 可追溯性与异常

- checkpoint SHA-256：`7065a461ca8c29eedb38264f329a0daa99a2f429277720925966b840572d28f4`。
- harness：0.4.12，commit `6d642546f4688648fced259eb3302efd36ece5af`。
- 正式评分共执行 112,919 个 log-likelihood 请求，无 limit。
- 首次训练尝试因默认 rootfs Hugging Face Arrow cache 填满 10GB 而在首个 step 前失败；迁移 cache 到 CPFS 后 attempt 2 正常完成。
- 原始 harness 结果见 `official_benchmarks.json`，完整执行条件见 `eval_manifest.json`。
