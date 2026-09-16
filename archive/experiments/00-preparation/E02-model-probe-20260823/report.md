# E02-model-probe-20260823

## 状态

Stage2 completed，阶段门通过。Dense/MoE 结构、单卡 forward/backward、三组 8 卡 100-step 和 seed42 resume 均有可复现证据。

## 目标

1. 从输入 token 解释 Dense/MoE 模型结构、Tensor shape、logits 与 loss；
2. 验证 Dense/MoE 单卡 BF16 forward/backward；
3. 验证 64M Dense 在 8×L20 上连续运行 100 optimizer steps；
4. 验证 seed42 从 step50 恢复至 step100，optimizer、LR 与 SwanLab run 连续；
5. 用 seed42/43/44 观察短探针稳定性。

## 基线

- Lab 直接基线：`9961a5119e4249c48da923f3630c2e8ba362df15`；
- 执行代码：`5ccb66702c2f1f1b969c79240fd5fe5e247131cd`；
- MiniMind source：`393e387e9ad99f0f04c296e4c5e7353f4444629f`；
- 数据 revision：`312afb4f76391145c6902f765bb51691c09a12f5`。

## 口径

- 模型：hidden 768、8 层、Dense、约 64M；
- 训练：BF16、8×L20、每卡 batch 4、seq 128、100 optimizer steps；
- 数据：由固定 revision 的官方 pretrain mini 前 8192 条生成预分词 probe 数据；
- 指标：loss、grad norm、tokens/s、samples/s、GPU 利用率、峰值显存；
- 边界：该实验只验证结构和运行时，不用于模型质量或收敛结论。

## 模型结构结果

| 模型 | 总参数 | 每 token 激活参数估算 | 单卡峰值 allocated | BF16 F/B |
|---|---:|---:|---:|---|
| Dense | 63,912,192 | 63,912,192 | 655.5MiB | finite |
| MoE | 198,416,640 | 63,936,768 | 1592.3MiB | finite |

MoE 总参数是 Dense 的 3.105 倍，但 top-1 每 token 激活参数约为 1.0004 倍。单次小 batch latency 有明显 warmup 噪声，不用于速度结论。

KV cache 从 prefix `[2,8,4,96]` 增长为 `[2,9,4,96]`，continued logits 为 `[2,1,6400]`。

## 8 卡训练结果

| Seed | 初始 loss | 最终 loss | tokens/s | samples/s | 平均 GPU util | SwanLab |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 8.8794 | 6.6754 | 50,641 | 437.2 | 89.0% | `iq14wfm1nc1ca8iigdbop` |
| 43 | 8.8939 | 6.5579 | 50,647 | 435.6 | 90.3% | `is8yx09hw3341ar8qvvfa` |
| 44 | 8.9409 | 6.5112 | 50,521 | 433.7 | 90.1% | `kngkosspce6sfuzkoztzm` |

三组平均 50,603 有效 tokens/s，seed 间标准差 71 tokens/s；峰值 PyTorch allocated 1393MiB/rank，nvidia-smi 观察峰值约 1956MiB/GPU。

## Resume 证据

Seed42 分两阶段运行：

1. step1-50，保存 model、AdamW optimizer、step、world size 和 SwanLab run id；
2. 从同一 checkpoint 和 SwanLab run 恢复 step51-100。

恢复后 optimizer state 包含 90 个参数条目；step50/51 的 LR 为 `2.75e-4 → 2.679e-4`，loss 为 `6.9928 → 6.8876`。run id 全程为 `iq14wfm1nc1ca8iigdbop`。

## 产物

- `results/architecture.json`：参数量、Tensor shape、单卡 F/B、KV cache；
- `results/metrics-seed*.jsonl`：每一步 loss、LR、grad norm 和吞吐；
- `results/summary-seed*.json`：每个 seed 汇总；
- `results/summary.json`：三 seed 聚合与 resume 结论；
- `docs/source_reading/02-model-architecture.md`：源码阅读笔记；
- `checkpoint-manifest.txt`：保留的 seed42 step100 resume checkpoint。

## 异常与边界

- 本地系统 `kubectl v1.35.3` 间歇出现 Go socket `bad file descriptor`；直接 `/readyz` 始终为 HTTP 200，实验编排临时使用官方 SHA256 校验过的 `kubectl v1.34.11`，未修改 ACL；
- NCCL 未找到外部 tuner plugin，明确回退 internal tuner，8 ranks 正常完成；
- SwanLab 提示有新版本，但本次保持已验收的 0.7.11，不在实验中升级依赖；
- 预分词 seq128 探针刻意隔离数据加载开销，吞吐不能直接外推到 Stage3 seq768 正式训练。

## 成本与保留

- 驱动开始：`2026-08-23T16:20:07Z`；完成：`2026-08-23T16:22:52Z`；
- 正式三 seed 与一次 resume 总流程约 3 分钟，不含代码开发；
- 仅保留 seed42 step100 resume checkpoint，约 732MiB；
- seed43/44 不保存冗余 checkpoint；官方数据和预分词 Tensor 不进入 Git。

## 结论

Stage2 通过：能从 token 输入解释到 logits/loss，Dense/MoE 两条结构路径均可训练；8 卡 BF16 探针稳定；checkpoint、optimizer、LR、step 和 SwanLab resume 连续。下一步可进入 Stage3 Dense Pretrain mini，正式质量结论必须使用固定评测协议。
