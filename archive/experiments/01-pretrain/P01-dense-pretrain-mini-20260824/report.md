# P01-dense-pretrain-mini-20260824

## Objective

在 8×NVIDIA L20 上按历史 Official Zero 配置完成 64M Dense 的 mini 数据预训练，建立 P02、S01 和后训练实验的首个正式 Base checkpoint。

## Direct baseline

这是本 Lab 的首个正式 Pretrain，因此没有可用于单变量结论的直接自身基线。历史 Official Zero 只作为配置参照；P01 现已成为 P02 和 S01 的直接基线。

## Configuration

- 模型：Dense 63,912,192 parameters，hidden 768，8 layers，vocab 6400；
- 硬件：8×NVIDIA L20 46GB，BF16；
- 训练：1 epoch，每卡 batch 4，累积 8，全局 sequence batch 256，seq 768，lr 5e-4；
- 数据：`pretrain_t2t_mini.jsonl`，1,270,238 行，1,241,043,656 bytes；
- 数据 SHA-256：`6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`；
- 训练 commit：`64d2908c8300985357993a66011e2b6114104fd1`；
- SwanLab：[P01-Pretrain-Mini-64M-Seq768](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nfax3tyg0j217j1cz8y0b)（由原 `MiniMind-Lab-Stage3/7iochx9kfe75qa2pt6d1u` 同步）。

## Training result

训练于 2026-08-24 03:16:40 UTC 从随机权重启动，04:02:29 UTC 正常完成，子进程退出码为 0。完成 `39,695/39,695` micro-steps，最终单点 training loss 为 1.8419；最后 100 和 1000 个日志点的均值分别为 2.0440 和 2.0817。首 100 个日志点均值为 6.3188，说明优化过程显著下降，但上游脚本没有独立 validation split，不能把这些数字称为 validation loss 或据此单独判定泛化收敛。

最终 Base checkpoint：

- 路径：`/data/artifacts/minimind-lab/P01-dense-pretrain-mini-20260824/checkpoints/p01_pretrain_768.pth`；
- 大小：137,684,380 bytes；
- SHA-256：`71efd40d9fcd494bc5472891b66ea7f17167ae27ac341968bcd258a5a24b94e9`。

## Evaluation result

使用固定的 `lm-evaluation-harness` 0.4.12（commit `6d642546f4688648fced259eb3302efd36ece5af`）执行 0-shot Base 评测，不加 chat template，batch 16，单张 L20，seed 42。主指标规则是在任务提供时选 `acc_norm`，否则选 `acc`。

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

宏平均高 0.66 个百分点只能作为“官方任务与调用口径对齐”的参考，不能声称 bitwise reproduction 或严格超过官方：官方 README 未固定 harness commit、dataset revision 和完整环境。C-Eval、CMMLU、ARC-Easy、Social IQA 仍低于官方；OpenBookQA 的 +5.40 个百分点需在 P02 同协议复核。

固定 greedy Base 续写仍明显退化：三个提示立即 EOS，算术提示重复 `1`，代码提示重复引号。因此该 checkpoint 是合格的实验基线，不是可直接发布的生成模型。原始结果、完整条件和样例见 [`../../../../experiments/01-pretrain/P01-dense-pretrain-mini-20260824/eval`](../../../../experiments/01-pretrain/P01-dense-pretrain-mini-20260824/eval)。

## Cost and system metrics

- wall-clock：45.83 分钟；
- 8 卡训练成本：6.11 GPU-hours；
- 活跃采样行平均 GPU 利用率：98.74%，中位数 99%；
- 活跃期单卡平均显存：3,505 MiB，峰值 3,612 MiB；
- 活跃期单卡平均功耗：171.47 W，峰值 201.12 W；
- token-slot 吞吐上界：354,794/s，包含 padding，不代表有效非 padding tokens/s。

## Failures and anomalies

- 训练本身未出现 OOM、NaN、NCCL 退出或重复任务；
- `lm-eval` 专用环境最初缺少支持 `TypedDict extra_items` 的新版 `typing_extensions`，升级到 4.16.0 后通过；
- CMMLU loader 硬编码 Hugging Face 主站地址，L20 直连超时；镜像元数据随后出现 HTTP 429。最终使用相同 `cmmlu_v1_0_1.zip` 的本地缓存，archive SHA-256 为 `22ecf70b28bef447ee7d8aa5fe144f56996762f901a8537b03b7693773c672a6`，正式评分全程离线；
- `requests` 报告 chardet/urllib3 版本告警，但未影响数据构建、112,919 个 log-likelihood 请求或结果保存；
- 固定续写样例退化，说明训练 loss 下降不能替代生成质量和 benchmark 评测。

## Conclusion

P01 已完成训练、checkpoint 校验、SwanLab 记录、固定续写、七项官方口径评测和系统指标闭环。它建立了可复现的 64M mini Base 基线，但尚不具备稳定自由生成能力。下一步应让 P02 仅扩大预训练数据规模，并完全复用本次评测协议，以判断数据规模是否改善中文任务、续写退化和逐项能力。
