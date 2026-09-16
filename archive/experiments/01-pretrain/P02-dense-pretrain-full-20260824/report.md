# P02-dense-pretrain-full-20260824

## Objective

在 8×NVIDIA L20 上从随机权重训练同一个 64M Dense 模型，只把 P01 的 `pretrain_t2t_mini` 替换为完整 `pretrain_t2t`，测量数据规模对 Base 能力、生成退化和单位成本的影响。

## Direct baseline

直接基线是 `P01-dense-pretrain-mini-20260824`。P01 七项 Base 宏平均为 31.44，训练耗时 45.83 分钟，固定续写仍存在立即 EOS 和重复字符。

## Configuration

- 模型、随机初始化、seed、dtype、batch、梯度累积、seq 768、学习率、epoch 和 8×L20 均与 P01 相同；
- 唯一主要变量：数据从 1,270,238 行 mini 扩大到 8,468,827 行 full，行数为 6.67 倍；
- 虽然源码文档对 full 数据建议 `max_seq_len≈380`，本次保留 P01 的 768，以避免同时改变数据规模和截断长度；
- 预计 264,651 个 micro-steps、约 33,082 次 optimizer update；按 P01 线性外推约 305.5 分钟。

## Data verification

- revision：`312afb4f76391145c6902f765bb51691c09a12f5`；
- 文件大小：8,275,074,893 bytes；
- 行数：8,468,827；
- SHA-256：`31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d`，与官方 LFS oid 一致；
- 首尾记录字段均为 `text`。

## Retrospective full-data audit

2026-09-02 按 MiniMind 实际 seq768 tokenization 对全部 8,468,827 行补审：得到 2,020,766,347 个 shifted loss target、6,504,059,136 个 padded compute slots，target 利用率 31.07%；331,996 行发生截断，丢弃 188,086,250 个 raw tail tokens。全量还发现 1,433 个 training-visible exact duplicate；与后来建立的 11,525 行共享 validation exact overlap 为 0。

固定七项 benchmark 的 full exact equality 为 0，但 query-in-document containment 为 115；20,480 行确定性 near-duplicate 样本中为 0。部分 containment 是明确题干重合，部分是通用短语碰撞，不能全部断言为确定泄漏，但 P02 不满足 P03 的零重复、零 containment 数据门槛。P02 单文件没有逐行来源、精确 source mix 或 license provenance；完整口径、P02/P03 对比和证据 SHA 见 [eval/data_audit_summary.md](eval/data_audit_summary.md)。

同一 P03 共享 validation（6.4M targets）上，P01/P02/P03 的 NLL 分别为 3.56414/3.19096/2.60432，PPL 分别为 35.3092/24.3117/13.5221。P02 相对 P01 的 token-level 泛化有提升，但仍弱于 P03；该 validation 在 P02 训练后建立，不是 P02 当时的训练期验证集。

## Training result

attempt 2 于 2026-08-24 12:04:45 UTC 从随机权重启动，训练 commit `10e2de6deb1d3dc827e896149b1d3e8ba1c3a85c`。step 10→100→180 的 loss 为 8.5419→7.3531→6.9985；8 张 L20 均约 99% 利用率、单卡约 3,476 MiB，早期 ETA 约 302 分钟。该窗口只证明训练健康，不能提前判定收敛。

## Evaluation result

训练结束后完全复用 P01 的固定 Base 续写与七项 0-shot 评测协议：`lm-evaluation-harness` 0.4.12、commit `6d642546f4688648fced259eb3302efd36ece5af`、0-shot、batch 16、单张 L20、seed 42，不使用 chat template。主指标固定为任务提供 `acc_norm` 时使用 `acc_norm`，否则使用 `acc`。

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

P02 仅在 CMMLU 和 ARC-Easy 上高于 P01，七项宏平均反而下降 0.54 个百分点。P02 相对官方表的宏平均高 0.12 个百分点，但官方没有固定 harness commit、dataset revision 和完整环境，因此只能表述为任务与调用口径对齐，不能声称严格超过官方。

五个固定 greedy Base 续写全部在第一个生成 token 输出 EOS，continuation 均为空。相比 P01 的三个立即 EOS、两个重复输出，P02 的自由续写没有因数据扩大而改善。

## Cost and system metrics

- 完成 `264,651/264,651` micro-steps，最终单点 training loss 1.8701；最后 100/1000 个日志点均值为 1.7157/1.7213；
- wall-clock 285.78 分钟，8 卡合计 38.10 GPU-hours，分别为 P01 的 6.24 倍；
- 活跃采样行平均 GPU 利用率 98.70%，中位数 99%；单卡平均显存 3,490 MiB，峰值 3,750 MiB；
- 单卡平均功耗 175.32 W，峰值 199.34 W；
- token-slot 吞吐上界 379,311/s，比 P01 高 6.91%，但包含 padding，不代表有效非 padding tokens/s。

## Failures and anomalies

首次正式启动在进入第一个训练 step 前失败：8 个 rank 使用默认 `/root/.cache/huggingface/datasets` 并发生成 full JSON 的 Arrow cache，rank 7 报 `No space left on device`。GPU 尚未进入训练，未生成 checkpoint。

修复为强制 cache 位于 CPFS `/data/cache/huggingface/datasets`，并在 torchrun 前通过单进程预构建并验证 8,468,827 行共享 Arrow cache。只删除了本次失败产生的 217,068,495-byte 临时 cache，历史 cache 未动。

attempt 2 正常完成，无 OOM、NaN、NCCL 退出或重复训练任务。正式评分执行 112,919 个 log-likelihood 请求，无 limit；`requests` 版本告警未影响结果保存。

训练成功 run 已同步到统一 SwanLab 项目：[P02-Pretrain-Full-64M-Seq768](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3i1muwq039fpfv89fq4ru)。首次失败 run 也作为异常证据同步为：[P02-Pretrain-Full-64M-Seq768-Failed-Cache](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/q2lnh08i1dkvwtgey5d7z)。原始 run 分别位于 `MiniMind-Lab-Stage5/bs7n0qfcxykk13fammxis` 和 `MiniMind-Lab-Stage5/ij3ujuvmbni6fouaz0r1f`。

## Conclusion

P02 已完成训练、checkpoint 校验、SwanLab 统一归档、固定续写、七项评测和系统指标闭环。虽然 full 数据将训练末段平均 loss 从 P01 的 2.0440 降到 1.7157，数据规模扩大 6.67 倍也带来约 6.91% 的 token-slot 吞吐提升，但七项宏平均下降 0.54 个百分点，五个固定续写全部立即 EOS。因此在当前 64M、1 epoch、seq 768 配置下，单独扩大预训练数据没有改善综合 Base 能力，也不能用更低 training loss 推断泛化更好。

补全数据审计后还可进一步确认：P02 虽有 2.021B 有效 target，但 padding 后仅 31.07% 槽位形成 loss target，且存在 1,433 个 exact duplicate 和 115 个 benchmark containment 命中。P02 应保留为历史基线，不应视为满足当前 P03 数据验收规范的主线数据实验。

下一阶段不应只凭 loss 自动选择 P02。SFT 应固定同一数据与超参数，分别从 P01 和 P02 初始化做一组 Base-checkpoint 对照；若资源只允许保留一条主线，P01 是当前七项宏平均更高的基线。
