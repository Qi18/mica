# P02 数据与统一评测补全

本页补齐 `P02-dense-pretrain-full-20260824` 在最初实验中未统计的数据侧指标，并使用 P03 的固定验证集和污染协议重新建立可比较口径。完整机器可读结果见 [data_audit_full.json](data_audit_full.json)，统一验证结果见 [validation_shared_p03.json](validation_shared_p03.json)，审计程序见 [audit_p02_data.py](../audit_p02_data.py)。

## 结论

P02 确实比 P03 看过更多有效训练 target：P02 为 2.021B，P03 为 1.280B，前者是后者的 1.58 倍。但 P02 把每一行独立 padding 到 768，产生 6.504B 个固定计算槽位，有效 target 利用率只有 31.07%；P03 经过 tokenizer 对齐切块后为 72.04%。所以 P02 的 optimizer update 是 P03 的 3.66 倍，并不代表有效 token 也多 3.66 倍。

P02 的基础格式质量合格，但不能通过 P03 的正式数据验收门槛：全量发现 1,433 个 training-visible exact duplicate，以及 115 个 benchmark query-in-document containment 命中；P03 两项均为 0。P02 与后来建立的 11,525 行共享验证集没有 exact tokenized-chunk 重叠，但 P02 训练时没有独立 validation split。

## 审计口径

- 输入文件固定为 revision `312afb4f76391145c6902f765bb51691c09a12f5` 的 `pretrain_t2t.jsonl`；全量行数、字节数与 SHA-256 均重新读取校验。
- token 口径完全复现 MiniMind `PretrainDataset`：文本不加 special token，截断到 766，再拼接 BOS/EOS；shift 后的 loss target 数为 `min(raw_tokens, 766) + 1`。
- exact duplicate 按训练实际可见的 `[BOS] + truncated_text_tokens + [EOS]` SHA-256 判断；不同 raw 文本如果前 766 tokens 相同，也会被视为训练重复。
- benchmark exact/containment 使用与 P03 相同的 7 项固定 revision、29,322 个唯一 query 和 `NFKC + lowercase + 仅保留 Unicode alnum` 归一化。exact 与 containment 扫描全部 8,468,827 行。
- near duplicate 不是全量声明；使用确定性 bottom-k 抽取 20,480 行，样本量等于 P03 的 40 个 train shard × 每 shard 512 行。
- 共享 validation 是在 P02 训练之后建立的，只能用于回溯比较，不能补成 P02 当时不存在的训练期验证集。

## P02 全量数据指标

| 维度 | 结果 |
|---|---:|
| 文件大小 | 8,275,074,893 bytes |
| 物理行 / 有效行 | 8,468,827 / 8,468,827 |
| 空行 / 空文本 | 0 / 0 |
| 非法 UTF-8 / JSON / schema | 0 / 0 / 0 |
| raw text tokens | 2,200,383,770 |
| seq768 可见 text tokens | 2,012,297,520 |
| shifted loss targets | 2,020,766,347 |
| non-padding input tokens | 2,029,235,174 |
| padded compute slots | 6,504,059,136 |
| target 利用率 | 31.07% |
| non-padding input 利用率 | 31.20% |
| 被截断的行 | 331,996（3.92%） |
| 被丢弃的 raw tail tokens | 188,086,250（raw tokens 的 8.55%） |
| training-visible exact duplicates | 1,433（0.0169%） |
| 唯一 training-visible chunks | 8,467,394 |
| 训练可见词表覆盖 | 6,351 / 6,400（99.23%） |

长度分布基于未截断 raw token 数：p50=209、p90=418、p95=571、p99=1,434、max=17,792。实际 seq768 训练中 p99 和 max 都被截为 766。这解释了为什么文件后半段的扫描速度显著下降，也说明单看行数无法表示 P02 的真实数据量。

## 文本构成代理

原文件没有语言或来源标签，只能给出可复核的 Unicode script 代理，不能把它解释为语言分类：

- 93.35% 的行至少含一个 CJK 字符；
- 45.46% 的行至少含一个 ASCII 拉丁字母；
- 38.84% 的行同时含 CJK 与 ASCII 拉丁字母；
- 归一化 alnum 字符中，CJK 占 80.00%，ASCII 拉丁字母占 17.41%，数字占 2.33%，其他 Unicode alnum 占 0.26%。

这些比例互有重叠，只能说明文本形态，不能代替中文/英文占比，也不能恢复数据源权重。

## 重复、验证集与 benchmark 污染

| 检查 | P02 | P03 train | 判定 |
|---|---:|---:|---|
| training-visible exact duplicate | 1,433 | 0 | P02 不满足 P03 的零重复门槛 |
| 与共享 validation exact overlap | 0 | 0 | 通过 exact chunk 检查 |
| benchmark exact equality | 0 | 0 | 通过 |
| benchmark containment | 115 | 0 | P02 不满足 P03 的零 containment 门槛 |
| benchmark near duplicate | 0 / 20,480 抽样 | 0 / 20,480 train 抽样 | 抽样内未发现，不能外推为全量 0 |

115 个 containment 命中按任务分布为：ARC-Easy 72、CMMLU 8、OpenBookQA 21、PIQA 14；C-Eval、HellaSwag、Social IQA 为 0。样例中既有较明确的题干级重合，例如 CMMLU 的 Windows 剪贴板、机器语言题，也有 `Which statement is true?`、`Inherited characteristics` 这类通用短语碰撞。因此应表述为“协议检测到 115 个 containment pattern-document pair”，不能把 115 个全部断言为确定的数据泄漏；但按 P03 的零容忍验收标准，P02 仍然失败。

共享 validation 有 11,525 行、6.4M targets，自身 exact duplicate 为 0；P02 与它的训练可见 exact overlap 为 0。回溯评测结果：

| Checkpoint | validation NLL | PPL |
|---|---:|---:|
| P01 | 3.56414 | 35.3092 |
| P02 | 3.19096 | 24.3117 |
| P03 | 2.60432 | 13.5221 |

因此 P02 相对 P01 在统一 validation NLL 上有提升，但仍明显弱于 P03；这与 P02 七项宏平均低于 P01、自由生成全部立即 EOS 并不矛盾，因为三类评测测量的是不同能力。

## P02 与 P03 数据对比

| 维度 | P02 | P03 train | P02 / P03 |
|---|---:|---:|---:|
| 行数 | 8,468,827 | 2,313,483 | 3.66× |
| shifted loss targets | 2.021B | 1.280B | 1.58× |
| padded compute slots | 6.504B | 1.777B | 3.66× |
| 平均 targets / row | 238.61 | 553.28 | 0.43× |
| target 利用率 | 31.07% | 72.04% | 0.43× |
| optimizer updates | 33,082 | 9,038 | 3.66× |
| 估算 targets / update | 61.1K | 141.6K | 0.43× |
| exact duplicates | 1,433 | 0 | — |
| benchmark containment | 115 | 0 | — |
| 训练期独立 validation | 无 | 11,525 行 / 6.4M targets | — |
| 行级来源与处理 provenance | 无 | 完整 sidecar | — |

两次训练的全局 sequence batch 都是 256，但 sequence 的有效长度完全不同。P02 的 update 横轴每步平均只承载约 61.1K target，P03 约 141.6K；这些是用全数据 target 总数除以 optimizer updates 得到的近似值，不包含 P02 DistributedSampler 为整除 8 ranks 重复的 5 行。因此不能直接按 optimizer update 对齐两条 loss 曲线，也不能把 P02 多 3.66 倍 updates 解释为多 3.66 倍有效训练量。

P03 的来源配比有明确记录：ChineseWebText2 45%、FineWeb-Edu 30%、Wikipedia zh/en 各 5%、FineMath 10%、Stack v3 permissive code 5%。P02 的固定文件只有 `text` 字段。[该 revision 的数据卡](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/README.md)只说明包含通用文本、对话整理、蒸馏补充，以及匠数、Magpie-Align 等公开来源，没有逐行来源、精确权重或逐行 license。仓库数据卡同时标记 Apache-2.0 与 CC-BY-NC-2.0，因此不能据此宣称 P02 是逐行可追溯或完全可商用的数据集。

## seq380 反事实分析

官方对 full 文件建议 `max_seq_len≈380`。在不重新训练、只用同一 tokenizer 重算的反事实口径下：

| 指标 | 实际 seq768 | 反事实 seq380 |
|---|---:|---:|
| loss targets | 2.021B | 1.822B |
| padded compute slots | 6.504B | 3.218B |
| target 利用率 | 31.07% | 56.61% |
| 截断行比例 | 3.92% | 13.52% |

seq380 会减少 50.52% 的固定计算槽位，但只减少 9.85% 的 loss targets，代价是截断行比例升高。这个结果支持官方约 380 的吞吐建议，也解释了 P02 在 seq768 下为何 optimizer updates 多、wall time 长，却没有按比例获得更多有效 target。

## 证据与边界

- 原始数据 SHA-256：`31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d`；
- 完整审计 JSON SHA-256：`cfe509c33f3e8986d2f4373e68ce68c41d673c5077eb27745e27403426baefa7`；
- 审计程序 SHA-256：`88526d82c814e885d2471760a88b8a9ddbd0b3cd49ecbccb2d55a5cab7ae460f`；
- 固定 benchmark pattern cache SHA-256：`d5c43ce673a1d20418921196a2b4e3d84e1a39ae487b16876fbb97c24040c74b`；
- 共享 validation SHA-256：`0a7e8503f01bc185740b3e26e26326c43ca00309452ae1eec081d2ac2d9105cb`；
- P02 validation 结果 JSON SHA-256：`3352ac7e0b960c44215c90d669c42d48630aca006b2084152255b1ca2c7d23e9`；
- tokenizer.json SHA-256：`71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668`。

本次补全没有重新训练 P02，也没有改变 checkpoint。source mix、逐行 lineage 和训练期 validation 无法从历史单文件中恢复；near duplicate 仍是确定性抽样结论。其余 rows、质量、token、截断、exact duplicate、共享 validation exact overlap 与 benchmark exact/containment 均为全量扫描结果。
