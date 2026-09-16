# Pretrain v1 数据构建与验收协议

更新时间：2026-09-02
当前状态：`accepted`

## 1. 正式入口

- 配置：`configs/data/pretrain/pretrain_v1.yaml`、`pretrain_shards_v1.yaml`、`contamination_eval_v2.yaml`
- builder：`scripts/data/pretrain/build_pretrain_v1.py`
- remixer：`scripts/data/pretrain/remix_pretrain_v1.py`
- 独立 auditor：`scripts/data/pretrain/audit_pretrain_v1.py`
- launcher：`experiments/00-preparation/D01-training-data-v1-20260828/launch_pretrain_v1_1b28.sh`
- 修复 launcher：`experiments/00-preparation/D01-training-data-v1-20260828/launch_pretrain_v1_remix.sh`
- 详细报告：`experiments/00-preparation/D01-training-data-v1-20260828/report.md`

历史 3.2M smoke、320M pilot 和首次失败的 1.28B final 只用于追溯。正式可训练版本仅指 `final-remix-v1/` 且 `_SUCCESS.status=accepted` 的数据树。

## 2. Token 口径

MiniMind `PretrainDataset` 使用 `add_special_tokens=False`，文本最多保留 766 tokens，再添加 BOS/EOS 并右侧 padding 到 768。

```text
loss_target_tokens = len(truncated_raw_tokens) + 1 EOS
nonpad_input_tokens = loss_target_tokens + 1 BOS
padded_compute_tokens = 768 × valid_rows
```

正式预算使用 `loss_target_tokens`，不使用 `rows × sequence_length`，也不把 BOS 计为 shifted loss target。

## 3. 来源与精确配额

固定 candidate pool 由 54 个 pinned objects 构成。长文按 Tokenizer 对齐切块，保留文档尾部；确定性 hash 只在有界 candidate window 内选择，因此不能表述成完整上游语料的全局均匀抽样。

| 来源 | train loss-target tokens | validation | 权重 |
|---|---:|---:|---:|
| ChineseWebText2.0 | 576,000,000 | 2,880,000 | 45% |
| FineWeb-Edu | 384,000,000 | 1,920,000 | 30% |
| Wikipedia zh | 64,000,000 | 320,000 | 5% |
| Wikipedia en | 64,000,000 | 320,000 | 5% |
| FineMath 4+ | 128,000,000 | 640,000 | 10% |
| The Stack v3.1 permissive subset | 64,000,000 | 320,000 | 5% |
| 合计 | **1,280,000,000** | **6,400,000** | **100%** |

Stack v3 只接受 permissive、非 vendor、许可证非空且属于固定语言集合的对象；`no_license` 不用于补预算。

## 4. 正式流水线

```text
pinned sources + licenses
  → materialize fixed objects
  → normalize / whitespace gate
  → tokenizer-aligned chunking
  → deterministic candidate selection
  → exact/containment contamination filtering
  → exact quota replacement and remix
  → 40 train + 1 validation shards
  → independent recount / loader dry-run / full audit
  → atomic audit report
  → atomic _SUCCESS.status=accepted
```

首次 audit 发现 whitespace-only、exact benchmark overlap 和短 query containment 假阳性。修复版将 exact 下限设为 1 个规范化字符、containment 下限设为 20，并在最终选择前排除 258 条候选：226 containment、5 exact、27 empty/whitespace；替换后各来源预算仍精确命中。

## 5. 已接受结果

| 指标 | train | validation | 合计 |
|---|---:|---:|---:|
| rows | 2,313,483 | 11,525 | 2,325,008 |
| loss-target tokens | 1,280,000,000 | 6,400,000 | 1,286,400,000 |
| nonpad input tokens | 1,282,313,483 | 6,411,525 | 1,288,725,008 |
| padded compute tokens | 1,776,754,944 | 8,851,200 | 1,785,606,144 |

- 输出 shard：40 train + 1 validation；
- 数据 shards：3,804,930,498 bytes；provenance sidecars：1,333,318,467 bytes；
- 完整 accepted tree：128 files，5,138,671,539 bytes；
- invalid JSON/schema/UTF-8、blank、empty：全部 0；
- train exact duplicate、validation exact duplicate、跨 split exact overlap：全部 0；
- 独立 auditor：14/14 checks 通过，`passed=true`。

## 6. 污染检查边界

固定七项 benchmark 共得到 29,322 个唯一规范化 query。exact 和 query-in-document containment 全量扫描全部 2,325,008 个有效文档，结果均为 0。

near duplicate 不是全量扫描：每个物理 shard 使用 deterministic SHA-256 bottom-k 抽取 512 条，共审计 20,992 条 eligible documents，结果为 0。报告中必须始终写成“分层确定性样本未发现 near overlap”，不能写成“全量 near overlap 为 0”。

## 7. 接受证据与使用方式

- accepted data：`/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/`
- audit：`experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_remix_audit.json`
- verify：`experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_remix_verification.json`
- marker：`final-remix-v1/_SUCCESS`

训练入口必须使用 quoted glob：

```bash
--data_path '/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/train-*.jsonl'
```

本次目录整理改变了正式脚本路径，历史 marker 仍绑定当时的脚本/config 哈希，不应被重写。历史 accepted 数据与其原始审计证据保持不可变；用新路径重建数据时，应生成新的运行证据和版本绑定。

## 8. 许可证与发布边界

配置和审计证明每个来源均绑定 revision 与声明许可证，但组合数据的再分发条件仍需逐源复核。Git 不保存正文；向 ModelScope/Hugging Face 发布前，必须单独完成 license compatibility、数据卡和来源 attribution 审核。
