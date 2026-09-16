# D01-training-data-v1-20260828

状态：`accepted`（正式 remixed Pretrain v1 已通过 verify、audit 和 marker read-back；接受时间 2026-08-31 11:16:03 UTC）

## 结论与阶段边界

- SFT 代理流水线已经完成，用来验收 schema、分组去重、token-aware 监督和 benchmark 去污染；它不是目标 160M assistant-token SFT。
- 3.2M Pretrain smoke 是历史验收产物。
- 320M pilot 是历史已完成产物：五源、533,376 行、320,001,838 nonpad input tokens，对应 319,468,462 loss-target tokens。它使用旧的单对象前缀和逐文档截断口径，不能当作正式 1.28B 的前 25%。
- 首次正式 run `pretrain-v1-1b28-run-20260831T042840Z-246795` 的 `materialize -> mix -> verify` 成功，audit 发现旧 final 中 10 条真实 whitespace-only 数据行、6 个 exact 命中；旧 containment 把很短的 query 当任意子串，记录了 4,249,137 次 pattern-entry/document 命中，主要由短 query 假阳性驱动，不能解释为 4,249,137 个污染文档，因此 run 失败且没有 `_SUCCESS`。旧 `final/` 原样保留作失败追溯。
- 修复将所有非空规范化 query 保留为 exact 候选（`exact_min_normalized_chars=1`），只允许长度至少 20 的 query 进入 query-in-document containment；remixer 在选择前过滤原始记录及切分片段，从已固定的 54-object candidate pool 重新补齐精确来源 quota，共排除 258 条候选。
- 修复 run `pretrain-v1-remix-run-20260831T085409Z-311234` 已完成 `remix -> verify -> audit -> acceptance_readback`。可训练产物是新的 `final-remix-v1/`，不是旧 `final/`。

## 关键入口

- Pretrain 配置：`configs/data/pretrain/pretrain_v1.yaml`
- 固定对象池：`configs/data/pretrain/pretrain_shards_v1.yaml`
- benchmark 配置：`configs/data/pretrain/contamination_eval_v2.yaml`
- 首次构建 launcher：`experiments/00-preparation/D01-training-data-v1-20260828/launch_pretrain_v1_1b28.sh`
- 修复验收 launcher：`experiments/00-preparation/D01-training-data-v1-20260828/launch_pretrain_v1_remix.sh`
- builder：`scripts/data/pretrain/build_pretrain_v1.py`
- remixer：`scripts/data/pretrain/remix_pretrain_v1.py`
- 独立 auditor：`scripts/data/pretrain/audit_pretrain_v1.py`
- 阶段报告：`experiments/00-preparation/D01-training-data-v1-20260828/report.md`
- 旧运行追溯：`experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_run_metadata.json`

## 两次正式运行

| run / stage | UTC 区间 | 时长 | 状态 |
|---|---|---:|---|
| 首次 `materialize` | 04:28:40–05:39:02 | 1:10:22 | success |
| 首次 `mix` | 05:39:02–06:06:34 | 0:27:32 | success |
| 首次 `verify` | 06:06:34–06:45:43 | 0:39:09 | success |
| 首次 `audit` | 06:45:43–07:22:06 | 0:36:23 | failed，exit 1 |
| 首次 run 合计 | 04:28:40–07:22:06 | 2:53:26 | failed at audit |
| 修复 `remix` | 08:54:09–10:02:41 | 1:08:32 | success |
| 修复 `verify` | 10:02:41–10:41:06 | 0:38:25 | success |
| 修复 `audit` | 10:41:06–11:16:04 | 0:34:58 | success |
| 修复 `acceptance_readback` | 11:16:04 | <1 秒（日志秒级精度） | success |
| 修复 run 合计 | 08:54:09–11:16:04 | 2:21:55 | success，exit 0 |

每个 run 目录均保存 `metadata.json`、`commands.tsv`、`events.tsv`、`status.env` 和 `run.log`；元数据记录 Git dirty、输入及脚本 SHA，不记录环境变量、访问令牌或凭据。

## 数据路径

| 用途 | 路径 |
|---|---|
| 历史 3.2M smoke | `/data/datasets/minimind-lab/data-v1/pretrain-smoke-3m2/` |
| 历史 320M pilot | `/data/datasets/minimind-lab/data-v1/pretrain-pilot-320m/` |
| 正式 1.28B data root | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/` |
| 对象级候选与 done manifest | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/work/candidates/` |
| 首次失败 final（保留、无 marker） | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final/` |
| 已接受 remixed final | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/` |
| remixed verify 报告 | `experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_remix_verification.json` |
| remixed audit 报告 | `experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_remix_audit.json` |
| remixed 接受标记 | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/_SUCCESS` |

训练时必须给 loader 传 quoted glob：`--data_path '/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/train-*.jsonl'`。

## 精确来源 quota 与分片

| 来源 | train loss-target tokens | validation loss-target tokens |
|---|---:|---:|
| ChineseWebText2.0 | 576,000,000 | 2,880,000 |
| FineWeb-Edu | 384,000,000 | 1,920,000 |
| Wikipedia zh | 64,000,000 | 320,000 |
| Wikipedia en | 64,000,000 | 320,000 |
| FineMath 4+ | 128,000,000 | 640,000 |
| The Stack v3.1 permissive subset | 64,000,000 | 320,000 |
| 合计 | **1,280,000,000** | **6,400,000** |

每个来源和 aggregate 都是精确 quota，delta 为 0，不是“至少达到”或允许一个 chunk overshoot。最终为 40 个 train shards、1 个 validation shard；train 2,313,483 行，validation 11,525 行。32M 只是 40 个 train shard 的期望平均 loss-target tokens，分片本身按确定性 hash 顺序 row round-robin。

## 过滤、verify 与 audit 结果

- remixer 排除 258 条候选：226 containment、27 whitespace-only、5 exact；train/validation 分别 251/7。排除都发生在 original candidate 阶段，ledger、snapshot 和 benchmark snapshot 均有 SHA 绑定。
- verify 独立重计数得到 train/validation 1,280,000,000/6,400,000 loss-target tokens，40+1 shard 全部 `status=ok`；MiniMind `PretrainDataset` loader dry-run 为 `status=ok`，sample shape `[768]`。
- audit 加载固定 7 个 benchmark source、124 个 config、29,638 行、29,322 个唯一规范化 query，无 load/pin error。
- exact 与 containment 对全部 2,325,008 个有效文档扫描，最终 exact/containment 均为 0。
- near 使用 `stratified`、每个物理 shard 512 条 deterministic bottom-k；共审计 20,992/2,308,050 个 eligible documents，结果为 0。它不是 near 的全语料扫描，不能写成“near 全量通过”。
- empty/blank/invalid UTF-8/invalid JSON/invalid schema 均为 0；train exact duplicate、validation exact duplicate、train/validation overlap 均为 0。
- `build_evidence_chain=true`，repair evidence、candidate manifest、来源 quota、manifest、verification 与 tokenizer 绑定全部通过；audit `passed=true`。

## 接受证据

| 证据 | SHA-256 / fingerprint |
|---|---|
| candidate manifest SHA | `07e2d6638d061a53524cc2c9dfb87e0d7a2e9507677ad4f7d83324fb1815b125` |
| final manifest SHA | `1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6` |
| verify report SHA | `ad58496ddd6a3d80dc590910299555db0d405c54579e0a6c9b4a9d03626fac3d` |
| audit report SHA | `bfe58ccbf2a1b0aefe13b1a684d717094cae53db3b37b3563fd572015da8a58a` |
| `_SUCCESS` SHA | `8c9c728ed063b214ebb74e905e0f6ccc4a510c7078408b87cda5d0c990639c7c` |
| manifest fingerprint | `ee7323cd3513b84aca2ba5f0c0543a6dced8d9e87db4cee0c4c3c24c48f1bc25` |
| dataset fingerprint | `cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd` |

Final manifest 仍保持不可变的 `pending_external_audit`；正式接受态由外部 `_SUCCESS` 的 `status=accepted` 表达。Marker 原子写在 audit report 之后，并绑定 audit report、manifest、verification、candidate manifest、config/eval config、tokenizer、auditor、dataset fingerprint 和 repair evidence；launcher 最后再次 read-back。仅凭 final 目录或 manifest 本身不能宣称已接受。
