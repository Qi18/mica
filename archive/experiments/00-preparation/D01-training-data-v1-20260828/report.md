# D01 数据 v1 阶段报告

更新时间：2026-08-31
状态：`accepted`（remixed Pretrain v1 于 2026-08-31 11:16:03 UTC 由外部 `_SUCCESS` 接受，并通过 launcher read-back）

## 结论

SFT 代理流水线、3.2M Pretrain smoke 和 320M Pretrain pilot 均为历史已完成阶段。320M pilot 已通过文件 SHA 与独立 tokenizer 重计数，但它采用旧的五源、单对象前缀和逐文档截断口径，只用于历史对照，不能视为正式 1.28B 的已完成部分。

首次正式 run `pretrain-v1-1b28-run-20260831T042840Z-246795` 已完成 materialize、mix 和 verify，但 audit 在旧 `final/` 中发现 10 条真实 whitespace-only 数据、6 个 exact 命中；旧 containment 又错误地把很短的 query 当作任意子串，产生 4,249,137 个缺乏污染含义的命中。该 run 以 exit 1 失败，未生成 `_SUCCESS`，旧 `final/` 原样保留用于追溯。

修复后，exact 接受所有至少 1 个规范化字符的 query，containment 只接受至少 20 个规范化字符的 query；remixer 从固定 54-object candidate pool 重新选择并精确补齐 quota，共排除 258 条候选。修复 run `pretrain-v1-remix-run-20260831T085409Z-311234` 的 remix、verify、audit 和 acceptance read-back 全部成功，正式可训练产物为新的 `final-remix-v1/`。

## SFT 构建与验收

| 阶段 | train | validation | test | 说明 |
|---|---:|---:|---:|---|
| 完整会话精确去重后初分 | 5,330 | 316 | 309 | 合计 5,955 |
| near 组件重分桶 | 5,329 | 315 | 311 | 仅移动 3 条 |
| tokenizer 零目标隔离后 | 5,292 | 309 | 307 | 隔离 47 条 |
| benchmark 去污染后 | 5,292 | 309 | 306 | 从 test 隔离 1 条 |
| 最终 | 5,292 | 309 | 306 | 合计 5,907 |

near grouping 对规范化用户消息使用 char 5-gram、Jaccard 0.85。5,955 条形成 5,895 个连通组件，跨 split exact/near 均为 0。tokenizer 口径复用 MiniMind chat template、sequence length 768 head truncation 和 `SFTDataset.generate_labels` 等价 assistant mask，最终零 assistant target 为 0。

七项 benchmark 共加载 29,638 条。唯一初次 exact 命中是 Glaive test 会话中的 `How do you find the area of a triangle?`，流水线隔离完整会话后，最终 5,907 条全语料 exact/containment/near 为 0/0/0。长度小于 20 的 6,137 条规范化 benchmark query 只做 exact，这是已记录的检测边界。

| 最终 SFT 代理指标 | 结果 |
|---|---:|
| train / validation / test | 5,292 / 309 / 306 |
| train input tokens | 2,091,820 |
| train assistant target tokens | 1,618,976 |
| train truncated rows | 1,085（20.503%） |
| train zero assistant target | 0 |
| 跨 split exact / near | 0 / 0 |
| 七项 exact / containment / near | 0 / 0 / 0 |

该代理集只用于验收流水线，不等于目标 160M assistant-token SFT。

## 历史 Pretrain smoke 与 320M pilot

3.2M smoke 已完成：5,306 行、3,201,486 nonpad input tokens，invalid/empty 为 0。smoke 总 raw tokens 为 10,270,025，49.5854% 行发生截断，证明旧的逐文档截断会丢弃大量长文尾部。

320M pilot 已完成，manifest status 为 `ok`，五源文件 SHA、pinned revision 和独立 tokenizer 重计数一致：

| 来源 | 目标 nonpad tokens | 实际 nonpad tokens | 行数 |
|---|---:|---:|---:|
| ChineseWebText2.0 | 144,000,000 | 144,000,231 | 262,615 |
| FineWeb-Edu | 112,000,000 | 112,000,683 | 172,615 |
| Wikipedia zh | 16,000,000 | 16,000,501 | 25,813 |
| Wikipedia en | 16,000,000 | 16,000,421 | 25,637 |
| FineMath | 32,000,000 | 32,000,002 | 46,696 |
| 合计 | 320,000,000 | 320,001,838 | 533,376 |

该 pilot 对应 319,468,462 loss-target tokens。旧 pilot 每个来源只读取一个 pinned object 的前缀，每篇长文只保留前 766 个 text tokens；它是历史实验，不是正式 1.28B 的组成部分。

- smoke：`/data/datasets/minimind-lab/data-v1/pretrain-smoke-3m2/`
- pilot：`/data/datasets/minimind-lab/data-v1/pretrain-pilot-320m/`

## 正式 1.28B Pretrain v1

正式输出来自固定的 54 个 pinned objects。长文按 tokenizer 切成不超过 766 个 text tokens 的片段并保留尾部；每条训练记录的 loss-target 为 `text_tokens + EOS`。Stack v3 仅接受 `license_type=permissive`、`is_vendor=false`、`detected_licenses` 非空且属于固定 14 种主流语言的文件，`no_license` 不用于补足预算。

sampling boundary 是固定多对象池的 bounded candidate window 内 hash selection，并不等价于扫描完整上游数据源后的全局均匀抽样。

### 精确 quota 与输出规模

| 来源 | train loss-target tokens | validation loss-target tokens | delta |
|---|---:|---:|---:|
| ChineseWebText2.0 | 576,000,000 | 2,880,000 | 0 |
| FineWeb-Edu | 384,000,000 | 1,920,000 | 0 |
| Wikipedia zh | 64,000,000 | 320,000 | 0 |
| Wikipedia en | 64,000,000 | 320,000 | 0 |
| FineMath 4+ | 128,000,000 | 640,000 | 0 |
| The Stack v3.1 permissive subset | 64,000,000 | 320,000 | 0 |
| 合计 | **1,280,000,000** | **6,400,000** | **0** |

每个来源和 aggregate 都是精确 quota，不允许“至少达到”或由最后一个 chunk overshoot。最终输出为 40 个 train shards 和 1 个 validation shard；32M 只是 40 个 train shard 的期望平均 loss-target tokens。

| split | rows | loss-target tokens | nonpad input tokens | padded input tokens | text tokens |
|---|---:|---:|---:|---:|---:|
| train | 2,313,483 | 1,280,000,000 | 1,282,313,483 | 1,776,754,944 | 1,277,686,517 |
| validation | 11,525 | 6,400,000 | 6,411,525 | 8,851,200 | 6,388,475 |
| 合计 | 2,325,008 | 1,286,400,000 | 1,288,725,008 | 1,785,606,144 | 1,284,074,992 |

### 首次失败与修复口径

首次 run 的 audit v1.1.0 对 2,324,986 行进行了检查。预算、duplicate、split overlap、shard、near 和 evidence gate 通过，但下列三项失败：

- `empty_rate`：train 中存在 10 条真实 whitespace-only 行；
- `eval_exact_overlap`：发现 6 个 exact 命中；
- `eval_query_in_document_containment`：旧规则包含任意长度 query，短词如 `pan`、`rag`、`hat`、`soap`、`Seeds` 仅作为其他文本的子串就被计数，累计 4,249,137 次，语义不成立。

修复规则为 `exact_min_normalized_chars=1`、`containment_min_normalized_chars=20` 和 `not text.strip()` whitespace 过滤。remixer 在最终选择前同时过滤 original candidate 和切分后的 piece，并在尾部只剩 1 token 等边界上做事务性 replacement，使 quota 仍然精确。

| remixer 排除原因 | 数量 |
|---|---:|
| containment | 226 |
| whitespace-only / empty | 27 |
| exact | 5 |
| 合计 | **258** |

258 条排除全部发生在 original candidate 层，train/validation 分别为 251/7；来源分布为 ChineseWebText2 61、FineMath 74、FineWeb-Edu 100、Stack v3 20、Wikipedia en 3、Wikipedia zh 0。exclusion ledger、repair snapshot 和 benchmark snapshot 都被 evidence chain 的 SHA 固定。

### 两次正式运行

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
| 修复 `acceptance_readback` | 11:16:04 | <1 秒（日志为秒级精度） | success |
| 修复 run 合计 | 08:54:09–11:16:04 | 2:21:55 | success，exit 0 |

每个 run 目录保存 `metadata.json`、`commands.tsv`、`events.tsv`、`status.env` 和 `run.log`。metadata 绑定 config、sources、eval config、tokenizer、candidate manifest、builder、remixer、auditor、launcher SHA 及 Git dirty 状态，不记录密钥。

### Verify、audit 与边界

- verify 独立重计数 train/validation 为 1,280,000,000/6,400,000 loss-target tokens；40+1 shards 的状态全部为 `ok`。
- MiniMind `PretrainDataset` loader dry-run 为 `status=ok`，样本 shape 为 `[768]`；训练必须使用 quoted glob `--data_path '/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/train-*.jsonl'`。
- audit 加载固定 7 个 benchmark source、124 个 config、29,638 行、29,322 个唯一规范化 query；23,394 个长度至少 20 的 query 进入 containment，5,928 个短 query 跳过 containment；无 load 或 pin error。
- exact 与 containment 均扫描全部 2,325,008 个有效文档，结果为 0/0。
- near 不是全量扫描：它按每个物理 shard 512 条做 stratified deterministic SHA bottom-k，共审计 20,992/2,308,050 个 eligible documents，near overlap 为 0。
- blank、empty、invalid UTF-8、invalid JSON、invalid schema 均为 0；train exact duplicate、validation exact duplicate、train/validation exact overlap 均为 0。
- `build_evidence_chain=true`，repair evidence 与 manifest、verification、candidate manifest、quota、config/eval/tokenizer/auditor/dataset fingerprint 的绑定均通过；audit `passed=true`。

### 正式路径与接受证据

| 产物 | 路径 |
|---|---|
| data root | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/` |
| candidates | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/work/candidates/` |
| 首次失败 final（保留，无 marker） | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final/` |
| 已接受 final | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/` |
| 首次 run | `experiments/00-preparation/D01-training-data-v1-20260828/runs/pretrain-v1-1b28-run-20260831T042840Z-246795/` |
| 修复 run | `experiments/00-preparation/D01-training-data-v1-20260828/runs/pretrain-v1-remix-run-20260831T085409Z-311234/` |
| verify report | `experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_remix_verification.json` |
| audit report | `experiments/00-preparation/D01-training-data-v1-20260828/pretrain_v1_remix_audit.json` |
| accepted marker | `/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1/_SUCCESS` |

| 证据 | SHA-256 / fingerprint |
|---|---|
| candidate manifest SHA | `07e2d6638d061a53524cc2c9dfb87e0d7a2e9507677ad4f7d83324fb1815b125` |
| final manifest SHA | `1d14286c760e33884a5bc8d8afd1ac95e9d084f1b52b9bf2f582fc3743b694d6` |
| verify report SHA | `ad58496ddd6a3d80dc590910299555db0d405c54579e0a6c9b4a9d03626fac3d` |
| audit report SHA | `bfe58ccbf2a1b0aefe13b1a684d717094cae53db3b37b3563fd572015da8a58a` |
| `_SUCCESS` SHA | `8c9c728ed063b214ebb74e905e0f6ccc4a510c7078408b87cda5d0c990639c7c` |
| manifest fingerprint | `ee7323cd3513b84aca2ba5f0c0543a6dced8d9e87db4cee0c4c3c24c48f1bc25` |
| dataset fingerprint | `cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd` |

Final manifest 是构建时不可变证据，状态仍为 `pending_external_audit`；正式接受态只由外部 `_SUCCESS` 的 `status=accepted` 表达。marker 在 audit report 之后原子写入，绑定 report/manifest/verification/candidate/config/eval/tokenizer/auditor/dataset/repair evidence，并由 launcher 最终 read-back。因此仅凭 final 目录或 manifest 不能宣称验收完成。

## 后续训练与评测缺口

1. 基于 `final-remix-v1/train-*.jsonl` 完成正式 Pretrain 训练，并保存训练配置、SwanLab run、checkpoint 与故障记录；
2. 定稿摘要和显式 reasoning 数据，扩展到 160M assistant target tokens；
3. 正式跑分前冻结 lm-evaluation-harness commit 与评测 prompt/tokenizer 口径；
4. 训练官方等 token 基线、自建 v1，并完成 code/math 来源消融。
