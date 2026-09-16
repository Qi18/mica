# E01-tokenizer-dataset-20260823

## 状态

Stage1 已完成，阶段门通过。固定数据版本、Tokenizer 观察、Dataset 语义、数据质量和源码阅读均已有可复现证据。

## 目标

1. 解释官方 ByteLevel BPE Tokenizer 的训练边界与特殊 token；
2. 证明 Pretrain/SFT/DPO/RLAIF/AgentRLDataset 的实际输入输出语义；
3. 对官方数据抽样统计压缩率、截断、padding 和 schema；
4. 建立可复现的数据 manifest，不把训练数据提交到 Git。

## 直接基线

- Lab baseline：`080179c9d1ec2a0752e2a21155af899e7509c5b9`；
- MiniMind upstream source：`393e387e9ad99f0f04c296e4c5e7353f4444629f`；
- Dataset revision：`312afb4f76391145c6902f765bb51691c09a12f5`；
- Tokenizer：仓库自带 `minimind/model`，本阶段不训练或替换。

## 数据完整性

| 文件 | 大小 | 行数 | 坏 JSON |
|---|---:|---:|---:|
| pretrain mini | 1.24GB | 1,270,238 | 0 |
| SFT mini | 1.74GB | 905,718 | 0 |
| DPO | 53.7MB | 17,166 | 0 |
| RLAIF | 23.8MB | 19,502 | 0 |
| Agent RL | 82.0MB | 39,988 | 0 |
| Agent RL Math | 18.4MB | 20,000 | 0 |

6 个文件均匹配固定 revision 的大小和 SHA256。官方数据存放在 `/data/datasets`，未提交 Git。

## 观察结果

每个文件固定 seed 42 抽样 2000 条，统一用 768 tokens 观察：

| 数据 | 平均 tokens | P90 | 超过 768 |
|---|---:|---:|---:|
| pretrain mini | 261.9 | 408 | 4.20% |
| SFT mini | 493.3 | 719 | 6.55% |
| DPO | 472.5 | 837 | 15.75% |
| RLAIF | 317.3 | 557 | 0.05% |
| Agent RL | 611.0 | 1177 | 29.30% |
| Agent RL Math | 408.0 | 443 | 0% |

长样本风险集中在 Agent RL 和 DPO。768 是统一观察口径，不是后续各脚本的最终配置。

六类文本各 20 条均完成压缩率统计：中文 1.389、英文 2.898、代码 1.922、数学 1.569、新闻 1.432、领域 1.627 字符/token。分类为启发式规则，不是官方标签。

## Dataset 语义

- Pretrain：BOS/text/EOS 后 padding，所有非 pad token 都参与 loss；
- SFT：仅 assistant span 参与 loss，fixture 为 33 valid / 95 ignored；
- DPO：chosen/rejected 分别 shift，assistant mask 为 20/15；
- RLAIF：最后答案移出 prompt，返回待 rollout 的 generation prompt；
- Agent RL：返回 `messages[:-1]`、tools 与 gt，不在 Dataset 内 tokenization。

## 产物

- `data_manifest.json`：外部数据文件版本、大小、SHA256 和行数；
- `results/dataset_audit.json`：schema、长度、截断和 padding；
- `results/compression_metrics.json`：六类文本各 20 条的压缩率；
- `results/sample_transformations.json`：合成样本从 JSON 到 prompt/input_ids/labels/mask；
- `data_quality_report.md`：数据质量结论与口径边界；
- `docs/source_reading/01-data-tokenizer.md`：源码阅读结论。

## 异常与修复

1. L20 无法直连 HuggingFace，改用 hf-mirror；文件最终由官方大小和 SHA256 验证；
2. SSH ProxyCommand 的 socket stdin 触发误导性的 6443 `bad file descriptor`，在 Skill 中通过管道规范化修复；
3. 第一版观察器遗漏字符串 `tool_calls` 的反序列化，导致 SFT 统计只有 1823 条；按官方 `SFTDataset` 语义修正后，2000 条全部进入最终统计；
4. 第一版新增文件补丁行数错误导致脚本截断；增加 `summary.json` 非空验收后修复并完整复跑。

## 结论

Stage1 通过。官方 Tokenizer 保持固定；数据身份和质量边界已建立；能够从原始 JSON 解释到 template、input_ids、labels 与 mask。下一步进入 Stage2 模型结构和 100-step probe。
