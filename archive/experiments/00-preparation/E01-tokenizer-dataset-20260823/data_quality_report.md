# Stage1 数据质量报告

## 数据身份

- 来源：`jingyaogong/minimind_dataset`；
- revision：`312afb4f76391145c6902f765bb51691c09a12f5`；
- 文件：pretrain/SFT mini、DPO、RLAIF、Agent RL、Agent RL Math；
- 总大小：3,158,062,501 bytes；
- 下载端点：hf-mirror，仅作为传输代理；最终身份由固定 revision、大小和官方 SHA256 校验。

## 完整性

| 文件 | 行数 | 坏 JSON | 空行 |
|---|---:|---:|---:|
| pretrain_t2t_mini.jsonl | 1,270,238 | 0 | 0 |
| sft_t2t_mini.jsonl | 905,718 | 0 | 0 |
| dpo.jsonl | 17,166 | 0 | 0 |
| rlaif.jsonl | 19,502 | 0 | 0 |
| agent_rl.jsonl | 39,988 | 0 | 0 |
| agent_rl_math.jsonl | 20,000 | 0 | 0 |

行数、坏 JSON 和空行来自全文件扫描。文件 SHA256 与大小见 `data_manifest.json`。

## 长度、截断与 padding

方法：每个文件 reservoir sample 2000 条，seed 42，使用仓库固定 Tokenizer，统一以 768 tokens 计算潜在截断和 padding。

| 文件 | 样本 | 平均 | P50 | P90 | P99 | 最大 | 截断率 | 平均 padding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrain mini | 2000 | 261.9 | 213 | 408 | 1420 | 2053 | 4.20% | 529.7 |
| SFT mini | 2000 | 493.3 | 491 | 719 | 933 | 1292 | 6.55% | 280.7 |
| DPO | 2000 | 472.5 | 449 | 837 | 1069 | 1299 | 15.75% | 314.6 |
| RLAIF | 2000 | 317.3 | 350 | 557 | 731 | 771 | 0.05% | 450.7 |
| Agent RL | 2000 | 611.0 | 545 | 1177 | 1600 | 1969 | 29.30% | 255.6 |
| Agent RL Math | 2000 | 408.0 | 412 | 443 | 455 | 464 | 0% | 360.0 |

边界：这是观察分布，不是对所有记录的全量 tokenization；768 也不是 DPO/Agent 阶段的最终配置。Stage3 以后必须按具体 `max_seq_len` 重算。

## Tokenizer 压缩率

官方 pretrain/SFT 抽样，启发式分类，每类 20 条：中文 1.389、英文 2.898、代码 1.922、数学 1.569、新闻 1.432、领域 1.627 字符/token。分类结果仅用于观察，不作为官方领域标签或质量打分。

## Dataset 语义检查

- Pretrain fixture：128 长度中 45 个有效 label，其余为 padding `-100`；
- SFT fixture：33 个 assistant label、95 个忽略位置；
- DPO fixture：chosen/rejected assistant mask 分别 20/15；
- RLAIF fixture：最后答案被移出 prompt，生成提示包含 assistant 起始段；
- Agent fixture：返回 user 前缀消息、解析后的 tools 和 `gt=437`。

## 结论

Stage1 数据门禁通过：固定版本的 6 个文件全部通过大小/SHA256/JSON 检查；六类压缩率样本齐全；五类 Dataset 的输入输出语义已由可复现 fixture 验证。长样本风险集中在 Agent RL 和 DPO，进入对应训练阶段时必须重新选择序列长度并报告截断成本。
