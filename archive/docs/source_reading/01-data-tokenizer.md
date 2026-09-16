# MiniMind Tokenizer 与 Dataset 源码阅读

## 1. 模块拓扑

```text
官方 JSONL
  ├─ PretrainDataset ── text → BOS/text/EOS → pad → labels
  ├─ SFTDataset ─────── conversations → chat template → assistant-only labels
  ├─ DPODataset ─────── chosen/rejected → shifted x/y → assistant loss mask
  ├─ RLAIFDataset ───── conversations[:-1] → generation prompt
  └─ AgentRLDataset ─── messages[:-1] + tools + gt
                         │
minimind/model/tokenizer.json + tokenizer_config.json
                         │
                 input_ids / labels / masks
```

主线文件：

- [`train_tokenizer.py`](../../../trainer/train_tokenizer.py)：展示 ByteLevel BPE 的训练过程；
- [`tokenizer.json`](../../../tokenizer/tokenizer.json)：固定词表与 merge 结果；
- [`tokenizer_config.json`](../../../tokenizer/tokenizer_config.json)：特殊 token 与 chat template；
- [`lm_dataset.py`](../../../dataset/lm_dataset.py)：五类 Dataset 的实际输入输出逻辑。

## 2. 为什么不替换官方 Tokenizer

`train_tokenizer.py` 第一行就说明脚本只供学习，不建议重训主线 Tokenizer。当前实现是：

- ByteLevel 预分词的 BPE；
- `vocab_size=6400`；
- 36 个预留 token 位置；
- `bos=<|im_start|>`、`eos=<|im_end|>`、`pad/unk=<|endoftext|>`；
- `model_max_length=131072`；
- `<tool_call>`、`<tool_response>`、`<think>` 等 token 被加入词表，但配置为普通 added token，便于模板输出和文本匹配。

更换 Tokenizer 会同时改变 token id、Embedding 行含义、LM Head 行含义和 token 数量，因此已有权重与后续对照都失去可比性。Stage1 固定仓库自带 Tokenizer，只观察不替换。

## 3. Chat Template 的实际结构

无工具时，每条消息使用：

```text
<|im_start|>{role}\n{content}<|im_end|>\n
```

assistant 消息还会插入：

```text
<think>\n{reasoning_content}\n</think>\n\n{content}
```

有工具时，模板把工具 JSON schema 放入 system 段；assistant 的调用被包在 `<tool_call>` 中，tool 返回被包在 `<tool_response>` 中。`add_generation_prompt=True` 会追加 assistant 起始标记，并根据 `open_thinking` 决定是否打开思考段。

## 4. 五类 Dataset 的实际逻辑

### 4.1 PretrainDataset

1. 对 `text` 编码，最多保留 `max_length-2`；
2. 手工添加 BOS/EOS；
3. 右侧补 pad 到固定长度；
4. `labels=input_ids.clone()`，仅 pad 位置改为 `-100`。

因此预训练对 BOS 之后的所有非 pad token 计算 next-token loss，不存在 assistant-only mask。

### 4.2 SFTDataset

1. `pre_processing_chat` 对非工具样本有 20% 概率增加 system prompt；工具样本完整保留；
2. 字符串形式的 `tools/tool_calls` 会先反序列化；
3. chat template 生成完整 prompt；
4. `post_processing_chat` 默认有 80% 概率删除空 `<think>` 段；
5. `generate_labels` 搜索 `<|im_start|>assistant\n` 到 `<|im_end|>\n`，只复制 assistant span，其余位置保持 `-100`。

合成 fixture 在长度 128 下得到 33 个有效 label 和 95 个忽略位置，证明用户、system 与 padding 不直接进入监督目标。

### 4.3 DPODataset

chosen/rejected 分别套 chat template 并 pad/truncate，再构造：

- `x=input_ids[:-1]`；
- `y=input_ids[1:]`；
- `mask=assistant_loss_mask[1:]`。

合成 fixture 的 chosen/rejected 有效 mask token 分别为 20/15；偏好损失只比较 assistant 输出区域。

### 4.4 RLAIFDataset

它取 `conversations[:-1]` 形成 generation prompt，最后一个 assistant 答案不进入 prompt。`thinking_ratio` 随机选择开放 `<think>` 还是插入空思考段，返回值中的 `answer` 为空，后续由 rollout 生成。

### 4.5 AgentRLDataset

它不做 tokenization，而是返回：

- `messages=conversations[:-1]`；
- system 消息中解析出的 `tools`；
- 数据给出的最终校验目标 `gt`。

Tokenizer 与 rollout engine 在后续 Agent 阶段才介入。

## 5. Stage1 观察结果

固定官方数据 revision `312afb4f76391145c6902f765bb51691c09a12f5`。全文件扫描未发现坏 JSON 或空行；token 长度使用每文件 2000 条、seed 42 的 reservoir sample，统一按 768 tokens 观察。

| 数据 | 行数 | 平均 tokens | P90 | 超过 768 |
|---|---:|---:|---:|---:|
| pretrain mini | 1,270,238 | 261.9 | 408 | 4.20% |
| SFT mini | 905,718 | 493.3 | 719 | 6.55% |
| DPO | 17,166 | 472.5 | 837 | 15.75% |
| RLAIF | 19,502 | 317.3 | 557 | 0.05% |
| Agent RL | 39,988 | 611.0 | 1,177 | 29.30% |
| Agent RL Math | 20,000 | 408.0 | 443 | 0% |

这里的 768 是统一观察口径，不等于每个训练脚本的代码默认值。尤其 Agent/DPO 的长样本比例更高，后续阶段必须按各自配置重新评估截断。

## 6. 压缩率

从官方 pretrain/SFT 的固定抽样中按启发式规则选择六类文本，每类 20 条；仅提交文本 SHA256 与聚合指标，不提交原文。

| 类型 | 字符/token |
|---|---:|
| 中文 | 1.389 |
| 英文 | 2.898 |
| 代码 | 1.922 |
| 数学 | 1.569 |
| 新闻 | 1.432 |
| 领域 | 1.627 |

这些值适合比较同一 Tokenizer 对不同文本形态的相对开销，不代表语言能力。类别来自关键词与字符比例启发式分类，不是数据集官方标签。

## 7. 可复现证据

- [数据 manifest](../../experiments/00-preparation/E01-tokenizer-dataset-20260823/data_manifest.json)
- [数据质量报告](../../experiments/00-preparation/E01-tokenizer-dataset-20260823/data_quality_report.md)
- [样本变换结果](../../experiments/00-preparation/E01-tokenizer-dataset-20260823/results/sample_transformations.json)
- [分析脚本](../../../scripts/eval/analyze_tokenizer_dataset.py)

官方原始数据保存在 CPFS `/data/datasets`，不进入 Git。仓库只保存版本、文件大小、SHA256、行数、聚合指标和自行编写的合成 fixtures。
