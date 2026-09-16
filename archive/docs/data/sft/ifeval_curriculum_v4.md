# IFEval curriculum v4 数据卡

## 目的

该数据集用于验证 64M MiniMind 能否从独立的、机械可验证的指令课程迁移到冻结的 IFEval。它不是正式通用 SFT 数据集，也不替代开放域 Chat、人工偏好或 Tool 数据。

## 构建入口

```bash
HF_DATASETS_OFFLINE=1 /data/venvs/minimind-eval/bin/python \
  scripts/data/sft/build_ifeval_curriculum_v4.py \
  --output-root /data/datasets/minimind-lab/data-v1/sft-ifeval-curriculum-v4 \
  --broad-root /data/datasets/minimind-lab/data-v1/sft-chat-repair-v2 \
  --formal-smoke-root /data/datasets/minimind-lab/data-v1/sft-v1-32m/final/smoke \
  --broad-replay 20000 --seed 20260907
```

## 构成与验收

- 16,657 条去重后的 broad replay、6,343 条 formal smoke replay、18,720 条新生成的约束课程。
- 新课程覆盖公开 IFEval registry 的 25 类单约束和 14 类兼容组合。
- 每条生成答案均由 lm-eval IFEval strict checker 验证；失败样本不会写入数据集。
- train/validation/test 使用不同主题词和编号空间；跨 split exact/normalized prompt overlap 均为 0。
- 使用冻结的 `google/IFEval` 541 prompts 做污染扫描；三个 split 的 exact/normalized overlap 均为 0。
- 训练集 4,163,325 assistant target tokens，zero target=0，`max_seq_len=768` 下截断=0。

## 证据边界

训练数据使用了与最终 benchmark 相同的公开约束类型和 verifier 语义，因此结果只能表述为“IFEval-family 指令遵循提升”。即使原题重合为 0，也不能把该结果外推成任意对话能力提升；必须同时报告原有 Chat/Tool 回归和七项知识评测。
