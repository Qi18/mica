# MiniMind 模型结构与训练路径

本文对应 MiniMind Lab Stage2，源码基线为 `minimind/model/model_minimind.py` commit `393e387e9ad99f0f04c296e4c5e7353f4444629f`，观察实验代码基线为 Lab commit `5ccb66702c2f1f1b969c79240fd5fe5e247131cd`。

## 1. 模块拓扑

```text
input_ids [B,T]
  -> Embedding [B,T,768]
  -> 8 x MiniMindBlock
       -> RMSNorm
       -> GQA Attention + residual
       -> RMSNorm
       -> Dense FFN 或 MoE FFN + residual
  -> RMSNorm
  -> tied LM Head [B,T,6400]
  -> shifted cross entropy
```

主线 64M Dense 配置为 hidden size 768、8 层、8 个 query heads、4 个 KV heads、head dim 96。词表为 6400，Embedding 与 LM Head 权重共享。

## 2. 一层 Block 的实际逻辑

### 2.1 Pre-Norm 与残差

每个 Block 先对输入做 RMSNorm，再送入 Attention，输出与原输入相加。第二段再次 RMSNorm，经过 FFN 后与 Attention 段的输出相加：

```text
h = x + Attention(RMSNorm(x))
y = h + FFN(RMSNorm(h))
```

RMSNorm 以 float32 计算均方根，再转换回原 dtype，因此 BF16 训练时归一化部分仍保持更稳定的数值计算。

### 2.2 GQA Attention

对 hidden `[B,T,768]`：

| Tensor | RoPE 前 shape | Attention 计算 shape |
|---|---|---|
| Q | `[B,T,8,96]` | `[B,8,T,96]` |
| K | `[B,T,4,96]` | repeat 后 `[B,8,T,96]` |
| V | `[B,T,4,96]` | repeat 后 `[B,8,T,96]` |

Q head 数是 KV head 的 2 倍，因此每两个 query heads 共享一组 K/V。训练配置 `flash_attn=True` 时，代码实际调用 PyTorch `scaled_dot_product_attention`；这表示走 SDPA 的 fused/flash 后端选择机制，并不是单独引入 FlashAttention-3 包。

RoPE 只应用于 Q/K。频率表以 `rope_theta=1e6` 预计算，KV cache 保存 RoPE 后、repeat 前的 K/V，因此缓存维度仍是 4 个 KV heads。

### 2.3 Dense FFN

Dense 分支使用 SwiGLU：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

hidden 768 对应 intermediate size 2432。每层包含 gate/up/down 三个无 bias 线性层。

### 2.4 MoE FFN

MoE 分支保留相同 Attention，但将每层 Dense FFN 替换为：

1. Router 对每个 token 输出 4 个 expert 概率；
2. 选择 top-1 expert；
3. 仅执行被选中的 expert；
4. 按 routing weight 合并输出；
5. 训练时计算 load-balancing aux loss。

当前默认 4 experts、每 token 激活 1 expert。总参数明显增加，但每 token 的激活参数接近 Dense。

## 3. 参数量拆解

| 模块 | Dense | MoE |
|---|---:|---:|
| Embedding + tied LM Head | 4,915,200 | 4,915,200 |
| Attention | 14,157,312 | 14,157,312 |
| Norm | 13,056 | 13,056 |
| MLP/MoE | 44,826,624 | 179,331,072 |
| 总参数 | 63,912,192 | 198,416,640 |
| 每 token 激活参数估算 | 63,912,192 | 63,936,768 |

MoE 总参数是 Dense 的 3.105 倍，但 top-1 下激活参数只约为 Dense 的 1.0004 倍。该数字描述参数路径，不等同于端到端 FLOPs 或吞吐；Router、dispatch 和多个 expert kernel 仍会引入额外开销。

## 4. Logits 与 Loss

模型输出 hidden states `[B,T,768]`，经共享权重 LM Head 得到 logits `[B,T,6400]`。训练 loss 做标准 next-token shift：

```text
x = logits[:, :-1, :]
y = labels[:, 1:]
cross_entropy(x, y, ignore_index=-100)
```

因此位置 `t` 的 logits 预测位置 `t+1` 的 label。MoE 总 loss 还要加各层 Router aux loss。

## 5. KV Cache

单卡观察中，batch 2、prefix 8 tokens 时，layer0 key shape 为 `[2,8,4,96]`；再输入 1 个 token 后变为 `[2,9,4,96]`，新 logits 为 `[2,1,6400]`。

生成时模型只把尚未缓存的新 token 送入网络，K/V 沿序列维追加。这避免每生成一个 token 都重新计算整个 prefix。

## 6. 单卡结构探针

| 模型 | Loss | Aux loss | forward+backward | 峰值 allocated | finite |
|---|---:|---:|---:|---:|---|
| Dense | 8.9281 | 0 | 0.2149s | 655.5MiB | 是 |
| MoE | 8.9323 | 0.00476 | 0.1986s | 1592.3MiB | 是 |

输入为 `[2,16]` 随机 token。该单次时延包含 CUDA warmup 差异，不能据此声称 MoE 比 Dense 快；它只验证两条结构路径均可完成 BF16 forward/backward，loss 和 gradient 均为有限值。

## 7. 8 卡 100-step 探针

正式探针使用 Dense 64M、BF16、8×L20、每卡 batch 4、seq 128。数据由固定官方 pretrain mini 的前 8192 条构造为预分词 Tensor，因此吞吐主要反映模型与 DDP，而不是 JSON 解析和 tokenization。

| Seed | 初始 Loss | 最终 Loss | 有效 tokens/s | samples/s | 平均 GPU 利用率 |
|---:|---:|---:|---:|---:|---:|
| 42 | 8.8794 | 6.6754 | 50,641 | 437.2 | 89.0% |
| 43 | 8.8939 | 6.5579 | 50,647 | 435.6 | 90.3% |
| 44 | 8.9409 | 6.5112 | 50,521 | 433.7 | 90.1% |

三组均无 NaN、Inf、OOM 或 NCCL 错误。平均吞吐 50,603 有效 tokens/s，seed 间标准差 71 tokens/s。

Seed42 在 step50 保存 model、AdamW optimizer、step、world size 和 SwanLab run id，随后恢复到 step100。恢复后 optimizer state 有 90 个参数条目；step50/51 的 LR 从 `2.75e-4` 连续到 `2.679e-4`，SwanLab run id 保持 `iq14wfm1nc1ca8iigdbop`。

## 8. 结论与边界

Stage2 证明了模型结构、Dense/MoE 分支、KV cache、8 卡 DDP、BF16、SwanLab 和 resume 链路正确。它没有证明模型已经收敛，也不能与 Stage3 的 seq 768 正式 Pretrain 吞吐或质量直接比较。

可复现证据位于 `experiments/00-preparation/E02-model-probe-20260823/`，原始模型与 optimizer checkpoint 只保留在 CPFS，由 `checkpoint-manifest.txt` 记录 SHA256。
