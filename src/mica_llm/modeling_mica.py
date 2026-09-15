# Derived from MiniMind; see NOTICE and minimind/LICENSE.
"""Mica 模型定义：配置、RMSNorm、RoPE、GQA Attention、Dense/MoE FFN 与因果语言模型。

只依赖 PyTorch 与 transformers 的基础设施（PretrainedConfig / PreTrainedModel / GenerationMixin），
不复用 transformers 的 Llama 实现，因此从 config 到自回归采样的每一步都能直接读到。
主线 Dense 配置为 hidden_size=768、8 层、8 个 query head、4 个 KV head、head_dim=96、vocab_size=6400。
"""
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     Mica Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MicaConfig(PretrainedConfig):
    """Mica 的超参容器，继承 PretrainedConfig 以便 from_pretrained / save_pretrained 直接可用。

    显式参数只有三个：hidden_size（隐藏维度）、num_hidden_layers（Block 层数）、use_moe（FFN 是否走 MoE），
    其余超参一律从 kwargs 取默认值，实验脚本只需覆盖关心的字段。几个不直观的默认值：
    - head_dim 默认由 hidden_size // num_attention_heads 推出；num_key_value_heads 小于
      num_attention_heads 即为 GQA。
    - intermediate_size 默认取 hidden_size * pi 再向上对齐到 64 的倍数，让矩阵维度对硬件友好。
    - rope_theta 取 1e6 而非常见的 1e4，配合 max_position_embeddings=32768 支持长上下文。
    - inference_rope_scaling 为 True 时才生成 YaRN 的 rope_scaling 字典，训练阶段保持 None。
    - num_experts 等 MoE 字段在 use_moe=False 时不生效。
    """
    model_type = "mica"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)  # 先让 PretrainedConfig 消化 transformers 的通用字段
        self.hidden_size = hidden_size  # 隐藏维度，主线 64M 配置为 768
        self.num_hidden_layers = num_hidden_layers  # Block 层数
        self.use_moe = use_moe  # FFN 走 Dense 还是 MoE
        self.dropout = kwargs.get("dropout", 0.0)  # 默认关闭，预训练数据量大时一般不需要
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 必须与 model/tokenizer.json 的词表大小一致
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 数据集在每条样本前补的起始 token
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 样本结尾，也是 generate 的停止条件
        self.flash_attn = kwargs.get("flash_attn", True)  # 允许走 SDPA；是否真的走还要看运行时条件
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # query 头数
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # KV 头数，小于 query 头数即 GQA（8:4 → n_rep=2）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每头维度，默认 768 // 8 = 96
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # FFN 激活函数，silu 配 SwiGLU 门控
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # 768*pi≈2412，向上对齐 64 得 2432
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # RoPE 表预生成长度，不等于训练长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 的除零保护
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 底数，取 1e6 让低频维转得更慢，利于长文本
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # Embedding 与 LM Head 共享权重
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)  # 仅推理期开 YaRN，训练期保持关闭
        self.rope_scaling = {
            "beta_fast": 32,  # ramp 上界对应的波长阈值，高于它的维度基本不缩放
            "beta_slow": 1,  # ramp 下界，低于它的维度按 factor 全量缩放
            "factor": 16,  # 频率压缩倍数，等价于把可用上下文放大 16 倍
            "original_max_position_embeddings": 2048,  # 原始训练长度，目标长度超过它才触发插值
            "attention_factor": 1.0,  # 注意力温度补偿，1.0 表示不补
            "type": "yarn"  # 标记插值类型，供外部工具识别
        } if self.inference_rope_scaling else None  # 关闭时置 None，precompute_freqs_cis 走原始 RoPE
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)  # 专家总数，决定总参数量
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每 token 激活几个专家，1 即 top-1 路由
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 单个专家的 FFN 宽度，默认与 Dense 同宽
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 把选中的 topk 权重重新归一化到和为 1
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 负载均衡损失系数，置 0 即关闭该项

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     Mica Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """RMS 归一化：只按均方根缩放，不减均值也不加 bias，比 LayerNorm 少一半统计量。"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()  # nn.Module 的注册机制依赖父类初始化，必须先调
        self.eps = eps  # 防除零，config 实际传 1e-6
        self.weight = nn.Parameter(torch.ones(dim))  # 每维一个可学习缩放，初值 1 相当于恒等变换

    def norm(self, x):
        """对最后一维做均方根归一化，eps 防止输入接近全零时除零。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # rsqrt(mean(x^2)) 即除以均方根，keepdim 保证能广播回原形状

    def forward(self, x):
        """先升到 float32 完成归一化再乘可学习权重，最后转回入参 dtype，保证 BF16 训练下归一化数值稳定。"""
        return (self.weight * self.norm(x.float())).type_as(x)  # float() 升精度做归一化，type_as 再转回 bf16 / fp16

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE 的 cos/sin 表，返回两个 [end, dim] 张量供所有层共享。

    基础频率为 1 / rope_base^(2i/dim)，低维转得快、高维转得慢。传入 rope_scaling 且目标长度超过原始
    训练长度时按 YaRN 插值：先用 beta_fast / beta_slow 反解出需要缩放的维度区间 [low, high]，再在该
    区间内用线性 ramp 把频率平滑压缩 factor 倍——高频维度（负责局部相对位置）几乎不动，低频维度
    （负责全局位置）压缩最多，从而在不重训的情况下外推上下文长度。
    cos/sin 各自 cat 一次是为了配合 rotate_half 的前后半拼接约定；attn_factor 是 YaRN 的注意力温度补偿。
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0  # 基础频率 1/base^(2i/dim)：i 越大波长越长
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        # 一次取出 YaRN 的 5 个参数，缺省值与 MicaConfig 里的 rope_scaling 字典保持一致
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:  # 只有目标长度超过原始训练长度才需要插值，否则原样使用
            # 波长 b 对应的维度下标，由 base^(2i/dim) = orig_max / (2*pi*b) 反解而来
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  # 夹到合法范围，得到需过渡的维度区间
            # 线性 ramp：low 以下为 0（不缩放）、high 以上为 1（全量缩放），中间线性过渡
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)  # 按 ramp 混合原频率与压缩 factor 倍后的频率
    t = torch.arange(end, device=freqs.device)  # 位置下标 0..end-1
    freqs = torch.outer(t, freqs).float()  # 外积得 [end, dim/2]：每个位置 × 每个频率 = 旋转角
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # 前后半重复一次凑成 [end, dim]，与 rotate_half 的切分方式对应
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # sin 表同理
    return freqs_cos, freqs_sin  # 由 MicaModel 注册成 buffer 后全层共用

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把旋转位置编码作用到 q/k：x * cos + rotate_half(x) * sin，等价于对每一对相隔半维的分量做二维旋转。

    q/k 形状为 [B, T, H, head_dim]，cos/sin 形状为 [T, head_dim]，unsqueeze_dim=1 在 T 之后插入
    head 维（变成 [T, 1, head_dim]）以广播到所有 head。RoPE 只改 q/k 不改 v，注意力分数因此只依赖相对位置差。
    """
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)  # 后半维取负搬到前面：(x1, x2) -> (-x2, x1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)  # .to(dtype) 防止 float32 的 cos/sin 把 q 升精度
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)  # k 同样处理；v 不加位置信息
    return q_embed, k_embed  # 形状与输入一致

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA 的 KV 展开：把 [B, T, n_kv_heads, head_dim] 沿 head 维每个 KV head 重复 n_rep 次，对齐 query head 数。

    n_rep == 1 即退化为 MHA，直接返回原张量。KV cache 始终按 n_kv_heads 存储，只有真正参与注意力
    计算时才在这里展开，因此缓存显存不会被 n_rep 放大。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape  # 输入为 [B, T, n_kv_heads, head_dim]
    if n_rep == 1: return x  # MHA 时无需展开，省一次拷贝
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))  # 先 expand 出 n_rep 份视图，再 reshape 合并成 n_kv_heads*n_rep 个头

class Attention(nn.Module):
    """GQA 自注意力：query 用 num_attention_heads 个头，key/value 只用 num_key_value_heads 个头，KV cache 因此小 n_rep 倍。

    q_norm / k_norm 是作用在 head_dim 上的 QK-Norm，用来抑制训练过程中注意力 logits 的数值爆炸。
    self.flash 标记当前 PyTorch 是否提供 scaled_dot_product_attention 且 config 允许启用。
    """
    def __init__(self, config: MicaConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads  # 未指定 KV 头数时退化为 MHA
        self.n_local_heads = config.num_attention_heads  # query 头数（命名沿用上游，此处无张量并行含义）
        self.n_local_kv_heads = self.num_key_value_heads  # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头要被多少个 query 头共享
        self.head_dim = config.head_dim  # 单头维度
        self.is_causal = True  # 恒为因果注意力，两条计算路径都据此加掩码
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)  # [hidden] -> [n_heads * head_dim]
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 只输出 n_kv_heads 份，这是 GQA 省参数与省 cache 的根源
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 同上
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)  # 多头拼接后投回 hidden_size
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # QK-Norm：作用在 head_dim 上，抑制 attention logits 爆炸
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 同上，作用于 k
        self.attn_dropout = nn.Dropout(config.dropout)  # 只在手写路径里对 softmax 后的注意力权重生效
        self.resid_dropout = nn.Dropout(config.dropout)  # 输出投影之后的 dropout
        self.dropout = config.dropout  # 存一份标量，SDPA 路径需要直接传 dropout_p
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn  # 运行时是否具备 SDPA 且 config 允许

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """一次注意力前向，返回 (输出 [B, T, hidden_size], 供下一步复用的 KV cache)。

        流程：qkv 投影 -> 拆多头 -> QK-Norm -> RoPE -> 与 past_key_value 沿时间维拼接 -> 记录 cache
        -> repeat_kv 对齐 head 数。随后二选一：满足条件（有 flash、seq_len>1、无 cache、mask 全 1）时
        走 SDPA 融合算子；否则手写 scores = q·k^T / sqrt(head_dim)，用上三角 -inf 施加因果掩码（只作用
        于最后 seq_len 列，以兼容带 cache 的增量解码），再叠加 padding mask。
        注意 cache 在 repeat_kv 之前保存，存的是未展开的 KV。
        """
        bsz, seq_len, _ = x.shape  # x 为 [B, T, hidden_size]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # 三个线性投影；k/v 的输出宽度比 q 小 n_rep 倍
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 拆成 [B, T, n_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [B, T, n_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 同上
        xq, xk = self.q_norm(xq), self.k_norm(xk)  # QK-Norm 在 RoPE 之前做，先规范化每个头的向量长度
        cos, sin = position_embeddings  # 上层按当前位置切好的 RoPE 表
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 给 q/k 注入位置信息
        if past_key_value is not None:  # 增量解码：把历史 KV 接到前面
            xk = torch.cat([past_key_value[0], xk], dim=1)  # dim=1 是时间维（此时形状仍为 [B, T, H, D]）
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 同上
        past_kv = (xk, xv) if use_cache else None  # 缓存未 repeat 的 KV，比缓存展开后省 n_rep 倍显存
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))  # 统一转成 [B, H, T, D] 并把 KV 展开到 n_heads 份
        # 只有四个条件同时成立才走 SDPA；单 token 解码、带 cache、带 padding mask 三种情况都退回下面的手写路径
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)  # 融合算子内部自己加因果掩码并顺带 dropout
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 手写路径：q·k^T / sqrt(head_dim)，得 [B, H, T, T_kv]
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)  # 只给最后 seq_len 列加上三角 -inf：历史 cache 全可见，新 token 之间保持因果
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9  # padding 位减 1e9，softmax 后概率趋近 0
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv  # softmax 用 float32 提高数值稳定性，再转回原 dtype
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # 回到 [B, T, n_heads * head_dim]
        output = self.resid_dropout(self.o_proj(output))  # 输出投影加 dropout
        return output, past_kv  # past_kv 交由上层逐层收集

class FeedForward(nn.Module):
    """SwiGLU 前馈网络：gate_proj 过激活后与 up_proj 逐元素相乘，再由 down_proj 投回 hidden_size。

    三个矩阵都不带 bias。intermediate_size 缺省取 config 的值，MoE 中的专家会传入更小的宽度。
    """
    def __init__(self, config: MicaConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size  # MoE 专家会传 moe_intermediate_size，Dense 时取 config 默认值
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 门控分支
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 收缩回 hidden_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 值分支，与门控同宽后逐元素相乘
        self.act_fn = ACT2FN[config.hidden_act]  # 按名字取激活，config 默认 silu

    def forward(self, x):
        """down(silu(gate(x)) * up(x))，即 SwiGLU 的门控形式。"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  # 乘法门控比单层 MLP 多一个矩阵，同参数量下效果更好

class MOEFeedForward(nn.Module):
    """稀疏 MoE 前馈：gate 打分后每个 token 只走 num_experts_per_tok 个专家，总参数量增加而单 token 计算量基本不变。"""
    def __init__(self, config: MicaConfig):
        super().__init__()
        self.config = config  # 前向里要反复读 num_experts_per_tok / norm_topk_prob 等字段
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # 路由打分层，输出每个专家的 logit
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])  # num_experts 个独立 FFN，各自宽度为 moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]  # 保留字段，实际激活在每个专家内部

    def forward(self, x):
        """token 级路由的 MoE 前向，输出形状与输入一致，同时把负载均衡损失写入 self.aux_loss。

        实现要点：
        - 先把 [B, T, H] 摊平成 token 维，softmax 后 topk 选专家；norm_topk_prob 为真时对选中的权重重新归一化。
        - 按专家循环，用 mask 取出该专家负责的 token，加权结果 index_add_ 回输出，避免逐 token 分支。
        - 训练时若某个专家一个 token 都没拿到，补一项 0 * 参数和，让它仍留在计算图里；
          否则 DDP 会因该专家没有梯度而报「参数未参与反向」。
        - aux_loss = num_experts * Σ(每个专家被选中的频率 × 其平均路由概率) * router_aux_loss_coef，
          推理时置零。该项由训练脚本加到主损失上，作用是鼓励路由分布均匀。
        """
        batch_size, seq_len, hidden_dim = x.shape  # 记住原形状，最后要还原
        x_flat = x.view(-1, hidden_dim)  # 摊平成 [B*T, H]，路由以 token 为单位
        scores = F.softmax(self.gate(x_flat), dim=-1)  # 每个 token 对全部专家的概率分布 [N, E]
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)  # sorted=False 省一次排序，后续不依赖顺序
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)  # 让选中的 k 个权重和为 1；+1e-20 防除零
        y = torch.zeros_like(x_flat)  # 输出累加器，按 token 位置累加各专家贡献
        for i, expert in enumerate(self.experts):  # 按专家循环而非按 token，循环次数固定为 num_experts
            mask = (topk_idx == i)  # [N, k] 布尔，标出哪些 token 的哪一路选了专家 i
            if mask.any():  # 该专家至少被选中一个 token
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 在 k 维上压一次，得到选中它的 token 下标
                weight = topk_weight[mask].view(-1, 1)  # 与 token_idx 一一对应的路由权重
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))  # 只对这批 token 算 FFN，加权后累加回原位置
            elif self.training:  # 训练时该专家一个 token 都没被选中
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())  # 补一项恒为 0 的假贡献，把专家参数拉进计算图，否则 DDP 报「参数未参与反向」
        if self.training and self.config.router_aux_loss_coef > 0:  # 仅训练且系数为正时才算负载均衡损失
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # 在 token 维求均值，得 [k, num_experts] 的实际选中频率
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef  # 频率 × 平均路由概率求和，分布越不均这项越大
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()  # 推理路径给一个 0 标量，dtype / device 跟随 scores
        return y.view(batch_size, seq_len, hidden_dim)  # 还原成 [B, T, H]；aux_loss 由上层读取

class MicaBlock(nn.Module):
    """一层 Transformer Block：Pre-Norm + GQA Attention + Pre-Norm + FFN，两段各带一次残差。

    mlp 按 config.use_moe 在 Dense FFN 与 MoE FFN 之间二选一，除此之外 Dense 与 MoE 的结构完全一致。
    layer_id 仅为与上游签名保持一致，本实现未使用。
    """
    def __init__(self, layer_id: int, config: MicaConfig):
        super().__init__()
        self.self_attn = Attention(config)  # 注意力子层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 进注意力前的 Pre-Norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 进 FFN 前的 Pre-Norm（命名沿用 Llama）
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # Dense 与 MoE 只在这一行分叉

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """h = x + Attention(RMSNorm(x))，y = h + FFN(RMSNorm(h))，返回 (y, 本层 KV cache)。"""
        residual = hidden_states  # 先存下输入，注意力算完要加回去
        # 注意力吃的是归一化后的输入（Pre-Norm），原输入只给残差用
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # 原地加残差；上一步已产出新张量，所以原地改不会污染输入
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 第二段：再归一化走 FFN，然后加残差
        return hidden_states, present_key_value  # 顺带把本层 KV cache 抛给上层

class MicaModel(nn.Module):
    """主干网络：Embedding + N 层 MicaBlock + 末尾 RMSNorm，不含 LM Head。

    RoPE 的 cos/sin 表在构造时算好并注册为 persistent=False 的 buffer：全模型共享一份，且不写入
    checkpoint（可由 config 重算，也避免把 max_position_embeddings 固化进权重文件）。
    """
    def __init__(self, config: MicaConfig):
        super().__init__()
        self.config = config  # forward 里重算 RoPE 表时还要用
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers  # 常用字段提到实例上
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # token 嵌入表，可能与 lm_head 共享权重
        self.dropout = nn.Dropout(config.dropout)  # 嵌入之后的 dropout
        self.layers = nn.ModuleList([MicaBlock(l, config) for l in range(self.num_hidden_layers)])  # 堆 num_hidden_layers 层；传入的 layer_id 并未被使用
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 末层归一化，接 LM Head 前的最后一步
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)  # 构造时一次算好 RoPE 表
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # persistent=False：随模型搬 device，但不写进 state_dict
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # 同上

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """主干前向，返回 (末层归一化后的 hidden_states, 每层 KV cache 列表, MoE aux_loss 之和)。

        past_key_values 约定为「每层一个 (k, v) 元组」的列表；transformers 新版传入的 Cache 对象
        （带 layers 属性）与该约定不兼容，直接丢弃退化为无缓存。start_pos 由已缓存长度推出，用来在
        共享的 RoPE 表上切出本次 token 对应的位置切片，从而让增量解码拿到正确的绝对位置。
        Dense 模型的 aux_loss 恒为 0 标量，保证返回结构与 MoE 一致。
        """
        batch_size, seq_length = input_ids.shape  # batch_size 本函数并未用到，只靠 seq_length 做位置切片
        if hasattr(past_key_values, 'layers'): past_key_values = None  # 新版 transformers 传的 Cache 对象不合本实现约定，弃用退化为无缓存
        past_key_values = past_key_values or [None] * len(self.layers)  # 统一成每层一个槽位，便于下面 zip
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 已缓存的 token 数，即本次输入的起始位置
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # [B, T] -> [B, T, H]
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:  # 正常表首元素是 cos(0)=1；为 0 说明 buffer 被 meta 初始化清空了
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)  # 用 config 重算一份
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)  # 赋值覆盖 buffer，并搬到与输入同一 device
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])  # 从共享表切出 [start_pos, start_pos+T) 这一段
        presents = []  # 收集每层的 KV cache
        for layer, past_key_value in zip(self.layers, past_key_values):  # 逐层前向，上一层输出就是下一层输入
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)  # use_cache=False 时 present 为 None，列表里存的就是一串 None
        hidden_states = self.norm(hidden_states)  # 末层归一化
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())  # 汇总所有 MoE 层的 aux_loss；初值用 new_zeros，Dense 模型直接得 0 标量
        return hidden_states, presents, aux_loss  # 三元组交给 MicaForCausalLM

class MicaForCausalLM(PreTrainedModel, GenerationMixin):
    """在主干之上接 LM Head 的因果语言模型，负责与 transformers 生态（from_pretrained / save_pretrained / generate）对接。

    tie_word_embeddings 为 True 时 Embedding 与 LM Head 共享同一份权重矩阵，在 vocab_size=6400 的
    小模型上能省下可观比例的参数量。
    """
    config_class = MicaConfig  # 供 from_pretrained 反序列化 config 用
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 告知 transformers 这两个权重共享，存取时不重复处理
    def __init__(self, config: MicaConfig = None):
        self.config = config or MicaConfig()  # 允许不传 config 直接用默认超参
        super().__init__(self.config)  # PreTrainedModel 初始化，必须在建子模块之前
        self.model = MicaModel(self.config)  # 主干网络
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 投到词表的输出头，无 bias
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight  # 直接指向同一张量，两边从此是同一份参数
        self.post_init()  # transformers 的收尾：权重初始化与权重共享校验

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """跑主干再投影到词表，返回 MoeCausalLMOutputWithPast。

        logits_to_keep 为 int 时只对最后若干个位置算 logits（解码时取 1 可省下整段词表投影）。
        传入 labels 会自动做移位交叉熵：logits 去掉最后一位、labels 去掉第一位；ignore_index=-100
        用于屏蔽 padding 以及 SFT 中不计损失的 prompt 段。aux_loss 原样透传，由训练脚本决定是否累加。
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)  # 走主干；返回的 presents 直接覆盖到同名变量上
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # int 时转成 slice(-n, None)；0 得 slice(0, None) 即全取
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # 只对需要的位置做词表投影
        loss = None  # 不传 labels 就只返回 logits
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()  # 移位对齐：用第 t 位预测第 t+1 位
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)  # 摊平成 [N, V] 与 [N]；-100 的位置不计损失
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)  # aux_loss 单独一个字段，是否加进总损失由训练脚本决定

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """自实现的自回归采样循环，替代 GenerationMixin.generate，短到可以逐行核对。

        每步依次：按 past_len 只把新增 token 喂进 forward -> 取最后一位 logits 除以 temperature ->
        repetition_penalty 对已出现过的 token 打折（正 logits 除、负 logits 乘）-> top_k 截断 ->
        top_p 核采样（累积概率越过阈值的尾部置 -inf，且保留第一个以免全被屏蔽）->
        multinomial 采样或 argmax 贪心。
        num_return_sequences 通过复制 batch 实现；finished 记录每条序列是否已经吐出 EOS，已结束的
        序列后续强制补 EOS，全部结束则提前退出。streamer 用于逐 token 回吐；return_kv=True 时
        额外返回 KV cache 以便后续续写。
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)  # inputs 与 input_ids 两种传法都兼容；按 batch 复制实现多条返回
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None  # mask 跟着一起复制
        past_key_values = kwargs.pop("past_key_values", None)  # 支持传入已有 cache 接着往下写
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)  # 每条序列是否已经吐过 EOS
        if streamer: streamer.put(input_ids.cpu())  # 先把 prompt 交给 streamer
        for _ in range(max_new_tokens):  # 最多生成这么多个新 token
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0  # 已缓存长度，决定下一行只喂新增部分
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)  # 首步喂全部 prompt，之后每步只喂 1 个 token
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None  # 新 token 一律可见，mask 补 1
            logits = outputs.logits[:, -1, :] / temperature  # 只要最后一位；temperature 越小分布越尖
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):  # 逐条序列处理，batch 大时这里是瓶颈
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)  # 出现过的 token：正 logit 除、负 logit 乘，两种情况都是降低其概率
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')  # 低于第 k 大者全部屏蔽，只留概率最高的 k 个
            if top_p < 1.0:  # 核采样：只保留累积概率达到 top_p 的高概率候选集
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 按 logit 降序排
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p  # 累积概率越过阈值的尾部标记待屏蔽
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0  # 整体右移一位并强制保留第一名，防止最高概率 token 也被屏蔽
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')  # 把排序空间的 mask 散射回原词表下标再屏蔽
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)  # do_sample 决定采样还是贪心
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)  # 已结束的序列强制补 EOS，保持 batch 长度对齐
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # 接到序列尾部
            past_key_values = outputs.past_key_values if use_cache else None  # 不用 cache 时下一轮重算全序列
            if streamer: streamer.put(next_token.cpu())  # 逐 token 回吐
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)  # 累积标记，一旦结束不会再翻回
                if finished.all(): break  # 全部结束就提前退出
        if streamer: streamer.end()  # 通知 streamer 收尾
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}  # 需要续写时把 cache 一并返回
        return input_ids  # 默认只返回 [B*num_return_sequences, prompt + 新生成] 的 token id
