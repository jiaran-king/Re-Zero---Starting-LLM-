---
type: concept
domain: 推理与部署
status: active
---

# MQA 与 GQA

> [!note]
> 两种通过减少 KV 头数来压缩 KV Cache 的注意力变体。MQA 让所有 Query 头共享同一组 K/V，GQA 则按组共享——在缓存节省和表达能力之间做折中。LLaMA-2/3 采用 GQA，是当前主流推理优化方案之一。

---

## 1. 从 MHA 说起

标准 Multi-Head Attention（MHA）中，每个头都有独立的 Q、K、V：

$$Q, K, V \in \mathbb{R}^{B \times H \times S \times d}$$

$H$ 个头意味着 $H$ 份 K 和 V。推理时每层都要缓存所有历史 token 的 K/V，KV Cache 大小与头数成正比——这是 MHA 在长序列推理中的主要瓶颈。

MQA 和 GQA 的出发点相同：**大部分 KV Cache 的冗余来自多头 K/V 的重复性，能不能减少独立 KV 的份数？**

## 2. MQA：全共享

Multi-Query Attention 的做法很直接——Q 保持多头不变，K 和 V 压缩到只有 1 份：

$$Q \in \mathbb{R}^{B \times H \times S \times d}, \qquad K, V \in \mathbb{R}^{B \times 1 \times S \times d}$$

所有头共享同一组 K 和 V 来计算注意力分数。KV Cache 直接缩减为原来的 $1/H$。

代价也很明确：不同头无法学习差异化的注意力模式，表示多样性下降。实际效果通常略低于完整 MHA，尤其在需要复杂表示的任务上。

## 3. GQA：分组共享

GQA（Grouped-Query Attention）可以看作是 **MHA 和 MQA 之间的折中方案**。

它的核心思想是：**每个注意力头仍然保留独立的 Query，但 Key 和 Value 不再是每个头各自一份，而是把多个头分组，每组共享一份 K/V。**

设 batch size 为 $B$、序列长度为 $S$、query 头数为 $H$、kv 头数为 $G$、单头维度为 $d$（总隐藏维度 $D = H \cdot d$），则：

$$Q \in \mathbb{R}^{B \times H \times S \times d}, \qquad K, V \in \mathbb{R}^{B \times G \times S \times d}$$

其中 $1 < G < H$。Q 仍然是完整的多头结构，而 K/V 的头数被压缩成了 $G$ 个。每个 kv head 会服务一组 query heads，因此叫做 Grouped-Query Attention。

### 原理理解

标准 MHA 中，每个 query head 都有自己独立的 K/V，表达能力最强但 KV Cache 最大。

MQA 中所有 query heads 共享同一组 K/V，KV Cache 最小但压缩过于激进，容易损失表示能力。

GQA 介于两者之间——保留 $G$ 组 K/V，让每组服务若干个 query heads。既能减少 KV Cache，又能比 MQA 保留更多头间差异性。

如果每个 kv head 对应 $\frac{H}{G}$ 个 query heads，推理时 KV Cache 的大小大约缩小为原来的 $\frac{G}{H}$。

当 $g = H$ 时退化为 MHA，$g = 1$ 时退化为 MQA。LLaMA-2 70B（$g=8$）和 LLaMA-3 系列均采用 GQA。研究表明，适中的 $g$ 值（如 $H/8$ 左右）可以在接近 MHA 效果的同时获得接近 MQA 的推理速度。

### 优点

1. **推理更高效**——缓存的 K/V 头数减少，KV Cache 显存占用下降，读写带宽压力也更小，生成速度通常比 MHA 更快
2. **效果通常比 MQA 更稳**——仍然保留了多组 K/V，比单组更能维持不同注意力模式的多样性，效果通常更接近标准 MHA
3. **工程上很实用**——本质上只是在 MHA 的基础上减少 `num_kv_heads`，实现简单，部署友好

### 缺点

1. 虽然比 MQA 保留了更多表达能力，但仍不是完整的 MHA，理论上表示能力弱于每头独立 K/V 的标准多头注意力
2. 需要处理 **query heads 与 kv heads 的映射关系**——实现时通常要把 K/V 按组扩展或广播到 query head 维度，逻辑比最标准的 MHA 稍复杂一些

## 4. 三者的统一视角

三者本质上都可以统一为一个参数 $H_{kv}$，即 **K/V 的头数**：

| | $H_{kv}$ | 含义 |
|:--:|:--:|:--|
| MHA | $H_{kv} = H$ | 每头独立 K/V |
| MQA | $H_{kv} = 1$ | 全部共享 1 份 K/V |
| GQA | $1 < H_{kv} < H$ | 分组共享 |

统一写法：

$$Q \in \mathbb{R}^{B \times H \times S \times d}, \qquad K, V \in \mathbb{R}^{B \times H_{kv} \times S \times d}$$

三者的本质区别不在 attention 公式本身，而在于 **K/V 到底有多少个 heads**。

## 5. 对比总结

| | MHA | GQA | MQA |
|:--:|:--:|:--:|:--:|
| KV 头数 $H_{kv}$ | $H$ | $G$（$1<G<H$） | 1 |
| KV Cache 大小 | 基准 | 约 $G/H$ | 约 $1/H$ |
| 表达能力 | 最强 | 接近 MHA | 较弱 |
| 推理速度 | 基准 | 接近 MQA | 最快 |
| 代表模型 | 多数预训练模型 | LLaMA-2/3 | PaLM, GLM |

> [!warning]
> - MQA/GQA 减少的是 **KV Cache 的存储**，不是 attention 计算量本身——Q 仍然是多头的，attention score 矩阵大小不变
> - "GQA 效果等于 MHA" 是近似说法，复杂任务上仍有可测量的差距
> - 选择 $G$ 值时需要在效果和速度之间权衡，没有万能最优值

---

## 6. PyTorch 实现

下面是一份统一风格的实现，只保留核心 attention 结构（不含 mask、RoPE、KV cache），通过 `num_kv_heads` 一个参数同时覆盖三种模式。

```python
import math
import torch
import torch.nn as nn


class BaseAttention(nn.Module):
    """统一的 MHA / MQA / GQA 实现，差异仅在于 num_kv_heads"""

    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        if d_model % num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        # Q 始终投影到 H * d
        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        # K/V 投影到 H_kv * d
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H*d] -> [B, H, S, d]
        B, S, _ = x.shape
        return x.view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H_kv*d] -> [B, H_kv, S, d]
        B, S, _ = x.shape
        return x.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        # kv: [B, H_kv, S, d] -> [B, H, S, d]
        # MHA 时 H_kv = H，不需要扩展
        # MQA / GQA 时，把每个 kv head 复制给对应的一组 query heads
        if self.num_kv_heads == self.num_q_heads:
            return kv
        return kv.repeat_interleave(self.group_size, dim=1)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, S, D]
        q = self._shape_q(self.q_proj(x))      # [B, H, S, d]
        k = self._shape_kv(self.k_proj(x))     # [B, H_kv, S, d]
        v = self._shape_kv(self.v_proj(x))     # [B, H_kv, S, d]

        k = self._expand_kv(k)                 # [B, H, S, d]
        v = self._expand_kv(v)                 # [B, H, S, d]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, H, S, S]

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)            # [B, H, S, d]

        out = out.transpose(1, 2).contiguous().view(
            x.size(0), x.size(1), self.d_model)
        out = self.o_proj(out)                 # [B, S, D]
        return out
```

### 三种实例

```python
# MHA：num_kv_heads = num_q_heads，每头独立 K/V
class MHA(BaseAttention):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__(d_model=d_model,
                         num_q_heads=num_heads,
                         num_kv_heads=num_heads)


# MQA：num_kv_heads = 1，所有头共享同一组 K/V
class MQA(BaseAttention):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__(d_model=d_model,
                         num_q_heads=num_heads,
                         num_kv_heads=1)


# GQA：num_kv_heads = G，按组共享 K/V
class GQA(BaseAttention):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__(d_model=d_model,
                         num_q_heads=num_q_heads,
                         num_kv_heads=num_kv_heads)


# 使用示例
mha = MHA(d_model=512, num_heads=8)           # K/V 各 8 个 head
mqa = MQA(d_model=512, num_heads=8)           # K/V 只有 1 个 head
gqa = GQA(d_model=512, num_q_heads=8, num_kv_heads=2)  # 2 个 kv head 服务 8 个 q head
```

### 代码层面的本质区别

从实现角度看，MHA、MQA、GQA 的差异几乎只在 **K/V 投影输出多少个 heads** 这一件事上：

```python
num_kv_heads = num_heads      # MHA
num_kv_heads = 1              # MQA
num_kv_heads = G              # GQA
```

关键操作在 `_expand_kv` 方法中：当 `num_kv_heads < num_q_heads` 时，通过 `repeat_interleave` 将每个 kv head 按组复制给对应的 query heads，使后续的 attention 计算可以统一执行。

---

## 关联

- 属于：[推理优化](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md)
- 相关：[Attention](Attention.md) [KV Cache](KV%20Cache.md) [Transformer](Transformer.md) [MLA](MLA.md) [Flash Attention](Flash%20Attention.md)
- 用于：大模型推理实验

## 相关概念

- [KV Cache](KV%20Cache.md)
- [MLA](MLA.md)
- [Attention](Attention.md)
