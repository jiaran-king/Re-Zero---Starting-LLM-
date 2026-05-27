---
type: concept
domain: 基础架构
status: active
---
# Attention

> [!note]
> Transformer 的核心计算机制。通过 Query-Key-Value 三元组计算序列中任意两个位置之间的相关性权重，再对 Value 做加权求和，实现信息聚合。标准 self-attention 公式：$\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d}) V$。它是 Transformer 替代 RNN/CNN 的关键创新——任意位置之间一跳直达，路径长度 $O(1)$。

---

## 标准形式

给定输入 $X \in \mathbb{R}^{S \times D}$：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

Multi-Head Attention 将 $D$ 维拆成 $H$ 个头并行计算，每个头学习不同的注意力模式，最后拼接输出。

## 变体谱系

Attention 机制在 LLM 发展中演化出多条优化路线：

| 优化方向 | 目标 | 代表技术 |
|---------|------|---------|
| **KV Cache 压缩** | 减少推理时 K/V 缓存 | [MHA变体：MQA与GQA](MHA%E5%8F%98%E4%BD%93%EF%BC%9AMQA%E4%B8%8EGQA.md)、[MLA](MLA.md) |
| **计算量降低** | 将 $O(S^2)$ 降到近似线性 | [稀疏注意力](%E7%A8%80%E7%96%8F%E6%B3%A8%E6%84%8F%E5%8A%9B.md) |
| **IO 优化** | 减少 HBM 读写瓶颈 | [Flash Attention](Flash%20Attention.md) |
| **位置信息注入** | 为 attention 加入顺序信号 | [位置编码](%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) |
| **投影增强** | 增加 Q/K/V 表示灵活性 | [QKV Bias](QKV%20Bias.md) |

> [!tip]
> 这些变体通常**组合使用**而非互斥。一个现代 LLM 可能同时采用 GQA + RoPE + Flash Attention + 滑动窗口 attention。

---

> [!warning]
> - "Attention" 有时指整个机制族（含所有变体），有时特指标准 scaled dot-product attention，需根据上下文区分
> - 复杂度 $O(S^2)$ 指的是标准稠密 attention；稀疏变体可降到 $O(Sw)$ 或更低
> - $\sqrt{d}$ 缩放因子防止点积值过大导致 softmax 饱和（来自原始论文）

---

## 关联

- 属于：[Transformer](Transformer.md)
- 相关：[MHA变体：MQA与GQA](MHA%E5%8F%98%E4%BD%93%EF%BC%9AMQA%E4%B8%8EGQA.md) [MLA](MLA.md) [Flash Attention](Flash%20Attention.md) [稀疏注意力](%E7%A8%80%E7%96%8F%E6%B3%A8%E6%84%8F%E5%8A%9B.md) [位置编码](%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) [QKV Bias](QKV%20Bias.md)
- 用于：大模型推理实验

## 相关概念

- [Transformer](Transformer.md)
- [MHA变体：MQA与GQA](MHA%E5%8F%98%E4%BD%93%EF%BC%9AMQA%E4%B8%8EGQA.md)
- [MLA](MLA.md)
- [Flash Attention](Flash%20Attention.md)
