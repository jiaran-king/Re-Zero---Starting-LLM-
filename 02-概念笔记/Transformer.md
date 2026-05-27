---
type: concept
domain: 基础架构
status: active
---
# Transformer

> [!note]
> 基于 self-attention 的序列建模架构，由 Vaswani 等人于 2017 年提出（"Attention Is All You Need"）。它摒弃了 RNN/CNN 的递归或卷积结构，完全依赖注意力机制捕捉序列中的依赖关系，成为现代大语言模型的统一基础架构。当前主流 LLM（GPT、LLaMA、Qwen、DeepSeek 等）均基于 Transformer 的 Decoder-only 变体。

---

## 核心组件

一个 Transformer 层由两个主要子层组成：

| 子层 | 作用 | 计算 |
|------|------|------|
| **Multi-Head Self-Attention** | 让每个位置关注序列中其他位置，建模 token 之间的依赖关系 | $\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d}) V$ |
| **Feed-Forward Network (FFN)** | 对每个位置独立做非线性变换，提供表达能力 | $\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$ |

每个子层后接残差连接和 Layer Norm：

$$\text{Output} = \text{LN}(x + \text{SubLayer}(x))$$

---

## 两种主要结构

### Encoder–Decoder

传统 Transformer 的完整结构，用于翻译、摘要等 seq2seq 任务。

**Encoder**：由多层 self-attention + FFN 组成，$Q, K, V$ 均来自输入序列本身，负责将输入编码为包含全局语义的表示。

**Decoder**：由三层组成：
- **Masked Self-Attention**：$Q, K, V$ 来自 decoder 输入，加 causal mask 使每个位置只能看到自己和之前的 token
- **Cross-Attention**：$Q$ 来自 Decoder，$K, V$ 来自 Encoder 输出，让 Decoder 在生成时读取输入信息
- **FFN**

核心作用：**先理解输入，再条件化地生成输出**。适合输入和输出边界清晰的任务。

### Decoder-only

现代大语言模型最主流的结构。去掉 Encoder 和 cross-attention，只保留带因果掩码的 self-attention 与 FFN，通过自回归方式完成生成。

每层的 $Q, K, V$ 都来自当前层输入，加 causal mask 后每个位置只能访问自身及之前的 token。训练目标通常是 next-token prediction——给定前文预测下一个 token。

核心优势：把各种任务统一为"给定前缀，继续生成后文"，结构简单、扩展性强、适合开放式文本生成、对话和代码生成。GPT 系列、LLaMA、Qwen、DeepSeek 均采用此结构。

### 两者对比

| | Encoder–Decoder | Decoder-only |
|--|-----------------|--------------|
| 注意力类型 | Self-attn + Cross-attn | Masked self-attn |
| 典型任务 | 翻译、摘要等 seq2seq | 通用文本生成 |
| 信息流 | 输入 → 编码 → 条件解码 | 前缀 → 自回归续写 |
| 结构复杂度 | 较高（双塔 + 交叉注意力） | 低（单塔） |
| 主流程度 | 特定领域仍使用 | 当前 LLM 绝对主流 |

---

## 为什么 Transformer 取代了 RNN

| 维度 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 长距离依赖 | 需逐步传播，梯度消失/爆炸 | 任意位置之间直接 attention，一跳直达 |
| 并行化 | 序列步骤串行，无法并行 | 所有位置同时计算 |
| 路径长度 | $O(N)$ | $O(1)$（任意两点间） |
| 固定表示 | 整个序列压缩到一个状态 | 每个位置保留独立上下文化表示 |

Transformer 的 $O(1)$ 路径长度是其成功的关键理论优势——信息在两层之间可以直接从任意位置传到任意另一位置，不受序列长度影响。

---

> [!warning]
> - "Transformer" 有时被泛指为整个模型架构（含 embedding、position encoding、head 等），有时特指 attention 层本身，需根据上下文区分
> - Decoder-only 不是"去掉了 Encoder 的功能"——它通过足够的层数和自回归机制同样能学习到强大的表示能力
> - Pre-LN（Layer Norm 在 sublayer 之前）比 Post-LN 训练更稳定，是现代实现的主流选择

---

## 关联

- 属于：[基础架构](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84.md)
- 相关：[Attention](Attention.md) [位置编码](%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) [MHA变体：MQA与GQA](MHA%E5%8F%98%E4%BD%93%EF%BC%9AMQA%E4%B8%8EGQA.md) [MLA](MLA.md) [Flash Attention](Flash%20Attention.md) [KV Cache](KV%20Cache.md)
- 用于：所有 LLM 相关笔记的上位概念

## 相关概念

- [Attention](Attention.md)
- [位置编码](%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
- [MHA变体：MQA与GQA](MHA%E5%8F%98%E4%BD%93%EF%BC%9AMQA%E4%B8%8EGQA.md)
- [MLA](MLA.md)
