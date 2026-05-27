---
type: concept
domain: 基础架构
status: seed
---

# QKV Bias

> [!note]
> 注意力层中 Query、Key、Value 三个线性投影各自携带的可学习偏置项。在标准权重矩阵之外增加平移自由度，让 Q/K/V 的输出不必严格经过原点。

---

## 形式

带 bias 的投影：

$$Q = XW_Q + b_Q, \quad K = XW_K + b_K, \quad V = XW_V + b_V$$

不带 bias 则退化为纯线性变换 $Q = XW_Q$。

## 作用

bias 在两个层面提供额外可学习自由度：

- **注意力打分层**：$b_Q$ 和 $b_K$ 直接影响 $QK^\top$ 的分数分布，改变模型"关注什么"
- **输出表示层**：$b_V$ 影响加权求和后的最终输出

不是简单的"加一个常数"，而是在注意力的权重分配和内容表达两个维度上都增加了可调参数量。

> [!warning]
> - QKV bias 是模型容量的一部分，不是正则化手段——它增加的是表达能力，不是约束
> - 部分实现默认不使用 QKV bias（如某些 Transformer 变体的简化版本），这不影响机制正确性，只是少了一些表示灵活性

---

## 关联

- 属于：[Attention](Attention.md)
- 相关：[Attention](Attention.md) [Transformer](Transformer.md)
- 用于：

## 相关概念

- [Attention](Attention.md)
