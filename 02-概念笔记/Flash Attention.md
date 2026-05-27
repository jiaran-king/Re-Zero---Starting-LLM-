---
type: concept
domain: 推理与部署
status: active
---
# Flash Attention

> [!note]
> 一种 IO 最优的 Attention 实现算法，通过 Online Softmax + 分块计算（Tiling）避免物化 $N \times N$ 的中间矩阵，将显存占用从 $O(N^2)$ 降到 $O(N)$，HBM IO 达到理论下界。由 Tri Dao 等人提出（2022），是当前大模型推理和训练的事实标准实现。

---

## 核心直觉

标准 Attention 的做法：先算完所有两两相似度存成 $N \times N$ 矩阵 → 再做 softmax → 再乘 V。每一步都要把大矩阵写入 HBM 再读出来。

Flash Attention 的做法：只算当前 tile 需要的那部分相似度 → 立刻合并进 softmax 和输出 → 算完就丢，不保存完整 $S$ 或 $P$。

节省 IO 的根本原因：**避免物化大规模中间矩阵 + 利用片上存储（SRAM）做分块复用**。

---

## 1. 标准 Attention 回顾

给定 $Q, K, V \in \mathbb{R}^{N \times d}$：

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

对第 $i$ 行展开：

$$O_i = \frac{\sum_{j=1}^{N} e^{s_{ij}} V_j}{\sum_{j=1}^{N} e^{s_{ij}}}, \quad s_{ij} = \frac{Q_i K_j^T}{\sqrt{d}}$$

### 标准实现的瓶颈

```
1. S = QK^T / sqrt(d)   → 写入 HBM，O(N²)
2. P = softmax(S)        → 读 S 写 P 到 HBM，O(N²)
3. O = PV                → 读 P 和 V，写 O
```

中间矩阵 $S$ 和 $P$ 都是 $N \times N$，需要 $O(N^2)$ 显存且反复读写 HBM。

---

## 2. Online Softmax：动态更新的数学基础

### 2.1 Safe Softmax

标准 softmax 会数值溢出，实践中减去最大值：

$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j$$

需要两遍扫描：先求 $m$，再算指数和。

### 2.2 单遍扫描的 Online Softmax

维护两个量，边扫描边更新：

- $m^{(k)} = \max(x_1, \ldots, x_k)$：当前最大值
- $\ell^{(k)} = \sum_{j=1}^{k} e^{x_j - m^{(k)}}$：当前指数和

当第 $k+1$ 个元素到来时：

**更新 max：**

$$m^{(k+1)} = \max(m^{(k)},\; x_{k+1})$$

**修正历史 + 加入新元素：**

$$\ell^{(k+1)} = \ell^{(k)} \cdot e^{m^{(k)} - m^{(k+1)}} + e^{x_{k+1} - m^{(k+1)}}$$

其中 $e^{m^{(k)} - m^{(k+1)}}$ 是修正因子——当 max 没变时等于 1，历史量无需调整。

### 2.3 推广到 Attention 输出向量

定义未归一化加权和 $a^{(k)} = \sum_{j=1}^{k} e^{x_j - m^{(k)}} V_j$，更新规则：

$$a^{(k+1)} = a^{(k)} \cdot e^{m^{(k)} - m^{(k+1)}} + e^{x_{k+1} - m^{(k+1)}} V_{k+1}$$

最终输出 $O_i = a^{(N)} / \ell^{(N)}$。整个过程只需维护 $(m, \ell, a)$ 三个量，不需要存储完整的 $N \times N$ 矩阵。

---

## 3. 分块策略（Tiling）

### 3.1 切分方式

沿序列维度切分：
- $K, V$ 切为 $T_c = \lceil N / B_c \rceil$ 块，每块 $B_c \times d$
- $Q$ 切为 $T_r = \lceil N / B_r \rceil$ 块，每块 $B_r \times d$

块大小受 SRAM 容量 $M$ 约束。每次 SRAM 需同时容纳：$K_j$ 块、$V_j$ 块、$Q_i$ 块、局部 score $S_{ij}$（$B_r \times B_c$）、输出累积 $O_i$、统计量 $m_i, \ell_i$。

论文典型选择：$B_c = \lceil M / (4d) \rceil$，$B_r = \min(\lceil M / (4d) \rceil, d)$。

### 3.2 算法流程

```
初始化：每个 Q 块 i，设 m_i = -∞, ℓ_i = 0, O_i = 0（HBM）

外层循环：for j = 1 to T_c（遍历 K/V 块）
    加载 K_j, V_j 到 SRAM
    内层循环：for i = 1 to T_r（遍历 Q 块）
        加载 Q_i, O_i, m_i, ℓ_i 到 SRAM

        ① 局部 score：S_ij = Q_i @ K_j^T / sqrt(d)     ∈ R^{B_r × B_c}
        ② 局部统计：m̃_ij = rowmax(S_ij)
                    P̃_ij = exp(S_ij - m̃_ij)
                    ℓ̃_ij = rowsum(P̃_ij)
        ③ 更新全局统计（Online Softmax 块级版）：
           m_i^new = max(m_i, m̃_ij)
           ℓ_i^new = ℓ_i · exp(m_i - m_i^new) + ℓ̃_ij · exp(m̃_ij - m_i^new)
        ④ 更新输出累积：
           O_i^new = O_i · (ℓ_i · exp(m_i - m_i^new) / ℓ_i^new)
                   + P̃_ij · exp(m̃_ij - m_i^new) / ℓ_i^new @ V_j
        ⑤ 写回 O_i^new, m_i^new, ℓ_i^new 到 HBM
```

### 3.3 第④步的严格推导

设在处理第 $j$ 块前已累积状态 $(O_i^{(j-1)}, m_i^{(j-1)}, \ell_i^{(j-1)})$，处理后需变为：

$$O_i^{(j)} = \frac{\sum_{t=1}^{j} \sum_k e^{s_{ik} - m_i^{(j)}} V_k}{\ell_i^{(j)}}$$

分子拆为历史部分 + 新块贡献：

$$\text{分子} = \underbrace{e^{m_i^{(j-1)} - m_i^{(j)}} \cdot \ell_i^{(j-1)} \cdot O_i^{(j-1)}}_{\text{历史修正}} + \underbrace{e^{\tilde{m}_{ij} - m_i^{(j)}} \cdot \tilde{P}_{ij} V_j}_{\text{新块贡献}}$$

这正是算法第④步的公式来源。

---

## 4. Flash Attention 2 的改进

### 4.1 延迟 rescaling

V1 每步都除以 $\ell_i^{(j)}$ 做归一化；V2 只维护未归一化的累积量 $\tilde{O}_i$，最后一次性除以 $\ell_i^{(T_c)}$。减少了每步的非矩阵乘操作（GPU 上较慢）。

### 4.2 循环顺序调换

| | V1 | V2 |
|--|----|----|
| 外层 | 遍历 K/V 块 | 遍历 Q 块 |
| 内层 | 遍历 Q 块 | 遍历 K/V 块 |

V2 的好处：每个 Q 块的 $O_i, m_i, \ell_i$ 在整个内层循环中**常驻 SRAM**，只在外层切换时读写 HBM 一次。V1 每次内层迭代都要加载/写回这些量。

直观理解：V1 是"列向施工"——竖着一溜一溜地画，所有颜料盘要来回倒腾；V2 是"行向施工"——顺着行方向走完全程，几个 Q 一次搞定再写回。

### 4.3 并行化改进

- V1：在 batch × heads 维度并行（并行度 = $B \times H$）
- V2：额外在 Q 序列块维度并行（并行度 = $B \times H \times T_r$），更充分利用 GPU SM

---

## 5. 复杂度分析

### 5.1 HBM IO 对比

| 方法 | HBM IO | 显存峰值 |
|------|--------|---------|
| 标准 Attention | $O(Nd + N^2)$ | $O(N^2)$（存 S 和 P） |
| Flash Attention | $O(N^2 d^2 / M)$ | $O(N)$（只存 $O, m, \ell$） |

典型值：$M = 20\text{MB}$，$d = 128$ → 加速比约 **600×**（IO 角度）。实际加速没这么夸张（计算也是瓶颈之一），但 **IO 不再是主要瓶颈**。

### 5.2 FLOPs 不变

分块只改变计算顺序，不增加也不减少计算量。FLOPs 仍为 $O(N^2 d)$。

### 5.3 IO 下界证明（Theorem 2）

论文证明了 SRAM 大小为 $M$ 时，任何精确计算 Attention 的算法 HBM IO 下界为：

$$\Omega\left(\frac{N^2 d^2}{M}\right)$$

Flash Attention 达到这个下界，因此是 **IO 最优** 的。

### 5.4 为什么显存降到 $O(N)$

关键在于 **Online Softmax 打破了 softmax 必须看到完整行的依赖**——通过维护 $(m, \ell)$ 两个标量逐块更新，永远只需要当前块的局部 $S_{ij}$（$B_r \times B_c$ 大小），算完即弃。任意时刻 HBM 中只有 $Q, K, V, O, m, \ell$，总计 $O(Nd) = O(N)$。

训练时反向传播需要 $P$ 矩阵算梯度，Flash Attention 的策略是**不保存 $P$，反向时重计算**——以重计算换显存，训练时显存仍保持 $O(N)$。

---

## 6. 数值等价性

Flash Attention 与标准 Attention **数学上严格相等**（非近似）。处理完所有块后：

$$m_i = \max_j s_{ij}, \quad \ell_i = \sum_j e^{s_{ij} - m_i}, \quad O_i = \frac{\sum_j e^{s_{ij} - m_i} V_j}{\ell_i} = \text{softmax}(s_i) \cdot V$$

分块和 online 更新只改变了**计算顺序**，不改变最终结果。

---

## 总结图示

```
标准 Attention:
  Q,K ──→ [S = QK^T] ──→ [P = softmax(S)] ──→ [O = PV]
             ↕ HBM            ↕ HBM              ↕ HBM
          N×N 矩阵         N×N 矩阵           读 N×N

Flash Attention:
  对每个 (Q块_i, K块_j, V块_j):
    SRAM 内: S_小块 → softmax → 累积到 O_i
    只读写 O(Bd) 大小的块到 HBM
    ⇒ 永远不 materialize N×N 矩阵
```

> [!warning]
> - Flash Attention 加速的是 **IO 而非 FLOPs**——计算量不变，但消除了 HBM 瓶颈使 GPU 能跑满算力
> - "Flash Attention 让模型支持更长序列"是不准确的说法——它减少的是显存占用和 IO 开销，序列长度上限仍受模型架构本身约束
> - 训练时的"重计算"策略以额外的 FLOPs 换取显存节省，前向传播无此开销

---

## 关联

- 属于：[推理优化](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md)
- 相关：[Attention](Attention.md) [Transformer](Transformer.md) [KV Cache](KV%20Cache.md) [MHA变体：MQA与GQA](MHA%E5%8F%98%E4%BD%93%EF%BC%9AMQA%E4%B8%8EGQA.md) vLLM
- 用于：大模型推理实验

## 相关概念

- [Attention](Attention.md)
- [KV Cache](KV%20Cache.md)
- [Transformer](Transformer.md)
