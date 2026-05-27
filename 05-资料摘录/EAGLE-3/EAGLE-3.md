---
type: paper-note
status: summarized
domain: LLM inference
tags:
  - speculative-decoding
  - EAGLE
  - inference-acceleration
  - draft-model
  - training-time-test
aliases:
  - EAGLE-3
  - Training-Time Test
  - EAGLE-3论文笔记
source: https://arxiv.org/abs/2503.01840
arxiv: 2503.01840v3
canvas:
  - 08-图片/eagle3_mainflow.canvas
---
# EAGLE-3：通过 Training-Time Test 扩展大模型推理加速

> [!info] 原始来源
> [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)

> [!note] 一句话
> EAGLE-3 的核心不是换了一个更大的 draft model，而是把 EAGLE 原来“预测 target model hidden feature”的中间目标去掉，改成直接优化 draft token；同时用 training-time test 让 draft model 在训练时就经历多步自回归 draft，从而既释放表达能力，又避免第二步以后因为输入混入自身输出而崩掉。

## 0 阅读地图

![eagle3_mainflow](../../08-%E5%9B%BE%E7%89%87/canvas-preview/eagle3_mainflow.svg)

[打开原始 Canvas](../../08-%E5%9B%BE%E7%89%87/eagle3_mainflow.canvas)

这张 Canvas 只画一条主线：从自回归解码瓶颈，到 speculative sampling，再到 EAGLE 的 feature prediction constraint，最后落到 EAGLE-3 的 training-time test、multi-layer feature fusion 和加速结果。正文可以按这条链路顺序阅读。

## 1 论文定位

论文：**EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test**  
来源：[arXiv:2503.01840](https://arxiv.org/abs/2503.01840)  
作者：Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang  
版本：arXiv:2503.01840v3，2025-04-23

这篇论文解决的是 **LLM 推理阶段的解码加速**，具体属于 speculative sampling / speculative decoding 路线。

论文的出发点可以概括为三个观察：

1. 普通自回归解码每生成一个 token 都要调用一次 target model，decode 阶段慢且贵。
2. EAGLE/EAGLE-2 通过复用 target model 的 top-layer feature，让 draft model 更接近 target model，已经比 vanilla speculative sampling 更强。
3. 但作者发现：给 EAGLE 扩更多训练数据，收益很有限。瓶颈不是数据不够，而是 **feature prediction constraint 限制了 draft model 的表达能力**。

所以 EAGLE-3 的目标是：在保持 speculative sampling lossless 性质的前提下，让 draft model 更能从训练数据扩展中受益，并提高多步 draft token 的接受率。

---

## 2 背景：Speculative Sampling 为什么能加速

普通 LLM 解码是自回归的：

```text
prefix -> target model forward -> 1 token
prefix + 1 token -> target model forward -> next token
```

每个 token 都要访问一次大模型参数。Speculative sampling 的思路是：

```text
便宜 draft model 先猜多个 token
昂贵 target model 一次 forward 并行验证这些 token
```

如果当前前缀是：

```text
How can
```

draft model 先生成：

```text
I help you today
```

target model 一次 forward 可以并行计算这些位置的 token probability，然后从前到后判断哪些 token 可以接受。

在采样场景下，draft 分布记为 $\hat p$，target 分布记为 $p$。对 draft token $\hat t$，接受概率是：

$$
\min\left(1, \frac{p(\hat t)}{\hat p(\hat t)}\right)
$$

如果 token 被拒绝，则从修正分布中采样：

$$
\mathrm{norm}(\max(0, p - \hat p))
$$

这个接受-拒绝规则保证最终生成分布与直接使用 target model 自回归采样一致，所以 speculative sampling 是 **lossless acceleration**：加速不改变目标模型输出分布。

> [!tip] 加速来自哪里
> Transformer 在给定完整 token 序列时，可以用 causal mask 并行计算多个位置的 next-token distribution。draft model 先把未来 token 猜出来，target model 才能一次性验证多个位置。

---

## 3 EAGLE/EAGLE-2 的基线思想

Vanilla speculative sampling 通常用一个独立小模型做 draft model。问题是：小模型和 target model 分布差距可能很大，导致 draft token 很快被拒绝。

EAGLE 的改法是：**不只看 token，而是复用 target model 已经算出的内部 feature。**

EAGLE 使用 target model 的 top-layer feature，也就是 LM head 前面的 hidden state。它的流程是：

```text
target model top-layer feature f
        ↓
EAGLE draft model 预测下一个 feature f_hat
        ↓
target model LM head
        ↓
draft token distribution
```

也就是说，EAGLE 的 draft model 不是直接预测 token，而是做 feature-level autoregression：

$$
f_1, f_2, \ldots, f_t \rightarrow \hat f_{t+1}
$$

然后把 $\hat f_{t+1}$ 接到 target model 的 LM head 上得到 draft token。

EAGLE 还会输入上一时刻采样出的 token embedding。原因是 feature 只表示概率分布，不表示实际采样路径。比如同一个 hidden state 可能对应：

```text
I: 0.45, you: 0.25, we: 0.15, ...
```

但一旦实际采样成 `I`、`you` 或 `we`，后续生成路径就不同。因此 draft model 必须知道实际 sampled token。

EAGLE-2 在 EAGLE 基础上加入 **dynamic draft tree**：不是用固定 draft tree，而是根据 draft model confidence 动态扩展和剪枝。EAGLE-3 继续兼容并采用 EAGLE-2 的动态 draft tree。

---

## 4 EAGLE 的强点和限制

EAGLE 的训练目标有两部分：

$$
L = l_{fea} + l_{token}
$$

其中：

| Loss | 作用 | 直觉 |
|---|---|---|
| $l_{fea}$ | 让 $\hat f_{t+1}$ 接近 target model 真实 feature $f_{t+1}$ | 内部表示要像 target model |
| $l_{token}$ | 让 $\mathrm{LMHead}(\hat f)$ 的 token 分布接近 target model | 最终 token 行为要像 target model |

feature prediction 是 EAGLE 强的原因：

1. **降低学习难度**：连续 feature 比离散 token 更容易建模。
2. **复用 target model 表示**：draft model 不必从原始 token 重新理解上下文。
3. **稳定多步 draft**：第一步输出的 $\hat f$ 接近真实 $f$，第二步继续喂回 draft model 时仍然像一个正常 feature。

但 EAGLE-3 指出：feature prediction 也是 EAGLE 的限制。

因为 speculative decoding 的最终目标不是：

```text
预测出来的 feature 是否像 target model hidden state
```

而是：

```text
预测出来的 draft token 是否会被 target model 接受
```

从最终目标看，$l_{fea}$ 是一个额外约束。它要求 draft model 的输出既要能产生好的 token distribution，又必须长得像 target model 的 top-layer feature。可能存在某个向量 $a$ 并不像真实 feature，但经过 LM head 后能给出更好的 draft token 分布。EAGLE 的 feature loss 会排斥这种向量。

> [!warning] 关键矛盾
> EAGLE 靠 feature prediction 获得稳定性，但 feature prediction 同时把 draft model 锁在 target model 的 feature manifold 附近，限制了它为了 token acceptance 学到更自由的中间表示。

---

## 5 EAGLE-3 的第一个改动：移除 Feature Prediction Constraint

EAGLE-3 不再要求 draft model 输出 $\hat f_{t+1} \approx f_{t+1}$。它让 draft model 输出一个自由向量：

$$
a_{t+1}
$$

这个向量不需要像 target model 的 hidden feature，只需要经过 LM head 后能预测出好的 token：

$$
\mathrm{softmax}(\mathrm{LMHead}(a_{t+1}))
$$

因此，EAGLE-3 的目标从：

```text
预测 target model feature，再通过 LM head 预测 token
```

变成：

```text
直接学习一个服务于 token prediction 的 unconstrained vector
```

论文 Figure 4 显示：只移除 feature prediction constraint 后，第一个 draft token 的接受率 $0\text{-}\alpha$ 明显提高。原因很直接：draft model 不再同时满足“feature 像”和“token 对”两个目标，而是更自由地服务 token prediction。

但这会带来新问题：第二步会崩。

---

## 6 为什么不能只是删掉 Feature Loss

假设当前前缀是：

```text
How can
```

target model forward 后，我们得到真实 fused feature：

$$
g_{How}, g_{can}
$$

第一步 draft model 输出：

$$
a_I
$$

并通过 LM head 预测出 draft token：

```text
do
```

第一步效果可能很好，因为输入是真实 target model feature。

但第二步要继续预测时，理想输入应该包含 target model 对 `I` 产生的真实 feature：

$$
g_I
$$

问题是：`I` 还只是 draft token，target model 还没有验证它，所以我们拿不到 $g_I$。只能用上一轮 draft model 自己输出的 $a_I$ 代替：

```text
理想输入：g_How, g_can, g_I
实际输入：g_How, g_can, a_I
```

如果训练时 draft model 只见过真实 feature $g$，推理时却要处理自己生成的自由向量 $a$，就会出现 train-test mismatch。

更关键的是：$a$ 没有被 feature loss 约束，它可能是一个适合 LM head 预测 token 的向量，但不一定是一个适合作为下一步 draft model 输入的“正常 feature”。

所以简单删掉 feature loss 的结果是：

| 阶段 | 现象 | 原因 |
|---|---|---|
| 第一步 draft | 接受率上升 | 输出空间更自由，直接服务 token prediction |
| 第二步及以后 | 接受率下降 | 输入混入自身输出 $a$，偏离训练分布 |

EAGLE-3 的关键不是“删掉 feature loss”本身，而是删掉之后如何重新获得多步稳定性。

---

## 7 EAGLE-3 的第二个改动：Training-Time Test

**Training-time test** 是 EAGLE-3 的核心机制。

这个名字容易误解。它不是 test-time training，也不是推理时继续训练模型，而是：

> 训练时就模拟测试/推理时的多步 draft 流程。

推理时，draft model 的输入会逐渐从纯 target feature 变成：

$$
g_1, g_2, \ldots, g_t, a_{t+1}, a_{t+2}, \ldots
$$

因此训练时也要让模型看到这种混合输入。

训练流程可以理解为：

1. Step 1：使用真实 fused features $g_1, \ldots, g_t$，生成 $a_{t+1}$，并预测 token。
2. Step 2：把刚生成的 $a_{t+1}$ 喂回 draft model，生成 $a_{t+2}$。
3. Step 3：再把 $a_{t+2}$ 喂回，继续训练后续 draft。

这样模型在训练阶段就学会：

```text
我不仅要会基于 target model 的真实 feature 预测 token，
还要会基于我自己上一轮产生的向量继续预测 token。
```

可以这样对比：

| 方法 | 多步稳定性来源 |
|---|---|
| EAGLE | 输出被 feature loss 拉向真实 feature |
| EAGLE-3 | 训练时就模拟多步推理，让模型适应自身输出 |

> [!tip] 核心理解
> EAGLE 的稳定性来自“让输出像真实 feature”；EAGLE-3 的稳定性来自“训练时就让模型习惯自己的输出”。

---

## 8 EAGLE-3 的第三个改动：Multi-Layer Feature Fusion

EAGLE/EAGLE-2 主要复用 target model 的 top-layer feature，也就是 LM head 前最后一层 hidden state。

top-layer feature 很适合预测 next token，因为它已经高度对齐 LM head logits。但它的问题是：过于贴近下一个 token，对 next-next token、next-next-next token 的信息未必充分。

EAGLE-3 既然移除了 feature prediction loss，就不再必须把输入限制在 top-layer feature 上。它改为融合 target model 的低层、中层、高层 feature：

$$
g_i = \mathrm{FC}([l_i; m_i; h_i])
$$

其中：

| Feature | 含义 | 可能提供的信息 |
|---|---|---|
| $l_i$ | low-level feature | token 形式、局部模式、词法信息 |
| $m_i$ | middle-level feature | 结构、语义组合、上下文关系 |
| $h_i$ | high-level feature | 接近最终 next-token prediction 的信息 |

拼接后通过 FC layer 压回 target model hidden size，得到 fused feature $g_i$。

这和 remove feature constraint 是配套的：既然输出不再拟合 top-layer feature，输入也可以更自由地选择对多步 draft 更有用的信息。

---

## 9 推理流程：g 和 a 的交替

以论文 Figure 5 的例子理解。

当前 prefix：

```text
How can
```

target model forward 后生成下一个 token：

```text
I
```

同时记录 target model 的低/中/高层特征，并融合成：

$$
g_{How}, g_{can}
$$

第一步 draft 要在 `How can I` 的上下文下预测后续 token。draft model 输入：

```text
g_How, g_can + sampled token embedding e_I
```

输出：

$$
a_I
$$

通过 LM head 采样得到 draft token：

```text
do
```

第二步时，target model 还没有验证 `I`，拿不到 $g_I$，于是用 $a_I$ 代替：

```text
g_How, g_can, a_I + token embedding e_do
```

输出：

$$
a_{do}
$$

再通过 LM head 预测下一个 token，例如：

```text
it
```

因此 EAGLE-3 的多步 draft 输入是：

$$
g_1, g_2, \ldots, g_t, a_{t+1}, a_{t+2}, \ldots
$$

这里 $g$ 和 $a$ 维度相同，但身份不同：

| 符号 | 来源 | 含义 |
|---|---|---|
| $g$ | target model | 已验证上下文的 fused feature |
| $a$ | draft model | 未验证 draft 上下文的临时自由向量 |

---

## 10 Training-Time Test 为什么需要特殊 Attention Mask

Training-time test 不只是把 $a$ 喂回模型，还要处理 attention mask。

普通自回归训练是线性序列，用标准 lower triangular causal mask 即可。但 training-time test 会形成树状 draft 结构。

例如训练序列是：

```text
How can I
```

第一轮 draft 可能在三个位置分别预测：

```text
are, we, do
```

这三个 token 不是同一条线性句子里的连续 token，而是三条不同分支：

```text
How are
How can we
How can I do
```

如果把 `are we do` 简单拼成线性序列并使用普通下三角 mask，`do` 可能看到 `are` 和 `we`。但真实推理中，`How can I do` 这个分支不应该看到兄弟分支信息。

因此 Figure 6 的 attention mask 要模拟 draft tree 的依赖关系：每个节点只能看到自己所在分支上的祖先节点，不能看到其他分支的 sibling nodes。

> [!warning] 容易忽略的点
> Training-time test 的关键不只是多跑几步 draft，而是要让训练时的上下文可见性和真实推理一致。否则模型会在训练中看到推理时不存在的信息。

---

## 11 和 HASS 的区别

HASS 也会在训练时模拟多步 draft，并修改 attention 机制，所以容易和 EAGLE-3 混淆。

论文强调二者动机和目标不同：

| 方法 | 是否保留 feature prediction loss | 输入是否必须是 top-layer feature | 核心动机 |
|---|---:|---:|---|
| HASS | 是 | 是 | 缓解 EAGLE feature prediction 不准造成的误差累积 |
| EAGLE-3 | 否 | 否 | 移除不必要约束，增强 draft model 表达能力 |

HASS 仍然是在 feature prediction 框架内做稳定性改进；EAGLE-3 则是把 feature prediction 这个中间目标直接拿掉，转向 direct token prediction。

---

## 12 实验结论

### 12.1 整体效果

EAGLE-3 在 chat model 和 reasoning model 上都提升明显，实验模型包括 Vicuna 13B、LLaMA-Instruct 3.1 8B、LLaMA-Instruct 3.3 70B、DeepSeek-R1-Distill-LLaMA 8B。

评测任务覆盖：

| 任务 | 数据集 |
|---|---|
| 多轮对话 | MT-bench |
| 代码生成 | HumanEval |
| 数学推理 | GSM8K |
| 指令跟随 | Alpaca |
| 摘要 | CNN/Daily Mail |

核心结果：

- EAGLE-3 相对 vanilla autoregressive decoding 达到约 **3.0x 到 6.5x** speedup。
- 相比 EAGLE-2，EAGLE-3 通常有 **20% 到 40%** 的速度提升。
- 在 Vicuna 13B + HumanEval + temperature=0 上，EAGLE-3 达到 **6.47x speedup**，平均接受长度 $\tau=7.54$。
- 在 LLaMA-Instruct 3.1 8B 上，temperature=0 的 mean speedup 从 EAGLE-2 的 **3.23x** 提升到 EAGLE-3 的 **4.44x**。
- 在 LLaMA-Instruct 3.3 70B 上，temperature=0 的 mean speedup 从 EAGLE-2 的 **2.85x** 提升到 EAGLE-3 的 **4.12x**。
- 在 DeepSeek-R1-Distill-LLaMA 8B 上，temperature=0 的 mean speedup 从 EAGLE-2 的 **3.26x** 提升到 EAGLE-3 的 **4.16x**。

### 12.2 Acceptance Rate

Figure 7 显示：随着输入中 self-predicted values 增多，EAGLE 的 acceptance rate 明显下降，而 EAGLE-3 基本保持稳定。

这正好验证了 training-time test 的作用：

```text
EAGLE：自己的预测越多，误差累积越明显
EAGLE-3：训练时已经适应自己的输出，所以多步接受率更稳
```

### 12.3 Ablation Study

在 LLaMA-Instruct 3.1 8B 上的消融结果：

| 方法 | MT-bench Speedup | MT-bench τ | GSM8K Speedup | GSM8K τ |
|---|---:|---:|---:|---:|
| EAGLE-2 | 3.16x | 4.05 | 3.39x | 4.24 |
| + remove feature constraint | 3.82x | 5.37 | 3.77x | 5.22 |
| + fused features | 4.40x | 6.13 | 4.48x | 6.23 |

结论很清楚：

1. 移除 feature constraint 本身已经显著提高 speedup 和 acceptance length。
2. 再加入 multi-layer fused features 后继续提升。
3. 两个改动都不是装饰，而是分别贡献了有效增益。

### 12.4 生产框架中的吞吐

论文还在 SGLang 和 vLLM 中测试大 batch 吞吐。这个部分很重要，因为 speculative decoding 常被认为只适合 batch size 小、decode 阶段 memory-bound 的场景。

SGLang v0.4.4 + H100 + LLaMA-Instruct 3.1 8B：

- batch size = 64 时，EAGLE-3 仍有 **1.38x throughput improvement**。
- batch size = 1 时，throughput 从 baseline 的 **158.34 tokens/s** 提升到 EAGLE-3 的 **373.25 tokens/s**。
- 相同设置下 EAGLE-2 是 **244.10 tokens/s**。

vLLM 实验中，EAGLE-3 在大 batch 下也比 EAGLE 更稳；例如 batch size = 56 时，EAGLE 已降到 0.71x，而 EAGLE-3 仍约 1.01x。

---

## 13 Scaling Law：为什么论文标题强调 Scaling up

论文不是只说“EAGLE-3 更快”，还强调：**EAGLE-3 让 draft model 能真正从更多训练数据中受益。**

作者把训练数据从 ShareGPT 基准规模扩到 2x、4x、8x，发现：

```text
EAGLE-2：扩数据收益有限，曲线趋于停滞
EAGLE-3：speedup 和 acceptance length 随数据规模继续提升
```

这说明 EAGLE 原来的瓶颈并非单纯数据不足，而是 feature prediction constraint 限制了模型表达空间。EAGLE-3 移除这个约束后，draft model 的能力才更符合“数据越多越强”的 scaling behavior。

> [!tip] 论文标题里的 scaling up
> 这里的 scaling 不是扩大 target model，而是扩大 draft model 的训练数据，并让 inference acceleration 本身出现可扩展收益。

---

## 14 和 DeepSeek MTP 的关系

论文在 EAGLE/EAGLE-2 背景中提到，EAGLE 启发了 DeepSeek-v3 预训练中的 multi-token prediction，而 DeepSeek-v3 的设计又反过来启发了 EAGLE-3。

二者有一个共同点：都希望模型不要只服务于“下一个 token”，而要能更好地支持多步预测。

但二者定位不同：

| 方法 | 所在阶段 | 主要目标 |
|---|---|---|
| MTP | 预训练结构/辅助 loss | 让主模型表示具备多 token 预测能力，也可用于 speculative decoding |
| EAGLE-3 | 推理加速 draft model | 训练一个更强的 draft model，提高 speculative sampling 接受长度 |

可以把 EAGLE-3 理解为：在推理加速场景下，系统性重做了“多步 token draft”这件事。

---

## 15 一张逻辑链记住 EAGLE-3

```text
自回归解码慢
    ↓
speculative sampling：draft 多个 token，target 一次验证
    ↓
vanilla draft model 太弱
    ↓
EAGLE：复用 target top-layer feature，做 feature-level autoregression
    ↓
feature prediction 稳定，但限制表达能力，扩数据收益有限
    ↓
EAGLE-3：移除 feature prediction constraint，改成 direct token prediction
    ↓
第一步更强，但第二步输入混入 unconstrained vector 会产生 train-test mismatch
    ↓
training-time test：训练时模拟多步 draft，把自身输出 a 喂回去
    ↓
为了更丰富的信息，输入从 top-layer feature 改成 low/mid/high feature fusion
    ↓
配合 tree-aware attention mask 和 EAGLE-2 dynamic draft tree
    ↓
更高 acceptance length、更高 speedup，并能从更多训练数据中继续受益
```

---

## 16 最重要的记忆点

EAGLE 可以概括为：

```text
predict feature f_hat -> LM head -> draft token
```

EAGLE-3 可以概括为：

```text
produce unconstrained vector a -> LM head -> draft token
```

EAGLE 的稳定性来自：

$$
\hat f \approx f
$$

EAGLE-3 的稳定性来自：

```text
training-time test：训练时模拟推理时的多步自回归 draft
```

EAGLE/EAGLE-2 的输入主要是：

$$
f_1, f_2, \ldots, f_t
$$

EAGLE-3 的输入变成：

$$
g_1, g_2, \ldots, g_t, a_{t+1}, a_{t+2}, \ldots
$$

其中：

$$
g_i = \mathrm{FC}([l_i; m_i; h_i])
$$

一句话总结：

> **EAGLE-3 释放了 draft model 的表达空间，让它不再模仿 target model 的 top-layer feature；同时用 training-time test 和 tree-aware attention mask 让模型在训练时就适应推理时的多步自生成输入，从而提高 draft token 接受率、平均接受长度和实际推理速度。**

---

## 17 容易混淆的点

> [!warning] lossless 不是“draft model 总是猜对”
> Lossless 指最终采样分布与 target model 原始自回归采样一致。draft token 可以被拒绝，拒绝后用修正分布补偿。

> [!warning] feature prediction constraint 不是完全没用
> 它在 EAGLE 中提供了多步稳定性。EAGLE-3 不是证明 feature loss 一无是处，而是用 training-time test 替代了它原本承担的稳定作用。

> [!warning] training-time test 不是 test-time training
> 它发生在训练阶段，不是在部署时更新参数。名字里的 test 指训练时模拟测试/推理流程。

> [!warning] g 和 a 不是同一个东西
> $g$ 是 target model 的 fused feature，来自已验证上下文；$a$ 是 draft model 自己生成的 unconstrained vector，用来临时代替还拿不到的未来 $g$。

> [!warning] top-layer feature 不一定最适合多步 draft
> top-layer feature 与 next-token logits 高度对齐，但多步预测需要更多层次的信息，因此 EAGLE-3 融合低/中/高层 feature。

---

## 18 可拆分概念

后续可以从这篇笔记拆成以下独立概念页：

- Speculative Decoding：draft-verify 框架、接受-拒绝规则、lossless 性质
- EAGLE：feature-level autoregression、feature uncertainty、shifted token input
- EAGLE-2：dynamic draft tree、tree attention
- Training-Time Test：训练时模拟多步推理、解决 train-test mismatch
- Feature Prediction Constraint：为什么 feature imitation 既稳定又限制表达能力
- Draft Model：推理加速中的候选生成器
- Multi-Layer Feature Fusion：低/中/高层 feature 融合
- Acceptance Length：speculative decoding 的关键性能指标

## 关联

- 相关：Speculative Decoding [MTP](../Deepseek-V4/2.1Deepseek%20Moe/MTP.md) LLM推理加速 [KV Cache](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/KV%20Cache.md) EAGLE EAGLE-2
- 对比：Medusa HASS Lookahead Decoding Hydra
- 系统框架：SGLang、vLLM

## Handoff

目标：
- 将 EAGLE-3 论文整理为长期可复用的分析笔记，并用 Canvas 固化方法主线。

已知：
- 这是一篇综合论文解读笔记，保留在 `05-资料摘录`，不直接挂入主题图谱。
- 已生成 `08-图片/eagle3_mainflow.canvas`，用于展示 EAGLE-3 从 feature constraint 到 training-time test 的主流程。
- 资料索引中已登记官方来源 [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)，状态为 `summarized`。

未定：
- 是否将 `Speculative Decoding`、`Training-Time Test`、`Feature Prediction Constraint` 拆成正式概念笔记。
- 是否单独画一张 `g/a` 交替推理流程 Canvas，用于解释 Figure 5 的细节。

看这些文件：
- `05-资料摘录/EAGLE-3/EAGLE-3.md`
- [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- `08-图片/eagle3_mainflow.canvas`
- `05-资料摘录/资料索引.md`

不要重复：
- 不要把这篇综合论文笔记直接挂入主题页；后续应先拆概念，再让概念进入图谱。
- 不要再用 Mermaid 重画本文主线；本轮已按 Canvas 规范生成原生 `.canvas`。

继续建议：
- 下一步优先拆 `Speculative Decoding` 或 `Training-Time Test`，再根据概念笔记规范补三向链接。
