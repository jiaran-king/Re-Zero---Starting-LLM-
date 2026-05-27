
# DeepSeek 的 MTP 学习笔记

> [!info] 原始来源
> - [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
> - [DeepSeek-V4 官方模型卡 / Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

## 1. MTP 是什么

MTP 的全称是 **Multi-Token Prediction**，中文可以理解为“多 token 预测”。它的核心思想很简单：普通自回归语言模型在每个位置只预测下一个 token，而 MTP 让模型在同一个位置上额外预测更后面的 token。

普通 next-token prediction 是：

```text
x_i → x_{i+1}
```

MTP 则变成：

```text
x_i → x_{i+1}, x_{i+2}, x_{i+3}, ...
```

也就是说，普通语言模型只被问：“下一步是什么？”  
MTP 还会继续追问：“再下一步大概是什么？”

这会给模型带来一种轻量的 lookahead 训练压力：当前位置的 hidden state 不仅要能解释眼前的下一个 token，还要对后续 token 的结构有所预判。
-
## 2. 用例子理解 next-token 和 MTP

假设训练文本是：

```text
我 喜欢 机器 学习
```

普通语言模型训练时大致是：

```text
看到「我」       → 预测「喜欢」
看到「我 喜欢」  → 预测「机器」
看到「我 喜欢 机器」→ 预测「学习」
```

每个位置只有一个监督目标。

如果使用 MTP，在位置「我」这里，模型不只学习：

```text
我 → 喜欢
```

还要额外学习：

```text
我 → 机器
```

更一般地，如果 MTP depth 更深，还可能继续预测更远的 token。但在 DeepSeek-V3/V4 的配置里，MTP depth 设为 1，也就是额外预测一个未来 token。

因此，DeepSeek 中常见的理解是：

```text
主 LM 目标：预测 x_{i+1}
MTP 额外目标：预测 x_{i+2}
总效果：每个位置学习预测未来两个 token
```

这里容易误解的一点是：**MTP depth = 1 不是说模型总共只预测一个 token，而是说在普通 next-token 之外，额外加一个预测深度。**

## 3. DeepSeek 的 MTP 不是简单“多个预测 head”

一种最朴素的多 token 预测方式可能是直接在同一个 hidden state 上接多个 head：

```text
hidden state h_i
├── head 1 → 预测 x_{i+1}
├── head 2 → 预测 x_{i+2}
└── head 3 → 预测 x_{i+3}
```

但这种做法有一个问题：预测 `x_{i+2}` 时，没有显式依赖 `x_{i+1}`。而真实的自回归生成过程是有顺序的：生成第二个未来 token 时，第一个未来 token 应该已经成为上下文的一部分。

DeepSeek 的 MTP 更强调 **sequential prediction**，即顺序地预测额外 token，并保留完整的因果链。DeepSeek 的 MTP module 会结合上一深度的表示和未来 token 的 embedding，再通过 Transformer block 得到当前深度的表示；embedding 和 output head 与主模型共享。

可以粗略理解为：

```text
主模型 hidden state h_i
      ↓
预测 x_{i+1}
      ↓
MTP module 结合 h_i 与 x_{i+1} 的 embedding
      ↓
预测 x_{i+2}
```

所以 DeepSeek 的关键不是“并行猜多个 token”，而是：

```text
先建立第 1 个未来 token 的条件，
再基于这个条件预测第 2 个未来 token。
```

这就是它和简单多 head 方法的本质区别。

## 4. DeepSeek MTP 的结构直觉

DeepSeek-V4 的架构图中，主干模型经过 embedding、Transformer blocks、prediction head 后产生普通 LM loss；同时，顶部还有 MTP modules，对应 MTP loss。也就是说，MTP 是附加在主模型上的训练目标和模块，而不是替代主模型的 next-token objective。

一个简化图可以这样记：

```text
输入序列：
x1, x2, x3, x4, x5

主模型：
h1, h2, h3, h4
 ↓   ↓   ↓   ↓
预测 x2 x3 x4 x5       ← 普通 next-token prediction

MTP module：
结合 h1 + emb(x2) → 预测 x3
结合 h2 + emb(x3) → 预测 x4
结合 h3 + emb(x4) → 预测 x5
```

在位置 `x1` 上：

```text
主模型负责：x1 → x2
MTP 负责：x1 + x2 → x3
```

这就是“预测未来第二个 token”时保留自回归因果链的含义。

## 5. MTP 的训练目标

普通语言模型的 loss 是：

```text
L_LM = CrossEntropy(预测 x_{i+1}, 真实 x_{i+1})
```

MTP 会额外加入预测更远 token 的 loss。一般可以写成：

```text
L_total = L_LM + λ · L_MTP
```

如果有多个 MTP depth，则可以理解为：

```text
L_MTP = average(L_MTP^1, L_MTP^2, ..., L_MTP^D)
```

其中 `λ` 是 MTP loss 的权重，用来控制额外预测目标对主训练目标的影响。你上传的 DeepSeek-V4 报告中给出了具体训练设置：DeepSeek-V4 的 multi-token prediction depth 设为 1；在训练的大部分阶段，MTP loss weight 设为 0.3，在学习率开始衰减时降为 0.1。

这说明 DeepSeek 把 MTP 当作一个有明确权重的辅助目标，而不是让它无限制地影响主语言建模目标。

## 6. 为什么 MTP 可能有用

第一，MTP 让训练信号更密集。普通 next-token prediction 中，一个位置通常只有一个监督信号；MTP 中，同一个位置还要参与预测更远的 token，相当于让同一段文本提供更多训练约束。你上传的笔记也强调，这种机制可能迫使 hidden representation 学到更强的局部规划能力。

第二，MTP 迫使模型“看远一点”。普通 next-token prediction 可能让模型过度关注局部模式，只要下一个 token 猜对即可；但 MTP 要求模型对后续 token 也有预测能力，因此 hidden state 需要承载更多关于未来结构的信息。

例如代码补全时，模型看到：

```python
for i in range(n):
```

它不仅要知道下一个 token 可能是换行或缩进，还要预判后面可能出现循环体、变量更新、条件判断等结构。MTP 的训练压力会让模型更倾向于形成这种“后续结构感”。

第三，MTP 可能帮助推理时加速。MTP 本身是训练目标和模型模块，但训练好的 MTP module 可以在推理时作为一种内置 draft module，用于 speculative decoding。也就是说，它可以先草拟额外 token，再由主模型验证。你上传的笔记中提到，DeepSeek-V3 报告里第二个 token 的接受率在不同生成主题上约为 85% 到 90%，并带来约 1.8 倍 TPS 提升。

## 7. MTP 和 speculative decoding 的关系

这两个概念很容易混淆。

**MTP 是训练目标 / 模型结构。**  
它解决的是：训练时如何让模型在每个位置额外学习预测未来 token。

**Speculative decoding 是推理加速方法。**  
它解决的是：推理时如何先用 draft 模型或 draft 模块猜一些 token，再由主模型验证，从而减少完整解码次数。

DeepSeek 的巧妙之处在于，MTP module 训练完成后，可以被用作 speculative decoding 的 draft module。这样它既能在训练时提升主模型 representation，又能在推理时尝试加速 decoding。

但要注意：即使推理时不使用 speculative decoding，MTP 仍然可能通过辅助训练目标提升主模型本身。另一方面，如果不想增加推理复杂度，也可以直接丢弃 MTP modules，让主模型按普通方式生成。

## 8. DeepSeek-V4 中 MTP 的位置

DeepSeek-V4 并没有把 MTP 当作主要的新创新点，而是把它作为从 DeepSeek-V3 继承下来的有效策略继续保留。DeepSeek-V4 报告明确说，V4 相比 V3 保留了 DeepSeekMoE 框架和 MTP strategy，同时主要新增了 hybrid attention、mHC 和 Muon optimizer 等架构与优化技术。

这意味着你学习 DeepSeek-V4 的 MTP 时，要分清两层：

```text
MTP：沿用 V3 的训练/预测策略
CSA/HCA、mHC、Muon：V4 的主要新架构与优化升级
```

所以，DeepSeek-V4 的百万 token 长上下文能力，主要不是由 MTP 带来的，而是由 CSA/HCA 等长上下文注意力机制与系统优化带来的；MTP 更像是继续保留的预训练辅助目标和潜在解码加速模块。

## 9. 和 GRPO、RL、CoT 的区别

MTP 不是 RL，也不是 GRPO，也不是 Chain-of-Thought。

它发生在更底层的语言模型训练阶段，主要目标是让模型的 token-level representation 具备更好的未来预测能力。GRPO/RL 更偏向后训练阶段，用奖励信号优化模型在数学、代码、agent、指令遵循等任务上的行为。DeepSeek-V4 报告中也提到，其 post-training pipeline 会对不同领域进行 SFT 和 GRPO，再通过 on-policy distillation 合并专家能力。

可以这样区分：

```text
MTP：
预训练/结构层面的辅助目标，让模型学会更好地预测未来 token。

SFT：
用高质量样本教模型按照期望格式和行为回答。

GRPO/RL：
用奖励信号进一步强化某些任务表现，比如数学、代码、agent 行为。

CoT：
推理时或训练数据中的显式思维链表达方式。
```

MTP 不直接教模型“反思”或“分步推理”，但它可能让底座模型的表示更适合承载后续推理能力。你上传的笔记也把它概括为一种“预训练阶段的轻量 lookahead 机制”。

## 10. 常见误区

误区一：认为 MTP depth = 1 就是只预测一个 token。  
正确理解是：普通 LM 已经预测 `x_{i+1}`，MTP depth = 1 表示额外预测 `x_{i+2}`，所以整体相当于预测未来两个 token。

误区二：认为 MTP 就是多个 output head。  
DeepSeek 的 MTP 不是简单并行多个 head，而是顺序预测额外 token，并尽量保持自回归因果链。

误区三：认为 MTP 一定会增加推理成本。  
MTP modules 可以在普通推理时丢弃，因此主模型仍可独立运行；也可以在 speculative decoding 中作为 draft module 使用。

误区四：认为 DeepSeek-V4 的长上下文主要来自 MTP。  
DeepSeek-V4 的长上下文效率主要来自 CSA/HCA 等注意力设计；MTP 是从 V3 继承的策略。

## 11. 一句话总结

**MTP 的本质是：让模型在每个位置不仅学习“下一个 token 是什么”，还学习“再往后的 token 会是什么”。**

DeepSeek 的关键点是：

```text
普通 LM：x_i → x_{i+1}

DeepSeek MTP：
x_i → x_{i+1}
x_i + x_{i+1} → x_{i+2}
```

它不是简单多 head，而是顺序预测额外 token；它在训练时提供更密集的监督信号，在推理时既可以丢弃，也可以用于 speculative decoding 加速。

最适合记忆的一句话是：

```text
普通 LM 学“下一步怎么走”，MTP 让模型顺便学“接下来几步会怎么走”。
```

## 12. 建议复习问题

你可以用下面几个问题检查自己是否真正理解了 MTP：

1. 普通 next-token prediction 和 MTP 的目标有什么区别？

   **简答**：普通 next-token prediction 在位置 `i` 只要求模型根据当前上下文预测 `x_{i+1}`。MTP 则在普通 LM 目标之外，额外要求模型预测更远的未来 token，例如 `x_{i+2}`。这样同一个位置的 hidden state 不只服务于预测下一个 token，也要包含对后续 token 和局部结构的预判信息。

2. 为什么 DeepSeek 不只是简单接多个 prediction head？

   **简答**：简单多个 head 会从同一个 hidden state 并行预测多个未来 token，预测 `x_{i+2}` 时没有显式依赖 `x_{i+1}`，不符合自回归生成的因果链。DeepSeek 的 MTP 强调 sequential prediction，会结合当前位置 hidden state 和前一个未来 token 的 embedding，再经过 MTP Transformer block 得到新的表示，用这个表示预测更远的 token。

3. MTP depth = 1 到底预测几个未来 token？

   **简答**：MTP depth = 1 不是说模型总共只预测 1 个 token，而是指在普通 next-token prediction 之外，额外增加 1 个预测深度。普通 LM 已经预测 `x_{i+1}`，MTP depth = 1 额外预测 `x_{i+2}`，所以整体上每个位置学习预测未来两个 token。

4. MTP loss 如何并入总训练 loss？

   **简答**：MTP loss 作为辅助目标加入主语言模型 loss，常见形式可以写成 `L_total = L_LM + λ · L_MTP`。其中 `L_LM` 是普通 next-token prediction loss，`L_MTP` 是额外未来 token 的预测 loss，`λ` 控制 MTP 对主训练目标的影响强度。多个 MTP depth 时，`L_MTP` 通常可以理解为多个深度 loss 的平均或加权组合。

5. MTP loss 是只作用于 MTP module 自身，还是也作用于主模型？

   **简答**：通常不只作用于 MTP module 自身，而是会沿着 MTP module 依赖的计算路径反向传播，同时训练 MTP module 和主模型 backbone。因为 MTP module 的输入依赖主模型 hidden state，`L_MTP` 会对主模型表示施加额外约束，让 hidden state 不只服务于预测 `x_{i+1}`，也要支持预测更远的未来 token。若某个实现显式 stop-gradient / detach hidden state，则可能只主要训练 MTP module，但这不是 DeepSeek MTP 作为辅助训练目标的主要直觉。

6. MTP 为什么可能提升代码、数学、推理类任务的底座能力？

   **简答**：代码、数学、推理类任务通常依赖局部规划、步骤一致性和因果链建模。MTP 在训练时要求 hidden state 不只预测下一个 token，还要支持预测更远的未来 token，因此会迫使模型学习后续结构感。例如代码中的缩进、括号闭合、变量使用，数学中的公式变换，推理中的步骤衔接，都受益于这种 lookahead 训练压力。

7. MTP 和 speculative decoding 是什么关系？

   **简答**：MTP 是训练目标 / 模型结构，主要在训练时通过额外预测未来 token 来约束 hidden state；speculative decoding 是推理加速方法，核心流程是 draft model / draft module 先连续草拟多个 token，再由主模型一次性验证这些 draft tokens，并按顺序接受一部分。如果中途某个 token 被拒绝，后续 draft token 会被丢弃，主模型在当前位置重新生成，然后进入下一轮。DeepSeek 的 MTP module 训练完成后，可以作为 speculative decoding 的内置 draft module，但 MTP 本身不等于 speculative decoding。

8. DeepSeek-V4 中 MTP 是新设计，还是继承自 V3？

   **简答**：DeepSeek-V4 中的 MTP 更适合理解为从 DeepSeek-V3 继承下来的策略，而不是 V4 的主要新创新。V4 主要新增或强化的是 CSA/HCA、mHC、Muon optimizer 等架构与优化技术；MTP 在 V4 中继续作为预训练辅助目标和潜在的 speculative decoding draft module 保留。

掌握这些问题后，基本就可以比较清楚地理解 DeepSeek 的 MTP 机制了。
