---
type: concept
domain: 参数高效微调
status: active
---
# LoRA 学习笔记

> [!note]
> LoRA（Low-Rank Adaptation）通过在冻结的预训练权重旁并联一个低秩修正分支 $BA$，将可训练参数减少数万倍，同时不引入额外推理延迟。它解决的是全量微调代价过高的问题。

大模型时代有一个核心矛盾：预训练模型越来越大，但针对每个下游任务都做全量微调，代价越来越高。全参数微调不仅需要更新全部参数，还要为每个任务保存一整套新权重；模型越大，训练显存、存储成本和部署成本就越难承受。LoRA 论文以 GPT-3 175B 为例指出，全量微调的任务实例化成本已经非常高。

在 LoRA 之前，参数高效微调有两条主路线：Adapter 通过在网络中插入额外模块进行适配，Prefix / Prompt Tuning 通过优化输入侧的连续提示来适配任务。它们都能减少可训练参数，但也各有缺点——Adapter 会引入额外推理延迟，Prefix 类方法会占用序列长度，且优化稳定性并不总是理想。LoRA 的目标是同时绕开这两类问题：既减少训练参数，又尽量不改变推理图。

> [!tip] 核心思想
> 下游任务适配真正需要的，不一定是对整个大权重矩阵做自由更新，而可能只需要在少数关键方向上做低维修正。LoRA 在 GPT-3 175B 上将可训练参数减少 10,000 倍、训练显存降低约 3 倍，效果与全量微调相当。

---

## 低秩分解与数学基础

设一个矩阵 $\Delta W \in \mathbb{R}^{d \times k}$，它的秩为 $r$。秩可以理解为：这个矩阵真正独立的方向只有 $r$ 个。虽然矩阵看起来很大，但它的列空间维度并不大，变化本质上集中在少数主方向里。

如果 $\mathrm{rank}(\Delta W) \le r$，那么一定存在两个矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$，使得 $\Delta W = BA$。从线性变换的角度理解：$\Delta W$ 是一个从 $k$ 维到 $d$ 维的线性映射，若秩只有 $r$，这个映射可以拆成两步——先把输入压缩到 $r$ 维，再从 $r$ 维映射到输出空间：

$$x \xrightarrow{A} \mathbb{R}^r \xrightarrow{B} \mathbb{R}^d$$

SVD 告诉我们，任意矩阵都可以写成 $M = U \Sigma V^\top$，其中 $\Sigma$ 是按大小排列的奇异值。如果大部分"能量"集中在前几个奇异值上，只保留最大的前 $r$ 个就能得到很好的低秩近似。

> [!note] LoRA 并不是先求 $\Delta W$ 再分解
> LoRA 不是先得到一个完整的 $\Delta W$ 再做 SVD，而是从一开始就只允许模型学习形如 $BA$ 的低秩更新——这是"前验约束"而非"后验分解"：$W = W_0 + BA$。

---

## 核心公式与机制

对于原始线性层权重 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 冻结 $W_0$，只学习一个低秩增量 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。前向计算为：

$$h = W_0 x + \frac{\alpha}{r} B A x$$

$\alpha$ 是 scaling 超参，用于在不同 rank 下保持输出尺度稳定。论文指出通常直接把 $\alpha$ 设为初始 rank 即可。

从结构上看，LoRA 不是替换原线性层，而是在旁边并联一个低秩修正支路——主干 $W_0 x$，LoRA 分支 $\frac{\alpha}{r}BAx$，最终输出为两者相加。可训练参数大幅减少，冻结参数不需存梯度和优化器状态，且 LoRA 分支可在部署时合并回原权重。

### 初始化

$A$ 随机高斯初始化，$B$ 零初始化。这样一开始 $BA = 0$，训练起点与原始预训练模型完全一致，模型不会因 LoRA 分支的随机输出而偏离。虽然 $B=0$，但 $A$ 已经是非零的，所以 $B$ 会先收到有效梯度开始更新，训练可以平稳启动。

---

## Transformer 中的接入位置

LoRA 理论上可以加到 Transformer 中任意线性层上。论文的主实验只在 attention 权重上加，并且默认重点放在 $W_q$ 与 $W_v$ 上。作者的消融实验表明，在 GPT-3 175B 上同样约 18M 参数预算时，适配 $W_q + W_v$ 的效果优于只改 $W_q$、只改 $W_k$ 等方案。从机制上看，$W_q$ 决定"如何去查询别的 token"，$W_v$ 决定"从别的 token 取回什么内容"，两者合在一起通常已能对任务适配产生较强影响。

| 策略 | 覆盖模块 | 适合场景 |
|:--:|:--:|:--:|
| 保守 | `q_proj` + `v_proj` | 入门、显存紧张 |
| 均衡 | `q/k/v/o_proj` | 效果与成本平衡 |
| 激进 | 上述 + MLP 层 | 追求上限、资源充足 |

扩展到 $W_k$、$W_o$ 或 MLP 层，参数量会线性上升（额外参数约为 $r(d_{\text{in}} + d_{\text{out}})$），但表达能力更强。

---

## 权重合并与推理零延迟

训练时某个线性层的前向是 $y = W_0 x + \frac{\alpha}{r} B A x$。因为两项都是线性变换，可以直接合并：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

推理阶段直接计算 $y = W_{\text{merged}} x$，不需要额外分支。

> [!note] 与 Adapter 的根本区别
> Adapter 在网络中增加了新的顺序模块，推理图变深，产生额外延迟。而 LoRA 分支与原线性层是并联关系，且二者都属于线性映射，可以提前合并成一个新矩阵。推理图恢复成原模型的普通矩阵乘法。

| 阶段 | 策略 | 原因 |
|:--:|:--|:--|
| 训练 | 不 merge，保留 `base + lora` 计算图 | 便于对 $(A, B)$ 求梯度更新 |
| 推理 | merge 为 $W_0 + \frac{\alpha}{r}BA$ | 恢复为单一权重，无额外开销 |

---

## 从 LoRA 到 QLoRA

LoRA 解决了"不全量训练"的问题，但冻结的底座模型仍要以 FP16/BF16 完整驻留显存。QLoRA 进一步把冻结底座量化到 4-bit 存储：

$$\text{QLoRA} = \text{LoRA} + \text{Quantized Frozen Base Model}$$

QLoRA 不只是"量化底座"，而是配合了一整套数值设计：

- NF4（NormalFloat 4-bit）：更适合近似正态分布权重的 4-bit 表示方式
- Double Quantization：连量化过程中的 scale 等附加信息也进一步量化
- Paged Optimizer：用分页思路缓解训练时的显存峰值压力

核心思想是存储低比特，计算时解量化到合适精度，底座冻结仅训练 LoRA。LoRA 与 QLoRA 不是替代关系而是递进关系——LoRA 解决"不全量训练"，QLoRA 进一步解决"冻结底座也太占显存"。显存充足用普通 LoRA 更简单稳妥；显存受限想微调更大模型时 QLoRA 更有优势。

---

## 工程实现要点

自己实现 LoRA，最重要的四件事：原始权重冻结、前向公式 $W_0 x + \frac{\alpha}{r}BAx$ 正确、初始化 $A$ 随机 $B$ 零、merge/unmerge 逻辑正确且不重复。

> [!warning] 常见坑点
> - 重复 merge：merge 两次会把 $\Delta W$ 加两次，结果直接出错
> - dtype 不一致：基座可能是 fp16/bf16，LoRA 参数可能是 fp32，merge 时需处理类型转换
> - 训练态直接 merge：训练时不应长期保持 merge 状态，一般 train 不 merge，eval/deploy 时再 merge
> - 量化模型 merge：底座是 4-bit/8-bit 量化权重时，不能简单原地加 $\Delta W$
> - rank 和模块范围一上来开太大：更推荐先从 `q_proj + v_proj + 小 rank` 起步

---

## 关联
- 属于：[[参数高效微调]]
- 相关：[[QLoRA]] [[Adapter]] [[PEFT]] [[全量微调]] [[Prompt Tuning]]
- 用于：[[SFT]] [[大模型微调项目]]

## 相关概念
- [[QLoRA]]
- [[PEFT]]
- [[SFT]]
- [[Transformer]]
