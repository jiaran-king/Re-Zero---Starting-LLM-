---
type: concept
domain: 后训练
status: active
---
# RLHF：从预训练到人类对齐

> [!note]
> RLHF 是将语言模型从"会接话"训练到"有用、安全、遵从指令"的核心方法。覆盖 InstructGPT 的完整训练流程（SFT → RM → PPO）以及 PPO 算法在大模型对齐场景中的关键设计。

---

## 为什么需要 RLHF

语言模型的预训练目标是 next token prediction，这个目标本身不保证模型输出有用、安全、能正确遵循用户指令。RLHF 要解决的核心问题就是让模型的优化目标从"预测下一个 token"转向"符合人类偏好"。

InstructGPT 论文给出的完整流程是：

$$\text{Pretrain} \rightarrow \text{SFT} \rightarrow \text{RM Training} \rightarrow \text{PPO}$$

预训练产出基座模型，SFT 让模型初步学会遵循指令，RM 学习人类偏好的奖励函数，PPO 利用这个奖励函数进一步优化策略。

---

## SFT 阶段

SFT 的目标很直接：用人工标注的高质量 prompt-response 对微调预训练模型。InstructGPT 使用了大约 13k 条 prompt，训练 16 个 epoch。

一个有意思的观察是 SFT 模型在训练集上明显过拟合了——1 个 epoch 后验证 loss 就开始上升，但继续训练反而能提升 RM 给出的分数和人类偏好评分。这说明 RM 捕捉到的"好回答"特征与 SFT 训练集上的 loss 不完全一致，过拟合到少量标注数据反而可能在偏好维度上更好。

但 SFT 有天然瓶颈：标注成本高，数据量小，且标注者之间的主观判断差异大。单纯靠 SFT 难以覆盖用户实际使用中遇到的丰富场景。

> [!note] SFT 的根本局限
> SFT 只有正例信号，缺乏对比信号。模型学习"好回答长什么样"，但没有"远离坏回答"的信号。SFT 的本质是 behavior cloning，天花板就是训练数据的质量，模型只能模仿不能超越。RL 教模型"区分好坏并探索更好的"，模仿有天花板，探索无上限。

---

## 奖励模型

### 训练方式

RM 本质上是一个打分模型，但它不是直接学绝对分数，而是通过成对比较来学习。对于同一个 prompt 的 K 个不同回答，人工标注员给出偏好排序，RM 学习预测这种排序。训练 loss 基于 Bradley-Terry 模型：

$$\mathcal{L} = -\mathbb{E}\left[\log \sigma\left(r(x, y_w) - r(x, y_l)\right)\right]$$

$y_w$ 是标注者偏好的回答，$y_l$ 是不偏好的回答。Sigmoid 函数将两者的分数差映射到偏好概率上，差值越大，$y_w$ 被判定为更好的概率越接近 1。

当同一个 prompt 有 K 个排序后的回答时，可以提取出 $\binom{K}{2}$ 个比较对。为防止某个 prompt 主导梯度，需要在组内做平均：

$$\mathcal{L} = -\frac{1}{\binom{K}{2}} \mathbb{E}\left[\log \sigma\left(r(x, y_w) - r(x, y_l)\right)\right]$$

### 为什么用成对比较而非绝对打分

绝对打分的问题在于标注者之间的评分尺度不一致——同一篇回答，有人打 3 分有人打 5 分，但两人对"A 比 B 好"的判断更容易达成一致。InstructGPT 中标注者间的一致率约为 73%。

### 训练效率：按 prompt 打包

对于同一个 prompt 的 K 个回答，朴素做法是将所有 $\binom{K}{2}$ 个比较对打散后按 batch 抽取。但这样做同一个回答（如 A）出现在 K-1 个比较对中，每次都要重新过一遍大模型，总共 $K(K-1)$ 次前向计算；且同一个回答被分散在不同 batch 中反复更新梯度，容易过拟合。

InstructGPT 的做法是把同一个 prompt 衍生出的所有比较对作为一个不可分割的 batch element。先对 K 个回答各做一次前向传播，得到 K 个标量分数，然后在显存中直接做两两相减算 loss。每个回答只过一次模型，且针对该 prompt 只执行一次反向传播。

---

## PPO 阶段

更细的算法展开见 [PPO](PPO.md)。这里保留它在 RLHF 流水线中的角色和关键设计。

### 优化目标

PPO 在 RLHF 中的优化目标是：

$$\max_\theta \; \mathbb{E}_{(x,y) \sim \pi_\theta}\left[r(x, y)\right] - \beta \cdot D_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right]$$

前半部分要求回复质量高（RM 打分高），后半部分的 KL 约束防止模型偏离 SFT 模型太远。InstructGPT 中 $\beta = 0.02$。

> [!tip] KL 惩罚的两种等价视角
> 正则化视角：把 KL 项看作正则项，限制策略更新幅度，防止过拟合 RM。偏好视角：KL 项等价于给每个 token 一个负奖励 $-\beta \log \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}$，模型在提升 RM 分数的同时要为偏离参考策略付出代价。

KL 惩罚的根本目的是防止 reward hacking——模型找到 RM 的漏洞来刷高分，但生成的内容实际上毫无意义甚至有害。

### 四个模型协作

| 模型 | 作用 | 是否更新 |
|------|------|:------:|
| Actor（策略模型） | 生成回答 | 更新 |
| Critic（价值模型） | 估计每个 token 位置的状态价值 $V(s_t)$ | 更新 |
| RM（奖励模型） | 对回答打分 | 冻结 |
| Reference（参考模型） | 计算 KL 散度的锚点 | 冻结 |

> [!note] 训练结束后只保留 Actor
> Critic、RM、Reference 都是训练阶段的"脚手架"。推理时只需要 Actor，其余三个模型的参数全部丢弃。

### 信用分配与 GAE

RM 只在整个回答生成完毕后给出一个标量奖励，但 PPO 需要在每个 token 上计算 advantage 来指导策略更新。这就是信用分配（credit assignment）问题：回答末尾的奖励如何分配到生成过程中的每一步？

通过 Critic 引入逐 token 的价值估计 $V(s_t)$，计算 TD 残差：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

再通过 GAE 从后往前递推，聚合为每个 token 的 advantage：

$$\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}$$

Critic 估计越准，GAE 聚合出的 advantage 越可靠，Actor 的梯度方向就越正确。如果 Critic 很差——比如对所有状态都输出一个常数——那 $\delta_t$ 就退化成原始奖励信号，Actor 收到的梯度方向时对时错，训练就不稳定甚至不收敛。因此 Critic 通常需要比 Actor 更快地收敛，很多实现里 Critic 的 loss 权重会设成 Actor 的 2 到 3 倍。

---

## PPO-ptx：缓解对齐税

PPO 训练会导致模型在 SQuAD、DROP、HellaSwag 等 NLP benchmark 上的性能下降，这种现象被称为对齐税。原因是 PPO 纯粹面向 RM 优化，忽略了模型原有的语言建模能力。

PPO-ptx 通过在目标函数中混入预训练数据的 log-likelihood 项来缓解：

$$\max_\theta \; \mathbb{E}_{(x,y)}\left[r(x, y) - \beta \cdot D_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right]\right] + \gamma \cdot \mathbb{E}_{x \sim D_{\text{pretrain}}}\left[\log \pi_\theta(x)\right]$$

其中 $\gamma = 27.8$，使用 8 倍于 RL episode 数量的预训练数据。PPO-ptx 大幅恢复了对齐税带来的性能回退，甚至在 HellaSwag 上超越了 GPT-3，代价是训练时间翻倍。

---

## Transformer Loss：从交叉熵到工程实现

理解 PPO 的训练，需要先理解 Transformer 的基础 loss 是怎么从信息论交叉熵一步步演化出来的。

信息论中交叉熵的经典公式：$H(P,Q) = -\sum_x P(x) \log Q(x)$。在语言模型语境下，$P(x)$ 是真实分布，$Q(x)$ 是模型的预测分布。

next token prediction 中标准答案只有一个确定正确的词，真实分布中正确词的概率是 1，其余都是 0，化简为负对数似然损失：$H(P,Q) = -\log Q(x_c)$。

Transformer 最后一层输出长度为 $V$ 的 Logits 向量 $z$，通过 Softmax 转化为概率分布后代入：

$$\text{Loss}_{\text{single\_token}} = -\log \left( \frac{e^{z_c}}{\sum_{j=1}^{V} e^{z_j}} \right) = -z_c + \log \left( \sum_{j=1}^{V} e^{z_j} \right)$$

反向传播时模型做两件事：拉高正确词 $c$ 的原始得分 $z_c$，压低词表中所有词得分的指数和。

对长度为 N 的序列，总 loss 是每个时间步预测正确词汇 loss 的平均值：

$$\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log Q_t(y_t)$$

实际训练中还有两个常见技巧：Padding Mask 通过 mask 矩阵将 `<PAD>` 对应位置的 loss 乘以 0，防止无意义的填充词干扰梯度更新；Label Smoothing 将正确答案的概率从 1 削减到 0.9，剩余 0.1 平均分配给词表中其他词——一旦开启标签平滑，前面 One-Hot 化简中"消失"的词表遍历求和会重新出现。

---

## 实验结果与局限

1.3B 参数的 InstructGPT 在人类评估中优于 175B 的 GPT-3，说明对齐比单纯的模型规模更重要。175B InstructGPT 在 85±3% 的情况下被偏好于 175B GPT-3，幻觉率从 41% 降到 21%，毒性在礼貌提示下降低约 25%。模型还能处理训练数据中极少出现的非英语指令和代码任务，说明学到了"遵循指令"这一泛化能力。

已知局限：Reward hacking（被指示生成有毒内容时 InstructGPT 比 GPT-3 更毒）；标注偏差（约 40 人标注团队，73% 一致率意味着约四分之一的偏好判断存在分歧）；对齐目标模糊——"有用、安全、遵从指令"本身有内在张力。

| 指标 | 数值 |
|------|------|
| SFT 训练数据 | ~13k prompts |
| RM 训练数据 | ~33k prompts，K=4~9 |
| RM 规模 | 6B |
| 标注者间一致率 | ~73% |
| KL 系数 $\beta$ | 0.02 |
| 预训练损失系数 $\gamma$ | 27.8 |
| 175B InstructGPT vs GPT-3 偏好率 | 85±3% |
| 幻觉率（InstructGPT vs GPT-3） | 21% vs 41% |

---

## 关联
- 属于：[后训练与对齐](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%90%8E%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%AF%B9%E9%BD%90.md)
- 相关：SFT [DPO](DPO.md) [GRPO](GRPO.md) [PPO](PPO.md) [Transformer](Transformer.md) Reward Model Bradley-Terry
- 用于：InstructGPT 复现 大模型对齐项目

## 相关概念
- SFT
- [DPO](DPO.md)
- [GRPO](GRPO.md)
- [PPO](PPO.md)
- [Transformer](Transformer.md)
- [KV Cache](KV%20Cache.md)
