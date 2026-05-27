---
type: concept
domain: 后训练
status: active
---
# DPO：直接偏好优化

> [!note]
> DPO 绕过了 RLHF 中的 Reward Model 训练和 PPO 强化学习，直接从人类偏好数据优化策略模型。数学上利用最优策略与最优奖励函数之间的解析映射关系，把 RM 的训练目标重新参数化为策略模型本身的目标，变成一个类似 SFT 的分类问题。

---

## 定位与动机

RLHF 的经典三阶段需要同时维护四个模型，PPO 阶段的工程复杂度尤其高。DPO 的核心问题是：能不能跳过 RM，直接从人类偏好数据优化策略？

| 方法   | 需要的模型                             | 训练方式                | 工程复杂度 |
| ---- | --------------------------------- | ------------------- | ----- |
| PPO  | 4 个（Actor, Critic, RM, Reference） | 在线采样 + GAE + 多步更新   | 很高    |
| DPO  | 2 个（Policy, Reference）            | 离线，类似 SFT           | 很低    |
| GRPO | 2 个（Policy, Reference）+ RM 打分     | 在线组采样 + z-score 归一化 | 中等    |
|      |                                   |                     |       |

---

## 完整数学推导

DPO 的推导建立在 RM 的 Bradley-Terry 模型之上。推导的出发点是 RLHF 的 KL 约束优化目标：

$$\max_\pi \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi}\big[r(x, y)\big] - \beta \, D_{KL}\big[\pi(y|x) \,\|\, \pi_{ref}(y|x)\big]$$

固定 $x$，将期望展开为求和，加入概率归一化约束，构建拉格朗日函数后对 $\pi(y|x)$ 求偏导并令其为 0。反解出最优策略的闭式解：

$$\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

最优策略在参考模型的基础上，对高奖励的生成路径做指数级的概率放大，$Z(x)$ 保证归一化。对闭式解两边取对数并移项，得到用策略表示隐式奖励的关系：

$$r(x, y) = \beta \log\frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

> [!tip] 关键洞察
> 如果有一个最优语言模型 $\pi$，它自身就已经隐式地定义了一个奖励函数——奖励值等于当前模型概率相对于参考模型的提升幅度，加上一个只依赖 $x$ 的常数项。

将这个隐式奖励代入 Bradley-Terry 模型，计算 $r(x, y_w) - r(x, y_l)$ 时，两个 $\beta\log Z(x)$ 完全相同，做差即消。化简后代入负对数似然损失，得到最终的 DPO 损失函数：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

---

## 直觉理解

DPO 定义了一个隐式奖励 $\hat{r}(x, y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$，即模型当前对 $y$ 的概率相对于参考模型的对数概率比乘以 $\beta$。DPO 损失要求 chosen response 的隐式奖励高于 rejected response 的隐式奖励。

如果模型已经正确地给 $y_w$ 高分、给 $y_l$ 低分，$\sigma(\cdot)$ 接近 1，loss 很小，梯度近零。如果模型犯错给 $y_l$ 更高分，$\sigma(\cdot)$ 接近 0，产生巨大的正惩罚。

实践中观察到 DPO 倾向于只降低 $y_l$ 的概率，而不怎么提升 $y_w$ 的概率。这是一个已知问题，后续的 IPO、SimPO、KTO 等变体尝试修复。

---

## 梯度分析

对 DPO 损失求 $\theta$ 的偏导：

$$\nabla_\theta \mathcal{L}_{DPO} = - \beta \mathbb{E}_{(x, y_w, y_l)} \left[ \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right) \right]$$

梯度可以拆成两个部分。更新方向 $\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)$ 告诉模型增加好回复的概率、降低坏回复的概率。动态权重 $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 像一个自适应调节的误差项——模型预测正确且很自信时梯度消失，预测错误时梯度被放大。这个机制有效防止了过度优化和模型崩溃。

---

## 与 RM 损失的本质联系

并排对比两者的损失函数，唯一的区别在于"奖励"的定义方式：

| | RM | DPO |
|---|---|---|
| 奖励来源 | 独立神经网络 $r_\phi$ | 策略模型自身的概率比值 |
| 需要的模型 | 单独训练一个 RM | 只需 $\pi_\theta$ 和冻结的 $\pi_{ref}$ |
| 底层框架 | Bradley-Terry 偏好模型 | 同一个 Bradley-Terry 偏好模型 |

DPO 就是把"生成模型自身"当作了"奖励模型"，直接在偏好数据上计算 Bradley-Terry 损失，把 RM 训练和策略优化合成了一步。

---

## On-policy 与 Off-policy

PPO/GRPO 与 DPO 的一个根本差异在于数据来源。On-policy（同策略）是指模型用自己当前生成的回答来计算奖励并更新自己；Off-policy（异策略）是指模型对着提前构建好的静态数据集进行拟合。

| 维度 | On-policy（PPO / GRPO） | Off-policy（DPO） |
|------|------------------------|-------------------|
| 数据状态 | 在线生成 | 离线静态数据集 |
| 计算开销 | 极大 | 极小 |
| 核心优势 | 反馈信号精准，上限高 | 工程简单，训练稳定 |
| 核心痛点 | 训练不稳定，工程复杂 | Distribution Shift，性能天花板受限于数据 |

---

## DPO 的架构精简

PPO 需要四个模型而 DPO 只需两个，这不是偷工减料，而是数学推导带来的两次"裁员"。

PPO 依赖独立训练的 RM 给 $y_w$ 和 $y_l$ 打绝对分数，DPO 直接让 Actor 配合 Reference，通过计算对数概率比值来隐式表达奖励。Actor 本身兼任了自己行为的评判者。

标准 PPO 中文本生成被视作马尔可夫决策过程，Value Model 预测每个状态的未来期望回报，为 Actor 计算 Advantage 提供基准线。DPO 的推导直接绕过了整个轨迹采样和优势估计，把一个需要逐 token 算 Advantage 的多步决策问题等价转换成单步的分类损失。

GRPO 介于两者之间——也砍掉了 Critic，但保留了显式的 RM 打分，用组内 z-score 归一化替代了 Value Model 的 baseline 功能。

---

## DPO 变体

### Iterative / Online DPO：解决分布偏移

标准 DPO 的致命弱点是偏好数据由旧策略生成，随着模型不断更新，数据分布与当前策略越来越不匹配。Iterative DPO 每隔一段时间用当前最新的 $\pi_\theta$ 重新生成 response，标注偏好后用新数据继续训练，本质上是把 DPO 从纯 off-policy 拉向 on-policy。需要注意的是，Iterative DPO 并没有省掉 RM——仍然需要一个 RM（或人类标注）来给新生成的 pair 打分。

### IPO：放松 Bradley-Terry 假设

DPO 的推导建立在 Bradley-Terry 模型上，而 BT 模型隐含了偏好传递性（A>B 且 B>C 则 A>C）且奖励差可以用标量精确捕获的强假设。现实中人类偏好经常违反传递性，标注噪声也大。IPO 直接优化偏好对的 log-ratio 差距，不经过 BT 模型，用 MSE loss 替代 log-sigmoid loss：

$$\mathcal{L}_{IPO} = \mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

| 维度 | DPO | IPO |
|------|-----|-----|
| 损失类型 | log-sigmoid（交叉熵） | 平方损失（MSE） |
| 优化目标 | 鼓励差距越大越好 | 让差距稳定在 $\frac{1}{2\beta}$ 附近 |
| 过拟合倾向 | 容易过度拉大差距 | MSE 自带正则化 |

DPO 的 log-sigmoid loss 结构上仍鼓励无限拉大差距；IPO 的 MSE loss 直接惩罚差距偏离目标值，自带"刹车"效果。

### SimPO：去掉 Reference Model + 消除 Length Bias

SimPO 解决两个工程痛点。一是 Reference Model 的显存开销，直接用序列的平均 log 概率作为隐式奖励，从"两驱"变成"单驱"：

$$\hat{r}_{SimPO}(x, y) = \frac{1}{|y|} \log \pi_\theta(y|x)$$

二是 Length Bias。标准 DPO 中 $\log \pi_\theta(y|x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})$ 是所有 token 的 log 概率之和，更长的序列天然有更大的负数绝对值，模型容易学到"长回答 = 好回答"的捷径。SimPO 通过除以序列长度做归一化消除了这个偏差，并加入 margin 项 $\gamma$ 要求 chosen 和 rejected 之间的奖励差必须超过最低阈值：

$$\mathcal{L}_{SimPO} = -\mathbb{E}\left[\log\sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)\right]$$

### 其他变体

KTO 不需要成对偏好数据，只需要"好/坏"的二元标签，理论基础来自行为经济学的前景理论。DPOP 针对"只降 rejected 不升 chosen"的问题，在 loss 中增加显式正则项惩罚 chosen 概率的下降。

| 问题 | 对应变体 | 核心改动 |
|------|---------|---------|
| Off-policy 分布偏移 | Iterative / Online DPO | 数据层面刷新 |
| BT 模型假设太强 | IPO | MSE loss 替代 log-sigmoid |
| Reference Model 开销 + Length Bias | SimPO | length-normalized log prob |
| 需要成对数据 | KTO | 只需二元标签 |
| 只降 rejected 不升 chosen | DPOP | 增加 chosen 概率下降惩罚项 |

---

## 关联
- 属于：[[后训练与对齐]]
- 相关：[[GRPO]] [[PPO]] [[Expert Iteration]] [[IPO]] [[SimPO]] [[KTO]]
- 用于：[[大模型对齐项目]]

## 相关概念
- [[RLHF]]
- [[GRPO]]
- [[SFT]]
- [[PPO]]
