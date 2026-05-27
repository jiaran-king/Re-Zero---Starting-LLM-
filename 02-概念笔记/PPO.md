---
type: concept
domain: 后训练
status: active
aliases:
  - Proximal Policy Optimization
  - 邻近策略优化
---
# PPO：邻近策略优化

> [!note]
> PPO 是当前最经典的策略梯度稳定化算法之一。它解决的不是“如何做强化学习”这个大问题，而是“在用策略梯度更新模型时，怎样避免单步更新过猛导致训练崩溃”。在 LLM 的 RLHF 场景中，PPO 具体承担的是最后一跳：利用 Reward Model 的打分和 KL 约束，把 SFT 模型继续优化成更符合人类偏好的策略模型。

---

## 在 RLHF 中的位置

从训练流水线看，PPO 不是孤立算法，而是 RLHF 的最后一阶段：

```text
Pretrain → SFT → Reward Model → PPO
```

四阶段的分工不同：

| 阶段 | 解决的问题 | 产出 |
|------|-----------|------|
| Pretrain | 学会语言规律与世界知识 | Base Model |
| SFT | 学会按指令回答 | SFT Model |
| Reward Model | 把人类偏好变成可学习的数值信号 | Reward Model |
| PPO | 在偏好方向上继续优化策略，同时防止跑偏 | 最终 RLHF 模型 |

进入 PPO 训练后，典型实现里会同时出现四个角色：

| 模型 | 作用 | 是否更新 |
|------|------|:------:|
| Actor | 当前策略，负责生成回答 | 更新 |
| Critic | 估计状态价值 $V(s_t)$ | 更新 |
| Reward Model | 对完整回答打分 | 冻结 |
| Reference | 提供 KL 锚点，防止策略偏离 SFT 太远 | 冻结 |

> [!tip]
> PPO 在 RLHF 里的工程代价高，核心原因不是 clip 本身复杂，而是需要同时维护 Actor、Critic、Reward Model、Reference 四个角色，并处理逐 token 的信用分配问题。

---

## 为什么需要 PPO

最朴素的策略梯度方法直接用：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t\right]
$$

它的问题不在于方向错，而在于更新太不稳定：如果某批样本的优势特别大，一步更新就可能把策略拉得过远，导致新策略和采样这批数据时的旧策略严重失配，训练直接崩掉。

PPO 的核心思想是：

- 仍然沿用策略梯度的大框架
- 允许一批 on-policy 数据做多轮更新，提高样本利用率
- 但强行限制每一步不要偏离旧策略太远

也就是说，PPO 优化的是“稳定地进步”，而不是“每一步都走得尽量大”。

---

## 从 Actor-Critic 到优势函数

为了降低策略梯度的方差，PPO 通常采用 Actor-Critic 架构：

| 组件 | 职责 |
|------|------|
| Actor | 输出动作分布 $\pi_\theta(a \mid s)$ |
| Critic | 估计状态价值 $V_\phi(s)$ |

Critic 不负责选动作，它的职责是给当前状态一个“平均水平”估计。这样就能把原始回报 $G_t$ 换成优势函数：

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

直觉上：

- $A_t > 0$，说明这个动作比平均更好，应该提高概率
- $A_t < 0$，说明这个动作比平均更差，应该降低概率

引入基线 $b(s_t)=V(s_t)$ 的关键好处是：不改变梯度期望，但能显著降低方差。

---

## GAE：PPO 中最关键的信用分配工具

在 RLHF 场景里，Reward Model 往往只在整条回答结束后给一个总分，但策略更新需要逐 token 的训练信号。PPO 的经典做法是先构造 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

再用 GAE（Generalized Advantage Estimation）递推优势：

$$
\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}
$$

对应的递推实现形式是：

$$
\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}
$$

它解决的是偏差和方差之间的折中：

| 估计方式 | 偏差 | 方差 |
|----------|------|------|
| 1-step TD | 大 | 小 |
| Monte Carlo 全回报 | 小 | 大 |
| GAE | 可调 | 可调 |

$\lambda=0$ 时更像一步 TD，稳定但偏差大；$\lambda\to1$ 时更接近完整回报，准确但噪声更大。工程上最常用的是 $\lambda=0.95$ 左右。

> [!tip]
> PPO 在 LLM 中难训，很大一部分难点其实不在 clip，而在 Critic 和 GAE 是否能给出足够可靠的逐 token 优势信号。

---

## PPO 的核心公式

### 重要性采样比率

PPO 先用旧策略 $\pi_{\theta_{old}}$ 采样，再用新策略 $\pi_\theta$ 更新。因此需要一个新旧策略的概率比率：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

这个比率衡量的是：对同一个动作，新策略相对旧策略到底放大了多少概率。

### Clip 目标

PPO 最经典的版本是 PPO-Clip：

$$
L^{clip}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

其中 $\epsilon$ 常取 $0.1 \sim 0.2$。

它的直觉是：

- 好动作本来应该继续增大概率
- 坏动作本来应该继续减小概率
- 但一旦新旧策略差太远，就强行截断收益，不再鼓励继续猛冲

分情况理解：

| 情况 | 如果没有 clip | PPO 的处理 |
|------|---------------|------------|
| $\hat{A}_t > 0$ 且 $r_t$ 过大 | 继续猛增好动作概率 | 收益封顶，停止鼓励继续增大 |
| $\hat{A}_t < 0$ 且 $r_t$ 过小 | 继续猛压坏动作概率 | 惩罚封顶，防止过度打压 |

所以 PPO 的“邻近”不是要求 $r_t$ 必须等于 1，而是要求每一步都别偏离旧策略太远。

---

## RLHF 里的 Reward Shaping

在 LLM 的 RLHF 训练里，PPO 使用的奖励通常不是单纯的 Reward Model 分数，而是：

$$
Reward_t = Score_t - \beta \cdot KL_t
$$

这里有两个看起来相似、实际上职责完全不同的比值：

| 机制 | 比较对象 | 作用 |
|------|----------|------|
| KL 惩罚 | New vs Reference | 约束全局偏离，防止 reward hacking |
| Clip 比率 | New vs Old | 约束单步更新幅度，防止训练崩盘 |

一句话记忆：

- KL 看的是“离最初的 SFT 锚点有多远”
- Clip 看的是“这一步比上一轮走得是不是太猛”

这也是 PPO 在 RLHF 里最容易混淆、但最关键的结构差异。

### 工程里的非负 KL 估计

PPO 实现里还经常会看到一个容易记混的 KL 估计形式：

$$
(r - 1) - \log r
$$

也可以写成：

$$
r - \log r - 1
$$

它不是把 reward 本身改成这个形式，而是把逐样本 KL 估计项改成一个**期望不变、单样本非负**的形式。

设样本来自旧策略 $\pi_{old}$，并定义：

$$
r = \frac{\pi_\theta(a \mid s)}{\pi_{old}(a \mid s)}
$$

那么：

$$
-\log r
$$

是 $KL(\pi_{old} \| \pi_\theta)$ 的一个逐样本无偏估计，但单个样本可能为负。由于在 $a \sim \pi_{old}$ 下有：

$$
\mathbb{E}[r - 1] = 0
$$

所以可以加上控制变量 $r-1$：

$$
\hat{KL} = (r - 1) - \log r
$$

它的期望仍然不变，同时因为经典不等式：

$$
\log r \le r - 1
$$

所以对所有 $r>0$ 都有：

$$
r - 1 - \log r \ge 0
$$

并且当 $r=1$ 时正好等于 $0$。

对应代码通常是：

```python
log_ratio = new_logprob - old_logprob
ratio = torch.exp(log_ratio)
approx_kl = (ratio - 1) - log_ratio
```

> [!note]
> 这里正确形式是 $r - 1 - \log r$，不是 $r - \log r + 1$。如果代码里把 ratio 反过来定义，比如 $r=\pi_{old}/\pi_\theta$，公式里的符号也要跟着换，核心条件是采样分布下被加进去的控制变量期望为 $0$。

---

## 完整训练流程

PPO 一轮训练通常按这个顺序进行：

1. 用旧策略 $\pi_{old}$ 对一批 prompt 生成回答
2. 用 Reward Model 打分，并结合 KL 惩罚得到每个位置的 $Reward_t$
3. 用 Critic 计算 $V(s_t)$，再算 TD 误差 $\delta_t$ 与 GAE 优势 $\hat{A}_t$
4. 构造 Critic 的训练目标 $R_t = \hat{A}_t + V(s_t)$
5. 固定这批数据，在同一批样本上做若干轮 mini-batch 更新
6. 同时优化三部分损失：策略损失、价值损失、熵奖励
7. 更新完成后令 $\pi_{old} \leftarrow \pi_\theta$，再进入下一轮采样

总损失常写成最小化形式：

$$
L = -L^{clip} + c_1 L^{value} - c_2 L^{entropy}
$$

其中：

- $L^{clip}$ 负责更新 Actor
- $L^{value} = (V_\phi(s_t) - R_t)^2$ 负责训练 Critic
- $L^{entropy}$ 用来保留探索，避免策略过早塌缩

---

## PPO 的优势与代价

| 维度 | PPO 的优点 | PPO 的代价 |
|------|-----------|-----------|
| 稳定性 | 比朴素策略梯度稳得多 | 仍然需要细致调参 |
| 样本利用率 | 一次采样可做多轮更新 | 仍属于 on-policy，数据不能长期复用 |
| 理论与工程平衡 | 比 TRPO 简单得多 | 比 DPO / SFT 复杂得多 |
| LLM 场景适配 | 能处理在线反馈与探索 | 需要 Critic、GAE、RM、Reference，工程很重 |

这也是为什么后续很多方法都把 PPO 当成参照系：

- [DPO](DPO.md) 试图绕过 PPO 的在线 RL 复杂度
- [GRPO](GRPO.md) 试图去掉 PPO 中最吃显存、最难学的 Critic
- [RLHF](RLHF.md) 则把 PPO 作为经典完整路线的一部分




---

> [!warning]
> - PPO 里的 KL 和 clip 不是一回事：前者是 New vs Reference，后者是 New vs Old
> - PPO 的稳定性不只来自 clip，还高度依赖 Critic 和 GAE 的质量
> - 在 LLM 中，PPO 的最大工程负担往往来自四模型协作和逐 token 信用分配，而不是公式本身
> - PPO 是 on-policy 算法，一批数据可以多轮更新，但不能像离线方法那样长期反复复用

## 关联

- 属于：[后训练与对齐](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%90%8E%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%AF%B9%E9%BD%90.md)
- 相关：[RLHF](RLHF.md) [GRPO](GRPO.md) [DPO](DPO.md) SFT Reward Model
- 用于：InstructGPT 复现 大模型对齐项目 DeepSeek-R1 复现

## 相关概念

- [RLHF](RLHF.md)
- [GRPO](GRPO.md)
- [DPO](DPO.md)
- SFT
- Reward Model
