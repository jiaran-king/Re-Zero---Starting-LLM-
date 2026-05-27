# PPO 知识体系


---

## 第一部分：全景图 —— RLHF 四阶段

PPO 不是孤立的算法，它是大语言模型训练流水线中的最后一环。理解全局有助于理解 PPO 中每个设计决策的动机。

### 1.1 四阶段概览

| 阶段 | 输入 | 目标 | 产出 |
|------|------|------|------|
| **PreTrain** | 海量文本（网页、书籍、代码） | 学会语言规律与世界知识 | Base Model（会续写，但不会对话） |
| **SFT** | 高质量问答对 | 学会按指令回答 | SFT Model（能对话，但上限取决于标注质量） |
| **Reward Model** | 人类对同一问题多个回答的排序 | 把人类偏好转化为数值信号 | Reward Model（自动打分的"裁判"） |
| **PPO** | RM 的分数 + KL 约束 | 在人类偏好方向上持续优化策略 | 最终 RLHF 模型 |

**核心比喻**：PreTrain 是「读万卷书」，SFT 是「模仿标准答案」，Reward 是「培养考官」，PPO 是「在考官指导下刷题进化」。

阶段一 PreTrain → 产出 Base Model（后续阶段的起点，但不直接对应任何一个模型）
阶段二 SFT → 产出 SFT Model → 这个 SFT Model 进入 PPO 阶段后，分裂成两个角色：

复制一份冻结起来 → 基准模型 (Reference)
另一份作为可训练的 → 训练模型 (Actor)

阶段三 Reward → 产出 Reward Model → 直接进入 PPO 阶段，冻结使用 → 奖励模型 (Reward)
阶段四 PPO 内部额外初始化的 → 状态价值模型 (Critic)，它没有对应的独立训练阶段，通常从 SFT Model 初始化并加一个 Value Head，在 PPO 训练过程中和 Actor 一起动态更新。

### 1.2 Reward Model 的训练细节

训练数据是 **(Question, Chosen, Rejected)** 三元组，而非让人类直接打分。

**为什么用比较而非打分？**
人类对绝对分数的标准不一致（不同人的"8分"含义不同），但对比两个回答谁更好则更稳定、噪声更小。

**损失函数**：

$$Loss_{RM} = -\log\sigma(r_{chosen} - r_{rejected})$$

其中 $\sigma$ 是 Sigmoid 函数。当 $r_{chosen}$ 远大于 $r_{rejected}$ 时，损失趋近于 0。

### 1.3 PPO 阶段的四模型架构

进入 PPO 训练后，内存中需要同时加载四个模型：

| 模型 | 角色 | 参数状态 | 输出 |
|------|------|----------|------|
| **基准模型 (Reference)** | 锚点 / 监督者 | 冻结 | 概率分布（用于计算 KL 散度） |
| **训练模型 (Actor)** | 决策者 | **优化中** | 生成回答（LM Head → 词表大小） |
| **奖励模型 (Reward)** | 裁判 | 冻结 | 标量分数（Score Head → 1维） |
| **状态价值模型 (Critic)** | 评论家 | **优化中** | 期望价值（Value Head → 1维） |

**工程优化**：实践中 Actor 与 Critic 共享 Transformer 底座（双头结构），Reference 与 Actor 可通过 LoRA 共享权重，显存接近减半。

####  Actor 与 Critic 共享 Transformer 底座

标准情况下，Actor 和 Critic 各持一个完整 Transformer，参数量翻倍。但两者接收相同输入（Prompt + Response），对语言的"理解"过程高度相似，区别仅在最后一步输出。
实际应用中，通常共用同一组 Transformer 层，仅在顶端分叉出两个 Head：

- **LM Head**（Actor）：隐藏状态 $H$ → 词表大小 $V$，输出 token 概率分布
- **Value Head**（Critic）：隐藏状态 $H$ → 标量 $1$，输出状态价值

```python
hidden = shared_transformer(input_tokens)   # [batch, seq_len, H]
logits = lm_head(hidden)                    # [batch, seq_len, vocab_size]  ← Actor
value  = value_head(hidden)                 # [batch, seq_len, 1]          ← Critic
```

底座参数量数十亿，两个 Head 各自仅一个线性层，参数量可忽略。
PPO 需同时加载四个模型。以 70B 半精度为例，不共享需 ~560GB 显存。共享后 Actor + Critic 从「两个 70B」变为「一个 70B + 两个线性层」，显存近乎减半。

Actor（策略损失）和 Critic（价值回归损失）的优化目标不同，共享参数时梯度会互相干扰。因此部分实现选择让 Critic 用更小的独立模型（如 Actor 70B / Critic 7B），在显存与训练质量之间折中。

---

## 第二部分：强化学习基础概念

### 2.1 核心元素

- **Environment**（环境）：智能体交互的外部系统
- **Agent**（智能体）：根据观测选择动作
- **State / Observation**：$s_t$（环境状态）/ $o_t$（实际可见的观测）
- **Action**：$a_t$（时刻 $t$ 的行为）
- **Reward**：$r_t$（即时反馈信号）

### 2.2 轨迹与回报

一条轨迹：$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$

折扣回报（从时刻 $t$ 开始）：

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}, \quad \gamma \in (0,1]$$

优化目标：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$$
$\gamma$ 越小越重视短期收益，越大越重视长期收益。

$\theta$ 是**策略网络（Actor）的参数**，也就是 Transformer 所有可学习权重的集合。

具体来说，$\pi_\theta(a_t \mid s_t)$ 表示"参数为 $\theta$ 的网络，在状态 $s_t$ 下选择动作 $a_t$ 的概率"。在 LLM 场景中，$s_t$ 是当前已有的 token 序列，$a_t$ 是下一个要生成的 token，$\pi_\theta$ 就是模型输出的词表概率分布。

$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$ 的含义就是：在当前这组参数 $\theta$ 下，策略所能获得的期望回报。PPO 的整个训练过程，就是不断调整 $\theta$，使 $J(\theta)$ 最大化。

---

## 第三部分：价值函数与优势函数

### 3.1 三个关键函数

| 函数 | 定义 | 含义 |
|------|------|------|
| 状态价值 $V^\pi(s)$ | $\mathbb{E}_\pi[G_t \mid s_t = s]$ | 在状态 $s$ 下遵循策略 $\pi$ 的期望回报 |
| 动作价值 $Q^\pi(s,a)$ | $\mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$ | 在状态 $s$ 下执行动作 $a$ 的期望回报 |
| 优势函数 $A^\pi(s,a)$ | $Q^\pi(s,a) - V^\pi(s)$ | 动作 $a$ 相对于"平均水平"好多少 |

### 3.2 为什么要用优势函数

- $A(s,a) > 0$：该动作优于平均，应增大其概率
- $A(s,a) < 0$：该动作劣于平均，应减小其概率

用优势替代直接回报，能显著降低策略梯度估计的方差。

---

## 第四部分：策略梯度与 Actor-Critic

### 4.1 策略梯度推导（Log-Derivative Trick）

从目标函数出发：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_{\tau} \pi_\theta(\tau) R(\tau)$$

对参数求导，利用 $\nabla_\theta p = p \nabla_\theta \log p$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(\tau) R(\tau)\right]$$

由于轨迹概率中环境转移项与 $\theta$ 无关，最终得到：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\right]$$
(这里是把π θ​(τ)对应的那一条轨迹展开成了从t=0到t时刻的概率乘积，然后经过log变成了和)

实际用采样均值近似：

$$\hat{g} = \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n} G_t^n \, \nabla_\theta \log \pi_\theta(a_n^t \mid s_n^t)$$

### 4.2 为什么可以减去基线

#### 动机

策略梯度的原始形式是：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

实践中 $G_t$ 的方差很大，导致梯度估计不稳定。我们想找个办法降低方差，同时又不破坏梯度的正确性（保持无偏）。

做法是把 $G_t$ 替换为 $G_t - b(s_t)$，其中 $b(s_t)$ 是仅与状态有关的基线。替换后的梯度展开为：

$$\underbrace{\mathbb{E}\left[\nabla_\theta \log \pi_\theta \cdot G_t\right]}_{\text{原始梯度}} - \underbrace{\mathbb{E}\left[\nabla_\theta \log \pi_\theta \cdot b(s_t)\right]}_{\text{需要证明 = 0}}$$

因此只需证明被减掉的部分为零，即可说明减去基线不影响梯度期望。

#### 证明

加入仅与状态有关的基线 $b(s_t)$ 后：

$$\mathbb{E}_{a_t \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \, b(s_t)\right] = b(s_t) \nabla_\theta \underbrace{\sum_{a_t} \pi_\theta(a_t \mid s_t)}_{=1} = 0$$

因此减去基线不改变梯度期望（无偏），但能降低方差。取 $b(s_t) = V^\pi(s_t)$ 即得到优势函数形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^\pi(s_t, a_t)\right]$$

#### 核心区别：为什么 $b(s_t)$ 可以消零而 $G_t$ 不行

**$b(s_t)$ 能提出来使结果为零：** $b(s_t)$ 只依赖状态 $s_t$，不依赖动作 $a_t$。所以在对 $a_t$ 求期望时，$b(s_t)$ 是常数，可以提到期望外面：

$$\mathbb{E}_{a_t}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot b(s_t)\right] = b(s_t) \cdot \underbrace{\sum_{a_t} \nabla_\theta \pi_\theta(a_t \mid s_t)}_{= \nabla_\theta 1 = 0}$$

**$G_t$ 不能提出来：** $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$，其中 $r_t$ 取决于 $(s_t, a_t)$，所以 $G_t$ **依赖于 $a_t$**。它和 $a_t$ 绑定在一起，无法从对 $a_t$ 的求和中提出来，自然也就不会消为零。

**一句话概括：** $b(s_t)$ 与 $a_t$ 无关所以可提出、可消零；$G_t$ 与 $a_t$ 有关所以不可提出、不可消零。

#### 为什么选 $V^\pi(s_t)$ 作为基线

$V^\pi(s_t)$ 是对所有动作的期望回报，只与状态有关，满足"与 $a_t$ 无关"这个条件。减掉它既不引入偏差，又能降低方差。

此时 $G_t - V^\pi(s_t)$ 就是优势函数 $A^\pi(s_t, a_t)$ 的蒙特卡洛估计。

#### 总结

| 表达式 | 含义 |
|--------|------|
| $\nabla_\theta \log \pi_\theta \cdot G_t$ | 原始策略梯度 |
| $\nabla_\theta \log \pi_\theta \cdot b(s_t)$ | 被减掉的部分（已证明 = 0） |
| $\nabla_\theta \log \pi_\theta \cdot (G_t - b(s_t))$ | 减去基线后的策略梯度（与原式等价，但方差更小） |

### 4.3 Actor-Critic 架构

| 组件 | 职责 |
|------|------|
| **Actor**（策略网络） | 输出动作分布 $\pi_\theta(a \mid s)$ |
| **Critic**（价值网络） | 估计 $V_\phi(s)$，辅助构造优势 $A_t$ |

注意：Critic 不直接"选动作"，它负责评估状态好坏，帮助 Actor 更稳定地学习。

---

## 第五部分：偏差-方差权衡与 GAE

### 5.1 TD 误差

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

直白理解：「实际发生的（奖励 + 未来预期）」减去「之前的预测」。正值说明这一步超出预期。

### 5.2 不同步数的优势估计

| 步数 | 公式 | 偏差 | 方差 |
|------|------|------|------|
| 1-step | $A^{(1)} = \delta_t$ | 大 | 小 |
| 2-step | $A^{(2)} = \delta_t + \gamma\delta_{t+1}$ | ↓ | ↑ |
| K-step | $A^{(K)} = \sum_{b=0}^{K-1}\gamma^b\delta_{t+b}$ | ↓↓ | ↑↑ |
| MC（全回报） | $G_t - V(s_t)$ | 最小 | 最大 |

**为什么看得越远方差越大？** 步数越多，策略采样和环境转移的随机扰动累计越多。

### 5.3 GAE（广义优势估计）

GAE 用参数 $\lambda$ 在上述极端之间做连续插值：

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}$$

等价递推形式（代码实现的直接依据）：

$$\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}$$

| $\lambda$ 值 | 效果 |
|--------------|------|
| $\lambda = 0$ | 退化为 1-step TD（方差小、偏差大） |
| $\lambda \to 1$ | 退化为 MC（偏差小、方差大） |
| $\lambda = 0.95$（常用） | 平衡偏差与方差 |

![alt text](image-2.png)
蒙特卡洛方法通过实际走完一条完整轨迹，用沿途收集到的真实 reward 累加来估计回报 G_t。 它不做任何近似，直接用采样得到的真实数据，所以是无偏的；但因为每条轨迹的随机性不同（不同的状态转移、不同的动作选择），不同轨迹算出来的 G_t 可能差异很大，所以方差高。
而 GAE 正是在这两个极端之间做插值——λ 接近 1 时更信任真实采样（像 MC），λ 接近 0 时更信任价值函数的估计（像 TD）。这也是为什么上一个问题中我纠正了"无穷采样"的说法：MC 不是"无穷采样"，而是"沿一条轨迹完整采样到底"。

### 5.4 GAE 在 $\lambda=1$ 时退化为 MC 的证明

展开 $\hat{A}_t = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} + \cdots$ 后，相邻项中的 $V$ 逐项对消：

$$+\gamma V_{t+1} \text{ 与 } -\gamma V_{t+1}, \quad +\gamma^2 V_{t+2} \text{ 与 } -\gamma^2 V_{t+2}, \quad \cdots$$

最终剩下 $\hat{A}_t = \sum_{l=0}^{\infty}\gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$。

### 5.5 GAE 代码实现

```python
# 倒序循环：从最后一步往回算
lastgaelam = 0
for t in reversed(range(gen_len)):
    nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    advantages_reversed.append(lastgaelam)

advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
```

---

## 第六部分：从 TRPO 到 PPO

### 6.1 问题背景

策略梯度若每次更新过大，策略分布骤变，训练可能崩溃。TRPO 通过约束 KL 散度限制步长，但实现复杂（需二阶优化）。PPO 用更简单的方式达到类似效果。

### 6.2 重要性采样

用旧策略 $\pi_{\theta'}$ 采集的数据来更新新策略 $\pi_\theta$：

$$\mathbb{E}_{x\sim p}[f(x)] = \mathbb{E}_{x\sim q}\left[f(x)\frac{p(x)}{q(x)}\right]$$

定义比率：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

### 6.3 PPO 的两种形式

**PPO-Penalty（KL 惩罚版）**：

$$Loss^{pen}(\theta) = \mathbb{E}_t\left[r_t(\theta)\hat{A}_t - \beta\,\mathrm{KL}(\pi_{\theta_{old}} \| \pi_\theta)\right]$$

**PPO-Clip（裁剪版，更常用）**：

$$Loss^{clip}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

$\epsilon$ 常取 $0.1 \sim 0.2$。

### 6.4 PPO-Clip 的分段理解

**当 $\hat{A}_t > 0$（好动作）**：
- $r_t \le 1+\epsilon$：未触发上界，目标随 $r_t$ 增大
- $r_t > 1+\epsilon$：被裁剪为 $(1+\epsilon)\hat{A}_t$，继续增大比率也不再增益

**当 $\hat{A}_t < 0$（坏动作）**：
- $r_t \ge 1-\epsilon$：目标仍可优化
- $r_t < 1-\epsilon$：被下界裁剪，避免把坏动作概率降得过猛

**核心直觉**：`clip` 限制更新幅度，`min` 选择保守目标，两者共同保证训练稳定性。PPO 不追求"完全不变"，而是把每次更新限制在可信区间内。

---

## 补充：PPO Clip 机制问答

### Q1：clip 函数是怎么工作的？

`clip(x, a, b)` 的含义是把 x 限制在 $[a, b]$ 区间内：

- 若 $x < a$，输出 $a$
- 若 $x > b$，输出 $b$
- 若 $a \le x \le b$，输出 $x$ 本身

因此 $\text{clip}(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon)$ 就是把概率比率 $r_t(\theta)$ 强行限制在 $[1-\epsilon,\; 1+\epsilon]$ 内。例如 $\epsilon = 0.2$ 时，clip 后的值被卡在 $[0.8, 1.2]$。

**为什么要 clip？**

$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ 衡量新旧策略对同一个动作的概率之比。若 $r_t$ 偏离 1 太远，说明新策略和旧策略差距过大，用旧数据估计的优势 $\hat{A}_t$ 就不再可信，更新可能崩溃。clip 就是为了防止这种情况。

**外层 `min` 的作用：**

$$L^{clip} = \mathbb{E}_t\left[\min\left(r_t\hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

`min` 取两项中较小者，效果是"悲观选择"——只保留对策略更新更保守的值：

- **$\hat{A}_t > 0$（好动作）**：希望增大 $r_t$，但 `min` 在 $r_t > 1+\epsilon$ 时让收益封顶为 $(1+\epsilon)\hat{A}_t$，防止步子太大。
- **$\hat{A}_t < 0$（坏动作）**：希望减小 $r_t$，但 `min` 在 $r_t < 1-\epsilon$ 时让惩罚封顶为 $(1-\epsilon)\hat{A}_t$，防止压得过猛。

两个方向都设了"刹车"，这是 PPO 稳定性的核心来源。

| 情况 | 越界方向 | min 的选择 | 效果 |
|---|---|---|---|
| $\hat{A}_t > 0$，$r_t > 1+\epsilon$ | 步子太大（过度增加好动作） | 选 clip 项 | **截断收益，刹车** |
| $\hat{A}_t > 0$，$r_t < 1-\epsilon$ | 好动作反而减少了 | 选未 clip 项 | 保留梯度，允许修正 |
| $\hat{A}_t < 0$，$r_t < 1-\epsilon$ | 步子太大（过度惩罚坏动作） | 选 clip 项 | **截断惩罚，刹车** |
| $\hat{A}_t < 0$，$r_t > 1+\epsilon$ | 坏动作反而增加了 | 选未 clip 项 | 保留梯度，允许修正 |
---

### Q2：$L^{clip}(\theta)$ 中的 $L$ 是什么含义？

$L$ 是 **Loss（损失函数）** 的缩写，但 $L^{clip}$ 实际上更像一个**目标函数（Objective）**，PPO 的目标是**最大化**它。

看似矛盾，但在总损失公式中：

$$L = -L^{clip} + c_1 L^{value} - c_2 L^{entropy}$$

$L^{clip}$ 前面带了**负号**。交给优化器最小化 $L$ 时，等价于最大化 $L^{clip}$。

本质上，$L^{clip}(\theta)$ 衡量的是"在 clip 约束下，当前策略能获得多少期望收益"。它越大，策略越好。加负号只是为了适配深度学习框架统一的"最小化"接口。

---

### Q3：我们是通过更新 $r_t(\theta)$ 来降低损失吗？

不完全是。$r_t(\theta)$ 本身不是直接更新的对象，它只是一个**中间量**。

我们真正更新的是 **$\theta$**（Actor 网络的参数）。更新链条如下：

1. 优化器调整 $\theta$（网络权重）
2. $\theta$ 变了 → $\pi_\theta(a_t \mid s_t)$（新策略对每个动作的概率）就变了
3. $\pi_\theta$ 变了 → $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ 自然就变了
4. $r_t$ 变了 → $L^{clip}$ 的值就变了

$r_t(\theta)$ 是 $\theta$ 的函数，是梯度传播链条上的一个环节，不是直接操控的变量。

**比喻**：想调整一道菜的味道（$L^{clip}$），你能动的是火候和调料用量（$\theta$），咸度（$r_t$）会随调料改变而改变，但你不是直接去"设定咸度"，而是通过调调料间接影响它。

## 第七部分：PPO 中的 Reward Shaping

在 LLM 的 RLHF 场景中，PPO 的奖励由两部分组成：

$$Reward_t = Score_t - \beta \times KL_t$$

| 组成部分 | 来源 | 含义 |
|----------|------|------|
| $Score_t$ | Reward Model | 仅在最后一个 token 处给出非零分（如 3.8），代表整体回答质量 |
| $KL_t$ | 训练模型 vs 基准模型 | 每个 token 位置都有值，衡量训练模型偏离基准的程度 |
| $\beta$（如 0.2） | 超参数 | KL 惩罚系数 |

**示例计算**：
- 中间 token：$Score=0, KL=1.2 \Rightarrow Reward = 0 - 0.2 \times 1.2 = -0.24$
- 最后 token：$Score=3.8, KL=2.1 \Rightarrow Reward = 3.8 - 0.2 \times 2.1 = 3.38$

这就是「带着镣铐跳舞」——既要拿高分（高 Score），又不能跑出舞台边界（低 KL）。

### Q4：GAE 公式中，Critic 和 Reward 分别在哪一步发生作用？

GAE 公式本身是最终产物，Reward 和 Critic 的作用藏在它的"原料" $\delta_t$ 里面。

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}$$

完整的关系链如下：

**第一步：Reward Model 给出分数**

对于生成的回答，Reward Model 在每个 token 位置产出 $r_t$（中间 token 的 RM 分数为 0，最后一个 token 才有实际分数，再叠加每步的 KL 惩罚）。

**第二步：Critic 给出价值估计**

Critic 对每个位置输出 $V(s_t)$——"从这个位置往后，预期还能拿多少总回报"。

**第三步：两者相遇，构成 TD 误差**

$$\delta_t = \underbrace{r_t}_{\text{Reward}} + \gamma \underbrace{V(s_{t+1})}_{\text{Critic}} - \underbrace{V(s_t)}_{\text{Critic}}$$

这一步是 Reward 和 Critic **唯一交汇的地方**。$\delta_t$ 的含义是"这一步实际发生的，比 Critic 预期的好了多少"。

**第四步：GAE 把所有 $\delta_t$ 加权求和**

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}$$

这一步纯粹是对已经算好的 $\delta$ 序列做指数衰减加权，不再涉及 Reward 或 Critic 的新计算。

**一句话总结**：Reward 提供 $r_t$，Critic 提供 $V(s_t)$，两者在 $\delta_t$ 这一步合流，然后 GAE 只是把这些 $\delta_t$ 用 $(\gamma\lambda)^l$ 加权拼起来，得到最终的优势估计 $\hat{A}_t$。

---

## 第八部分：PPO 完整训练流程

1. 用当前策略 $\pi_{old}$ 与环境交互，收集一批轨迹
2. **用 Reward Model 对每条轨迹打分，结合 KL 惩罚得到每个位置的 $r_t$**
3. 用 Critic 计算采样阶段的价值估计 $V_{\text{old}}(s_t)$，结合 $r_t$ 计算 TD 误差 $\delta_t = r_t + \gamma V_{\text{old}}(s_{t+1}) - V_{\text{old}}(s_t)$，再计算 GAE 优势 $\hat{A}_t$
4. 计算 Critic 的回报目标 / value target：
   $$R_t = \hat{A}_t + V_{\text{old}}(s_t)$$
   这里的 $V_{\text{old}}(s_t)$ 是采样阶段用于计算 GAE 的价值估计，构造出 $R_t$ 后通常会固定下来，供后续多轮 mini-batch 更新使用。
5. 固定这批数据，进行 $K$ 轮 mini-batch 更新
6. 同时优化三部分损失：
   - **策略损失** $L_{clip}$：让 Actor 向好动作倾斜
   - **价值损失** $L_{value} = (V_\phi(s_t) - R_t)^2$：让 Critic 预测更准
   - **熵奖励** $L_{entropy}$：鼓励探索，防止策略过早坍缩
7. 更新后令 $\pi_{old} \leftarrow \pi_\theta$，进入下一轮采样

**总损失**（最小化形式）：

$$L = -L^{clip} + c_1 L^{value} - c_2 L^{entropy}$$

> **修正说明**：原版第2步直接写"计算 $\delta_t$ 与 GAE"，但 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 的计算依赖 $r_t$，而 $r_t$ 来自 Reward Model 打分。原版遗漏了这一步，导致 Reward Model 在训练流程中的位置不明确。修正版将其单独列为第2步，使 Reward → Critic → GAE 的完整关系链显式可见。

PPO 本质是 **On-Policy** 算法——虽然用了重要性采样，但数据不能长期反复使用。

---

### Q5：回报目标 $R_t = \hat{A}_t + V(s_t)$ 具体指的什么？

$R_t$ 是 **Critic 的训练目标**，也叫 **value target**。它不是即时奖励，也不是"最优动作的回报"，而是根据当前采样轨迹、Reward、KL 惩罚、Critic 估计和 GAE 共同构造出来的"未来总回报估计"。

需要注意：

$$
\hat{A}_t = R_t - V(s_t)
$$

所以：

$$
R_t = \hat{A}_t + V(s_t)
$$

这个公式本质上是从优势函数的定义移项得到的。它不是说"回报由优势和价值两部分物理相加组成"，而是说：

> 优势 $\hat{A}_t$ 表示"这次采样结果相对于 Critic 原本预测 $V(s_t)$ 好了多少或差了多少"；把这个差值加回原来的预测，就得到 Critic 应该学习的目标 $R_t$。

可以这样理解：

| 量 | 含义 |
|---|---|
| $V_{\text{old}}(s_t)$ | Critic 在采样阶段对状态 $s_t$ 的预测："从这里往后大概能拿多少总回报" |
| $\hat{A}_t$ | GAE 算出的优势："这次实际轨迹比原预测好/差多少" |
| $R_t$ | Critic 的监督目标："根据这次轨迹修正后，这个状态大概应该值多少" |

因此：

$$
R_t = V_{\text{old}}(s_t) + \hat{A}_t
$$

可以理解为：

> 原预测 + 预测误差修正 = 新的训练目标。

这里特意写成 $V_{\text{old}}(s_t)$，是为了和 value loss 里的 $V_\phi(s_t)$ 区分开：$R_t$ 通常在采样阶段构造好并固定复用，而 $V_\phi(s_t)$ 是 Critic 在后续 mini-batch 更新中重新前向计算出来、并随着参数更新不断变化的当前预测。

---

### 易错点：PPO 不是在最小化 $|\hat{A}_t|$

虽然从公式看：

$$
\hat{A}_t = R_t - V(s_t)
$$

Critic 训练会让 $V_\phi(s_t)$ 接近 $R_t$，但这并不意味着 PPO 的总体目标是让优势函数绝对值变小。

更准确地说：

- **Actor 使用 $\hat{A}_t$ 判断动作好坏**：
  - $\hat{A}_t > 0$：说明这个动作比平均水平好，应提高概率；
  - $\hat{A}_t < 0$：说明这个动作比平均水平差，应降低概率。
- **Critic 使用 $R_t$ 学习状态价值**：
  - 让 $V_\phi(s_t)$ 更接近 $R_t$；
  - 目的是让后续优势估计更准确，而不是直接把所有优势压成 0。

因此，Actor 和 Critic 的角色不同：

| 模型 | 使用的量 | 目标 |
|---|---|---|
| Actor | $\hat{A}_t$ | 根据优势调整动作概率 |
| Critic | $R_t$ | 拟合回报目标，使价值估计更准 |

一句话总结：

> $R_t = \hat{A}_t + V_{\text{old}}(s_t)$ 是构造 Critic 监督目标的方法；$L_{value}$ 是训练 Critic 预测状态价值的回归损失；PPO 的最终目标仍然是提升 Actor 的长期期望回报，而不是让所有优势函数都变成 0。

---

### Q6：$R_t$ 和 Reward Model 给出的 $r_t$ 有什么关系？

两者符号相似但含义完全不同：

| 符号 | 含义 | 来源 |
|------|------|------|
| 小写 $r_t$ | 单步即时奖励 | Reward Model 打分（+ KL 惩罚） |
| 大写 $R_t$ | 回报目标 | 由一整串 $r_t$ 经过 GAE 计算得来 |

$R_t$ 和 $r_t$ 之间有一条清晰的推导链：

**Reward Model 给出 $r_t$** → 结合采样阶段 Critic 的 $V_{\text{old}}(s_t)$ 算出 **TD 误差 $\delta_t = r_t + \gamma V_{\text{old}}(s_{t+1}) - V_{\text{old}}(s_t)$** → 经过 GAE 加权求和得到 **优势 $\hat{A}_t = \sum (\gamma\lambda)^l \delta_{t+l}$** → 移项得到 **回报目标 $R_t = \hat{A}_t + V_{\text{old}}(s_t)$**

比喻：$r_t$ 是每次考试的单科成绩，$R_t$ 是加权算出来的综合绩点。Critic 要学的不是预测某一次单科分数，而是预测这个综合绩点。

---

### Q7：为什么 value loss 要让 $V_\phi(s_t)$ 接近 $R_t$？

Critic 的职责是估计状态价值：

$$
V^\pi(s_t) = \mathbb{E}_\pi[G_t \mid s_t]
$$

也就是"从状态 $s_t$ 出发，按照当前策略继续走，未来能拿到的期望总回报"。

但是这个真实期望值无法直接得到，所以 PPO 用采样轨迹构造一个近似目标 $R_t$。然后用均方误差训练 Critic：

$$
L_{value} = (V_\phi(s_t) - R_t)^2
$$

这里：

- $R_t$ 是采样阶段已经算好的固定目标；
- $V_\phi(s_t)$ 是当前 Critic 在 mini-batch 更新时重新前向计算出来的预测；
- value loss 衡量的是 Critic 的预测误差。

所以 Critic 的目标不是"让优势函数消失"，而是：

> 根据当前采样轨迹得到的回报目标 $R_t$，校准自己的价值估计 $V_\phi(s_t)$，让它以后预测得更准。

Critic 预测越准，后续采样时计算出来的优势 $\hat{A}_t = R_t - V_{\text{old}}(s_t)$ 就越可靠，Actor 的更新方向也越稳定。

---

### Q8：$G_t^{GAE}$ 的计算公式是 $r_t + \gamma V(s_{t+1})$ 吗？

不是。$r_t + \gamma V_{\text{old}}(s_{t+1})$ 是 **1-step TD 目标**，只是 $G_t^{GAE}$ 在 $\lambda=0$ 时的特例。

$G_t^{GAE}$（即 $R_t$）的完整公式：

$$R_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l} + V_{\text{old}}(s_t)$$

三种估计方式对比：

| 估计方式 | 公式 | 特点 |
|----------|------|------|
| 1-step TD 目标 | $r_t + \gamma V_{\text{old}}(s_{t+1})$ | 只看一步，方差小但偏差大 |
| MC 回报 | $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ | 看到底，偏差小但方差大 |
| $G_t^{GAE}$（即 $R_t$） | $\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l} + V_{\text{old}}(s_t)$ | 通过 $\lambda$ 在两者之间插值 |

- $\lambda = 0$ 时退化为 1-step TD 目标：$R_t = \delta_t + V_{\text{old}}(s_t) = r_t + \gamma V_{\text{old}}(s_{t+1})$
- $\lambda = 1$ 时退化为 MC 回报 $G_t$（笔记第五部分已证明）

---

### Q9：$\delta_t$ 是偏差函数吗？训练 Critic 的目的是消去 $\delta_t$ 吗？

$\delta_t$ 不是偏差函数，是 **TD 误差**（Temporal Difference Error）：

$$\delta_t = r_t + \gamma V_{\text{old}}(s_{t+1}) - V_{\text{old}}(s_t)$$

含义是"这一步实际发生的（$r_t + \gamma V_{\text{old}}(s_{t+1})$）"减去"之前的预测（$V_{\text{old}}(s_t)$）"，也就是 **采样阶段 Critic 在这一步的预测偏差**。

但需要区分两件事：

- $\delta_t$ 是采样阶段用来构造 GAE 的中间量；
- Critic 更新时直接拟合的是固定下来的 $R_t$，不是在当前 mini-batch 里反复重算并强行把所有 $\delta_t$ 压成 0。

如果 Critic 对当前策略的价值估计足够准确，TD 误差的期望会更接近 0，优势估计的噪声也会更小。但 PPO 的训练目标不是"消灭所有 TD 误差"或"让所有优势都变成 0"，而是用更准确的价值估计辅助 Actor 稳定提升长期期望回报。

## 第九部分：调参经验与常见问题

### 常用超参数起点

| 参数 | 典型值 |
|------|--------|
| $\gamma$ | 0.99 |
| $\lambda$ | 0.95 |
| clip $\epsilon$ | 0.1 ~ 0.2 |
| update epochs | 3 ~ 10 |
| mini-batch size | 64 ~ 2048 |
| learning rate | $3 \times 10^{-4}$（Adam） |

### 训练不稳定时的排查清单

1. 优势是否做了标准化 $(A - \text{mean}) / \text{std}$
2. 奖励尺度是否过大（是否需要 reward scaling）
3. Critic 是否欠拟合（价值损失居高不下）
4. 策略熵是否过快塌陷（探索不足）
5. KL 是否飙升（策略更新过猛，减小学习率或 $\epsilon$）

### 典型现象解释

| 现象 | 可能原因 |
|------|----------|
| 回报先涨后崩 | 策略步长过大或价值估计漂移 |
| 值函数 loss 小但策略没提升 | Value 过拟合、优势质量差 |
| 熵快速接近 0 | 策略过早确定化，陷入局部最优 |

---

## 附录：关键结论速查

1. 「一步展开更稳、全回报更准」本质是偏差-方差权衡
2. GAE 正是用 $\lambda$ 连续控制这种权衡
3. PPO 的核心不是"让比率等于 1"，而是"限制偏离不要太激进"
4. `clip` 与 `min` 共同定义了保守更新边界，是 PPO 稳定性的关键
5. RLHF 中 PPO 的奖励 = RM 分数 − KL 惩罚，兼顾质量与稳定

![alt text](image-1.png)


Q:所以在PPO中， reward加入的KL是new和ref的比值， 而clip是new 和 old 的比值
A:
完全正确！你总结得非常精辟，一语道破了这两个机制在数学上的本质区别。

这两个比值在公式里虽然长得很像（都是概率相除），但它们约束的时空维度和物理意义完全不同：

### 1. KL 惩罚：New vs Ref（防偏离）
* **比值核心**：$\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ 
* **对比对象**：正在更新的模型 (New)  vs  SFT 阶段冻结的初始参考模型 (Ref)。
* **核心作用（风筝线）**：它衡量的是一种**“绝对距离”**。无论模型在强化学习阶段训练了多少个 Epoch，更新了多少步，它永远都要回头和最开始的“出厂设置”对齐。这是为了防止模型为了迎合奖励而走火入魔（比如输出一堆乱码也能拿高分），确保生成的文本依然具有正常的语言逻辑和常识。

### 2. Clip 截断：New vs Old（防侧翻）
* **比值核心**：$\frac{\pi_\theta(y|x)}{\pi_{\theta_{old}}(y|x)}$
* **对比对象**：正在更新的模型 (New)  vs  当前这**一个 Batch 训练开始前**的模型 (Old)。
* **核心作用（限速器）**：它衡量的是一种**“相对步长”**。强化学习的梯度通常很不稳定，如果某一步的奖励信号特别大，模型可能会一步跨得太猛导致参数崩盘。Clip 机制强行把单次更新的概率变化范围卡在 $[1-\epsilon, 1+\epsilon]$（比如 $[0.8, 1.2]$）之间。它保证了模型每一步都在一个安全的“信任域（Trust Region）”内平滑微调。

---

**一句话速记：**
* **KL** 看的是“全局坐标”，保证你**不忘初心**；
* **Clip** 看的是“局部车速”，保证你**单步不翻车**。

既然我们已经把目标函数里的 KL 和 Clip 完全扫清了，需要我为你详细拆解一下 GRPO 中那个极其巧妙的“组内均值 baseline + z-score 归一化”，看看它是如何完美替代 PPO 里那个吃显存又难训练的 Critic 网络的吗？





# PPO 邻近策略优化 —— QA 知识点总结

---

## Q1：朴素策略梯度和 PPO 的 Clipped Objective 有什么区别？

**朴素策略梯度：**

$$\frac{1}{N}\sum_i\sum_t \log\pi_\theta(a_t|s_t)\cdot\hat{A}_t$$

这是 REINFORCE + 优势函数的基本形式，没有任何机制限制更新幅度。如果某个 $\hat{A}_t$ 特别大，一步梯度更新可能把策略拉得很远，导致训练崩溃。而且它是纯 on-policy 的，数据用一次就得扔。

**PPO Clipped Objective：**

$$Loss^{clip}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。通过 importance sampling ratio + clip 机制，限制每次更新的幅度，保证新策略不会偏离旧策略太远。

---

## Q2：PPO 中的"邻近"（Proximal）具体体现在哪里？

体现在 $r_t(\theta)$ 和 clip 的配合上：

1. **$r_t$ 度量新旧策略的距离**：当 $\pi_\theta = \pi_{\theta_{old}}$ 时 $r_t = 1$；$r_t$ 偏离 1 越远，说明新旧策略差别越大。

2. **clip 强制策略"不走太远"**：将 $r_t$ 截断在 $[1-\epsilon, 1+\epsilon]$（如 $\epsilon=0.2$ 时为 $[0.8, 1.2]$），超出范围后梯度变为零，不再继续往远处推。

3. **min 操作的具体效果**：
   - $\hat{A}_t > 0$（好动作）：$r_t$ 增大有利，但最多只享受 $r_t = 1+\epsilon$ 的收益
   - $\hat{A}_t < 0$（差动作）：$r_t$ 减小有利，但同样截断在 $1-\epsilon$

这构成了一个"信任区域"，每次更新只允许策略在旧策略的邻域内移动。相比 TRPO 用 KL 散度硬约束 + 二阶优化，PPO 用 clip 这个简单的 trick 达到了类似效果。

---

## Q3：PPO 训练中，"旧策略"出现在哪一步？

PPO 一轮训练分为两个阶段：

**采样阶段（用旧策略 $\pi_{\theta_{old}}$）：**
- 用当前策略对一批 prompt 生成 response
- 记录每个 token 的概率 $\pi_{\theta_{old}}(a_t|s_t)$
- 用 Critic 和 RM 算好 $V$、$\hat{A}_t$、reward
- 这一步完成后，数据固定

**多轮梯度更新阶段（$\theta$ 在变，数据不变）：**
- 从第一步更新起，$\theta$ 就变了，$\pi_\theta \neq \pi_{\theta_{old}}$
- 但数据仍来自采样阶段，所以是在用旧策略的数据训练新策略
- 这就需要 importance sampling ratio 来修正分布偏差，clip 保证不走太远

生成 response 的那一刻，用的策略就已经是"旧策略"了。

---

## Q4：PPO 可以一次采样、多次梯度更新吗？

可以，这正是 PPO 的核心工程优势。

- 朴素策略梯度：采样一次 → 更新一次 → 扔掉数据
- PPO：采样一次 → 在同一批数据上做多个 epoch 更新（通常 3-4 个 epoch）

采样（尤其是 LLM 自回归生成）是训练中最贵的环节，能在一批数据上多榨几次梯度更新，节省的算力非常可观。但没有 clip 的话，多更新几步后新旧策略差距过大，importance sampling 方差爆炸，训练崩溃。clip 就是给"复用次数"设了一个安全上限。

---

## Q5：每轮梯度更新中，哪些值是复用的，哪些是重新计算的？

**采样时固定、全程复用的：**
- response 的 token 序列（不会重新生成）
- $\pi_{\theta_{old}}(a_t|s_t)$（采样时记录的旧策略概率）
- reward（RM 打的分）
- $\hat{A}_t$（GAE 算出的优势函数）
- $R_t$（GAE 目标值）

**每轮更新时重新计算的：**
- $\pi_\theta(a_t|s_t)$：对固定的 token 序列做一次前向传播，得到当前策略概率（远比自回归生成便宜）
- $V_\phi(s_t)$：Critic 也在同步更新，需要重新前向传播

**然后计算 loss：**
- Actor loss：$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 乘以固定的 $\hat{A}_t$，加 clip
- Critic loss：新的 $V_\phi(s_t)$ 与固定的 $R_t$ 的 MSE，即 $(V_\phi(s_t) - R_t)^2$

每轮更新的计算开销主要就是一次前向传播 + 一次反向传播，不需要自回归生成，也不需要跑 RM 重新打分。这就是多轮更新边际成本低的原因，而采样才是真正的瓶颈。
