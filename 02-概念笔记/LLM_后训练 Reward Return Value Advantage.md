

> 核心问题：**一个完整回答得到的 reward，如何变成每个 token 的训练信号？**

这份笔记把 LLM 后训练里常见的 PPO、GRPO、REINFORCE、RLOO、GAE、critic、value head、advantage、reward-to-go、KL penalty 放在同一张知识图里理解。

---

## 0. 怎么使用这份笔记

建议按四遍读：

1. **第一遍看地图**：先读第 1、2、3 节，建立 reward → return → value/baseline → advantage → policy loss 的主链路。
2. **第二遍学公式**：重点读第 4、5、6 节，掌握 policy gradient、baseline、MC、TD、TD($\lambda$)、GAE。
3. **第三遍看算法**：读第 7、8、9 节，把 PPO、GRPO、RLOO、ReMax 和 Long-CoT 问题放回 LLM 后训练场景。
4. **第四遍查漏补缺**：读第 10、11、12、13 节，检查工程细节、误区、速查公式和自测题。

整份笔记的学习目标不是背算法名，而是能回答三个问题：

$$
\text{估计对象是什么？}
$$

$$
\text{baseline 从哪里来？}
$$

$$
\text{bias 和 variance 如何权衡？}
$$

---

## 1. 一页总览：从完整回答的 reward 到 token 更新

LLM 后训练可以抽象成下面这条链：

$$
\boxed{
\text{reward}
\rightarrow
\text{return}
\rightarrow
\text{baseline / value}
\rightarrow
\text{advantage}
\rightarrow
\text{policy loss}
}
$$

对应到一句话：

> Reward 告诉我们“回答好不好”；return 把未来奖励累计起来；value/baseline 给出“正常水平”；advantage 判断“比正常水平好多少”；policy loss 把这个好坏信号施加到 token logprob 上。

LLM 后训练可以拆成三层：

| 层级                            | 核心问题                   | 典型概念 / 方法                                            |
| ----------------------------- | ---------------------- | ---------------------------------------------------- |
| Reward 设计                     | 什么回答算好？                | reward model、verifier、单测、人类偏好、AI judge、格式奖励          |
| Return / Value / Advantage 估计 | 已有 reward 后，怎样变成训练信号？  | MC、TD、TD($\lambda$)、GAE、baseline、GRPO、RLOO           |
| Policy optimization           | 拿到 advantage 后，怎样更新模型？ | REINFORCE、PPO、GRPO-style objective、KL regularization |

这份笔记重点是第二层：**return / value / advantage estimation**。但 reward 设计必须先讲清楚，因为后面的 return、value、advantage 都是在已有 reward 的基础上构造出来的。

---

## 2. Reward 设计：先定义“什么回答算好”

Reward 设计回答的是：

> 模型输出什么样的 response，应该被认为是好回答？

常见 reward 来源如下。

| Reward 来源 | 输入 | 输出 | 适合场景 | 主要风险 |
|---|---|---|---|---|
| Reward model | prompt + response | 标量分数 | 开放问答、聊天、写作、安全性、帮助性 | RM 可能判断错，被 policy exploit，出现 reward hacking |
| Verifier | 题目 + 答案 / 推理 | 正误或 correctness score | 数学、逻辑、可验证问答、工具调用 | 适用范围有限，很多开放回答难以严格验证 |
| 单测 | 代码 + 测试用例 | 通过率 / 0-1 | 代码生成、算法题、函数实现 | 测试覆盖不足，可能只过测试不真正正确 |
| 人类偏好 | 多个回答比较 | A better than B | 开放对话、摘要、写作、安全性 | 昂贵、慢、有噪声、标注标准不一致 |
| AI judge | 回答 + 评分规则 | 分数、排序、理由 | 大规模自动评测、数据筛选 | 可能偏好流畅但错误的回答，也可能有模型偏见 |
| 格式奖励 | response 字符串 | 是否满足格式 | JSON、工具调用、字段抽取、最终答案格式 | 只奖励形式，不保证内容正确 |

常见组合形式：

```text
总 reward
= 正确性 reward
+ 格式 reward
+ 长度惩罚
+ KL penalty
+ 其他辅助 reward
```

这里最容易犯的错误是**不重视 reward scale**。

如果格式奖励太大，模型可能只学会套格式，而不是提高正确率。如果 correctness reward 是稀疏的 0/1，训练信号会很粗。如果 KL penalty 按 token 累加，长回答会天然承担更多 KL 成本，从而影响 return、advantage 和训练动态。

所以 reward 设计不是简单“加几个分数”，而是在定义模型真正要优化的行为。

---

## 3. 把 LLM 生成写成 RL 问题

这一节的目的，是把语言模型生成过程翻译成强化学习语言，方便后面使用 reward、return、value、advantage、policy gradient、PPO、GRPO 这些概念。

它不是在说 LLM 天然就是传统游戏环境，而是在建立一组对应关系：

| LLM 生成概念 | RL 概念 | 含义 |
|---|---|---|
| prompt + 已生成前缀 | state | 当前能用于决定下一个 token 的上下文 |
| 下一个 token | action | 当前一步选择的动作 |
| 语言模型 | policy | 给定上下文后，对下一个 token 的概率分布 |
| 完整回答 | trajectory | 多个 token 动作依次组成的一条生成轨迹 |
| 回答得分 | reward / return | 这条回答最终得到的反馈 |

一句话：

> LLM 生成一个回答，可以看成 policy 在一串 state 中依次选择 action，最终形成一条 trajectory。

给定 prompt $x$，模型已经生成的前缀是：

$$
y_{<t}=(y_1,y_2,\ldots,y_{t-1})
$$

第 $t$ 步状态是：

$$
s_t=(x,y_{<t})
$$

动作是下一个 token：

$$
a_t=y_t
$$

策略就是当前语言模型：

$$
\pi_\theta(a_t\mid s_t)
=
\pi_\theta(y_t\mid x,y_{<t})
$$

完整回答是一条轨迹：

$$
\tau=(x,y_1,y_2,\ldots,y_T)
$$

也可以简写为：

$$
\tau=(x,y_{1:T})
$$

所以这一节真正想表达的是：

> 把 LLM 生成写成 RL 问题，就是把 prompt、prefix、token、response、reward 对应到 state、action、policy、trajectory、return，从而让我们可以用 RL 方法优化语言模型生成。

这一步很重要，因为只有先完成这个翻译，后面的问题才有明确对象：

```text
这条轨迹得了多少 reward？
从某个 prefix 开始，未来期望 reward 是多少？
某个 token 比当前状态下的平均选择好多少？
应该提高还是降低这个 token 的概率？
```

也就是说，下面这条主链路的起点，就是先把生成过程写成 state-action-trajectory 形式：

```text
reward -> return -> value / baseline -> advantage -> policy loss
```

如果 reward 只在回答结束后给出，则写成：

$$
R(x,y_{1:T})
$$

例如数学题答对为 1、答错为 0；代码通过全部单测为 1；reward model 对完整回答输出一个质量分。

如果每一步都有 reward，则写成：

$$
r_t=r(s_t,a_t)
$$

但在很多 LLM 后训练任务中，尤其是数学、代码、Long-CoT reasoning，reward 通常是 **sequence-level outcome reward**，也就是整条回答结束后才得到一个总分。

这会带来核心难题：

> 最终结果只给一个分数，但训练 loss 要作用在很多 token 上。

这就是 credit assignment 问题。

---

### 3.1 为什么 response-level reward 能进入 token-level loss

假设一条回答最后得分是：

$$
R(x,y_{1:T})
$$

最简单的 REINFORCE 形式会把这条轨迹的 reward 乘到整条回答的 logprob 上：

$$
\nabla_\theta J(\theta)
\approx
R(x,y_{1:T})
\nabla_\theta
\log \pi_\theta(y_{1:T}\mid x)
$$

而自回归语言模型满足：

$$
\pi_\theta(y_{1:T}\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t})
$$

取 log 后：

$$
\log\pi_\theta(y_{1:T}\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t})
$$

所以 policy gradient 可以写成 token-level 的和：

$$
\nabla_\theta J(\theta)
\approx
R(x,y_{1:T})
\sum_{t=1}^{T}
\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t})
$$

这里要注意：

> 不是 reward 自动变成了 token-level，而是 response logprob 可以拆成 token logprob。

如果最终 reward 是 1，那么直接使用 REINFORCE 会提高整条回答里所有 token 的概率；如果最终 reward 很低，就会降低整条回答里所有 token 的概率。

这种做法能训练，但 credit assignment 很粗：它没有告诉我们到底是哪一步推理导致成功，哪个 token 是关键错误，哪些中间步骤虽然最终答案错但本身仍然有价值。

这也是后面为什么需要 return、value、baseline、advantage、GAE、GRPO、RLOO 等方法：它们都在尝试把一个轨迹级反馈变成更稳定、更有比较意义的训练信号。

---

### 3.2 SFT 和 RL 后训练的区别

SFT 中，我们通常有标准答案 token，训练目标是模仿：

```text
提高数据集中标准答案 token 的概率。
```

RL 后训练中，我们不一定有逐 token 标准答案。模型先自己生成回答，然后外部 reward 机制给这条回答打分，训练目标变成：

```text
提高高 reward 轨迹的概率，降低低 reward 轨迹的概率。
```

因此，把 LLM 生成写成 RL 问题的更深含义是：

> 把训练语言模型从 next-token imitation，改写成优化一个会生成 token 动作序列的 policy。

没有这一步，value function 就不知道是在估计哪个状态的 future return；advantage 也不知道是在比较哪个 action；PPO / GRPO 的 token-level objective 也就没有清晰来源。

---

## 4. 核心概念：Reward、Return、Value、Q、Advantage

### 4.1 五个概念的直觉版

| 概念 | 一句话解释 | LLM 中的对应物 |
|---|---|---|
| Reward | 实际拿到的反馈 | verifier 分数、RM 分数、单测通过率、答案正确性 |
| Return | 从某个时刻开始累计的 reward | 某个 token 之后这条回答最终拿到的累计分数 |
| Value $V(s)$ | 在状态 $s$ 下继续生成的期望 return | 给定 prompt + 当前 prefix，继续写下去的期望得分 |
| Q-value $Q(s,a)$ | 在状态 $s$ 先选动作 $a$ 后的期望 return | 当前 prefix 下选某个 token 后，后续期望得分 |
| Advantage $A(s,a)$ | 这个动作比当前状态下的平均选择好多少 | 当前 token / response 相对 baseline 有多好 |

可以这样记：

$$
\text{reward 是实际反馈}
$$

$$
\text{return 是轨迹上的累计反馈}
$$

$$
\text{value 是还没发生之前的期望反馈}
$$

$$
\text{advantage 是相对平均水平的超额反馈}
$$

---

### 4.2 Reward

单步 reward：

$$
r_t
$$

完整回答 reward：

$$
R(x,y_{1:T})
$$

reward 是实际观察到的反馈。它不是期望，也不是 baseline。

---

### 4.3 Return

从第 $t$ 步开始往后的累计回报：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
$$

其中 $\gamma$ 是 discount factor。

在很多有限长度文本生成任务里，常取：

$$
\gamma=1
$$

如果只有最终 reward，并且设：

$$
r_T=R(x,y_{1:T}),\quad r_t=0\quad(t<T)
$$

则：

$$
G_t=\gamma^{T-t}R(x,y_{1:T})
$$

当 $\gamma=1$ 时：

$$
G_t=R(x,y_{1:T})
$$

也就是说，在只有最终 reward 且 $\gamma=1$ 时，每个 response token 的 return 都一样。这很简单，但 credit assignment 很粗糙。

---

### 4.4 Value Function

状态价值函数：

$$
V^\pi(s_t)=\mathbb E_\pi[G_t\mid s_t]
$$

LLM 语境下：

> 给定 prompt 和当前生成前缀，从这里继续生成，最终期望能拿到多少 reward？ 如果我从这个 prefix 开始，继续让模型采样很多次，平均 return 会是多少？

例如：

```text
Prompt: 解一道数学题
Prefix: “我们先设 x = ...”
V(prefix) = 0.63
```

可以理解为：从这个前缀继续生成，最终答对或拿高分的期望约为 0.63。

---

### 4.5 Action-Value Function

动作价值函数：

$$
Q^\pi(s_t,a_t)=\mathbb E_\pi[G_t\mid s_t,a_t]
$$

LLM 语境下：

> 在当前 prefix 下先选某个 token，再继续生成，最终期望能拿到多少 reward？

Q-value 在经典 RL 中很重要；在 RLHF / LLM 后训练中，更常见的是 value head 估计 $V(s)$，然后用 return 减去 baseline 构造 advantage。

---

### 4.6 Advantage Function

优势函数：

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

直觉：

```text
A > 0：这个 token / response 比当前状态下的平均选择更好，应提高概率
A < 0：这个 token / response 比当前状态下的平均选择更差，应降低概率
```

实践中常写成：

$$
\hat A_t=\hat G_t-b_t
$$

其中 $b_t$ 是 baseline，可以是 learned value，也可以是 group mean、leave-one-out mean、greedy baseline、batch baseline 等。

---

## 5. Policy Gradient 与 Baseline：为什么 reward 不能直接用

### 5.1 最基础的 REINFORCE

REINFORCE 的基本 policy gradient 形式是：

$$
\nabla_\theta J(\theta)
\approx
\sum_t
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

直觉：

> 如果一条轨迹 return 高，就提高这条轨迹中动作的概率；如果 return 低，就降低概率。

问题是：这种估计方差很高。

---

### 5.2 加入 baseline

更常见的形式是：

$$
\nabla_\theta J(\theta)
\approx
\sum_t
(G_t-b_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

令：

$$
\hat A_t=G_t-b_t
$$

于是单个 token / action 的梯度样本是：

$$
\boxed{
\hat g_t
=
\hat A_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
}
$$

也就是：

$$
\boxed{
\text{gradient sample}
=
\text{好坏信号}
\times
\text{提高当前动作概率的方向}
}
$$

其中：

- $\hat A_t=G_t-b_t$ 决定更新的**正负和强度**。
- $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 决定**如果要提高当前动作概率，参数应该怎么动**。

---

### 5.3 同样 reward，在不同 prompt 下可能方向相反

假设某回答 reward 都是 0.8。

| Prompt | 平均 reward / baseline | 某回答 reward | Advantage | 训练含义 |
|---|---:|---:|---:|---|
| 简单题 | 0.95 | 0.80 | -0.15 | 低于预期，不该强化 |
| 困难题 | 0.20 | 0.80 | +0.60 | 明显好于预期，应强化 |

结论：

$$
\boxed{
\text{绝对 reward 不够，训练真正看的是相对 baseline 的差。}
}
$$

---

### 5.4 Baseline 为什么不改变期望梯度

只要 $b_t$ 不依赖当前动作 $a_t$，它不会改变 policy gradient 的期望方向。

关键恒等式是：

$$
\mathbb E_{a\sim\pi}
[
\nabla_\theta\log\pi_\theta(a\mid s)
]
=0
$$

推导：

$$
\mathbb E_{a\sim\pi}
[
\nabla_\theta\log\pi_\theta(a\mid s)
]
=
\sum_a
\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)
$$

由于：

$$
\nabla_\theta\log\pi_\theta(a\mid s)
=
\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
$$

所以：

$$
\sum_a
\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)
=
\sum_a
\nabla_\theta\pi_\theta(a\mid s)
$$

又因为：

$$
\sum_a\pi_\theta(a\mid s)=1
$$

因此：

$$
\sum_a
\nabla_\theta\pi_\theta(a\mid s)
=
\nabla_\theta 1
=0
$$

所以：

$$
\mathbb E[
 b(s)\nabla_\theta\log\pi_\theta(a\mid s)
]
=0
$$

这就是 baseline 不改变期望梯度的原因。

注意：这要求 baseline 不依赖当前动作。它可以依赖状态 $s_t$，例如 prompt 和 prefix；但不能根据当前采样 token 本身来偏置。

---

### 5.5 Baseline 降低的到底是什么方差

单看 return，减去常数 baseline 不会改变方差：

$$
\operatorname{Var}(G-b)=\operatorname{Var}(G)
$$

所以 baseline 降低的不是 return 本身的方差。

policy gradient 里的随机变量是：

$$
G\nabla_\theta\log\pi_\theta(a\mid s)
$$

加入 baseline 后变成：

$$
(G-b)\nabla_\theta\log\pi_\theta(a\mid s)
$$

因此 baseline 降低的是：

$$
\boxed{
\text{整个 gradient estimator 的方差}
}
$$

直觉上，return 中可能有一部分是“与当前动作选择无关的公共分数”。这部分公共 offset 乘上 score function 会制造随机波动，但在期望上并不贡献学习方向。baseline 的作用就是去掉这种无效波动，让剩下的信号更接近：

$$
\text{这个动作比平均水平好多少？}
$$

---

### 5.6 极简数值例子

当前状态下有两个动作 A 和 B，概率都是 0.5。

$$
G(A)=101,\quad G(B)=99
$$

假设 score function 为：

$$
\nabla_\theta\log\pi(A\mid s)=+0.5
$$

$$
\nabla_\theta\log\pi(B\mid s)=-0.5
$$

不加 baseline：

$$
A: 101\times0.5=50.5
$$

$$
B: 99\times(-0.5)=-49.5
$$

样本梯度在 50.5 和 -49.5 之间大幅跳动。

加 baseline $b=100$：

$$
A:(101-100)\times0.5=0.5
$$

$$
B:(99-100)\times(-0.5)=0.5
$$

两个样本梯度都变成 0.5，方差大幅降低。

注意，return 只是从 $[101,99]$ 平移到 $[1,-1]$，return 的方差没有因为平移而改变。真正变稳定的是 gradient sample。

---

### 5.7 记忆版

最短记三句话：

$$
\text{gradient sample}
=
\text{advantage}
\times
\text{score function}
$$

$$
\text{advantage}
=
\text{return}
-
\text{baseline}
$$

$$
\text{baseline removes action-independent reward offset, reducing gradient variance.}
$$

中文就是：

> 梯度样本 = 好坏信号 × 提高当前动作概率的方向。Baseline 去掉与动作无关的平均回报成分，让梯度样本更少被无效波动污染。

---

## 6. 为什么 LLM 中会出现 token-level loss

### 6.1 不是 reward 变成了 token-level，而是 logprob 可以拆成 token-level

一个回答：

$$
o_i=(o_{i,1},o_{i,2},\ldots,o_{i,T_i})
$$

在自回归模型下：

$$
\pi_\theta(o_i\mid q)
=
\prod_{t=1}^{T_i}
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

取 log：

$$
\log\pi_\theta(o_i\mid q)
=
\sum_{t=1}^{T_i}
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

所以完整 response 的 policy gradient 可以拆到 token 级别：

$$
\nabla_\theta J(\theta)
\approx
A_i
\sum_{t=1}^{T_i}
\nabla_\theta
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

基础 policy-gradient loss 是：

$$
\ell_{i,t}^{\mathrm{PG}}
=
-A_i
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

这里要特别注意：

$$
\boxed{
\text{很多时候 reward / advantage 仍然是 response-level，}
\text{但 logprob、loss、gradient 是 token-level。}
}
$$

---

### 6.2 Token-level loss 不等于解决了 token-level credit assignment

如果最终 reward 是 response-level 的 0/1，那么把同一个 $A_i$ 乘到每个 token 上，只是把整条回答作为整体强化或削弱。

这并没有告诉我们：

```text
到底是哪一步推理导致成功？
到底哪个 token 是关键错误？
哪些中间步骤虽然最终答案错但本身是有价值的？
```

因此：

> Token-level loss 是优化形式上的 token-level；真正的 token-level credit assignment 需要更细的 process reward、step verifier、value estimate 或其他中间监督来改善。

---

### 6.3 Sample-level loss 和 token-level 聚合的差异

如果使用 response-level advantage $A_i$，常见两种聚合方式：

**按 token 求和：**

$$
L_i=-A_i\sum_{t=1}^{T_i}\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

长回答 token 更多，总梯度可能更大。

**按 token 平均：**

$$
L_i=-A_i\frac{1}{T_i}\sum_{t=1}^{T_i}\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

长短回答的总梯度规模更接近。

这不是无害工程细节。它会影响：

```text
长度偏置
CoT 变长还是变短
KL 成本如何累计
不同样本之间的梯度尺度
```

---

## 7. 从 DP 到 GAE：经典 RL 估计方法

这一章主线是 bias-variance tradeoff：

$$
\text{DP}
\rightarrow
\text{MC}
\rightarrow
\text{TD(0)}
\rightarrow
\text{n-step TD}
\rightarrow
\text{TD}(\lambda)
\rightarrow
\text{GAE}
$$

---

### 7.1 DP：知道环境模型时的 Bellman backup

DP 假设我们知道完整环境模型：

$$
P(s'\mid s,a),\quad R(s,a,s')
$$

policy evaluation 的 Bellman expectation equation：

$$
V^\pi(s)=
\sum_a\pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^\pi(s')\right]
$$

最优价值的 Bellman optimality equation：

$$
V^*(s)=
\max_a
\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^*(s')\right]
$$

DP 的核心直觉：

> 如果知道所有状态转移和奖励，就可以递推地算出每个状态的价值。

LLM 后训练中很少直接用 DP，因为状态空间是所有 prompt + prefix，巨大且不可枚举；reward 也常来自 verifier、reward model、人类偏好或外部工具，并不存在一个可枚举的完整环境模型。

因此，DP 更像理论起点。后面的 TD 可以看成：不知道完整模型时，用采样轨迹近似 Bellman backup。

---

### 7.2 MC：完整 rollout 后用真实 return

Monte Carlo 方法等 episode 结束后，用真实 return 更新。

$$
G_t=r_t+\gamma r_{t+1}+\cdots+\gamma^{T-t}r_T
$$

如果只有最终 reward 且 $\gamma=1$：

$$
G_t=R(x,y_{1:T})
$$

REINFORCE 更新：

$$
\nabla_\theta J(\theta)
\approx
\sum_{t=1}^{T}
G_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
$$

MC 的优点：

```text
简单
不需要 value model
不 bootstrap
使用完整真实 return，bias 低
```

MC 的缺点：

```text
方差高
需要完整 rollout
reward 稀疏时 credit assignment 很弱
长 CoT 中最终 0/1 reward 很难说明哪一步推理错了
```

---

### 7.3 Reward-to-go：只让当前动作负责未来

普通 trajectory-level return 可能把整条轨迹的 reward 都分配给每个动作。但在一般 RL 中，当前动作不应该为过去已经发生的 reward 负责。

所以使用 reward-to-go：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
$$

如果只有最终 reward，reward-to-go 和整条 reward 差别不大。

但如果存在 process reward、step reward、token-level reward、per-token KL penalty，差别就很重要。

例如：

```text
Step 1 正确: +0.2
Step 2 正确: +0.2
Step 3 错误: -0.5
最终答案错误: 0
```

这时第 1 步、第 2 步、第 3 步的 return 不应该完全相同。reward-to-go 能更精确地表达“当前 token 之后发生了什么”。

---

### 7.4 TD(0)：一步 bootstrap

TD 的全称是 Temporal Difference，中文常译为时序差分。

TD 的核心：

> 不等 episode 完整结束，而是用“一步真实 reward + 下一状态 value 估计”来更新当前 value。

TD(0) target：

$$
y_t^{\mathrm{TD}}=r_t+\gamma V(s_{t+1})
$$

TD error：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

更新：

$$
V(s_t)\leftarrow V(s_t)+\alpha\delta_t
$$

TD target 叫 bootstrap target，因为它没有等完整未来 reward 都发生，而是用了当前 value function 对未来的估计：

$$
\underbrace{r_t}_{\text{真实观测}}
+
\gamma
\underbrace{V(s_{t+1})}_{\text{模型估计}}
$$

MC target 是完整真实未来：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
$$

TD target 是一步真实 + 后面估计：

$$
r_t+\gamma V(s_{t+1})
$$

所以 MC 不 bootstrap，TD bootstrap。

---

### 7.5 TD 学习的目标不是让每个样本 TD error 都为 0

真实 value function 应满足 Bellman equation：

$$
V^\pi(s_t)=
\mathbb E_\pi[
 r_t+\gamma V^\pi(s_{t+1})
\mid s_t
]
$$

所以 TD 的深层目标是 Bellman 一致性：

$$
V(s)
\approx
\mathbb E[r+\gamma V(s')\mid s]
$$

更准确地说，TD 希望：

$$
\mathbb E[\delta_t\mid s_t]=0
$$

而不是每个样本都满足：

$$
\delta_t=0
$$

原因是环境、reward、策略采样都可能有随机性。

例如某状态下：

```text
50% 概率 reward = 0
50% 概率 reward = 2
```

真实 value 是 1。单次采样 reward=0 时 TD error 可能是 -1；reward=2 时 TD error 可能是 +1。单次误差不为 0，但期望为 0。

一句话：

> TD 学习希望消除 value function 的系统性 Bellman 误差，而不是消除每个随机样本上的误差。

---

### 7.6 n-step TD：TD(0) 和 MC 的桥梁

TD(0) 只看一步：

$$
G_t^{(1)}=r_t+\gamma V(s_{t+1})
$$

MC 看到 episode 结束：

$$
G_t=r_t+\gamma r_{t+1}+\cdots+\gamma^{T-t}r_T
$$

n-step TD 介于两者之间：

$$
G_t^{(n)}=
 r_t+\gamma r_{t+1}+\cdots+\gamma^{n-1}r_{t+n-1}
 +\gamma^n V(s_{t+n})
$$

边界情况：

$$
n=1\Rightarrow \text{TD(0)}
$$

$$
n\to T-t\Rightarrow \text{MC}
$$

核心 tradeoff：

| n 的大小 | 结果 |
|---|---|
| 小 n | 更依赖 bootstrap，bias 高一些，variance 低一些 |
| 大 n | 更接近真实 return，bias 低一些，variance 高一些 |

---

### 7.7 TD($\lambda$)：多个 n-step return 的加权平均

TD($\lambda$) 不固定一个 $n$，而是把不同长度的 n-step return 加权平均。

无限 horizon 的写法：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

展开：

$$
G_t^\lambda
=
(1-\lambda)G_t^{(1)}
+
(1-\lambda)\lambda G_t^{(2)}
+
(1-\lambda)\lambda^2G_t^{(3)}
+\cdots
$$

每个 $G_t^{(n)}$ 的权重是：

$$
w_n=(1-\lambda)\lambda^{n-1}
$$

这些权重非负，并且：

$$
\sum_{n=1}^{\infty}(1-\lambda)\lambda^{n-1}=1
$$

所以 TD($\lambda$) 是标准意义上的多个 n-step return 的加权平均。

边界情况：

$$
\lambda=0\Rightarrow \text{TD(0)}
$$

$$
\lambda=1\Rightarrow \text{MC}
$$

$\lambda$ 控制 bias-variance tradeoff：

| $\lambda$ | 含义 |
|---:|---|
| 接近 0 | 更短视，更依赖 value bootstrap，方差低但 bias 高 |
| 接近 1 | 更接近完整 MC return，bias 低但方差高 |

有限 episode 中最后通常会把剩余权重分给完整 MC return；核心直觉仍然是用 $\lambda$ 控制 TD 和 MC 的折中。

---

### 7.8 GAE：TD($\lambda$) 的 advantage 版本

GAE，即 Generalized Advantage Estimation，用来估计 advantage，而不是直接估计 value。

先定义 TD residual：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

然后把未来多个 TD residual 做指数加权：

$$
\hat A_t^{\mathrm{GAE}}
=
\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}
$$

终止状态一般令：

$$
V(s_{T+1})=0
$$

GAE 的直觉：

> 如果未来连续多个 TD residual 都说明“这条轨迹比 value 预期更好”，那当前 action 的 advantage 就应该更大。

GAE 输出 $\hat A_t$，通常进入 policy loss：

$$
L_\pi\sim -\hat A_t\log\pi_\theta(a_t\mid s_t)
$$

如果 $\hat A_t>0$，提高该 action 的概率；如果 $\hat A_t<0$，降低该 action 的概率。

TD($\lambda$) 和 GAE 的关系：

| 方法 | 输出 | 用途 | 公式核心 |
|---|---|---|---|
| TD($\lambda$) | $G_t^\lambda$ | 训练 value function / critic | 多个 n-step return 加权 |
| GAE | $\hat A_t^{\mathrm{GAE}}$ | 更新 policy / actor | 多个 TD residual 加权 |

在很多设定下，可以近似理解为：

$$
\hat A_t^{\mathrm{GAE}}
\approx
G_t^\lambda - V(s_t)
$$

也就是说：

```text
TD(lambda) 产生 return-like 的 value target
GAE 产生 return-like target 减 baseline 后的 advantage
```

GAE 的 $\lambda$ 不是神秘技巧，而是在控制：

$$
\text{更依赖 critic 的低方差估计}
\quad\leftrightarrow\quad
\text{更接近真实 rollout 的低偏差估计}
$$

---

## 8. 两条主线：critic-based 与 critic-free

LLM 后训练中的 advantage 估计大致分成两条线。

### 8.1 Critic-based 路线

核心思想：训练一个 value model / value head 来估计 $V(s_t)$。

典型流程：

$$
\text{rollout}
\rightarrow
\text{reward}
\rightarrow
\text{value head}
\rightarrow
\text{GAE}
\rightarrow
\text{PPO update}
$$

代表方法：PPO + GAE。

优点：

```text
可以做 token-level value 估计
GAE 在 actor-critic 框架中成熟稳定
对一般 RL 问题适用范围广
```

缺点：

```text
需要额外 value head / critic
显存和训练目标更复杂
critic 如果学坏，advantage 会被污染
```

---

### 8.2 Critic-free 路线

核心思想：不训练显式 $V(s)$，而用采样组、batch、greedy response 等构造 baseline。

典型形式：

$$
\hat A_i=R_i-b_i
$$

代表方法：REINFORCE、RLOO、GRPO、ReMax。

优点：

```text
不需要 value head，工程更简单
对 outcome reward 明确的数学、代码、verifier 场景很适合
同一个 prompt 多采样时，可以自然构造相对优势
```

缺点：

```text
需要多个 samples 才能得到稳定 baseline
对 reward 方差、group size、长度处理很敏感
credit assignment 仍然粗糙
```

---

### 8.3 方法总览表

| 方法 | 估计对象 | baseline 来源 | 是否 bootstrap | 是否需要完整 rollout | 是否需要 critic | LLM 中的位置 |
|---|---|---|---:|---:|---:|---|
| MC / REINFORCE | $G_t$ 或 $R$ | 可无，也可 batch baseline | 否 | 是 | 否 | 最简单的策略梯度 |
| TD(0) | $V(s)$ | learned value | 是 | 否 | 是 | critic 学习基础 |
| n-step TD | $G_t^{(n)}$ | learned value | 部分 | 部分 | 是 | MC 与 TD 的桥梁 |
| TD($\lambda$) | $V(s)$ | learned value | 混合 | 通常需要 rollout 片段 | 是 | GAE 的前置思想 |
| GAE | $A_t$ | $V(s_t)$ | 混合 | 通常 rollout 后计算 | 是 | PPO-RLHF 常见 |
| GRPO | group-relative advantage | 同 prompt 组均值/标准差 | 通常否 | 是 | 否 | reasoning RL 常见 |
| RLOO | leave-one-out advantage | 其他 samples 均值 | 否 | 是 | 否 | critic-free RLHF |
| ReMax | baseline-adjusted return | prompt 绑定的确定性 baseline | 否 | 是 | 否 | 简化 RLHF |

---

## 9. LLM 后训练中的主要算法

### 9.1 REINFORCE：最简单的策略梯度

如果一条 response 的 reward 是 $R_i$，可以用：

$$
L_{\pi}=-R_i\sum_t\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
$$

加入 baseline 后：

$$
L_{\pi}= -\hat A_i\sum_t\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
$$

其中：

$$
\hat A_i=R_i-b_i
$$

如果 $\hat A_i>0$，最小化 loss 会提高该 response 中 token 的概率；如果 $\hat A_i<0$，会降低它们的概率。

---

### 9.2 PPO + GAE：critic-based 经典路线

PPO 中常用的 clipped surrogate objective 是：

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb E_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

其中：

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

PPO 的核心不是 advantage 的定义，而是：

> 在更新 policy 时限制新旧策略比值，避免一步更新太大。

典型 PPO-RLHF 流程：

```text
prompt batch
→ 当前 policy rollout responses
→ reward model / verifier 给 outcome reward
→ 计算 reference model logprob，并加入 KL penalty
→ value head 预测每个 token prefix 的 V(s_t)
→ 用 GAE 计算 token-level advantage
→ 用 PPO clipped objective 更新 policy
→ 用 value loss 更新 value head
```

value head 的训练目标通常类似：

$$
L_V=\mathbb E_t[(V_\phi(s_t)-\hat G_t)^2]
$$

其中：

$$
\hat G_t=\hat A_t+V_\phi(s_t)
$$

这里的 $\hat G_t$ 是由 GAE 反推出来的 value target，不是 reward model 的原始分数。

---

### 9.3 KL penalty：LLM RLHF 的特殊项

LLM 后训练通常不会只最大化 reward model 分数，还会约束当前 policy 不要偏离 reference model 太远。

常见 per-token 近似写法：

$$
r_t^{\mathrm{KL}}
=-\beta\left[
\log\pi_\theta(y_t\mid s_t)
-
\log\pi_{\mathrm{ref}}(y_t\mid s_t)
\right]
$$

然后把它加入 reward 序列。

如果 outcome reward 只在最后一步给出，一个常见结构是：

$$
r_t=r_t^{\mathrm{KL}}\quad(t<T)
$$

$$
r_T=R(x,y_{1:T})+r_T^{\mathrm{KL}}
$$

这样 GAE 或 reward-to-go 会同时考虑最终任务分数和每个 token 的 KL 成本。

关键提醒：

$$
\boxed{
\text{KL penalty 会影响 return 和 advantage，不只是影响 policy loss。}
}
$$

尤其在长回答中，如果 KL 按 token 累加，回答越长，累计 KL 成本越大。

---

### 9.4 GRPO：同 prompt 组内相对优势

GRPO 的核心：对同一个 prompt 采样多个 responses，形成 group，然后用组内 reward 的相对高低构造 advantage。

对同一个 prompt 采样 $G$ 个回答：

$$
y_1,y_2,\ldots,y_G
$$

对应 reward：

$$
R_1,R_2,\ldots,R_G
$$

常见 group-relative advantage：

$$
\hat A_i=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}
$$

直觉：

> 不问这个回答绝对有多好，而问它比同一个 prompt 下的其他回答好多少。

GRPO 的优势：

```text
不需要 value head
适合数学、代码、verifier reward 明确的场景
同 prompt 多采样天然提供 baseline
```

GRPO 的风险：

```text
group size 太小，baseline 噪声大
组内 reward 方差很小，标准化可能不稳定
所有 samples 都错时，仍可能强化“相对没那么差”的错误轨迹
长度归一化、token-level loss 聚合方式会显著改变训练动态
```

一个重要区分：GRPO 的 group mean / std 是 group-relative 训练信号，不完全等同于经典“严格不依赖当前动作”的 value baseline。RLOO 的 leave-one-out baseline 在这个意义上更接近经典 baseline 的直觉。

---

### 9.5 RLOO：Leave-One-Out baseline

RLOO，即 REINFORCE Leave-One-Out。

同一个 prompt 采样 $G$ 个回答。第 $i$ 个回答的 baseline 使用其他 $G-1$ 个回答的平均 reward：

$$
b_i=\frac{1}{G-1}\sum_{j\ne i}R_j
$$

advantage：

$$
\hat A_i=R_i-b_i
$$

直觉：

> 评价一个回答好不好时，拿它和同 prompt 下其他回答比较，但不要让它自己的 reward 参与定义自己的 baseline。

GRPO 与 RLOO 对比：

| 方法 | baseline | 是否通常标准化 | 关键词 |
|---|---|---|---|
| GRPO | group mean | 常见做 group std normalization | group-relative |
| RLOO | leave-one-out mean | 不一定 | baseline 不包含自己 |

二者都是 critic-free advantage estimation。

---

### 9.6 ReMax：用确定性 baseline 简化 RLHF

ReMax 是 REINFORCE-style 的简化路线。它不训练额外 value model，而是利用和 prompt 绑定的 baseline 降低方差。

抽象地看：

$$
\hat A_i=R(x_i,y_i)-b(x_i)
$$

其中 $b(x_i)$ 可以来自确定性生成、greedy response 或其他低方差的 prompt-level baseline。具体形式取决于实现。

它提醒我们：

> LLM 后训练不一定总需要完整 actor-critic 框架。如果 reward 是 sequence-level，且 rollout 成本可控，一个设计良好的 prompt-level baseline 可能已经很有效。

---

## 10. GRPO / PPO 中 token-level ratio 和 clipped loss 的来源

### 10.1 Sequence-level ratio

完整回答的概率比可以写成：

$$
\rho_i(\theta)=
\frac{\pi_\theta(o_i\mid q)}{\pi_{old}(o_i\mid q)}
$$

由于自回归分解：

$$
\pi_\theta(o_i\mid q)=
\prod_t\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

所以：

$$
\rho_i(\theta)=
\prod_t
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{old}(o_{i,t}\mid q,o_{i,<t})
}
$$

也就是：

$$
\rho_i(\theta)=\prod_t \rho_{i,t}(\theta)
$$

问题是，长序列中乘很多 token-level ratio，极容易数值不稳定，方差也会很大。

---

### 10.2 Per-token ratio

实践中更常见的是使用 per-token ratio：

$$
\rho_{i,t}(\theta)=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{old}(o_{i,t}\mid q,o_{i,<t})
}
$$

实际实现通常保存 old logprob：

$$
\log\rho_{i,t}
=
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-
\log\pi_{old}(o_{i,t}\mid q,o_{i,<t})
$$

然后：

$$
\rho_{i,t}=\exp(\log\rho_{i,t})
$$

这就是 PPO / GRPO-style objective 中 token-level ratio 的来源。

---

### 10.3 Clipped policy loss

如果某个 token 的 advantage 是 $\hat A_{i,t}$，PPO-style clipped objective 常写成：

$$
\min\left(
\rho_{i,t}\hat A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}
\right)
$$

若实现为 loss，则通常取负号再最小化。

在 GRPO 中，$\hat A_i$ 常常是 response-level 的 group-relative advantage，然后 broadcast 到该 response 的所有有效 response tokens：

$$
\hat A_{i,t}=\hat A_i
$$

这说明：

```text
logprob 是 token-level
ratio 是 token-level
KL 常是 token-level
mask 是 token-level
loss 项是 token-level
但 advantage 可能仍是 response-level
```

---

### 10.4 Mask 很重要

训练时必须 mask 掉：

```text
prompt tokens
padding tokens
无效 response tokens
截断后不该训练的 tokens
```

policy loss 通常只对 response tokens 计算。

否则模型可能错误地优化 prompt 部分，或者让 padding 影响统计。

---

## 11. Long-CoT RL 的特殊难点

长 CoT reasoning 常见 reward 是：

$$
R\in\{0,1\}
$$

回答长度可能是几百到几千 token。这会带来几个问题。

---

### 11.1 Credit assignment 粗糙

最终答案正确，不代表每一步推理都正确。

最终答案错误，也不代表所有前置推理都错。

如果所有 token 都乘同一个 advantage，训练信号会很粗。

缓解方法包括：

```text
process reward / step reward
中间步骤 verifier
value head 估计 prefix value
树搜索 / 多样本比较
更细粒度的错误定位
```

但只要 reward 仍然是 response-level，token-level credit assignment 就没有被彻底解决。

---

### 11.2 长度偏置

如果 policy loss 对 token 求和：

$$
L\propto \sum_t A\log\pi(y_t\mid s_t)
$$

长回答因为 token 更多，可能产生更大梯度。

如果 policy loss 对 response tokens 平均：

$$
L\propto \frac{1}{T}\sum_t A\log\pi(y_t\mid s_t)
$$

长短回答的总梯度规模更接近。

所以 token-level loss 是 sum 还是 mean，会直接影响长度偏置。

---

### 11.3 KL penalty 与长度耦合

如果每个 token 都有 KL penalty：

$$
r_t^{\mathrm{KL}}
=-\beta(\log\pi_\theta-\log\pi_{ref})
$$

长回答累计 KL 成本更高。

这会和 CoT 长度、reward scale、答案正确率产生耦合。

---

### 11.4 Group baseline 的局限

如果同一个 prompt 下所有 samples 都错，group-relative 方法仍可能强化“相对没那么差”的错误轨迹。

如果 reward 分布极稀疏，group size 不够大时，advantage 估计会非常噪声。

如果组内 reward 方差接近 0，标准化会不稳定，需要 $\epsilon$、clipping 或特殊处理。

---

### 11.5 Reward scale 很关键

如果把这些东西混在一起：

```text
0/1 correctness reward
+ format reward
+ length penalty
+ KL penalty
+ process reward
```

那么 advantage 的数值范围会直接影响训练稳定性。

LLM RL 里经常要关注：

```text
reward clipping
advantage normalization
KL coefficient
length normalization
group normalization 粒度
```

---

## 12. 实现时最容易出错的细节

### 12.1 Advantage normalization

常见处理：

$$
\hat A_t\leftarrow
\frac{\hat A_t-\mu_A}{\sigma_A+\epsilon}
$$

它可以稳定训练，但会改变不同 prompt、不同 batch、不同长度 response 之间的梯度尺度。

必须明确 normalization 粒度：

```text
全 batch 做？
按 prompt group 做？
按 mini-batch 做？
按 token 做还是按 response 做？
```

---

### 12.2 Reward clipping

有时会裁剪 reward：

$$
R\leftarrow\operatorname{clip}(R,R_{\min},R_{\max})
$$

优点是避免极端 reward 破坏训练。

缺点是可能损失细粒度偏好信息。

---

### 12.3 Length normalization

关键问题：

$$
\text{sequence-level advantage 如何分配到 token-level loss？}
$$

常见选择：

| 聚合方式 | 效果 |
|---|---|
| token 求和 | 长 response 可能梯度更大 |
| token 平均 | 长短 response 梯度规模更接近 |
| 固定长度归一化 | 控制长度偏置，但可能引入新 bias |

---

### 12.4 Masking

必须 mask 掉：

```text
prompt tokens
padding tokens
无效 response tokens
```

policy loss 通常只对 response tokens 计算。

mask 错误会导致：

```text
优化 prompt tokens
padding 影响 loss / advantage normalization
old/new logprob 对齐错误
PPO clipping 失效
```

---

### 12.5 EOS 和截断

如果模型没有生成 EOS，而是达到最大长度被截断，需要明确：

```text
截断样本是否算失败？
是否给 length penalty？
是否仍然计算 verifier reward？
value bootstrap 是否应该保留？
是否 mask 掉截断后的无效部分？
```

这些都会影响 return 和 advantage。

---

### 12.6 Old logprob 与 importance ratio

PPO / GRPO-style objective 中：

$$
\rho_t(\theta)=
\frac{\pi_\theta(y_t\mid s_t)}{\pi_{old}(y_t\mid s_t)}
$$

实际实现通常保存 old logprob：

$$
\log\rho_t=
\log\pi_\theta(y_t\mid s_t)
-
\log\pi_{old}(y_t\mid s_t)
$$

然后：

$$
\rho_t=\exp(\log\rho_t)
$$

如果 old logprob、new logprob、mask、response token 对齐出错，PPO clipping 会失效。

---

## 13. 伪代码：把流程串起来

### 13.1 PPO + GAE

```python
for prompts in dataloader:
    # 1. Rollout
    responses, old_logprobs, values = policy.generate_with_values(prompts)

    # 2. Outcome reward
    outcome_rewards = reward_model_or_verifier(prompts, responses)

    # 3. Per-token KL penalty
    ref_logprobs = ref_model.logprobs(prompts, responses)
    kl_rewards = -beta * (old_logprobs - ref_logprobs)

    # 4. Build token-level reward sequence
    # KL reward on every response token,
    # outcome reward added to the last valid response token.
    rewards = kl_rewards
    rewards[last_response_token] += outcome_rewards

    # 5. Compute GAE
    advantages = compute_gae(
        rewards=rewards,
        values=values,
        gamma=gamma,
        lam=lam,
        mask=response_mask,
    )
    returns = advantages + values

    # 6. PPO policy loss
    new_logprobs = policy.logprobs(prompts, responses)
    ratio = exp(new_logprobs - old_logprobs)
    policy_loss = ppo_clip_loss(
        ratio=ratio,
        advantages=advantages,
        mask=response_mask,
        clip_eps=clip_eps,
    )

    # 7. Value loss
    value_loss = masked_mse(values, returns, response_mask)

    loss = policy_loss + value_coef * value_loss
    loss.backward()
    optimizer.step()
```

关键点：

```text
GAE 需要 value head
KL penalty 会进入 reward，从而影响 return 和 advantage
outcome reward 通常加在最后一个有效 response token 上
mask 对齐非常重要
```

---

### 13.2 GRPO

```python
for prompts in dataloader:
    # 1. Sample a group of responses for each prompt
    responses, old_logprobs = policy.sample_group(
        prompts,
        group_size=G,
    )

    # 2. Compute sequence-level rewards
    rewards = verifier_or_reward_model(prompts, responses)

    # 3. Compute group-relative advantages
    group_mean = mean(rewards, per_prompt=True)
    group_std = std(rewards, per_prompt=True)
    advantages = (rewards - group_mean) / (group_std + eps)

    # 4. Broadcast sequence-level advantage to response tokens
    token_advantages = broadcast_to_response_tokens(
        advantages,
        responses,
    )

    # 5. Policy update, often with ratio clipping and KL regularization
    new_logprobs = policy.logprobs(prompts, responses)
    ratio = exp(new_logprobs - old_logprobs)

    loss = clipped_policy_gradient_loss(
        ratio=ratio,
        advantages=token_advantages,
        mask=response_mask,
    )

    loss.backward()
    optimizer.step()
```

关键点：

```text
GRPO 不需要 value head
必须对同一个 prompt 多采样
advantage 是组内相对量，不是 value-based advantage
sequence-level advantage 如何聚合到 token-level loss，会影响长度偏置
```

---

### 13.3 RLOO

```python
for prompts in dataloader:
    responses = policy.sample_group(prompts, group_size=G)
    rewards = reward_model_or_verifier(prompts, responses)

    advantages = []
    for i in range(G):
        baseline_i = mean([rewards[j] for j in range(G) if j != i])
        advantages.append(rewards[i] - baseline_i)

    token_advantages = broadcast_to_response_tokens(advantages, responses)

    loss = reinforce_loss(
        responses=responses,
        advantages=token_advantages,
        mask=response_mask,
    )

    loss.backward()
    optimizer.step()
```

关键点：

```text
RLOO 不需要 critic
每个样本的 baseline 不包含自己
本质是 REINFORCE + 多样本方差降低
```

---

## 14. 如何选择方法

| 场景 | 更自然的方法 | 原因 |
|---|---|---|
| reward 稠密，环境有连续反馈 | TD / GAE / PPO | value bootstrap 有价值 |
| RLHF 偏好模型给 sequence reward | PPO + GAE 或 RLOO | PPO 成熟，RLOO 简单 |
| 数学 / 代码 / verifier 0-1 reward | GRPO / RLOO / REINFORCE-style | 不一定需要 critic，同 prompt 多采样有效 |
| 显存紧张，不想训练 value head | GRPO / RLOO / ReMax | critic-free |
| 希望 token-level credit 更细 | PPO + GAE 或 process reward | value / step reward 能提供更细信号 |
| 长 CoT reasoning | GRPO-style + careful length handling，或 PPO + GAE | 长度偏置、KL、reward scale 需要重点控制 |
| 有大量旧策略数据 / replay buffer | off-policy correction 方法 | 需要处理行为策略与目标策略不一致 |

简单判断：

```text
如果你有可靠 token-level value / process signal：PPO + GAE 更自然。
如果你只有 outcome reward，且能对同一 prompt 多采样：GRPO / RLOO 更自然。
如果你想最小化工程复杂度：从 REINFORCE-style baseline 方法开始。
```

---

## 15. Off-policy estimation：先知道它是什么

如果数据来自当前 policy，叫 on-policy。

如果数据来自旧 policy、其他模型、replay buffer 或离线数据，叫 off-policy。

off-policy 场景下，直接使用 MC / TD 估计会有分布偏差。因为采样数据来自行为策略 $\mu$，但我们想评估或优化目标策略 $\pi$。

经典修正是 importance sampling：

$$
\rho_t=\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}
$$

轨迹级修正：

$$
\rho_{1:T}=\prod_{t=1}^{T}
\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}
$$

但轨迹级 importance sampling 在长序列上方差极高。

所以还有：

```text
per-decision importance sampling
Retrace(lambda)
V-trace
Tree Backup
```

在 LLM 后训练中，PPO、GRPO、RLOO 通常偏 on-policy 或 near-on-policy，因此 off-policy correction 不是主线。

但如果使用：

```text
旧模型生成的数据
离线偏好数据
replay buffer
多个模型混合采样
```

就会重新遇到 off-policy estimation 问题。

---

## 16. 常见误区清单

### 误区 1：reward 等于 return

不对。reward 是某一步或某条轨迹实际拿到的反馈；return 是从某个时间步开始往后的累计 reward。

---

### 误区 2：return 等于 value

不对。return 是一条采样轨迹的实际结果；value 是在某个状态下继续生成的期望结果。

---

### 误区 3：value 等于 advantage

不对。advantage 是动作相对当前状态平均水平的好坏：

$$
A(s,a)=Q(s,a)-V(s)
$$

---

### 误区 4：GAE 是 reward model

不对。GAE 不判断回答好不好。它只是在已有 reward 和 value estimate 的基础上估计 advantage。

---

### 误区 5：GRPO 只是 PPO 去掉 critic

不准确。GRPO 的关键是 group-relative baseline。它不仅少了 critic，还改变了 advantage 的统计结构。

---

### 误区 6：最终答案正确就应该强化每个 token

不一定。最终答案正确不代表每个中间推理 token 都正确或必要。这正是 outcome reward 下 credit assignment 的难点。

---

### 误区 7：advantage normalization 只是工程细节

不对。它会改变梯度尺度，影响不同 prompt、不同 response 长度、不同 batch 之间的优化动态。

---

### 误区 8：KL penalty 只影响 policy loss

不对。KL penalty 通常会进入 reward 序列，因此会影响 return、advantage 和最终 policy update。

---

### 误区 9：baseline 降低的是 return 本身的方差

不准确。单看 return，有：

$$
\operatorname{Var}(G-b)=\operatorname{Var}(G)
$$

baseline 真正降低的是整个 gradient estimator 的方差：

$$
(G-b)\nabla_\theta\log\pi_\theta(a\mid s)
$$

---

### 误区 10：token-level loss 意味着 token-level reward

不对。token-level loss 常常来自自回归 logprob 分解；reward / advantage 仍可能是 response-level。

---

## 17. 公式速查表

| 名称 | 公式 | 记忆方式 |
|---|---|---|
| Return | $G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k$ | 从当前时刻往后累计 reward |
| Value | $V^\pi(s_t)=\mathbb E_\pi[G_t\mid s_t]$ | 当前状态的期望 return |
| Q-value | $Q^\pi(s_t,a_t)=\mathbb E_\pi[G_t\mid s_t,a_t]$ | 当前状态先选动作后的期望 return |
| Advantage | $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ | 动作比平均水平好多少 |
| REINFORCE | $\sum_tG_t\nabla\log\pi(a_t\mid s_t)$ | 高 return 轨迹提高概率 |
| Baseline PG | $\sum_t(G_t-b_t)\nabla\log\pi(a_t\mid s_t)$ | 用 baseline 降低梯度方差 |
| Gradient sample | $\hat g_t=\hat A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ | 好坏信号 × 提高当前动作概率的方向 |
| Score function mean | $\mathbb E_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a\mid s)]=0$ | baseline 不改变期望梯度的关键 |
| Return shift variance | $\operatorname{Var}(G-b)=\operatorname{Var}(G)$ | baseline 不降低 return 本身方差 |
| Gradient variance | $\operatorname{Var}((G-b)\nabla\log\pi)$ | baseline 降低的是整个梯度估计器方差 |
| TD target | $y_t^{\mathrm{TD}}=r_t+\gamma V(s_{t+1})$ | 一步真实 reward + 后续 value 估计 |
| TD error | $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ | 实际一步 + 下一状态估计 - 当前估计 |
| Bellman consistency | $V(s)\approx\mathbb E[r+\gamma V(s')\mid s]$ | value 应满足一步递推一致性 |
| TD error condition | $\mathbb E[\delta_t\mid s_t]=0$ | 不是每个样本误差为 0，而是条件期望为 0 |
| n-step target | $G_t^{(n)}=\sum_{i=0}^{n-1}\gamma^ir_{t+i}+\gamma^nV(s_{t+n})$ | 看 n 步后再 bootstrap |
| TD($\lambda$) | $G_t^\lambda=(1-\lambda)\sum_n\lambda^{n-1}G_t^{(n)}$ | 多个 n-step return 加权 |
| GAE | $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ | 多个 TD residual 加权 |
| PPO ratio | $\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{old}(a_t\mid s_t)$ | 新旧策略概率比 |
| Token logprob | $\log\pi_\theta(o_i\mid q)=\sum_t\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})$ | response logprob 可拆成 token logprob |
| GRPO advantage | $(R_i-\mu_G)/(\sigma_G+\epsilon)$ | 同 prompt 组内标准化 |
| RLOO advantage | $R_i-\frac{1}{G-1}\sum_{j\ne i}R_j$ | baseline 不包含自己 |
| KL reward | $r_t^{KL}=-\beta(\log\pi_\theta-\log\pi_{ref})$ | 约束 policy 不偏离 reference |

---

## 18. 学习路线

### 第一步：经典 value estimation

按这个顺序学：

$$
\text{DP}
\rightarrow
\text{MC}
\rightarrow
\text{TD(0)}
\rightarrow
\text{n-step TD}
\rightarrow
\text{TD}(\lambda)
$$

重点掌握：

```text
Bellman backup
bootstrap
return vs value
bias-variance tradeoff
```

---

### 第二步：Policy gradient + baseline

重点理解为什么：

$$
G_t
\quad\text{可以换成}\quad
G_t-b_t
$$

以及为什么：

$$
A_t=G_t-V(s_t)
$$

能降低梯度估计器方差。

---

### 第三步：GAE

重点公式：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
\hat A_t^{\mathrm{GAE}}=\sum_l(\gamma\lambda)^l\delta_{t+l}
$$

理解 $\lambda$ 如何控制 bias-variance。

---

### 第四步：PPO

重点理解：

```text
advantage 如何进入 PPO clipped objective
old_logprob 和 new_logprob 为什么重要
clip ratio 如何限制策略更新幅度
value loss 和 policy loss 如何同时训练
```

---

### 第五步：Critic-free LLM RL

学习：

```text
REINFORCE
RLOO
GRPO
ReMax
```

共同主题是：不用 learned critic，而用 group / batch / prompt-level baseline 来构造 advantage。

---

### 第六步：Long-CoT RL 工程细节

重点关注：

```text
sequence-level reward 如何分配到 token-level loss
loss 按 token 求和还是平均
KL penalty 如何累计
advantage 是否按 group 标准化
长度偏置如何处理
截断和 EOS 如何定义
```

---

## 19. 自测题

### 题 1

为什么同样 reward = 0.8，在简单 prompt 和困难 prompt 下可能对应相反的训练方向？

<details>
<summary>答案</summary>

因为训练看的是 reward 相对 baseline 的差值。简单 prompt 的 baseline 可能是 0.95，因此 advantage 为负；困难 prompt 的 baseline 可能是 0.2，因此 advantage 为正。

</details>

---

### 题 2

GAE 中的 $\lambda$ 越大，通常意味着什么？

<details>
<summary>答案</summary>

越接近 MC return，bias 更低但 variance 更高；越小则更依赖短步 bootstrap，variance 更低但 bias 更高。

</details>

---

### 题 3

为什么 GRPO 适合同一个 prompt 多采样？

<details>
<summary>答案</summary>

因为 GRPO 用同 prompt 下多个 responses 的 reward 均值和标准差构造 group-relative advantage。没有多采样，就难以形成稳定的组内比较。

</details>

---

### 题 4

PPO + GAE 为什么需要 value head？

<details>
<summary>答案</summary>

GAE 要计算 TD residual：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

因此需要 $V(s_t)$ 和 $V(s_{t+1})$ 的估计。

</details>

---

### 题 5

为什么长 CoT 中 token-level loss 按 sum 还是 mean 很重要？

<details>
<summary>答案</summary>

如果按 token 求和，长回答因为 token 更多，可能产生更大总梯度；如果按 token 平均，长短回答的总梯度规模更接近。这个选择会影响长度偏置和训练动态。

</details>

---

### 题 6

RLOO 的 baseline 为什么叫 leave-one-out？

<details>
<summary>答案</summary>

因为第 $i$ 个 response 的 baseline 使用同 prompt 下其他 $G-1$ 个 responses 的平均 reward，不包含第 $i$ 个 response 自己。

</details>

---

### 题 7

为什么 TD target $r_t+\gamma V(s_{t+1})$ 叫 bootstrap target？

<details>
<summary>答案</summary>

因为它没有等完整 episode 结束，而是使用“一步真实 reward + value model 对未来的估计”来构造训练目标。未来部分来自当前 value function 自己的估计，所以叫 bootstrap。

</details>

---

### 题 8

TD 学习的目标是让每个样本的 TD error 都等于 0 吗？

<details>
<summary>答案</summary>

不是。TD 希望的是：

$$
\mathbb E[\delta_t\mid s_t]=0
$$

也就是消除 value function 的系统性 Bellman 误差，而不是消除每个随机样本上的误差。

</details>

---

### 题 9

TD($\lambda$) 和 GAE 的核心区别是什么？

<details>
<summary>答案</summary>

TD($\lambda$) 主要服务于 critic，用多个 n-step return 的加权平均构造 value target；GAE 主要服务于 actor，用多个 TD residual 的加权和构造 advantage estimate。

</details>

---

### 题 10

为什么 TD($\lambda$) 可以说是多个 n-step return 的加权平均？

<details>
<summary>答案</summary>

因为每个 $G_t^{(n)}$ 的权重是：

$$
w_n=(1-\lambda)\lambda^{n-1}
$$

这些权重非负，并且：

$$
\sum_{n=1}^{\infty}(1-\lambda)\lambda^{n-1}=1
$$

所以它是标准意义上的加权平均。

</details>

---

### 题 11

Policy gradient 的单个梯度样本由哪两部分组成？

<details>
<summary>答案</summary>

可以写成：

$$
\hat g_t
=
\hat A_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

其中 $\hat A_t=G_t-b_t$ 是好坏信号，决定更新的正负和强度；$\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 是 score function，决定提高当前采样动作概率的参数方向。

</details>

---

### 题 12

为什么 baseline 不改变 policy gradient 的期望方向？

<details>
<summary>答案</summary>

只要 baseline 不依赖当前动作 $a_t$，就有：

$$
\mathbb E_{a\sim\pi}
[
\nabla_\theta\log\pi_\theta(a\mid s)
]
=0
$$

因此：

$$
\mathbb E[
 b(s)\nabla_\theta\log\pi_\theta(a\mid s)
]
=0
$$

所以 baseline 项在期望上不贡献梯度方向。

</details>

---

### 题 13

Baseline 降低的是 return 的方差吗？

<details>
<summary>答案</summary>

不是。单看 return，有：

$$
\operatorname{Var}(G-b)=\operatorname{Var}(G)
$$

baseline 降低的是整个 gradient estimator 的方差：

$$
(G-b)\nabla_\theta\log\pi_\theta(a\mid s)
$$

</details>

---

### 题 14

为什么 response-level reward 也能写成 token-level loss？

<details>
<summary>答案</summary>

因为自回归模型中完整回答概率可以分解为 token 概率的乘积，取 log 后变成 token logprob 的和：

$$
\log\pi_\theta(o_i\mid q)=\sum_t\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
$$

所以 response-level advantage 可以乘到每个 token 的 logprob 上。但这不等于 reward 本身变成了 token-level。

</details>

---

### 题 15

KL penalty 为什么会影响 return 和 advantage？

<details>
<summary>答案</summary>

因为 KL penalty 常作为 per-token reward 加入 reward 序列：

$$
r_t^{KL}=-\beta(\log\pi_\theta-\log\pi_{ref})
$$

return 是 reward 的累计，advantage 又来自 return 减 baseline，因此 KL penalty 会进入 return 和 advantage。

</details>

---

### 题 16

GRPO 和 RLOO 的关键区别是什么？

<details>
<summary>答案</summary>

GRPO 通常使用 group mean / std 构造 group-relative normalized advantage；RLOO 使用 leave-one-out mean，计算第 $i$ 个样本 baseline 时不包含第 $i$ 个样本自身。

</details>

---

### 题 17

为什么 sequence-level ratio 在长文本中不常直接使用？

<details>
<summary>答案</summary>

因为 sequence-level ratio 是很多 per-token ratio 的乘积：

$$
\rho_i=\prod_t\rho_{i,t}
$$

长序列中乘积容易数值不稳定，方差也很高。因此实践中常使用 per-token ratio 和 clipping。

</details>

---

### 题 18

什么情况下 PPO + GAE 比 GRPO / RLOO 更自然？

<details>
<summary>答案</summary>

当有可靠 token-level value、process reward 或更稠密的环境反馈时，PPO + GAE 更自然，因为 value bootstrap 和 GAE 能更细地估计 token-level advantage。

</details>

---

### 题 19

什么情况下 GRPO / RLOO 比 PPO + GAE 更自然？

<details>
<summary>答案</summary>

当只有 sequence-level outcome reward，并且可以对同一个 prompt 多采样时，GRPO / RLOO 更自然，因为它们不需要训练 critic，可以用同 prompt 下多个 responses 构造相对优势。

</details>

---

### 题 20

为什么 masking 是 LLM RL 实现中的高风险细节？

<details>
<summary>答案</summary>

因为 policy loss 通常只应该作用在有效 response tokens 上。如果 prompt、padding、截断无效 token 没有正确 mask，loss、advantage normalization、old/new logprob 对齐和 PPO ratio 都可能出错。

</details>

---

## 20. 最后一页：把所有东西压缩成三条主线

### 主线 1：Reward 设计

它解决的是：

```text
什么回答算好？
```

典型来源：

```text
reward model
verifier
单测
人类偏好
AI judge
格式奖励
```

---

### 主线 2：Return / Value / Advantage Estimation

它解决的是：

```text
已有 reward 后，如何把它变成训练信号？
```

核心链条：

$$
\text{reward}
\rightarrow
\text{return}
\rightarrow
\text{baseline / value}
\rightarrow
\text{advantage}
$$

经典 RL 给出的路线：

$$
\text{DP}
\rightarrow
\text{MC}
\rightarrow
\text{TD}
\rightarrow
\text{TD}(\lambda)
\rightarrow
\text{GAE}
$$

---

### 主线 3：Policy Optimization

它解决的是：

```text
拿到 advantage 后，如何更新模型？
```

LLM 后训练中的两大路线：

$$
\text{PPO + value head + GAE}
$$

$$
\text{GRPO / RLOO / ReMax 等 critic-free 方法}
$$

最后记住这一句：

$$
\boxed{
\text{LLM 后训练中的 return / value / advantage 估计，本质是在解决：}
\text{一个完整回答得到的 reward，如何转换成每个 token 的更新信号。}
}
$$
