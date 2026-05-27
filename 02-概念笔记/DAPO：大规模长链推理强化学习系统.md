---
type: analysis
status: mature
domain: 后训练
aliases:
  - DAPO论文
  - 'DAPO: An Open-Source LLM Reinforcement Learning System at Scale'
tags:
  - reinforcement-learning
  - reasoning
  - long-cot
  - post-training
---
# DAPO：大规模长链推理强化学习系统

> [!info] 原始来源
> [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)

> [!note] 一句话
> DAPO 不是推翻 [GRPO](GRPO.md) 的新范式，而是把 naive GRPO 修成一套更适合 long-CoT reasoning RL 的训练配方：重点解决熵塌缩、全对/全错组零梯度、长回答 token 信号被 sample-level 平均稀释，以及超长截断带来的 reward noise。

## 1. DAPO这篇论文要解决什么问题

这篇论文的出发点很直接：推理型 LLM 的能力提升越来越依赖大规模 RL，但社区对强推理模型的强化学习 recipe 了解得并不充分。即便知道 [PPO](PPO.md) 或 [GRPO](GRPO.md) 的大框架，真正把训练稳定地跑起来依然很难。

作者给出的对照也很明确：直接使用 naive GRPO 在 Qwen2.5-32B Base 上训练时，AIME 2024 只能做到约 30，明显低于对比对象 DeepSeek-R1-Zero-Qwen-32B 的 47。论文真正要回答的问题因此不是“能不能再发明一个新名字的 RL 算法”，而是：**为什么 naive GRPO 在 long-CoT 场景下不够强，以及怎样把它修成一套可扩展的工程系统。**

## 2. DAPO 的定位

DAPO 保留了 GRPO 的主干结构：

- 对每个 prompt 用旧策略采样多条回答
- 用组内 reward 的相对差异构造 advantage
- 用 clipped policy objective 更新策略

它的重点不在于另起炉灶，而在于识别出 long-CoT RL 中几个会反复导致失稳的系统问题，并逐个修补。

> [!tip] 这篇论文最值得记的定位
> DAPO 更像是一套工业级 training recipe，而不是数学结构完全不同的新算法。它站在 GRPO 路线上，但把 reasoning RL 真正会出问题的地方公开讲清楚了。

## 3. DAPO 要修的四个问题

| 问题 | 训练中的表现 | 本质后果 |
|------|-------------|---------|
| 熵塌缩，探索不足 | 策略越来越尖，回答越来越相似 | 模型过早失去探索能力 |
| 全对 / 全错组增多 | 组内 reward 没有差异 | advantage 接近 0，样本几乎不提供梯度 |
| sample-level 平均稀释长回答信号 | 长序列中有效 token 被平均掉 | long-CoT 的 credit assignment 失真 |
| 超长截断引入奖励噪声 | 被截断样本直接受罚 | “推理错”和“回答太长”混成脏信号 |

这四个问题分别对应 DAPO 的四个关键技术：Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping。

## 4. 四个关键技术

### 4.1 Clip-Higher：放宽探索 token 的上升空间

原始 PPO / GRPO 使用对称 clipping，而 DAPO 把 clipping 区间改成上下不对称：

$$
\varepsilon_{high} > \varepsilon_{low}
$$

论文中的设置是 $\varepsilon_{low}=0.2$、$\varepsilon_{high}=0.28$。直觉上，这等于放宽“往上抬”的空间，减轻 upper clip 对低概率探索 token 的压制。

在 long-CoT RL 中，很多真正有价值的推理 token 一开始恰恰是低概率的；如果上界太紧，模型会更快收缩到保守答案，进而出现 entropy collapse。Clip-Higher 的作用就是让这部分探索有更大的生存空间。

### 4.2 Dynamic Sampling：只保留真正有比较信号的组

GRPO 的有效训练信号来自组内比较。如果一个 prompt 采样出来的回答全对或全错，那么组内 advantage 近似为：

$$
\hat A_i \approx 0
$$

这组样本基本不提供有效梯度。随着模型变强，尤其在可验证任务上，“全对组”会越来越多，表面看 batch 没变，实际有效 batch size 却在下降。

DAPO 的做法是：对每个 prompt 采样一组回答后，如果组内全对或全错，就过滤掉这组，继续采样，直到 buffer 里装满真正具有组内差异的样本组。它不是在改 advantage 的数学定义，而是在提高**有效梯度密度**。

### 4.3 Token-Level Policy Gradient Loss：修正 long-CoT 的 credit assignment

原始做法更偏 sample-level reduction：先在单个 sample 内对 token loss 做平均，再跨 sample 做平均。这会让每条样本大致等权，结果是长回答里真正重要的 token 被严重稀释。

在 long-CoT 场景下，这种 reduction 会带来两个问题：

- 好的长推理模式学得不充分
- 长回答里的坏模式，例如重复、发散、无效展开，也罚得不够

DAPO 把 loss 改成更明确的 token-level 聚合方式，让长序列中的 token 能更真实地参与总体梯度更新。它解决的不是 reward 定义，而是 long-CoT 下的 **credit assignment** 问题。

### 4.4 Overlong Reward Shaping：把长度问题和正确性问题拆开

对于超长或被截断的样本，DAPO 不再简单地直接重罚，而是分成两步处理：

1. **Overlong Filtering**：先把这类样本从 loss 中屏蔽
2. **Soft Overlong Punishment**：再按长度分段平滑惩罚

这样做的原因是：一个回答可能推理路径本身是对的，只是因为太长没写完。如果直接给负奖励，就会把“推理错”和“输出太长”混成同一种训练信号。

Soft Overlong Punishment 的直觉是：正常长度不罚，接近上限时轻罚，超出很多时再更强地罚。它的本质是在控制 reward noise，而不是单纯限制模型多写。

## 5. 四个技术分别属于哪一层

| 技术 | 所在层面 | 主要作用 |
|------|---------|---------|
| Clip-Higher | policy objective / actor 更新 | 减轻探索 token 被 upper clip 压制 |
| Token-Level Policy Gradient Loss | loss reduction / credit assignment | 修正 long-CoT 的 token 信号稀释 |
| Dynamic Sampling | sample selection / effective gradient | 提高组内相对比较信号密度 |
| Overlong Reward Shaping | reward design / noise control | 分离长度控制和正确性判断 |

如果只记大类，可以把前两者视为更偏 actor 更新侧，把后两者视为更偏训练信号质量侧。

## 6. DAPO 的训练链路

如果把整套系统按训练流程展开，可以更清楚地看到四个技术分别插在什么位置：

```text
[1] 从数据集采样一批 prompt q
    |
    v
[2] 用旧策略 π_old 对每个 prompt 采样 G 个回答
    |
    |-- 这里希望采样保持多样性，避免太早熵塌缩
    |-- 对应技术：Clip-Higher（作用在后续 policy update 规则上，
    |   但它的目标之一就是反过来改善这里的探索质量）
    v
[3] 对每个回答计算 reward
    |
    |-- 基础 reward：基于规则验证答案对错
    |   正确 = +1，错误 = -1
    |
    |-- 若回答过长 / 被截断：
    |   用 Overlong Reward Shaping 处理
    |   避免把“太长”和“答错”混成一个脏信号
    v
[4] 对每个 prompt 的 G 个回答做组内检查
    |
    |-- 如果一组回答全对 or 全错
    |   -> 组内没有相对差异
    |   -> GRPO advantage 近似为 0
    |   -> 过滤掉
    |
    |-- 如果组内有对有错
    |   -> 保留进 dynamic buffer
    |
    |-- 对应技术：Dynamic Sampling
    v
[5] buffer 凑满后，计算 group-relative advantage
    |
    |-- A_i = (R_i - group_mean) / group_std
    |-- 训练信号来自“同题不同回答”的相对好坏
    v
[6] 构造 policy gradient loss
    |
    |-- ratio r = π_θ / π_old
    |-- 用 clipped objective 做更新
    |
    |-- 对应技术 1：Clip-Higher
    |   原本是对称 clip
    |   改成更宽的上界、更保守的下界
    |   目的：保探索，减轻 entropy collapse
    |
    |-- 对应技术 2：Token-Level Policy Gradient Loss
    |   不再先把每个 sample 内 token 平均成一个等权样本
    |   而是让 token 更直接参与总体梯度
    |   目的：改善 long-CoT 的 credit assignment
    v
[7] 更新当前策略 π_θ
    |
    v
[8] 把新策略拷贝成下一轮的旧策略 π_old
    |
    v
[9] 进入下一轮 rollout / sampling / update
```

如果只想记“每个技术插在哪个模块”，可以再压成一张更短的模块定位图：

```text
数据采样
   ↓
旧策略生成 G 个回答
   ↓
reward 计算  <------ Overlong Reward Shaping
   ↓
筛掉全对/全错组  <------ Dynamic Sampling
   ↓
组内归一化 advantage
   ↓
clipped policy update  <------ Clip-Higher
   ↓
token-level loss 聚合  <------ Token-Level Policy Gradient Loss
   ↓
更新 actor
```

这张链路图对应的阶段分工可以压缩为四句话：

- **Overlong Reward Shaping**：作用在 reward 计算阶段，负责把长度问题和正确性问题拆开。
- **Dynamic Sampling**：作用在样本筛选阶段，负责去掉没有组内差异的零梯度组。
- **Clip-Higher**：作用在 policy update 阶段，负责减轻上 clip 对探索 token 的压制。
- **Token-Level Loss**：作用在 loss 聚合阶段，负责让 long-CoT 中的 token 级信号不被 sample-level 平均稀释。

这也是 DAPO 的一个重要特点：它不是只改一个 loss，而是沿着 rollout、reward、筛样、更新这整条链路逐段修补。

## 7. DAPO 与 PPO、GRPO 的关系

| 方法 | 核心特征 | 在这篇论文里的位置 |
|------|---------|------------------|
| [PPO](PPO.md) | 有 value function，用 clipped objective 稳定更新 | 参照系 |
| [GRPO](GRPO.md) | 去掉 value，用组内相对 reward 估计 advantage | DAPO 的基础骨架 |
| DAPO | 保留 GRPO 主干，再修补 long-CoT RL 的关键失稳点 | 工程化强化版 |

所以最准确的理解不是“DAPO 替代 GRPO”，而是：**DAPO 把 GRPO 变成更适合 reasoning RL 的系统版本。**

> [!warning] 容易混淆的点
> Dynamic Sampling 不是新的 advantage 公式，Overlong Reward Shaping 也不是单纯的长度惩罚技巧。前者在解决有效梯度密度问题，后者在解决奖励污染问题。把它们都看成“奖励小改动”会低估这篇论文的重点。

## 8. 奖励设置与数据处理

### 8.1 规则可验证奖励

论文采用的是 rule-based verifiable reward，而不是 reward model。对于可验证任务，奖励非常直接：

- 答对：`+1`
- 答错：`-1`

这意味着 DAPO 尤其适合数学、代码、定理证明等可验证任务。它依赖的是强 verifier，而不是主观偏好打分。

### 8.2 数据转换本身就是系统的一部分

论文特别提到：数学题的答案形式往往很多样，规则验证不稳定，所以他们对数据做了转换，尽量把答案规整成整数形式，并构造出 DAPO-Math-17K 数据集。

这个细节很重要，因为它说明在 reasoning RL 里，**reward 能不能被稳定验证，不只是奖励函数设计问题，也是数据格式工程问题。**

## 9. 实验结果应该怎么读

### 9.1 主结果

论文报告称：在 Qwen2.5-32B Base 上，DAPO 在 AIME 2024 上做到 50，超过对比的 DeepSeek-R1-Zero-Qwen-32B 的 47，而且训练步数约为对方的一半。

这里最值得注意的不是“最终只高了几分”，而是这套 recipe 让训练过程本身也更高效。

### 9.2 最关键的消融结果

| 设置 | AIME 2024 |
|------|-----------|
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling（DAPO） | 50 |

从这张消融表最容易读出三件事：

1. **Naive GRPO 的确不够强。**
2. **DAPO 的提升不是单点神技，而是系统性修复。**
3. **Overlong 处理和 Dynamic Sampling 的收益都很大，说明训练信号质量是 long-CoT RL 的核心瓶颈之一。**

### 9.3 关键图表各自在证明什么

| 图表 | 主要结论 |
|------|---------|
| Figure 2 | Clip-Higher 提高 AIME，并让 entropy 更高、更稳定 |
| Figure 3 | upper clip 确实主要命中低概率探索 token；全对样本比例会随训练上升 |
| Figure 4 | token-level loss 改善训练动力学，熵和长度走势更健康 |
| Figure 5 | overlong 样本如果处理不当，会明显污染训练 |
| Figure 6 | Dynamic Sampling 虽增加采样成本，但收敛更快 |
| Figure 7 | 作者持续监控 response length、reward、entropy、mean probability，而不是只看最终分数 |

## 10. 训练中最值得盯的指标

论文 4.3 节强调的几个监控量，很适合作为 reasoning RL 的训练仪表盘：

| 指标 | 应该怎么看 |
|------|-----------|
| Response Length | 过短可能没展开推理，过长可能在重复或发散；合理增长往往是健康探索信号 |
| Reward Dynamics | reward 上升不等于泛化同步上升，可能只是过拟合到训练分布 |
| Entropy | 太低说明探索塌缩，太高可能在胡言乱语；缓慢上升更有利于推理能力生长 |
| Mean Probability | 与 entropy 互补，用来观察策略分布是否逐渐变得过尖或异常平坦 |

## 11. 这篇论文最重要的认识

### 11.1 DAPO 是 recipe 革命，不是公式革命

论文最强的价值，不是提出了一个与 GRPO 完全无关的新目标函数，而是把大规模 reasoning RL 真正会踩的坑拆开讲清楚，并给出一套可复现的系统方案。

### 11.2 reasoning RL 的瓶颈不只是 objective

真正限制效果的，往往不是“公式对不对”，而是训练动力学是否健康：

- 探索有没有塌缩
- batch 里是否还有有效梯度
- 长回答的 token 有没有被正确分配 credit
- reward 有没有被长度问题污染

### 11.3 可验证 reward 与数据格式同样关键

这类 RL 系统能跑好，靠的不只是优化器或 loss，更依赖任务本身是否可验证，以及数据是否被整理成容易稳定验证的形式。

## 12. 复习时最该记住的五点

1. DAPO 本质上仍然站在 [GRPO](GRPO.md) 路线上，不是完全新范式。
2. 四个技术分别对应四个很具体的训练失稳点。
3. Clip-Higher 和 Token-Level Loss 更偏 actor 更新与 credit assignment。
4. Dynamic Sampling 和 Overlong Reward Shaping 更偏训练信号质量控制。
5. 这篇论文的最大价值在于把 large-scale reasoning RL 的 recipe 明确公开出来。

## 可拆分概念

以下内容适合从本篇分析笔记中继续拆成独立概念页：

- **Clip-Higher**：不对称 clipping 如何缓解探索塌缩
- **Dynamic Sampling**：GRPO 中全对 / 全错组与零梯度问题
- **Token-Level Policy Gradient Loss**：long-CoT 下的 token 级 credit assignment
- **Overlong Reward Shaping**：长度惩罚与正确性奖励的解耦
- **可验证 reward 与数据格式工程**：为什么 verifier 强度和答案规整方式会决定 RL 成败

## 相关概念

- [GRPO](GRPO.md)
- [PPO](PPO.md)
- [DPO](DPO.md)
- [RLHF](RLHF.md)
- [Expert Iteration](Expert%20Iteration.md)
- Reward Model
- Qwen2
