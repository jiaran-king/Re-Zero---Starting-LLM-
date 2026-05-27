---
type: concept
domain: 后训练
status: active
aliases:
  - 经验采样
  - 轨迹采样
  - Policy Rollout
---
# Rollout：强化学习中的经验采样

> [!note]
> Rollout 是把当前策略放到环境里实际跑一遍或一批，从而生成轨迹数据的过程。在大模型后训练中，它通常指让当前或旧策略模型对一批 prompts 生成 responses，并记录 logprob、reward、value、mask 等训练所需信息，供 PPO、GRPO 等在线强化学习算法更新策略。

Rollout 解决的不是“怎样更新模型参数”，而是“从哪里拿到可用于更新的经验数据”。没有 rollout，PPO / GRPO 这类在线或近在线 RL 方法就没有当前策略分布下的样本，也就无法估计 advantage、计算新旧策略概率比或做 KL 约束。

## 1. 基本定义

在经典强化学习里，rollout 是按某个策略 $\pi$ 与环境交互，得到一条轨迹：

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T, a_T, r_T)
$$

其中 $s$ 是状态，$a$ 是动作，$r$ 是奖励。训练算法会用这些轨迹估计 return、value、advantage，再决定哪些动作应该被强化、哪些应该被压低。

放到 LLM 里，环境不再是游戏或机器人控制器，而是“prompt + 自回归生成过程 + reward / verifier”。状态是当前上下文前缀，动作是下一个 token，轨迹就是模型从 prompt 开始生成完整 response 的过程。

| RL 术语 | LLM 后训练里的对应物 |
|---|---|
| state $s_t$ | prompt 加上当前已生成 token 的上下文 |
| action $a_t$ | 第 $t$ 个生成 token |
| policy $\pi_\theta$ | 当前语言模型的 token 分布 |
| trajectory $\tau$ | 一条 prompt-response 生成过程 |
| reward $r$ | Reward Model、规则 verifier、人类偏好或任务得分 |
| rollout batch | 一批 prompts 及其生成结果和训练元数据 |

## 2. 在 LLM RL 流水线中的位置

一次典型 rollout 可以压缩成这条链：

```text
prompts
→ old/current policy 采样 responses
→ 记录 token logprobs、mask、必要时记录 value
→ reward model / verifier 打分
→ 构造经验 batch
→ 估计 advantage
→ policy update
```

Rollout 是采样阶段，不是更新阶段。PPO、GRPO 等算法真正更新参数时，会回头使用 rollout 阶段保存的旧策略 logprob、reward、value 或组内 reward 统计。

> [!tip]
> 在很多代码库里，`rollout` 既可能指“一条完整生成轨迹”，也可能指“批量采样并打包经验数据的整个阶段”。读论文或代码时要看上下文：如果出现 `old_logprobs`、`advantages`、`returns`、`values`，通常说的是后者。

## 3. Rollout 需要记录什么

不同算法记录的字段不同，但 LLM 后训练里通常会包含：

| 数据 | 作用 |
|---|---|
| prompt / response tokens | 还原生成轨迹，作为 policy loss 的输入 |
| response mask | 确保 loss 只作用在有效 response token 上，避开 prompt、padding 和截断无效位 |
| old logprob | 计算新旧策略概率比 $\rho_t = \exp(\log \pi_\theta - \log \pi_{old})$ |
| reference logprob | 计算 KL 约束，防止策略偏离 SFT / reference 太远 |
| reward | 表示整条 response 或中间步骤的好坏 |
| value | PPO / GAE 需要，用来估计 baseline 和 return；GRPO 通常不需要 |
| advantage | 把 reward 转换成“比基线好多少”的训练信号 |
| stop reason / length | 判断 EOS、最大长度截断、格式失败等边界情况 |

这里最容易出错的是 mask、old logprob 和 token 对齐。只要 response token、padding 或 old/new logprob 错位，PPO clipping 或 GRPO 的 token-level loss 就会失去意义。

## 4. PPO、GRPO、DPO 中的 rollout 差异

| 方法 | 是否依赖在线 rollout | rollout 的角色 | 关键记录 |
|---|:---:|---|---|
| [PPO](PPO.md) | 是 | 用旧策略采样 response，再用 RM + KL 构造 reward 和 GAE | old logprob、value、reward、ref logprob、mask |
| [GRPO](GRPO.md) | 是 | 对同一 prompt 采样一组 responses，用组内相对 reward 估计 advantage | group responses、old logprob、reward、ref logprob、mask |
| [DPO](DPO.md) | 通常否 | 标准 DPO 使用离线偏好对，不在训练时重新 rollout | chosen / rejected logprob、ref logprob |
| [Expert Iteration](Expert%20Iteration.md) | 是，但用途不同 | 用 rollout 生成候选答案，经 verifier 过滤后做 SFT | 候选 response、验证结果、过滤后的专家样本 |

PPO 和 GRPO 都需要 rollout，但目的不完全相同。PPO 侧重“采样后如何用 value / GAE 做逐 token 优势估计”；GRPO 侧重“同一 prompt 多采样，靠组内比较构造 critic-free advantage”。DPO 的标准形态则把 rollout 前置到数据集构建阶段，训练时更像离线分类目标。

## 5. 与 reward、return、advantage 的关系

Rollout 只产生样本，不能直接告诉模型该怎么更新。训练还需要把采样结果变成梯度信号：

```text
rollout trajectory → reward → return / baseline → advantage → policy gradient
```

在 LLM 里，reward 往往是 response-level 的，比如完整回答结束后由 RM 或 verifier 给一个总分。但 policy loss 是 token-level 的，因为语言模型的 response 概率可以拆成 token logprob 之和：

$$
\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t})
$$

这意味着 response-level advantage 可以广播到 response tokens 上参与 loss；但这不等于每个 token 都真的获得了同样精确的因果 credit。长 CoT 训练里的很多不稳定性，本质上都来自“整条回答的 reward 如何分配到每个 token”这个问题。

## 6. 工程视角

Rollout 常常是 LLM RL 最贵的一段，因为它要真的跑生成，而且可能要对同一 prompt 采样多条 response。GRPO 中的 group size 越大，组内 advantage 越稳定，但推理成本也越高。

常见优化方向包括：

- 用 vLLM / 高吞吐 serving 引擎承接采样
- 对同一 prompt 的多条采样复用 prefix KV cache
- 控制 max length、EOS、格式失败提前终止
- 异步 actor-learner，把采样和训练解耦
- 动态调整 group size，在训练早期保留探索，后期提高吞吐

这些问题的生命周期归属仍在后训练，但工程入口更适合从 [训练系统工程](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F%E5%B7%A5%E7%A8%8B.md) 看。

## 7. 易错点

> [!warning]
> - Rollout 不是 PPO / GRPO 的参数更新，而是更新前的数据生成与经验打包。
> - “模型生成了一段文本”不一定就是训练意义上的 rollout；如果没有记录 logprob、mask、reward 等训练字段，它只是普通推理。
> - `old policy` 和 `reference policy` 不是一回事：old policy 用来计算 PPO / GRPO 的新旧概率比，reference policy 用来提供 KL 锚点。
> - response-level reward 可以进入 token-level loss，但不等于每个 token 都有准确 reward。
> - on-policy rollout 数据不能像普通 SFT 数据那样长期无限复用，否则会产生策略分布偏移。

## 关联

- 属于：[后训练与对齐](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%90%8E%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%AF%B9%E9%BD%90.md)
- 相关：[PPO](PPO.md) [GRPO](GRPO.md) [RLHF](RLHF.md) [DPO](DPO.md) [Expert Iteration](Expert%20Iteration.md) Reward Model [训练系统工程](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F%E5%B7%A5%E7%A8%8B.md)
- 用于：大模型对齐项目 DeepSeek-R1 复现 推理能力提升项目

## 相关概念

- [PPO](PPO.md)
- [GRPO](GRPO.md)
- [RLHF](RLHF.md)
- [DPO](DPO.md)
- [Expert Iteration](Expert%20Iteration.md)
- [LLM_后训练 Reward Return Value Advantage](LLM_%E5%90%8E%E8%AE%AD%E7%BB%83%20Reward%20Return%20Value%20Advantage.md)

## 来源

- 整理自：小红书摘录《同学提问：大模型中常说的 Rollout 是什么》（原始摘录已删除）
