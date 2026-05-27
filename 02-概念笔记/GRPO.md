---
type: concept
domain: 后训练
status: active
---
# GRPO：Group Relative Policy Optimization

> [!note]
> GRPO 是 PPO 的轻量化变体，去掉了 Value Model，用同一 prompt 的组内相对 reward 做 z-score 归一化来估计 advantage。来自 DeepSeekMath (2024) 和 DeepSeek-R1 (2025)。核心公式：GRPO = Vanilla REINFORCE + 组内均值 Baseline + Z-score 归一化 + PPO-style Clipping + KL 惩罚。

---

## 从 REINFORCE 到 GRPO

REINFORCE 是最基础的策略梯度算法：增加带来高奖励的动作概率，降低低奖励动作的概率。

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) R \right]$$

$R$ 是采样得到的绝对奖励。如果环境的奖励全是正数（如 10、50、100 分），REINFORCE 会把所有动作的概率都调高，只是幅度不同，导致模型很难快速区分"平庸"和"优秀"，梯度更新也极不稳定。

为解决高方差问题，引入优势函数 $A = R - b$，其中 $b$ 是某种 baseline。它回答的不再是"我得了多少分"，而是"我比平均水平多得多少分"。传统 PPO 用一个与主模型同等规模的 Critic 网络来预测 baseline，显存翻倍。GRPO 直接用组内均值作为 baseline：对同一 prompt 采样 G 条 response，用 reward 的均值作为 baseline，再用标准差归一化：

$$\hat{A}_i = \frac{r_i - \mu}{\sigma}$$

Z-score 归一化剥离了奖励系统的绝对尺度，保证传入优化器的 advantage 信号始终在稳定、可控的数值范围内。

再叠加 PPO-style Clipping——引入新旧策略的概率比率 $\rho_i = \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{old}}(a_i|s)}$，对目标函数截断：

$$L_i^{CLIP} = \min \left( \rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) \hat{A}_i \right)$$

$\hat{A}_i > 0$ 时 clip 在 $1+\varepsilon$ 处截断，防止过度强化；$\hat{A}_i < 0$ 时 clip 在 $1-\varepsilon$ 处截断，防止过度惩罚。每次更新只在"信任域"内微调，防止 Policy Collapse。

GRPO 并没有发明复杂的数学结构，精妙之处在于回归最基础的 REINFORCE 并辅以最简单的组内 Z-score，实现了与 PPO 媲美的稳定 Advantage 估计，同时砍掉了庞大的 Critic 模型。

---

## PPO 在 LLM 场景下的困境

PPO 应用于 RLHF 时需要同时维护四个模型：Policy Model（生成 token）、Value Model（预测 $V(s_t)$）、Reward Model（对 response 打分）、Reference Model（KL 约束锚点）。其中 Policy 和 Value 都需要训练。

RM 只在 response 末尾给出一个标量奖励，但策略梯度需要在每个 token 位置都有 advantage 信号。PPO 用 Value Model 在每个 token 位置估计 $V(s_t)$，计算 TD 残差，再通过 GAE 递推。但 Value Model 通常与 Policy Model 同等规模，Reward 信号只在末尾出现导致中间 token 上几乎是在"盲猜"，四个大模型同时驻留显存的开销也很高。

---

## GRPO 的核心设计

### 三模型架构

去掉 Value Model，只保留三个模型：Policy Model（训练）、Reward Model（冻结）、Reference Model（冻结）。

### Group Sampling

对同一个 question $q$，从 old policy $\pi_{\theta_{old}}$ 独立采样 G 条 response。这 G 条 response 完全并行、互相独立。类比同一道数学题让同一个学生独立做 G 遍，每次"忘掉"之前做过。论文中 $G = 64$。

### 组内相对 Advantage

Reward Model 对每条 response 打分后做组内 z-score 归一化：

$$\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

同一条 response 内所有 token 共享同一个 advantage 值（Outcome Supervision 下）。这种 group relative 的方式和 reward model 的训练方式天然契合——reward model 本身就是在同一 question 的不同 response 之间做 pairwise comparison 训练出来的（Bradley-Terry），用组内相对排序来算 advantage 正好利用了 reward model 最擅长的判断模式。

---

## 目标函数

完整公式：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left\{\min\left[\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}\hat{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\varepsilon, 1+\varepsilon\right)\hat{A}_{i,t}\right] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]\right\}\right]$$

逐层拆解：最外层 $\mathbb{E}$ 对 question 数据集取期望（论文中 batch size = 1024）；$\frac{1}{G}\sum$ 对 G 条采样输出求平均；$\frac{1}{|o_i|}\sum_t$ 对单条输出的所有 token 做长度归一化；$\min[\cdots]$ 是 clipped surrogate objective；$-\beta D_{KL}$ 是 KL 正则项。

长度归一化 $\frac{1}{|o_i|}$ 让每个完整回答无论长短对策略改进的贡献趋于等权，避免长回答因包含更多 token 而产生更大的梯度模长。但这也带来了 Length Bias 的副作用——归一化后模型会发现短回答的 per-token reward 更高，倾向生成更短的回答。缓解方案包括对过短回答直接给负分的长度惩罚项、指数归一化 $1/|o_i|^\alpha$（$0.5 < \alpha < 1$）、以及用 KL 散度约束防止崩塌到极短的采样空间。

---

## KL 散度的无偏估计器

标准 KL 散度 $D_{KL}[\pi_\theta \| \pi_{ref}] = \mathbb{E}_{o_t \sim \pi_\theta}[\log \frac{\pi_\theta}{\pi_{ref}}]$ 要求 token 从 $\pi_\theta$ 采样，但训练时 token 是从 $\pi_{\theta_{old}}$ 采样的，无法直接用样本均值估计。而且 $\log \frac{\pi_\theta}{\pi_{ref}}$ 可以为负，导致 KL 惩罚变成"奖励"。

GRPO 使用的 Schulman 估计器设 $r = \frac{\pi_{ref}(o_t)}{\pi_\theta(o_t)}$，则：

$$f(r) = r - \log r - 1$$

这个函数有两个关键性质。严格非负：$f'(r) = 1 - \frac{1}{r} = 0 \Rightarrow r = 1$ 是全局最小值点，$f(1) = 0$。无偏性：当 $o_t \sim \pi_\theta$ 时，$\mathbb{E}[r] = 1$，$\mathbb{E}[-\log r] = D_{KL}$，合起来 $1 + D_{KL} - 1 = D_{KL}$。

> [!note] 与 PPO 中 KL 处理的对比
> PPO 用 $\log \frac{\pi_\theta}{\pi_{ref}}$（可为负），加在 reward 中影响 advantage 计算；GRPO 用 $r - \log r - 1$（保证非负），加在 loss 中与 advantage 解耦。

严格来说 token 从 $\pi_{\theta_{old}}$ 采样而非 $\pi_\theta$，估计并不严格无偏。但 $\mu=1$（单步更新）加上 clipping 机制保证 $\pi_\theta \approx \pi_{\theta_{old}}$，偏差可控。

---

## Reward 机制

### Rule-based vs Learned Reward Model

| 特性 | Rule-based | Learned RM |
|------|-----------|-----------|
| 适用场景 | 数学、代码、形式逻辑 | 创意写作、开放式对话 |
| 信号强度 | 稀疏但绝对准确 | 密集但存在噪声 |
| 可解释性 | 极高 | 极低 |
| Reward Hacking 风险 | 无 | 有 |

数学和代码等"非黑即白"的领域，rule-based reward 是更好的选择。答案对就是 1 错就是 0，规则无法被欺骗，且几乎不占算力。

DeepSeek-R1-Zero 使用了两种最简单的规则：Accuracy Reward（答案是否正确）和 Format Reward（是否将思考过程放在 `<thought>` 标签内）。这种 0/1 的二元奖励看似粗糙，却催生了自主思考能力——为了拿到那个唯一的 1，模型在不断的采样博弈中自发学会了重新审视步骤和自我纠错。

DeepSeek R1 实际采用混合流水线：先用少量高质量数据做 SFT（Cold Start），然后在数学/逻辑任务上完全依赖 rule-based reward 做强化学习，最后在通识和价值观对齐上引入 learned RM。Rule-based reward 是推理能力的"考官"，Learned RM 是表达风格的"导师"。

---

## Outcome Supervision 与 Process Supervision

Outcome Supervision（OS）下，RM 对每条完整 response 打一个总分，所有 token 共享同一个 advantage：

$$\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

如果一条 response 前半段推理精彩但最后答错了，所有 token 都会被惩罚；反之前面胡说八道但最后答对了，所有 token 都会被强化。

Process Supervision（PS）下，PRM 在每个推理步骤结束位置给出 reward，某个 token 位置 $t$ 的 advantage 是它之后所有步骤归一化 reward 的累加：

$$\hat{A}_{i,t} = \sum_{index(j) \geq t} \tilde{r}_i^{index(j)}$$

这像一个没有 discount factor 的 return，提供了更细粒度的信号。没有折扣因子是因为严格的逻辑链条中因果关系不会随时间衰减——中间某步犯了致命错误导致后续全部崩盘，这个巨大的负向收益会无损地累加到前面，精准惩罚导致错误的早期步骤。实验表明 GRPO+PS 优于 GRPO+OS。

---

## Iterative GRPO 与 Reference Model 策略

随着训练推进 policy 变强，reward model 可能跟不上。Iterative GRPO 的做法是：用当前 policy 采样新数据，混合 10% 历史数据继续训练 reward model，将当前 policy 设为新的 reference model，用新 reward model 继续训练 policy。论文做了两轮迭代，第一轮提升显著，第二轮提升较小。

Reference Model 的核心作用是通过 KL 散度提供正则化约束。固定 Reference Model（标准做法）是绝对稳定的"锚点"，最大程度保证输出连贯性和指令遵循能力，但当 policy 远超 SFT 水平时会限制模型上限——惩罚那些在 SFT 模型看来概率很低但实际上非常优秀的新颖推理路径。

滑动 Reference Model（Iterative GRPO）释放了探索空间，但多轮累加后 policy 可能完全遗忘初始 SFT 分布，导致 reward hacking 或 mode collapse。融合两者的策略包括：EMA 平滑更新 $\theta_{ref} \leftarrow \alpha \cdot \theta_{ref} + (1-\alpha) \cdot \theta_{policy}$、在 Loss 中始终保留一部分原始 SFT 数据的交叉熵损失作为硬兜底、以及动态调整 KL 系数（初期鼓励探索，后期或检测到多样性下降时增大 $\beta$）。

---

## 工程细节

### Z-score 退化

当组内所有 response 的 reward 完全相同时（全对或全错），分子为 0，分母趋近 0，模型陷入"奖励盲区"。简单加 $\epsilon$ 虽能防止 NaN，但可能因浮点误差导致 z-score 飙升引发梯度爆炸。

常见对策：Skip Batch（std 低于阈值直接跳过该组）、Global Norm（维护跨 batch 的 running mean 和 std）、保留 KL 梯度（即使 reward 全一样，KL 散度项通常不同）。

```python
rewards = torch.tensor([...])
std = rewards.std()
if std < 1e-6:
    advantages = torch.zeros_like(rewards)
else:
    advantages = (rewards - rewards.mean()) / (std + 1e-8)
```

全错意味着模型没有获得任何引导信号，这就是 Cold Start（SFT）如此重要的原因——没有 SFT 提供的基础分布，z-score 永远无法产生区分度，RL 无法起步。全对时 z-score 强制进行组内对比，可能导致模型为了追求极微小的变分而改变已经正确的逻辑。

### 采样效率优化

G=64 意味着每个 question 要做 64 次完整推理。Prefix Caching 让 64 个回答共享同一份 prompt 的 KV Cache，只需在第一次 forward 时计算一次，极大节省 prefill 阶段计算量。Early Stopping 在 rule-based reward 场景下特别有效——如果前 50 个 token 里模型没有输出要求的格式标签直接中断并赋 0 分，或第一步就得出违背常理的结论时提前掐断。

### Rejection Sampling 作为前置

与其在 RL 阶段硬扛 G=64 的开销，不如将压力前置到数据准备阶段。让当前 policy 在较高 temperature 下为每个 prompt 生成 N 个候选回答，用 RM 或 rule-based 验证器打分，只保留最高分的做 SFT。

概率视角的洞察：独立采样 N 次时获取的是分布的极值。即便模型平均能力平庸，只要有非零概率蒙对，N 次抽样就能"碰撞"出落在长尾上的完美解答。将其蒸馏回模型，本质上是强迫模型记住"超水平发挥"时的状态，将长尾低概率事件变成未来生成的平均基线。

作为 RL 的前置（Cold Start），先用几万道题做一轮拒绝采样，把碰巧做对的带详细步骤的样本挑出来做 SFT，让模型先学会分步思考的格式和基础逻辑，有了这个基础盘再启动 GRPO 就更容易在组内对比中找到提升方向。

拒绝采样和 GRPO 的核心区别在于：RS 的上限受限于模型当前的采样空间——如果一道极难的题采样 10000 次都做不对，RS 就拿不到任何数据。GRPO 在连续动作空间里做梯度探索，有可能引导模型走向它在 SFT 阶段从未踏足过的概率空间，涌现出前所未有的推理能力。

动态 Group Size 也是常用的优化策略——训练初期模型方差大需要较大的 G 提供稳定 baseline，后期策略收敛可以减小 G（如降到 16）换取更高训练吞吐量。

---

## PPO vs GRPO 对比

| | PPO | GRPO |
|---|---|---|
| 需要训练的模型 | Policy + Value Model | 仅 Policy |
| 冻结的模型 | Reference + Reward Model | Reference + Reward Model |
| Advantage 来源 | GAE（基于 Value Model） | 组内 z-score 归一化 reward |
| KL 正则位置 | 加在 reward 中 | 加在 loss 中 |
| 每个 question 采样量 | 1 条 | G 条（论文中 G=64） |
| Token 级 credit assignment | 有（通过 GAE） | 无（OS）/ 步骤级（PS） |
| 显存 | 高（4 模型，2 个需训练） | 低（3 模型，1 个需训练） |
| Trade-off | 显存换信号精度 | 推理计算换显存 |

---

## 训练超参数

| 超参数 | 值 |
|--------|-----|
| Policy learning rate | 1e-6 |
| KL 系数 $\beta$ | 0.04 |
| Group size $G$ | 64 |
| Max length | 1024 |
| Batch size | 1024 |
| 每次采样后更新次数 $\mu$ | 1 |
| 训练数据 | GSM8K + MATH 相关的 ~144K questions |
| Reward Model 初始化 | DeepSeekMath-Base 7B，lr = 2e-5 |

---

## 关联
- 属于：[后训练与对齐](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%90%8E%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%AF%B9%E9%BD%90.md)
- 相关：[DPO](DPO.md) [PPO](PPO.md) [RLHF](RLHF.md) [Expert Iteration](Expert%20Iteration.md) REINFORCE Reward Model
- 用于：DeepSeek-R1 复现 数学推理训练

## Reward Hacking 与强 Verifier

### GRPO 并不"消灭" Reward Hacking

严格说，GRPO **没有从算法层面彻底解决 reward hacking**。它在可验证任务上配合规则型奖励把 reward hacking 的空间压缩得更小，但对不可靠验证的任务仍是开放问题。

核心分三层理解：

**第一层**：GRPO 本身不是防作弊算法，它做的是更稳定、更省资源地优化策略——去掉 critic/value model，用组内相对 reward 做 z-score 归一化估计 advantage。这主要是为了减少训练资源、避免长 CoT 下 value model 难学的问题。

**第二层**：DeepSeek 在数学/代码/逻辑任务上控制住 reward hacking 的关键不在 GRPO，而在 **reward 设计**。R1 论文明确使用 rule-based rewards：
- **Accuracy reward**：检查最终答案是否正确、代码能否通过测试
- **Format reward**：检查输出格式（如 `think` 标签）

这类奖励可被外部规则或编译器直接验证，模型很难靠"讨好奖励模型"拿高分。论文刻意不在 reasoning task 上使用神经 reward model 作为主奖励，因为这类 RM 在大规模 RL 中容易被 reward hacking。

**第三层**：GRPO 的"组内相对比较"机制确实能减轻一部分投机行为——模型要拿到正优势通常得在同组样本里给出更正确的答案，而非学会固定表面模式。但当 reward 本身能被伪造时，GRPO 也无能为力。

### 什么是强 Verifier

**强 verifier** 是一个能对模型输出给出高可信、低歧义、难被投机利用的自动判定器。它回答的核心问题是：**这次输出到底有没有真正完成任务？**

强 verifier 通常满足四个性质：

| 性质 | 说明 |
|------|------|
| 可客观判定 | 输出好坏不依赖主观审美，按规则判断 |
| 误判率低 | 不频繁把错判对或对判错 |
| 难被表面模式欺骗 | 看"答案对不对"而非"像不像好答案" |
| 与任务目标一致 | verifier 判定标准与真实目标一致 |

### Verifier 强弱三档

| 档位 | 类型 | 示例 | 特点 |
|------|------|------|------|
| 最强 | 形式化/可执行验证 | 数学答案匹配、代码单测、proof checker | 接近"机器裁判"，最抗 hacking |
| 中等 | 规则化但不完全形式化 | 格式检查、关键词约束、结构约束 | 提供信号但可钻空子，适合做辅助奖励 |
| 最弱 | 模型打分/偏好近似器 | "这个回答更有帮助吗" | 最容易被 reward hacking |

R1 论文的工程经验也印证了这一点：**规则奖励可以长跑，模型奖励要短跑**。他们在第二阶段 RL 加入 general preference reward，但只在最后 400 steps 使用——更多步数会导致 reward hacking。

### 实用判断标准

- 如果一个任务能写成"给定输出，可以用程序几乎不依赖人类主观判断地判对错"→ 适合做强 verifier 驱动的 RL
- 如果只能写成"这个回答大概更好一些，但很难严格说明为什么"→ verifier 偏弱，reward hacking 风险更高

Agentic RL 场景中的强 verifier 来源：数学（答案匹配/程序求值）、代码（编译/单测/沙箱执行）、检索问答（文档精确支持）、工具调用（外部可验证条件）、形式化任务（proof checker/SQL 执行结果）。

---

## 相关概念
- [RLHF](RLHF.md)
- [DPO](DPO.md)
- [Expert Iteration](Expert%20Iteration.md)
- SFT
- [PPO](PPO.md)
