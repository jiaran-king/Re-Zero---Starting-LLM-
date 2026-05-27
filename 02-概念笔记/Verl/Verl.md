---
title: Verl 源码学习笔记
tags:
  - verl
  - 强化学习
  - RLHF
  - 源码阅读
status: 整理版
updated: '2026-05-24'
---
# Verl 学习笔记

## 0. 总览

verl 的核心不是某一个 PPO / GRPO loss 函数，而是一个 **LLM 强化学习的分布式 dataflow 编排系统**。

它做的事情可以概括为：

```text
用单进程 Controller 写清楚 RL 算法流程，
再把 rollout、logprob、value、reward、update、checkpoint
这些高成本计算交给分布式 Worker / Engine 执行。
```

所以学习 verl 时，不要一开始就陷进 vLLM、FSDP、Megatron 的细节。应该先抓住这条主线：

```text
prompt batch
    -> rollout 生成 response
    -> actor / ref / critic / reward 补齐训练字段
    -> 计算 advantage / returns
    -> 更新 actor / critic
    -> 同步 rollout 权重
    -> 保存 checkpoint
```

一句话记忆：

> verl = HybridFlow 编程思想 + Single Controller + WorkerGroup RPC + DataProto / TransferQueue 数据协议 + Model Engine + Rollout Engine。

---

## 1. RL 训练是一条 dataflow

普通深度学习训练里，dataflow 的节点通常是 matmul、softmax、loss、backward 这类低层算子。

LLM-RL / RLHF 里的 dataflow 节点更高层：

```text
rollout
logprob forward
reference logprob forward
value forward
reward 计算
advantage 计算
policy update
critic update
checkpoint
```

verl 的设计就是把这些高层 RL 操作拆成可调度的分布式任务。

它主要分成两层：

```text
Control Flow
    写 RL 算法流程：先采样、再打分、再算 advantage、再更新模型。

Computation Flow
    真正执行模型 forward / backward / optimizer / inference / checkpoint。
```

这就是 verl 文档里常说的 HybridFlow：**控制流和计算流分离**。

---

## 2. 总体架构：Single Controller + Distributed Workers

verl 的训练主循环通常在一个 controller / trainer 进程里写得像单机程序，但实际计算由 Ray worker 执行。

整体结构可以这样理解：

```text
Controller / Trainer
    |
    | 调用 WorkerGroup 上的远程方法
    v
ActorRolloutRefWorker
    - rollout 采样
    - actor logprob
    - reference logprob
    - actor update
    - rollout weight sync

Critic Worker
    - value 计算
    - critic update

Reward Worker / RewardManager
    - reward model 打分
    - rule-based reward
    - sandbox / verifier reward

Model Engine
    - FSDP / FSDP2 / Megatron / Automodel / VeOmni / TorchTitan
    - model init / optimizer / scheduler / sharding / checkpoint

Rollout Engine
    - vLLM / SGLang / HF rollout
    - 高吞吐生成 response

Checkpoint Engine
    - sharded checkpoint 保存
    - checkpoint 恢复
    - checkpoint merge / 权重同步
```

RL的各个功能，在 verl 中通常不是本地函数调用，而是 WorkerGroup RPC：

```text
generate_sequences       -> rollout 采样
compute_log_prob         -> actor 对生成结果算 old_log_probs
compute_ref_log_prob     -> reference policy 算 ref_log_probs
compute_values           -> critic 算 values
compute_scores           -> reward function / reward model 打分
compute_advantages       -> controller 侧计算 advantages / returns
update_actor             -> actor 做 PPO / GRPO loss + backward + optimizer
update_critic            -> critic 做 value loss + backward + optimizer
update_weights           -> 把 actor 新权重同步给 rollout engine
save_checkpoint          -> 保存 actor / critic / optimizer / scheduler / extra state
```

这张图把上面的角色关系压缩成一条系统链路：

![[08-图片/verl_architecture.canvas]]

---

## 3. 一次 PPO / GRPO 训练如何跑起来

可以把一次训练 step 简化成下面这条链路。

![[08-图片/verl_training_step.canvas]]

### 3.1 从 dataloader 取 prompt

初始 batch 通常包含：

```text
input_ids
attention_mask
position_ids
prompts
non_tensor_batch:
    uid
    data_source
    ground_truth
    extra_info
```

这些信息描述了：

```text
要让模型回答什么 prompt；
这个样本来自哪个数据源；
正确答案或验证信息是什么；
是否有额外 metadata。
```

### 3.2 Rollout：生成 response

入口通常类似：

```python
actor_rollout_wg.generate_sequences(gen_batch)
```

它会生成：

```text
responses
response_mask
完整 input_ids / attention_mask / position_ids
可能还有 rollout logprobs
```

关键理解：

```text
actor trainer model
    负责训练、反向传播、optimizer step。

rollout engine
    负责高吞吐生成 response，通常接 vLLM / SGLang。

weight sync
    actor 更新后，需要把新权重同步给 rollout engine。
```

所以 verl 里的 rollout 不是简单的 `model.generate()`，而是训练模型和推理引擎分离后的分布式采样。

### 3.3 Actor / Ref / Critic 补齐训练字段

rollout 后，controller 会继续调 worker 补齐训练需要的字段。

```text
actor compute_log_prob
    -> old_log_probs

reference compute_ref_log_prob
    -> ref_log_prob

critic compute_values
    -> values
```

这些字段后面会进入 PPO / GRPO loss。

### 3.4 Reward：得到 token-level reward

reward 可以来自：

```text
rule-based reward
    例如数学题答案是否正确、代码单测是否通过、格式是否符合要求。

reward model
    单独的模型 forward 得分。

sandbox / verifier
    例如代码执行环境、外部验证器、多轮 agent 环境。
```

典型流程：

```text
DataProto / batch 中有:
    responses
    response_mask
    data_source
    ground_truth
    extra_info

RewardManager / reward_fn:
    decode response
    根据 data_source 选择 score_fn
    调用 rule reward / reward model / verifier
    得到 score
    写入 token_level_scores / token_level_rewards
```

重要点：

> 即使 reward 看起来是 sequence-level 的，例如答对给 1、答错给 0，最后也要变成和 response token 对齐的 tensor，方便和 token-level logprob / advantage / loss 对齐。

### 3.5 Advantage：PPO 和 GRPO 的核心差异

PPO 使用 critic 和 GAE：

```text
token_level_rewards + values + response_mask
    -> GAE
    -> advantages, returns
```

GRPO 通常不需要 critic。它对同一个 prompt 采样多个 responses，然后用组内 reward 做 baseline：

```text
同一个 prompt 的多个 responses
    -> 每个 response 一个 reward
    -> group mean / std normalization
    -> relative advantages
```

可以粗略理解为：

```text
PPO:
    advantage = GAE(rewards, values)

GRPO:
    advantage = reward_i - mean(reward_group)
    可选再除以 std(reward_group)
```

源码层面要关注：

```text
algorithm.adv_estimator
    gae
    grpo
    reinforce_plus_plus
    reinforce_plus_plus_baseline
    rloo
    rloo_vectorized
    grpo_vectorized
```

### 3.6 Actor update：真正训练发生在 worker 里

controller 侧一般只是调用：

```python
actor_rollout_wg.update_actor(batch)
```

worker 侧才真正做训练：

```text
ActorRolloutRefWorker.update_actor
    -> TrainingWorker.train_mini_batch
        -> for ppo_epoch in ppo_epochs
            -> for mini_batch in batch
                -> model forward
                -> compute PPO / GRPO loss
                -> backward
                -> optimizer.step
```

actor loss 典型消费字段：

```text
responses
response_mask
old_log_probs
new_log_probs
advantages
ref_log_prob
token_level_rewards
loss_mask
clip_ratio
entropy coefficient
KL coefficient
```

PPO clipped objective 可以压缩成：

```text
new_log_prob = actor(prompt, response)
ratio = exp(new_log_prob - old_log_prob)

loss_unclipped = - advantage * ratio
loss_clipped   = - advantage * clip(ratio)

policy_loss = masked_mean(max(loss_unclipped, loss_clipped))
```

GRPO 的 actor update 形式也类似，只是 advantage 来自 group-relative reward，而不是 critic + GAE。

### 3.7 Critic update：PPO 需要，GRPO 通常不需要

PPO 的 critic 是 value model，用来预测 token-level value。

critic update 的目标是让：

```text
values -> 拟合 returns
```

字段流：

```text
batch:
    input_ids
    attention_mask
    position_ids
    responses
    response_mask
    values
    returns

critic_wg.update_critic(batch)
    -> value_model forward
    -> value_loss(values, returns)
    -> backward
    -> optimizer.step
```

如果正在看 GRPO，可以先跳过 critic 路径，把注意力放在 group reward、advantage 和 actor loss 上。

### 3.8 Weight sync 和 checkpoint

actor update 后，rollout engine 必须拿到最新 actor 权重，否则下一批 rollout 会使用旧策略。

```text
actor update
    -> latest trainer weights
    -> update_weights
    -> rollout engine 使用新权重生成下一批 response
```

checkpoint 不是普通的 `torch.save(model.state_dict())`。因为 actor / critic 可能是 FSDP 或 Megatron 分片模型，所以保存的是 sharded checkpoint：

```text
checkpoints/${project_name}/${experiment_name}/
  global_steps_${i}/
    actor/
      huggingface/
      fsdp_config.json
      model_world_size_${world_size}_rank_${rank}.pt
      optim_world_size_${world_size}_rank_${rank}.pt
      extra_state_world_size_${world_size}_rank_${rank}.pt

    critic/
      huggingface/
      fsdp_config.json
      model_world_size_${world_size}_rank_${rank}.pt
      optim_world_size_${world_size}_rank_${rank}.pt
      extra_state_world_size_${world_size}_rank_${rank}.pt

  latest_checkpointed_iteration.txt
```

训练恢复和最终导出是两件事：

```text
训练恢复 checkpoint
    需要 model shard、optimizer shard、scheduler、RNG、extra state。

最终导出 HuggingFace 模型
    通常需要把 sharded checkpoint merge 成完整模型。
```

---

## 4. DataProto：经典路线里的数据载体

在经典 verl 训练路径中，`DataProto` 是各个模块之间传数据的核心对象。

可以把它理解成：

```text
DataProto
  batch:
    TensorDict，保存 tensor 字段。

  non_tensor_batch:
    保存 uid、data_source、ground_truth、extra_info 等非 tensor 数据。

  meta_info:
    保存 temperature、top_p、max_tokens、timing、metrics 等 metadata。
```

一个训练 batch 在不同阶段会逐步补字段：

```text
初始 dataloader:
    input_ids
    attention_mask
    prompts
    uid
    data_source
    ground_truth

rollout 后:
    responses
    response_mask

actor / ref / critic forward 后:
    old_log_probs
    ref_log_prob
    values

reward 后:
    token_level_scores
    token_level_rewards

advantage 后:
    advantages
    returns
```

`DataProto.union()` 的作用就是把不同 worker 返回的新字段合并回同一个 batch。

学习 DataProto 时，重点看这些方法：

```text
DataProto.from_single_dict
DataProto.pop
DataProto.union
DataProto.select
DataProto.chunk
DataProto.concat
DataProto.make_iterator
```

最重要的观察角度：

> 每一步不是孤立函数，而是在同一个 batch 上补齐下一步训练需要的字段。

---

## 5. TransferQueue：新路线里把大 tensor 从 controller 身上移走

经典 DataProto 路线的问题是，大 tensor 经常需要经过 single controller 传来传去。batch 变大、sequence 变长、多模态输入增加后，controller 可能成为内存和带宽瓶颈。

原笔记提到 verl 正在推进更工程化的数据路径：

```text
main_ppo_sync.py
TransferQueue
ReplayBuffer
KVBatchMeta
AgentLoopManagerTQ
RewardLoopManager
LLMServerManager
CheckpointEngineManager
```

这条路线的核心变化：

```text
经典 DataProto:
    大 tensor 跟着 controller 流动。

TransferQueue:
    controller 主要调度 metadata / keys；
    大 tensor 存在数据平面里，worker 按 key 读写。
```

可以理解成两层：

```text
Control Plane
    跟踪每个 sample 哪些字段已经 ready；
    记录哪些任务可以消费这些字段；
    做 metadata-based scheduling。

Data Plane
    存储真实 tensor；
    提供 put_data / get_data / clear_data；
    支持不同 storage backend。
```

新路线的数据流更像：

```text
Controller
    发起任务，只传 metadata / keys / partition_id

Rollout / AgentLoop
    生成 response
    写入 TransferQueue

ReplayBuffer
    根据 metadata 找到 ready samples
    sample 可训练数据

Actor / Critic / Reward
    根据 KVBatchMeta 从 TransferQueue 读取字段
    计算后再写回 TransferQueue

Controller
    调度下一步
```

学习建议：

```text
先用 DataProto 路线理解主流程，
再看 TransferQueue 如何优化数据传输瓶颈。
```

这张图对比了两条数据路径的关键差异：

![[08-图片/verl_dataproto_vs_transferqueue.canvas]]

---

## 6. Model Engine：屏蔽训练后端差异

verl 不希望 trainer 直接写死 FSDP 或 Megatron 的细节，于是抽象出 Model Engine。

Model Engine 负责：

```text
model init
optimizer init
lr scheduler init
sharding
forward_step
train_batch / infer_batch
checkpoint save / load
weight export
```

不同 backend 可以挂在同一套训练流程下面：

```text
TrainingWorker
    -> BaseEngine
        -> FSDPEngineWithLMHead
        -> FSDPEngineWithValueHead
        -> MegatronEngineWithLMHead
        -> MegatronEngineWithValueHead
        -> AutomodelEngineWithLMHead
        -> VeOmniEngineWithLMHead
        -> TorchTitanEngineWithLMHead
```

这就是 verl 的一个关键工程价值：

```text
同一套 PPO / GRPO Trainer
    + 同一套 Worker 抽象
        + 不同 Model Engine
            = 支持不同训练后端
```

所以不能只把 verl 看成 PPO 代码实现。它更像是一个可插拔的 LLM-RL 执行框架。

---

## 7. 源码阅读路线

### 第一阶段：只看主循环

先不要看 FSDP、vLLM、Megatron 细节，先看训练流程怎么串起来。

重点文件：

```text
verl/trainer/ppo/ray_trainer.py
verl/trainer/main_ppo.py
verl/trainer/main_ppo_sync.py
```

目标是能背出这条线：

```text
dataloader
    -> generate_sequences
    -> compute_log_prob
    -> compute_ref_log_prob
    -> compute_values
    -> reward
    -> apply_kl_penalty
    -> compute_advantage
    -> update_critic
    -> update_actor
    -> update_weights
    -> save_checkpoint
```

### 第二阶段：看数据协议

重点文件：

```text
verl/protocol.py
verl/utils/tensordict_utils.py
```

目标是理解 DataProto 如何保存、切分、合并、迭代 batch。

### 第三阶段：看 Single Controller 和 WorkerGroup

重点文件：

```text
verl/single_controller/base/decorator.py
verl/single_controller/base/worker_group.py
verl/single_controller/ray/base.py
```

目标是理解：

```text
@register
dispatch_mode
execute_mode
dispatch_fn
collect_fn
RayWorkerGroup
ResourcePool
```

也就是：普通 Python 方法如何变成 Ray 分布式 RPC。

### 第四阶段：看 worker

重点文件：

```text
verl/workers/engine_workers.py
```

目标是理解：

```text
ActorRolloutRefWorker
TrainingWorker
init_model
generate_sequences
compute_log_prob
compute_ref_log_prob
compute_values
update_actor
update_critic
update_weights
save_checkpoint
```

### 第五阶段：看 engine

重点文件：

```text
verl/workers/engine/base.py
verl/workers/engine/fsdp/
verl/workers/engine/megatron/
```

目标是理解 model init、forward、backward、optimizer、checkpoint 如何被封装。

### 第六阶段：看 rollout 和 agent loop

重点文件：

```text
verl/workers/rollout/
verl/workers/rollout/llm_server.py
verl/experimental/agent_loop/
```

目标是理解：

```text
vLLM / SGLang 怎么被调用；
prompt 怎么变 response；
多轮 agent loop 如何形成 trajectory；
生成结果怎么写入 DataProto 或 TransferQueue。
```

### 第七阶段：看算法细节

重点文件：

```text
verl/trainer/ppo/core_algos.py
verl/workers/utils/losses.py
```

目标是理解：

```text
GAE
GRPO
RLOO
REINFORCE++
KL penalty
PPO clipped loss
loss aggregation mode
```

### 第八阶段：看 TransferQueue

重点文件：

```text
verl/trainer/main_ppo_sync.py
verl/utils/transferqueue_utils.py
TransferQueue 相关实现和文档
```

目标是理解：

```text
KVBatchMeta
ReplayBuffer
tq.kv_batch_put
tq.kv_batch_get
tags
partition_id
keys
controller 如何只调度 metadata
```

---

## 8. 五条核心链路

### 链路一：rollout 采样

```text
Trainer
    -> actor_rollout_wg.generate_sequences

ActorRolloutRefWorker
    -> self.rollout.generate_sequences
    -> vLLM / SGLang / AgentLoop / LLMServer

Output
    -> responses
    -> response_mask
    -> input_ids / attention_mask / position_ids
    -> optional rollout logprobs
```

重点问题：

```text
生成出来的 response 如何和原 prompt 拼接？
response_mask 如何构造？
多轮 agent loop 的 output 如何变成训练样本？
```

简要回答：

```text
1. 生成出来的 response 如何和原 prompt 拼接？

   单轮 rollout 中，response token 会直接接在 prompt token 后面：

       full input_ids = prompt_ids + response_ids

   模型 forward 需要看到 full sequence，因为 response token 的 logprob
   必须条件化在 prompt 上：

       log p(response | prompt)

   但训练 loss 通常只取 response 部分，不对 prompt token 做 RL loss。

   工程上一般不额外保存 response_start 字段，因为普通 prompt+response
   布局里：

       response_start = prompt_length

   prompt_length 可以从 prompts / input_ids 的 shape 推出。需要注意的是，
   自回归模型取 logprob 时有 shift：

       target response 起点 = prompt_length
       logits response 起点 = prompt_length - 1

2. response_mask 如何构造？

   response_mask 通常和 responses 对齐，而不是和 full input_ids 对齐：

       responses:      [R1, R2, EOS, PAD, PAD]
       response_mask:  [1,  1,  1,   0,   0]

   它表示哪些 response token 是有效生成 token，应该参与 reward、KL、
   advantage、actor loss / critic loss 等 token-level 计算。

   attention_mask 和 response_mask 的职责不同：

       attention_mask:
           对齐 full input_ids，告诉模型哪些 token 不是 padding。

       response_mask:
           对齐 response 维度，告诉 RL loss 哪些生成 token 要参与训练。

   如果需要 full-sequence loss mask，可以概念上构造：

       full_loss_mask = prompt_zeros + response_mask

   其中 prompt 部分为 0，response 有效 token 为 1，padding 为 0。

3. 多轮 agent loop 的 output 如何变成训练样本？

   多轮 agent loop 会维护一条 trajectory：

       prompt
           -> assistant tool_call
           -> tool observation
           -> assistant tool_call
           -> tool observation
           -> assistant final_answer

   每一轮模型先基于当前 trajectory 生成 assistant segment。
   如果是 final answer，rollout 结束；如果是 tool call，则执行工具，
   把 observation 追加回 trajectory，作为下一轮 generation 的上下文。

   最终训练样本通常来自完整 trajectory 的扁平化 token 序列：

       input_ids = prompt + assistant actions + tool observations + final answer

   但 actor loss 只应该作用在模型生成的 assistant action / final answer token 上，
   不应该作用在工具返回的 observation token 上。

       prompt token:        attention=1, loss/action mask=0
       assistant action:    attention=1, loss/action mask=1
       tool observation:    attention=1, loss/action mask=0
       padding:             attention=0, loss/action mask=0

   所以多轮 agent 场景里，关键不是只知道 response 从哪里开始，
   而是要用 response_mask / loss_mask / action_mask 区分：

       哪些 token 是 policy 生成的 action；
       哪些 token 只是环境 observation；
       哪些 token 是 padding 或不参与训练的上下文。
```

### 链路二：reward 计算

```text
Trainer
    -> rm_wg.compute_rm_score(batch)  # 可选
    -> reward_fn(batch)               # rule-based 或组合 reward

Reward output
    -> token_level_scores
    -> token_level_rewards after KL penalty
```

重点问题：

```text
当前任务是 rule reward、model reward、sandbox reward，还是组合 reward？
reward 是 sequence-level 还是 token-level？
最后写到哪个 batch key？
```

简要回答：

```text
1. 当前任务是 rule reward、model reward、sandbox reward，还是组合 reward？

   reward 的来源通常有几类：

       rule-based reward:
           用规则、答案匹配、格式检查、单元测试、verifier 等直接打分。
           常见于数学、代码、工具调用、格式约束等可验证任务。

       reward model:
           训练一个单独的 reward model，对 prompt + response 输出 scalar score。
           常见于开放式偏好、风格、安全性、主观质量等任务。

       human feedback:
           人工直接打分或做 preference 标注。
           训练主循环里通常不在线使用，更多用于训练 reward model、
           构造偏好数据或做离线评估。

       hybrid reward:
           把 rule reward、reward model score、格式奖励、长度惩罚、
           工具调用成功率、安全约束等组合起来。

   实际工程里常见的是 hybrid reward，但需要注意不同 reward component
   的尺度、权重、归一化和 metrics 记录。

2. reward 是 sequence-level 还是 token-level？

   sequence-level score 表示一条 response / trajectory 一个分数：

       sequence_level_score.shape = [batch_size]

   例如最终答案是否正确、代码单测通过率、reward model 对整段回答的打分。

   token_level_scores 表示把任务分数放到 response token 维度上：

       token_level_scores.shape = [batch_size, response_length]

   LLM policy 的 logprob、KL、PPO ratio、loss 通常天然是 token-level，
   所以即使原始 reward 是 sequence-level，最后也常常要映射成
   和 response_mask 对齐的 token-level tensor。

   常见映射方式：

       terminal reward:
           把 sequence score 放到最后一个有效 response token 上。

       broadcast:
           把 sequence score 复制到所有有效 response token 上。

       average:
           把 sequence score 均摊到所有有效 response token 上。

       step-level / agent reward:
           把每一步 tool call、final answer、环境反馈的分数映射到对应
           assistant action token 上，不映射到 tool observation token 上。

   需要区分：

       sequence_level_score:
           原始整条回答得分。

       token_level_scores:
           原始任务分数映射到 token 维度后的结果。

       token_level_rewards:
           token_level_scores 加上 KL penalty 等修正后的最终训练奖励。

3. 最后写到哪个 batch key？

   reward_fn / reward model / verifier 的任务奖励通常先写到：

       token_level_scores

   它表示纯任务分数，已经被整理成和 response_mask / action_mask 对齐
   的 token-level tensor。

   如果启用 KL penalty，会继续计算：

       token_level_rewards = token_level_scores - kl_coef * token_level_kl

   后续 compute_advantage 主要消费：

       token_level_rewards
       response_mask / action_mask
       values  # PPO / GAE 时需要

   如果有多个 reward component，例如 correctness、format、tool_success、
   length_penalty、reward_model_score，可以先写到 metrics、meta_info
   或 extra_info 中做记录，最终组合成 token_level_scores。
```

### 链路三：advantage 计算

```text
PPO:
    token_level_rewards + values + response_mask
        -> GAE
        -> advantages, returns

GRPO:
    grouped rewards by prompt uid
        -> group-relative normalization
        -> advantages, returns
```

重点问题：

```text
adv_estimator 是什么？
有没有 critic？
有没有 group index / uid？
是否按 std normalize？
```

简要回答：

```text
1. adv_estimator 是什么？

   adv_estimator 是 advantage estimator，也就是 advantage 的估计方法。
   它决定如何从 reward、value、response_mask、group 信息中得到：

       advantages
       returns

   advantage 表示某个 action / token / response 相比 baseline 好多少：

       advantage > 0:
           这个生成结果比预期好，应该提高概率。

       advantage < 0:
           这个生成结果比预期差，应该降低概率。

   常见 adv_estimator 包括：

       gae
       grpo
       reinforce_plus_plus
       reinforce_plus_plus_baseline
       rloo
       rloo_vectorized
       grpo_vectorized

   PPO 通常使用 critic + GAE；GRPO 通常使用组内相对优势。

2. 有没有 critic？

   PPO / GAE 通常有 critic。

   critic 是 value model，用来估计每个 token position 的 value：

       values = critic(prompt, response)

   GAE 使用：

       token_level_rewards
       values
       response_mask

   计算得到：

       advantages
       returns

   actor update 消费 advantages；critic update 用 returns 训练 value model。

   GRPO 通常没有 critic。
   它不用 value model 做 baseline，而是用同一个 prompt 下多条 responses
   的组内平均 reward 作为 baseline。

       PPO baseline:
           critic model 的 value prediction

       GRPO baseline:
           同组 responses 的 reward mean

3. 有没有 group index / uid？

   如果使用 GRPO / RLOO 这类 group-relative 方法，就必须知道哪些
   responses 来自同一个 prompt。

   通常依赖：

       uid
       group_id
       prompt_id

   同一个 uid 下会有多条 sampled responses：

       prompt uid = x
           response 1 -> reward r1
           response 2 -> reward r2
           response 3 -> reward r3
           response 4 -> reward r4

   GRPO 会在这个 group 内计算：

       group_mean
       group_std  # 可选
       relative advantage

   如果 group index / uid 错了，不同 prompt 的 response 会被混在一起比较，
   advantage 就会失去语义。

4. 是否按 std normalize？

   std normalize 指计算 advantage 时是否除以组内标准差。

   不除 std：

       adv_i = reward_i - group_mean

   除以 std：

       adv_i = (reward_i - group_mean) / (group_std + eps)

   它的作用是把“比组内平均高多少分”变成：

       比组内平均高多少个标准差

   好处：

       不同 prompt / task 的 reward 尺度更稳定；
       训练更新幅度更容易控制；
       对 GRPO 这类 group-relative 方法很常见。

   代价：

       会丢掉一部分 reward 绝对尺度信息；
       group size 很小时 std 估计不稳定；
       std 很小时可能放大噪声，所以通常要加 eps。

   需要区分：

       GRPO group std normalize:
           在同一个 prompt 的多条 responses 内 normalize。

       PPO / GAE advantage normalize:
           可能在整个 batch / mini-batch 的 token advantages 上 normalize。
```

### 链路四：policy / critic update

```text
actor_rollout_wg.update_actor(batch)
    -> ActorRolloutRefWorker.update_actor
    -> TrainingWorker.train_mini_batch
    -> TrainingWorker.train_batch
    -> BaseEngine.forward_step
    -> loss_fn
    -> backward / optimizer.step

critic_wg.update_critic(batch)
    -> TrainingWorker.train_mini_batch
    -> value_loss
```

重点问题：

```text
loss 用了哪些 batch 字段？
mini-batch 怎么切？
micro-batch 怎么防 OOM？
sequence length balancing 有没有生效？
```

简要回答：

```text
1. loss 用了哪些 batch 字段？

   policy / actor loss 主要消费：

       input_ids / attention_mask / position_ids
           完整上下文，通常是 prompt + response。
           计算 log p(response | prompt) 必须条件化在完整历史上。

       responses
           要取 logprob 的模型生成 token。

       response_mask / action_mask / loss_mask
           标记哪些生成 token 有效并参与 loss。
           prompt、padding、多轮 agent 中的 tool observation 通常不参与 actor loss。

       old_log_probs
           rollout policy 对这批 response 的 logprob。
           PPO ratio 需要：

               ratio = exp(new_log_probs - old_log_probs)

       advantages
           判断 token / response 的概率应该提高还是降低。

       ref_log_prob  # 可选
           用于 KL penalty 或 KL metrics。

   critic / value loss 主要消费：

       input_ids / attention_mask / position_ids
       responses
       response_mask
       values
       returns

   actor loss 用 advantages + logprob 更新 policy；
   critic loss 用 returns 监督 value model，让 values 拟合 returns。

2. mini-batch 怎么切？

   rollout batch 是一轮 rollout 得到的完整 on-policy 数据，里面已经包含
   responses、log_probs、rewards、advantages、returns 等字段。

   update 阶段会把 rollout batch 打乱后按 mini_batch_size 切分：

       rollout batch
           -> shuffle
           -> mini-batch 1
           -> mini-batch 2
           -> ...

   如果 ppo_epochs > 1，同一批 rollout 数据会被重复训练多轮。

   mini-batch 是一次逻辑 optimizer update 使用的数据子批。
   切分时必须保持所有字段对齐：

       input_ids
       responses
       response_mask
       old_log_probs
       advantages
       returns
       non_tensor_batch / uid / extra_info

   GRPO / RLOO 这类 group-relative 方法通常先在完整 rollout batch 上
   按 uid / group 算好 advantages，再打乱切 mini-batch。

3. micro-batch 怎么防 OOM？

   如果一个 mini-batch 一次性 forward / backward，activation、logits、
   中间张量和 attention buffer 可能导致 OOM。

   实际会把 mini-batch 再切成更小的 micro-batches：

       mini_batch_size = 256
       micro_batch_size = 16
       num_micro_batches = 16

       optimizer.zero_grad()

       for micro_batch in micro_batches:
           loss = forward(micro_batch)
           loss = loss / num_micro_batches
           loss.backward()
           # 当前 micro-batch 的 activations 用完后释放

       optimizer.step()
       optimizer.zero_grad()

   关键点是：

       optimizer.step() 不会再 forward / backward 所有样本。

   它只处理已经累积好的参数梯度：

       param.grad

   也就是处理：

       模型参数 param
       累积梯度 param.grad
       optimizer state

   例如 AdamW 主要更新：

       exp_avg
       exp_avg_sq
       param

   这里不会重新读取完整 mini-batch 的 input_ids、responses、attention_mask，
   也不会重新保存所有样本的 activations / logits。

   micro-batch 防 OOM 的核心原因是：

       大显存开销主要发生在 forward / backward 保存和使用 activation 时；
       每个 micro-batch backward 完后，计算图和中间激活通常就可以释放；
       最后只留下累积到参数上的梯度。

   梯度累积不是保存每个样本或每个 micro-batch 的独立梯度。默认情况下：

       param.shape == param.grad.shape

   每次 backward 做的是：

       param.grad += grad_from_micro_batch

   所以即使累积 16 个 micro-batches，param.grad 的 shape 也不会变成：

       [16, *param.shape]

   它仍然是一份和参数同 shape 的梯度 tensor。

   所以：

       mini-batch:
           决定一次逻辑参数更新用多少样本。

       micro-batch:
           决定单次 GPU forward / backward 塞多少样本，主要用于控制 activation 显存。

       optimizer.step:
           决定用已经累积好的梯度更新参数，不再重新处理所有样本。

   但 optimizer.step 仍然可能 OOM，只是原因不同。

   micro-batch 主要解决的是：

       activation OOM

   也就是一次 forward / backward 塞太多 token 导致显存爆掉。

   它不能完全解决这些常驻显存问题：

       parameters
       gradients
       optimizer states
       master weights  # mixed precision 时可能有
       FSDP / ZeRO 的 shard / 通信 buffer

   这些主要和模型大小、optimizer 类型、精度、并行策略、offload 策略有关，
   而不是和当前 mini-batch 有多少样本直接线性相关。

   例如 AdamW 通常会为每个参数维护两个状态：

       exp_avg
       exp_avg_sq

   如果 optimizer state 本身已经放不下，即使 micro_batch_size = 1，
   也可能在 optimizer 初始化或 optimizer.step 时 OOM。

   可以把训练显存拆成几类：

       parameters
           模型权重，常驻。

       gradients
           参数梯度，常驻到 optimizer.step / zero_grad。

       optimizer states
           Adam 的 m / v 等状态，常驻。

       activations
           forward 保存给 backward 用的中间结果，和 micro-batch size、sequence length 强相关。

       temporary buffers
           logits、attention buffer、通信 buffer、loss 中间张量等。

   micro-batch 主要降低：

       activations
       一部分 temporary buffers

   它不降低或很少降低：

       parameters
       gradients
       optimizer states

   一个具体例子：

       完整 mini-batch 一次算：

           parameters:       20 GB
           gradients:        20 GB
           optimizer states: 40 GB
           activations:      60 GB
           temporary:        10 GB
           total:            150 GB

       切成 micro-batch 后，常驻部分仍然是：

           parameters:       20 GB
           gradients:        20 GB
           optimizer states: 40 GB

       但 activation 和临时 buffer 可能降为：

           activations:       4 GB
           temporary:         2 GB
           total:            86 GB

       到 optimizer.step() 时，不会重新出现 60 GB activations。

   分布式训练里也是同理。
   在 FSDP / ZeRO 中，参数、梯度、optimizer state 可能是分片的：

       rank 0 持有一部分 shard
       rank 1 持有另一部分 shard
       ...

   optimizer.step() 通常在每个 rank 上处理自己负责的 shard，
   仍然不需要把所有样本的 activation 汇总起来。

   但 backward 或 step 周期中可能出现通信和峰值 buffer：

       all_reduce
       reduce_scatter
       all_gather

   这些也可能造成峰值显存压力，但不是“optimizer.step 重新处理所有样本”。

   需要警惕的情况：

       把 loss tensor 存进 list 且没有 detach，导致计算图被保留。
       不必要地 retain_graph=True。
       记录了大量 logits / hidden_states / per-token tensors，且没有 detach 或搬到 CPU。
       optimizer state 本身太大。
       gradient synchronization buffer 太大。
       loss scaling 不正确，导致梯度累积后数值不稳定。

   准确回答这个问题：

       optimizer.step 不会处理所有样本，
       也不会重新保存所有样本的 activation。

       不会因为 mini-batch 的所有样本在 optimizer.step 被集中计算而 OOM。

       但如果模型参数、梯度、optimizer state 本身太大，
       optimizer.step 仍然可能 OOM。

   一句话：

       micro-batch 把样本相关的 activation 显存拆开了；
       optimizer.step 只看参数相关的梯度状态，不再看所有样本。

4. sequence length balancing 有没有生效？

   sequence length balancing 指按 token 工作量做负载均衡。

   LLM 训练的计算和显存压力不只取决于样本数，更取决于：

       prompt length
       response length
       attention_mask.sum()
       full sequence length

   如果简单按样本数切分，可能出现：

       某个 rank / micro-batch 全是长序列，计算慢或 OOM；
       另一个 rank / micro-batch 全是短序列，算完后空等。

   sequence length balancing 会根据样本有效长度重新排列或分配样本，
   让每个 rank / micro-batch 的总 token 数尽量接近。

   它解决的是：

       样本数相同，但 token 工作量严重不均。

   需要注意：

       micro-batch 控制单次计算塞多少样本；
       sequence length balancing 控制这些样本的长度如何搭配和分发。
```

### 链路五：checkpoint 和 rollout 权重同步

```text
actor update
    -> latest trainer weights

update_weights
    -> export parameters
    -> rollout.update_weights
    -> vLLM / SGLang 使用新权重生成下一批 response

save_checkpoint
    -> sharded actor checkpoint
    -> sharded critic checkpoint
    -> optimizer / scheduler / RNG / dataloader state
```

重点问题：

```text
训练恢复用哪个 checkpoint？
最终部署模型是否需要 merge？
rollout engine 当前用的是不是最新 actor 权重？
```

简要回答：

```text
1. 训练恢复用哪个 checkpoint？

   训练恢复要用 training checkpoint，而不是最终导出的 HuggingFace 模型。

   也就是说，恢复训练时需要的是能还原训练现场的 checkpoint：

       actor model shard
       actor optimizer shard
       actor extra state / RNG state
       critic model shard  # 如果有 critic
       critic optimizer shard
       critic extra state / RNG state
       lr scheduler state
       global step / dataloader 进度等额外状态

   它通常位于：

       checkpoints/${project_name}/${experiment_name}/global_steps_${i}/

   或新版本目录命名中的：

       checkpoints/${project_name}/${experiment_name}/global_step_${i}/

   下面会有 actor / critic 等子目录。

   判断“用哪个 step 恢复”通常有两种模式：

       resume_mode = auto:
           自动从 trainer.default_local_dir 找最新 checkpoint。
           最新 step 一般由 latest_checkpointed_iteration.txt 记录。

       resume_mode = resume_path:
           显式从 trainer.resume_from_path 指定的 checkpoint 路径恢复。

       resume_mode = disable:
           不恢复，从头开始训练。

   所以，如果只是故障恢复、抢占恢复、断点续训，通常用：

       latest_checkpointed_iteration.txt 指向的最新 global step checkpoint

   例如：

       checkpoints/verl_examples/gsm8k/latest_checkpointed_iteration.txt
           -> 100

       实际恢复目录：

       checkpoints/verl_examples/gsm8k/global_steps_100/

   或：

       checkpoints/verl_examples/gsm8k/global_step_100/

   具体目录名要以当前 verl 版本和实际保存结果为准。

   如果要从某个历史 step 做 ablation、回滚或对比实验，就不要用 auto，
   而是显式指定：

       trainer.resume_mode=resume_path
       trainer.resume_from_path=/path/to/global_step_xxx

   需要特别区分：

       training checkpoint:
           用于恢复训练。
           包含 sharded model、optimizer、scheduler、RNG、extra state。

       merged HuggingFace model:
           用于推理、评测、部署或继续作为普通 pretrained model 加载。
           通常不包含 optimizer、scheduler、dataloader 进度等训练现场状态。

   因此：

       恢复训练不要只拿 actor/huggingface/ 或 merge 后的 HF 模型。

   只拿 HF 模型继续训练，最多相当于“从这个权重重新开始一段训练”，
   不是严格意义上的 resume。它会丢失 optimizer momentum、lr scheduler、
   global step、RNG 等状态，训练曲线和随机性都可能接不上。

   对 PPO / GRPO 这类 RL 训练，严格 resume 还要注意：

       actor 和 critic 要来自同一个 global step。
       optimizer / scheduler 状态要和模型参数匹配。
       rollout 重新开始后，要确保 rollout engine 加载的是恢复后的 actor 权重。
       不要把不同 step 的 actor、critic、optimizer shard 混用。

   一句话：

       训练恢复用 latest 或显式指定的 global step training checkpoint；
       最终导出的 HuggingFace 模型不是完整训练恢复 checkpoint。

2. 最终部署模型是否需要 merge？

   通常需要。

   原因是 verl 训练时保存的 checkpoint 往往是训练后端自己的分布式格式，
   不是普通推理框架可以直接加载的完整 HuggingFace 模型。

   例如 FSDP checkpoint 里通常是：

       actor/
           huggingface/
           fsdp_config.json
           model_world_size_${world_size}_rank_${rank}.pt
           optim_world_size_${world_size}_rank_${rank}.pt
           extra_state_world_size_${world_size}_rank_${rank}.pt

   Megatron checkpoint 里通常是：

       actor/
           huggingface/
           dist_ckpt/

   这些目录适合训练恢复，因为它们保留了 sharding、optimizer、extra state 等信息。
   但部署时通常只需要 actor 的最终权重，并且希望它变成普通 HF 模型目录：

       config.json
       tokenizer files
       model.safetensors / pytorch_model.bin shards
       generation_config.json  # 如果有

   所以最终部署一般流程是：

       1. 选择一个 global step checkpoint。
       2. 取 actor checkpoint，而不是 critic checkpoint。
       3. 用 model_merger 把 sharded actor checkpoint 转成 HuggingFace 格式。
       4. 用合并后的 HF 目录做评测、推理、部署或上传 HuggingFace Hub。

   典型命令类似：

       FSDP:

       python -m verl.model_merger merge \
           --backend fsdp \
           --local_dir checkpoints/${project}/${experiment}/global_step_${i}/actor \
           --target_dir /path/to/merged_hf_model

       Megatron:

       python -m verl.model_merger merge \
           --backend megatron \
           --tie-word-embedding \
           --local_dir checkpoints/${project}/${experiment}/global_step_${i}/actor \
           --target_dir /path/to/merged_hf_model

   注意这里 merge 的通常是：

       actor

   而不是：

       critic
       optimizer
       extra_state

   因为最终部署的是 policy model，也就是 actor。
   critic 主要用于 PPO 训练时估计 value，部署生成回答时通常不需要。

   什么时候不需要手动 merge？

       如果 checkpoint.contents 里保存了 hf_model，
       并且 actor/huggingface/ 下已经有完整可加载的 HF 权重，
       那可能可以直接用这个目录。

       如果训练后端或脚本已经在保存时自动导出了完整 HF 模型，
       也不需要再手动 merge。

       如果下游推理系统本身支持读取该训练后端的 sharded checkpoint，
       也可以不转 HF；但这不是最常见的部署路径。

   但大模型训练里通常不建议每次 checkpoint 都保存完整 hf_model，
   因为它会带来额外存储和保存开销。更常见的做法是：

       训练过程中保存 sharded training checkpoint；
       需要部署或评测时，再挑一个 step merge 成 HF 模型。

   还要注意：

       merge 出来的 HF 模型适合推理 / 部署；
       sharded training checkpoint 适合恢复训练。

   不要把两者混为一谈。

   如果只是做训练恢复，不要先 merge 再 resume。
   merge 后通常已经丢掉 optimizer、scheduler、RNG、extra state，
   不能完整恢复训练现场。

   如果是 GRPO / PPO 训练出的最终策略，一般部署 actor merge 后的模型即可。
   critic 不需要一起部署，除非你的下游任务还要 value head 做评估或继续 RL 训练。

   一句话：

       部署通常要把 actor 的 sharded checkpoint merge 成 HuggingFace 模型；
       训练恢复则继续用原始 sharded training checkpoint。

3. rollout engine 当前用的是不是最新 actor 权重？

   正常同步训练路径里，应该是最新的。

   你的理解是对的：

       每次 update_actor 之后，
       需要把 actor trainer model 的最新权重同步到 rollout engine，
       也就是 model sync / weight sync。

   原因很直接：

       actor trainer model:
           负责 forward / backward / optimizer.step，是真正被训练更新的模型。

       rollout engine:
           负责高吞吐生成 response，可能是 vLLM / SGLang / HF rollout。

   这两者在工程上经常不是同一个模型实例。
   actor update 改的是 trainer 侧权重；如果不把新权重同步给 rollout engine，
   下一轮 generate_sequences 就会用旧策略生成 response。

   正常同步 PPO / GRPO 流程可以理解为：

       rollout engine 用当前 actor 权重生成 responses
           -> actor / ref / critic / reward 补字段
           -> compute advantages
           -> update_actor
           -> actor 权重发生变化
           -> sync actor weights to rollout engine
           -> 下一轮 rollout 使用新 actor 权重

   所以链路上通常会有：

       update_actor
           -> export latest actor parameters
           -> rollout.update_weights / update_weights
           -> rollout engine 加载新权重

   在 colocated actor-rollout 或 hybrid engine 场景中，常见实现是：

       从 training engine 导出参数；
       通过 checkpoint engine / weight sync path 传给 rollout；
       rollout engine 调用 update_weights 加载这些参数。

   对 vLLM / SGLang 这类 rollout engine 来说，weight sync 的本质是：

       让推理引擎内部用于 generation 的模型权重，
       对齐刚刚 optimizer.step 后的 actor 权重。

   为什么必须做这件事？

       PPO / GRPO 通常希望 rollout 数据尽量 on-policy。
       如果 rollout engine 长时间不更新，生成 response 的 policy 会落后于 actor trainer，
       batch 就会变成 stale policy / off-policy 数据。

   轻微的一步延迟在某些 async / one-step-off-policy 设计里可能是有意为之，
   但普通同步训练里，期望是每轮 actor 更新后同步 rollout 权重。

   判断 rollout engine 是否是最新 actor 权重，可以看几个点：

       update_actor 之后有没有调用 update_weights / rollout.update_weights。
       weight sync 是否在下一次 generate_sequences 之前完成。
       日志里有没有 sync_rollout_weights / update_weights 相关耗时。
       actor 和 rollout 的参数版本号 / global step 是否一致。
       async rollout 场景是否允许 one-step stale。

   什么时候可能不是最新？

       async training / one-step-off-policy:
           rollout 可能故意使用上一版 actor 权重，以换取训练和采样 overlap。

       weight sync 被配置跳过或被 bug 影响:
           update_actor 后没有真正调用 rollout.update_weights。

       sync 还没完成就开始下一轮 rollout:
           可能出现 rollout 使用旧权重。

       rollout engine 有独立的 sleep / wake_up / cache 管理:
           如果权重同步和 engine wake_up 逻辑耦合，要确认 wake_up 时确实加载了新权重。

       load_format / backend 不匹配:
           例如 FSDP / Megatron / HF / dtensor 的权重加载路径配置错，可能导致同步失败或加载异常。

   所以更准确的说法是：

       在普通同步训练路径里，rollout engine 应该在每次 actor update 后同步到最新权重；
       但在 async / one-step-off-policy 或 sync 失败的情况下，它可能不是最新。

   这也是为什么日志和 metrics 里经常会单独记录：

       sync_rollout_weights
       update_weights time
       rollout wake_up / sleep
       actor global step
       rollout global step

   一句话：

       正常同步 PPO / GRPO 中，update_actor 后会 model sync，
       让 vLLM / SGLang rollout engine 使用最新 actor 权重生成下一批 response；
       如果不同步，训练就会逐渐变成 stale-policy / off-policy。
```

---

## 9. 最小实验：打印 batch 字段变化

学习 verl 最有效的方式之一，是跑一个小模型、小 batch，然后在关键位置打印 batch keys。

可以观察这些点：

```python
print("after dataloader", batch.batch.keys(), batch.non_tensor_batch.keys())

gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
print("after rollout", gen_output.batch.keys())

batch = batch.union(gen_output)
print("after union rollout", batch.batch.keys())

ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
print("ref keys", ref_log_prob.batch.keys())

values = self.critic_wg.compute_values(batch)
print("value keys", values.batch.keys())

reward_tensor = self.reward_fn(batch)
print("reward shape", reward_tensor.shape)

batch = compute_advantage(...)
print("advantages", batch.batch["advantages"].shape)
print("returns", batch.batch["returns"].shape)
```

你应该重点观察：

```text
每一步新增了哪些字段；
这些字段的 shape 是什么；
哪些字段来自 worker；
哪些字段在 controller 侧计算；
哪些字段最终进入 actor / critic loss。
```

---

## 10. 理解检查表

学完一个模块后，用这些问题检查自己是否真的理解。

```text
1. 这个模块的入口 RPC 是哪个？
   generate_sequences / compute_log_prob / update_actor / save_checkpoint？

2. 它运行在 controller 还是 worker？
   controller 只调度，还是 worker 真正 forward / backward？

3. 输入 batch 需要哪些字段？
   input_ids、responses、old_log_probs、advantages、returns？

4. 输出 batch 会新增哪些字段？
   responses、ref_log_prob、values、token_level_scores、advantages？

5. 数据是通过 DataProto 返回，还是通过 TransferQueue 写入？
   classic route or main_ppo_sync route？

6. 当前 worker 用的是哪个 backend？
   fsdp、fsdp2、megatron、automodel、veomni、torchtitan？

7. rollout engine 和 actor trainer 是否同步权重？
   update_weights 什么时候触发？

8. checkpoint 是训练恢复用，还是导出 HuggingFace 模型用？
   sharded checkpoint 是否需要 model_merger？
```

---

## 11. 最后再压缩成一张图

回看第 3 节的 `verl_training_step.canvas`，它就是这篇笔记最核心的一张图。

最终记住这句话：

> verl 的核心实现不是单个算法公式，而是把 LLM-RL 训练拆成一组可调度、可分布式执行、可替换后端的 dataflow 节点。源码阅读的关键，是追踪一个 batch 从 dataloader 到 update_actor 的字段变化。

---

## 12. 参考入口

原笔记中提到的主要参考入口：

```text
verl/trainer/ppo/ray_trainer.py
verl/trainer/main_ppo.py
verl/trainer/main_ppo_sync.py
verl/protocol.py
verl/workers/engine_workers.py
verl/workers/engine/
verl/workers/rollout/
verl/trainer/ppo/core_algos.py
verl/workers/utils/losses.py
```

相关文档方向：

```text
HybridFlow Programming Guide
Data interface
Engine Workers
Model Engine
PPO / GRPO 文档
Reward Function 文档
Checkpoint 文档
TransferQueue 文档
```
