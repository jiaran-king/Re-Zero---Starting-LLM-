# FSDP in veRL

---

## 0. 总览

### 0.1 概括

FSDP，Fully Sharded Data Parallel，本质是把普通数据并行中每张 GPU 重复保存的 **parameters、gradients、optimizer states** 切分到多个 data parallel ranks 上。普通 DDP 中每张 GPU 都保存完整模型副本；FSDP 在 forward/backward 之外让参数保持 sharded，计算前通过 all-gather 临时恢复完整参数，backward 中通过 reduce-scatter 得到 sharded gradients，optimizer 只更新本 rank 的 sharded parameters 和 sharded optimizer states。

PyTorch FSDP2 教程把这个过程描述为：参数在 forward/backward 外完全分片，forward/backward 前 all-gather，backward 内 reduce-scatter 梯度，optimizer 更新 sharded states；并且 FSDP 可视为把 DDP 的 all-reduce 分解成 reduce-scatter 和 all-gather。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.12.0+cu130 documentation"))

### 0.2 在 veRL 中最常影响什么

```text
actor:
    policy 训练显存、update_policy 吞吐、gradient clipping、checkpoint、actor-rollout 权重同步。

critic:
    value model 训练显存、optimizer states、activation、checkpoint。

reference model:
    ref logprob / KL 相关 forward 显存，通常不训练，可更适合 offload。

reward model:
    如果是模型型 RM，主要影响 forward 显存；如果训练 RM，则也涉及 optimizer states。

rollout:
    通常由 vLLM/SGLang 负责，瓶颈更多是 KV cache、gpu_memory_utilization、max_num_batched_tokens、max_num_seqs 和权重同步；不能只从 FSDP 角度看。
```

### 0.3 默认配置起点

```text
actor:
    backend: fsdp 或 fsdp2
    sharding: FULL_SHARD / reshard_after_forward=True
    wrap: Transformer block / decoder layer
    dtype: bf16
    gradient checkpointing: True
    forward_prefetch: 先 False，稳定后再尝试 True

ref:
    forward-only
    dtype 与 actor 对齐
    大模型可考虑 offload

rollout:
    vLLM/SGLang
    dtype 与 actor 对齐
    gpu_memory_utilization 先保守，再根据 OOM/吞吐调整
    注意 KV cache 和 actor-rollout 权重同步开销

checkpoint:
    训练中优先 SHARDED_STATE_DICT
    最终导出再 full state dict / merge weights / save_pretrained
```

### 0.4 OOM 先看哪里

```text
初始化 OOM:
    checkpoint 加载方式、meta init / empty init、cpu_ram_efficient_loading、sync_module_states。

forward 前 OOM:
    FSDP unit 太大、wrap 太粗、forward_prefetch、micro batch / token 上限、rollout/vLLM 占用。

backward 开始 OOM:
    activation 峰值、backward all-gather、gradient checkpointing、长序列、logits/entropy。

rollout OOM:
    KV cache、gpu_memory_utilization、max_num_batched_tokens、max_num_seqs、prompt/response 长度。

checkpoint OOM:
    是否保存 FULL_STATE_DICT、rank0 CPU 内存、optimizer state dict、NCCL timeout。
```

---

## 1. FSDP 解决什么问题

### 1.1 DDP 的冗余在哪里

普通 DDP 的核心问题是：**训练吞吐可以通过多卡数据并行提升，但单卡仍然要保存完整模型状态**。

模型训练中的显存大头通常包括四类：

```text
parameters
    模型权重。

gradients
    backward 后的梯度。

optimizer states
    AdamW 的 m/v、fp32 master weights 等。

activations
    forward 中为了 backward 保存的中间张量。
```

FSDP 主要解决前三类，也就是 model states；activation 仍然需要 activation checkpointing、sequence packing、flash attention、micro batch 调整等方法配合解决。PyTorch FSDP 文档把 FSDP 定义为一种跨 data parallel workers 对 module parameters 做 sharding 的 wrapper，并指出它受到 ZeRO Stage 3 的启发。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/fsdp.html?utm_source=chatgpt.com "FullyShardedDataParallel — PyTorch 2.12 documentation"))

### 1.2 一个显存估算公式

用 AdamW 混合精度训练时，可以用一个简化公式帮助记忆。假设参数量是 `P`，常见状态大致是：

```text
fp16/bf16 参数:       2P bytes
fp16/bf16 梯度:       2P bytes
fp32 master weights: 4P bytes
Adam m:              4P bytes
Adam v:              4P bytes

总模型状态约: 16P bytes
```

如果有 `D` 张 GPU，DDP 每张卡仍然近似保存 `16P`；FSDP `FULL_SHARD` / ZeRO-3 风格理想情况下让每张卡长期只保存约 `16P / D` 的模型状态。

注意，这只是常驻模型状态估算，不包括 activation、临时 all-gather 参数、通信 buffer、CUDA allocator fragmentation、KV cache 等。RL 后训练里尤其不能忽略 activation、logits、rollout KV cache 和 actor-rollout 共置带来的额外显存。

### 1.3 FSDP 和 ZeRO-1/2/3 的关系

ZeRO 是 DeepSpeed 提出的“减少数据并行冗余”的思想。DeepSpeed 官方文档把 ZeRO stage 0/1/2/3 分别描述为：不启用、optimizer state partitioning、optimizer + gradient partitioning、optimizer + gradient + parameter partitioning。([DeepSpeed](https://deepspeed.readthedocs.io/en/latest/zero3.html "ZeRO — DeepSpeed 0.19.1 documentation"))

可以这样对照：

| 策略 | 参数 | 梯度 | 优化器状态 | 直觉 |
|---|---|---|---|---|
| DDP / ZeRO-0 | 完整复制 | 完整复制 | 完整复制 | 最简单，最耗显存 |
| ZeRO-1 | 完整复制 | 完整复制 | 分片 | 主要省 Adam 状态 |
| ZeRO-2 | 完整复制 | 分片 | 分片 | 继续省梯度 |
| ZeRO-3 | 分片 | 分片 | 分片 | 最省显存，通信最多 |
| FSDP `FULL_SHARD` | 分片 | 分片 | 分片 | PyTorch 原生 ZeRO-3 风格 |

Hugging Face Accelerate 文档也给出了类似映射：`FULL_SHARD` 对应 DeepSpeed ZeRO Stage 3，`SHARD_GRAD_OP` 对应 ZeRO Stage 2，`NO_SHARD` 对应 ZeRO Stage 0，`HYBRID_SHARD` 则是在节点内做 ZeRO-3 风格分片、节点间复制。([Hugging Face](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp "Fully Sharded Data Parallel · Hugging Face"))

### 1.4 核心通信流程

以 `FULL_SHARD` 为例，一个 FSDP unit 的生命周期是：

```text
初始化:
    参数被切成 shard，每个 rank 只保存一片。

forward 前:
    all-gather 当前 FSDP unit 的完整参数。

forward 中:
    用完整参数计算。

forward 后:
    reshard，释放非本地完整参数。

backward 前:
    再次 all-gather 当前 unit 的完整参数。

backward 中:
    计算本地完整梯度。

backward 后:
    reduce-scatter 梯度，得到每个 rank 自己的 gradient shard；
    reshard 参数。

optimizer step:
    每个 rank 只更新自己的 parameter shard 和 optimizer state shard。
```

FSDP2 的 `fully_shard()` 文档明确说明：初始化时参数跨 data-parallel workers 分片；forward 前 all-gather；如果 `reshard_after_forward=True`，forward 后释放 unsharded parameters，并在 backward 前重新 all-gather；gradient 计算后释放完整参数并 reduce-scatter 完整梯度。([PyTorch Documentation](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html "torch.distributed.fsdp.fully_shard — PyTorch 2.12 documentation"))

最重要的通信算子是：

```text
all-gather:
    每张卡贡献自己的参数 shard；
    每张卡都拿到完整参数。

reduce-scatter:
    每张卡先参与梯度规约；
    规约后的梯度被切分，每张卡只保留一片。

all-reduce:
    DDP 常用；
    每张卡最后拿到完整同步梯度。
```

所以，DDP 是 `replicate parameters + all-reduce gradients`；FSDP 是 `shard model states + all-gather parameters + reduce-scatter gradients`。

---

## 2. FSDP 策略和术语映射

### 2.1 FSDP sharding strategy

PyTorch FSDP1 的主要策略包括 `FULL_SHARD`、`SHARD_GRAD_OP`、`NO_SHARD`、`HYBRID_SHARD`、`_HYBRID_SHARD_ZERO2`。官方文档对这些策略的定义是：`FULL_SHARD` 分片参数、梯度和优化器状态；`SHARD_GRAD_OP` 分片梯度和优化器状态，参数在 forward 后不立即 reshard；`NO_SHARD` 类似 DDP；`HYBRID_SHARD` 节点内 `FULL_SHARD`、节点间复制；`_HYBRID_SHARD_ZERO2` 节点内 `SHARD_GRAD_OP`、节点间复制。([PyTorch Documentation](https://docs.pytorch.org/docs/2.12/fsdp.html "FullyShardedDataParallel — PyTorch 2.12 documentation"))

| 策略 | 显存 | 通信 | 适用场景 |
|---|--:|--:|---|
| `NO_SHARD` | 最高 | 较低 | 小模型、debug、DDP 对照 |
| `SHARD_GRAD_OP` | 中高 | 中等 | 显存够，想减少 backward 前 all-gather |
| `FULL_SHARD` | 最低 | 最高 | 大模型训练默认优先项 |
| `HYBRID_SHARD` | 介于 FULL_SHARD 和复制之间 | 跨机通信更少 | 多机训练，节点内通信快、节点间通信慢 |
| `_HYBRID_SHARD_ZERO2` | 比 HYBRID_SHARD 更高 | 更偏吞吐 | 多机、显存较充足、希望少一次 backward all-gather |

### 2.2 FSDP1 与 FSDP2

很多训练框架仍使用 FSDP1 配置名，但 PyTorch 官方教程现在更强调 FSDP2。FSDP2 的核心变化是：`fully_shard()` 原地作用在 module 上，参数以 DTensor 表示，state dict、optimizer、gradient clipping 和 distributed checkpoint 更容易组合。PyTorch FSDP2 教程列出的优势包括：sharded parameters 表示为 DTensor、communication-free sharded state dict、更简单的 meta-device 初始化流程、更好的 frozen/non-frozen 参数混合支持等。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.12.0+cu130 documentation"))

常用对应表：

| FSDP1 | FSDP2 |
|---|---|
| `FSDP(model, auto_wrap_policy=...)` | `fully_shard(layer); fully_shard(model)` |
| `ShardingStrategy.FULL_SHARD` | `reshard_after_forward=True` |
| `ShardingStrategy.SHARD_GRAD_OP` | `reshard_after_forward=False` |
| `HYBRID_SHARD` | 2D `DeviceMesh` / HSDP |
| `MixedPrecision` | `MixedPrecisionPolicy` |
| `CPUOffload` | `CPUOffloadPolicy` |
| `use_orig_params` | FSDP2 默认更接近 original params，不再使用 FSDP1 flat parameter 语义 |

FSDP2 不再主要用 `ShardingStrategy` 这个入口，而是用 `fully_shard()`、`DeviceMesh`、`reshard_after_forward` 表达相同行为。PyTorch FSDP2 迁移文档给出的对应关系是：`FULL_SHARD` 对应 `reshard_after_forward=True`，`SHARD_GRAD_OP` 对应 `reshard_after_forward=False`，`HYBRID_SHARD` 对应 `reshard_after_forward=True + 2D device mesh`，`_HYBRID_SHARD_ZERO2` 对应 `reshard_after_forward=False + 2D device mesh`。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.12.0+cu130 documentation"))

PyTorch FSDP2 文档还说明，如果传入 1D mesh，参数在该 mesh 上 fully sharded；如果传入 2D mesh，则在第 1 维 shard、第 0 维 replicate，这就是 HSDP 语义。([PyTorch Documentation](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html "torch.distributed.fsdp.fully_shard — PyTorch 2.12 documentation"))

---

## 3. FSDP 关键机制

### 3.1 wrap 粒度：不要只包 root，也不要包太碎

FSDP 的 wrap 粒度决定“一次 all-gather 多大参数、一次 reduce-scatter 多大梯度、临时完整参数峰值多高、通信和计算能否 overlap”。

FSDP2 文档强调，`fully_shard()` 应该 bottom-up 调用；每次调用会形成一个通信 group，该 group 的参数会在一个 collective 中 all-gather，梯度会在一个 collective 中 reduce-scatter；把模型分成多组，尤其是 layer-by-layer，可以节省峰值显存并提供通信/计算 overlap，通常不应该只对最顶层 root module 调用 `fully_shard()`。([PyTorch Documentation](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html "torch.distributed.fsdp.fully_shard — PyTorch 2.12 documentation"))

LLM 中最常见的默认策略是：

```text
每个 Transformer block / decoder layer 单独作为一个 FSDP unit；
最后再 wrap root model。
```

FSDP2 教程示例也是先遍历 `model.layers` 对每个 layer 调用 `fully_shard(layer)`，再对 root model 调用 `fully_shard(model)`；这样 forward 到某一层时，其余层仍保持 sharded。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.12.0+cu130 documentation"))

实践判断：

```text
wrap 太粗:
    all-gather 参数块过大；
    临时完整参数峰值高；
    容易 OOM；
    通信次数少，但 overlap 机会少。

wrap 太细:
    大量小 all-gather / reduce-scatter；
    NCCL launch 开销和 Python/autograd hook 开销变大；
    带宽利用率差；
    吞吐下降。

推荐起点:
    Transformer block 级别 wrap。

显存还不够:
    再考虑拆 Attention / MLP。

吞吐太低且显存富余:
    检查是否 wrap 过细。
```

Hugging Face Transformers 的 FSDP 文档也建议 Transformer 模型使用 transformer-based wrapping，并通过 `fsdp_transformer_layer_cls_to_wrap` 指定要包的层，例如 `BertLayer`；它还说明 embedding 等剩余层通常留在最外层 FSDP unit 中，避免共享权重被拆入不同 FSDP unit。([Hugging Face](https://huggingface.co/docs/transformers/en/fsdp "FullyShardedDataParallel · Hugging Face"))

### 3.2 mixed precision：不是简单一句 bf16

FSDP mixed precision 至少要区分三类 dtype：

```text
param_dtype:
    unsharded parameter 的 dtype；
    决定 forward/backward compute 和 parameter all-gather 的 dtype。

reduce_dtype:
    梯度 reduce-scatter / all-reduce 的 dtype；
    可以让 compute 用 bf16，但梯度规约用 fp32。

output_dtype:
    forward 输出是否 cast 到某个 dtype；
    FSDP2 中用于处理不同 module 有不同 mixed precision policy 的情况。
```

PyTorch FSDP2 文档明确说明，`param_dtype` 指定 unsharded parameter 的 dtype，因此影响 forward/backward 计算和 parameter all-gather；optimizer step 使用原始 dtype 的 sharded parameter；`reduce_dtype` 指定梯度规约 dtype，如果未设置但设置了 `param_dtype`，则默认用 compute dtype 做 reduction；`output_dtype` 用于 cast forward outputs。([PyTorch Documentation](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html "torch.distributed.fsdp.fully_shard — PyTorch 2.12 documentation"))

典型配置理解：

```python
# 计算和通信都偏 bf16，吞吐和通信更省
param_dtype = torch.bfloat16
reduce_dtype = torch.bfloat16

# 计算用 bf16，梯度通信用 fp32，更稳但通信更重
param_dtype = torch.bfloat16
reduce_dtype = torch.float32
```

在 LLM 后训练中，bf16 通常比 fp16 更稳，尤其是 PPO/GRPO 这类包含 logprob、KL、entropy、advantage normalization 的训练链路。真正排查 NaN 时，要同时看模型输出、loss scale、gradient norm、reduce dtype、optimizer state dtype，而不是只看 `mixed_precision=bf16`。

### 3.3 optimizer、gradient clipping 与参数对象

一个很重要的工程顺序是：

```text
先构建 model；
再 FSDP wrap / fully_shard；
最后创建 optimizer。
```

FSDP2 教程中，`fully_shard()` 后模型参数会变成 DTensor；示例也是先 `fully_shard`，再 `torch.optim.Adam(model.parameters())`。文档还说明 optimizer state dict 也以 DTensor 表示，`torch.optim.Adam` 和 `torch.nn.utils.clip_grad_norm_` 可以作用于 DTensor 参数。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.12.0+cu130 documentation"))

错误顺序：

```python
model = build_model()
optimizer = torch.optim.AdamW(model.parameters())  # 错误：太早
fully_shard(model)
```

正确顺序：

```python
model = build_model()

for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=...)
```

在 PPO/GRPO 训练中，gradient clipping 很常见。如果用 FSDP1，框架里可能封装了 FSDP-aware 的 `clip_grad_norm_`；如果用 FSDP2/DTensor，PyTorch 教程说明标准 `clip_grad_norm_` 可以直接处理 DTensor 参数。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.12.0+cu130 documentation"))

### 3.4 checkpoint 与 state_dict

普通单卡训练里，checkpoint 往往是：

```python
torch.save(model.state_dict(), path)
torch.save(optimizer.state_dict(), path)
```

但 FSDP 下参数和 optimizer states 是分片的，所以必须区分 checkpoint 类型：

```text
FULL_STATE_DICT:
    聚合成完整权重；
    适合导出 HuggingFace 模型、推理、最终保存；
    保存时可能显存/内存峰值高。

SHARDED_STATE_DICT:
    每个 rank 保存自己的 shard；
    适合训练中间 checkpoint 和 resume；
    大模型更推荐。

LOCAL_STATE_DICT:
    更偏 rank-local 状态；
    使用场景较少，通常由框架封装。
```

PyTorch FSDP1 文档建议在保存 full state dict 时使用 `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` 这类配置，以减少 GPU 显存和 CPU 内存压力；它也提供 `ShardedStateDictConfig` 和 optimizer state dict config 来控制模型和 optimizer checkpoint 的保存方式。([PyTorch Documentation](https://docs.pytorch.org/docs/2.12/fsdp.html "FullyShardedDataParallel — PyTorch 2.12 documentation"))

Hugging Face Accelerate 文档明确建议 FSDP 中间 checkpoint 使用 `SHARDED_STATE_DICT`；它说明 `accelerator.save_state("ckpt")` 会把模型和 optimizer 以 per-process shard 形式保存，resume 时用 `accelerator.load_state("ckpt")`；如果需要把 sharded FSDP 权重合并成单文件，可以用 `merge_fsdp_weights`，但这是 CPU-bound 过程。([Hugging Face](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp "Fully Sharded Data Parallel · Hugging Face"))

工程记忆：

```text
训练中:
    优先 sharded checkpoint，便于 resume，避免 full all-gather 峰值。

最终导出:
    full state dict / merge weights / save_pretrained。

RL 训练:
    不只保存 actor 权重，还可能需要 optimizer、scheduler、global step、RNG states、trainer extra state。

OOM 风险:
    full checkpoint 保存和加载都可能触发大规模 all-gather 或 CPU/GPU 内存峰值。
```

veRL 配置文档也说明，默认 checkpoint 保存 model、optimizer 和 extra information；extra information 当前包括 RNG states；默认不保存 `hf_model`，并提供 `scripts/model_merge.py` 将 checkpoint 格式转换为 Hugging Face 格式。([Verl](https://verl.readthedocs.io/en/latest/examples/config.html "Config Explanation — verl  documentation"))

### 3.5 offload：用吞吐换容量

FSDP 可以通过 offload 进一步省 GPU 显存，但它通常用吞吐换容量。

FSDP2 的 `CPUOffloadPolicy` 会把 parameters、gradients、optimizer states offload 到 CPU；sharded parameters 在 all-gather 前从 host 拷到 device，backward 中 sharded gradients 从 device 拷回 host，optimizer step 在 CPU 上执行。文档也提到 pinned memory 可以提高 H2D/D2H 拷贝效率并与 compute overlap，但 pinned memory 不能被其他进程使用，如果 CPU 内存不足需要谨慎。([PyTorch Documentation](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html "torch.distributed.fsdp.fully_shard — PyTorch 2.12 documentation"))

经验判断：

```text
模型能放下:
    尽量不要开 param/optimizer CPU offload，吞吐会更好。

模型放不下:
    优先考虑 optimizer offload，再考虑 param offload。

reference model:
    在 RL 中通常 forward-only，且不训练；
    大模型 ref 可以更适合 offload。

rollout/vLLM:
    更多受 KV cache、gpu_memory_utilization、max_num_batched_tokens、并发数影响；
    不能只从 FSDP 角度看显存。
```

veRL 配置文档中也建议，对于大于 7B 的 reference model，默认推荐打开 offload。([Verl](https://verl.readthedocs.io/en/latest/examples/config.html "Config Explanation — verl  documentation"))

### 3.6 activation checkpointing：FSDP 不自动解决 activation memory

FSDP 主要省 model states，不直接省 activation。长上下文训练、较大 micro batch、较大 vocab logits、PPO 多次 forward/backward 都可能让 activation 成为主要显存瓶颈。

PyTorch 对 activation checkpointing 的解释是：activation memory 会随着 forward 累积，通常在 backward 开始处达到峰值；checkpointing 在 forward 中不保存某个区域内部的中间张量，只保存输入，backward 时重新执行该区域来恢复中间 activation，因此用额外计算换显存。([PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/ "Current and New Activation Checkpointing Techniques in PyTorch – PyTorch"))

在 LLM 后训练中，activation checkpointing 通常按 Transformer block 打开，和 FSDP wrap 边界保持一致最容易分析：

```text
FSDP:
    省 parameters / gradients / optimizer states。

activation checkpointing:
    省 forward 中保存给 backward 的中间激活。

flash attention / fused kernels:
    降低 attention、RMSNorm、SwiGLU、RoPE 等算子显存和开销。

micro batch / dynamic batch:
    控制每次 forward/backward 的 token 数。
```

veRL 性能调优文档建议打开 `actor_rollout_ref.model.enable_gradient_checkpointing=True` 和 `critic.model.enable_gradient_checkpointing=True`，这通常允许更大的 micro batch；也提到 activation offloading 可以和 gradient checkpointing 配合，以获得更大的 micro batch，并且该功能当前只在 FSDP backend 可用。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

### 3.7 prefetch 与通信/计算 overlap

FSDP 的核心性能优化之一是 overlap：在当前层计算时，提前发起下一层参数 all-gather，尽量把通信藏在计算之后。

FSDP2 教程说明，implicit prefetching 会在 layer i 前发起 all-gather i，并把 all-gather 排到自己的 CUDA stream 中；对于非 CPU-bound 的 Transformer 大 batch 负载，layer i+1 的 all-gather 可以和 layer i 的计算重叠；官方也建议用户先从 implicit prefetching 开始，再考虑 explicit prefetching。([PyTorch Documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.12.0+cu130 documentation"))

veRL 性能文档也提到，在训练阶段可以通过 `fsdp_config.forward_prefetch=True` 启用 FSDP forward prefetch，例如 `actor_rollout_ref.actor.fsdp_config.forward_prefetch=True`，以便在当前 forward 计算完成前预取下一次 forward all-gather。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

实践判断：

```text
forward_prefetch=True:
    可能提升吞吐；
    可能增加瞬时显存，因为当前 unit 和预取 unit 参数可能同时存在。

backward_prefetch:
    FSDP1 常见 BACKWARD_PRE；
    veRL 文档特别提示 BACKWARD_POST 在 nested module 情况可能有问题，因此不支持相关策略。

显存紧张时:
    先关闭或保守使用 prefetch。

通信成为瓶颈时:
    再尝试 forward prefetch / hybrid shard / 调整 wrap 粒度。
```

---

## 4. FSDP 在 veRL / RL 后训练中的位置

### 4.1 FSDP 不是算法本身

在 veRL 这种 RL 后训练框架里，FSDP 不是算法本身。PPO、GRPO、DAPO 等算法关心的是 rollout、reward、advantage、KL、policy loss、value loss、entropy、optimizer step；FSDP 关心的是 actor/critic/ref/reward model 如何在多卡上放置、forward/backward、同步梯度、保存 checkpoint。

veRL 文档说明，它的 PyTorch FSDP backend 实现了 actor、critic、reference、rollout 和 reward models 的 worker，并实现了 `FSDPVLLMShardingManager`，用于在 FSDP 和 vLLM 之间 reshard 权重；该 backend 优点是容易支持多种模型、便于组织每个模型的 forward/backward，缺点是面对 Llama 70B/405B 这类大规模模型扩展性较差，actor 与 rollout 之间的 resharding overhead 可能比 Megatron-LM backend 更大。([Verl](https://verl.readthedocs.io/en/v0.4.1/workers/fsdp_workers.html "PyTorch FSDP Backend — verl  documentation"))

### 4.2 veRL worker 组织

veRL 当前的 worker 组织可以这样记：

```text
ActorRolloutRefWorker:
    actor + rollout + optional reference policy；
    常用于 colocated PPO；
    rollout 可以是 vLLM / SGLang；
    actor 通常是训练引擎，可能用 FSDP/FSDP2。

TrainingWorker:
    通用训练 worker；
    用于 critic、reference model、reward model、SFT/DPO 等；
    内部 engine 可以是 fsdp / fsdp2 / megatron / automodel / veomni / torchtitan。
```

veRL 最新 engine worker 文档说明，`ActorRolloutRefWorker` 是 actor、rollout 和可选 reference policy 的 hybrid worker；`TrainingWorker` 是 “one engine + optimizer + profiler” 的通用 worker，也可用于 critic、reference model、reward model 和 SFT/DPO 训练。([Verl](https://verl.readthedocs.io/en/latest/workers/engine_workers.html "Engine Workers — verl  documentation"))

### 4.3 Actor

actor 是真正被 policy loss 更新的模型。它需要：

```text
训练阶段:
    forward 计算 logprob / entropy / loss；
    backward；
    gradient clipping；
    optimizer step；
    checkpoint。

FSDP 作用:
    降低 actor 参数、梯度、optimizer state 显存；
    影响 update_policy 的吞吐和 OOM。
```

actor 通常是 FSDP 最重要的目标，因为它既要训练，又可能需要和 rollout engine 同步权重。

### 4.4 Rollout

rollout 负责生成 response，通常使用 vLLM/SGLang 等推理引擎。rollout 的瓶颈常常不是 FSDP 训练显存，而是：

```text
KV cache；
gpu_memory_utilization；
max_num_batched_tokens；
max_num_seqs；
sampling 并发；
prompt/response 长度；
权重从 actor 同步到 rollout 的开销。
```

veRL 配置文档说明，`actor_rollout_ref.rollout.dtype` 应与 FSDP/Megatron backend 中 actor model 参数类型对齐；`rollout.gpu_memory_utilization` 控制 vLLM 或 SGLang 实例使用的 GPU 显存比例；`rollout.load_format` 中 `dtensor` 是 FSDP backend 且 `StateDictType.SHARDED_STATE_DICT` 时推荐的 Hugging Face weight loader，而 `hf` 对应 `FULL_STATE_DICT`，但会产生更高峰值显存。([Verl](https://verl.readthedocs.io/en/latest/examples/config.html "Config Explanation — verl  documentation"))

### 4.5 Reference model

reference policy 通常用于 KL 约束或 reward 中的 KL。它多为 forward-only，不训练。veRL 文档说明，当 `actor.use_kl_loss` 或 `algorithm.use_kl_in_reward` 为 True 时，会启用 reference model；对于大于 7B 的模型，推荐默认开启 ref offload。([Verl](https://verl.readthedocs.io/en/latest/examples/config.html "Config Explanation — verl  documentation"))

工程上：

```text
ref 不需要 optimizer；
可以更激进地 offload；
log_prob_micro_batch_size_per_gpu 可以比 actor training micro batch 更大；
但如果 ref 与 actor colocate，也要考虑同一组 GPU 上的显存竞争。
```

### 4.6 Critic

critic 用于 value function 训练，PPO 中常见，GRPO 中可能不需要。critic 也可能用 FSDP，因为它要训练，有 forward/backward/optimizer state。veRL 性能文档建议 actor/critic 都可以打开 gradient checkpointing，并且 critic/reward 的 micro batch 或 max token limit 通常可以比 actor 更大，因为 actor 的最后 vocab head 带来的 logits 显存更重。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

### 4.7 Reward model

reward model 可以是规则函数，也可以是模型。如果是模型型 RM，通常是 forward-only 或少量训练，显存策略更接近 reference model。veRL PPO 架构文档说明，reward model 可以通过 `TrainingWorker` 加入资源映射，默认 worker 支持典型 Hugging Face `AutoModelForSequenceClassification` 布局。([Verl](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html "PPO Example Architecture — verl  documentation"))

---

## 5. veRL 配置怎么读

### 5.1 先区分 global batch 和 per-GPU micro batch

在 RL 后训练中，batch 相关配置非常容易混淆。veRL 性能文档给出的核心原则是：算法指标如 train batch size、PPO mini-batch size 是 global 参数；性能相关参数如 micro batch size、dynamic batch 的 max token length 是 local/per-GPU 参数，并且建议使用 `*micro_batch_size_per_gpu` 而不是即将废弃的 `*micro_batch_size`。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

常见配置含义：

```text
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu:
    actor update_policy 中每张 GPU 的训练 micro batch。

actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu:
    reference policy 计算 ref_log_prob 的 forward-only micro batch。

actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu:
    rollout 或 actor 重新计算 log_prob 的 forward-only micro batch。

critic.ppo_micro_batch_size_per_gpu:
    critic update 的训练 micro batch。

reward_model.forward_micro_batch_size_per_gpu:
    reward model forward 的 micro batch。
```

veRL 文档还说明，forward-only 参数，例如 reference logprob、rollout logprob、critic forward，可以比 actor/critic training micro batch 更大；critic 和 reward model 的 micro batch 也可以比 actor 更大，因为 actor final vocab layer 的 logits 显存更重。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

### 5.2 dynamic batch 看 token 上限

Dynamic batch 的思想是让每次 forward/backward 处理相近数量的 token，而不是固定样本数。veRL 文档说明，开启 `use_dynamic_bsz=True` 后，不再主要调 `*micro_batch_size_per_gpu`，而应调 `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`、`critic.ppo_max_token_len_per_gpu`、`ref.log_prob_max_token_len_per_gpu`、`rollout.log_prob_max_token_len_per_gpu`、`reward_model.forward_micro_batch_size_per_gpu` 等 token 上限。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

### 5.3 读配置时的顺序

```text
1. 先看 backend:
    fsdp / fsdp2 / megatron / rollout-only。

2. 再看哪些模型参与训练:
    actor 一定训练；critic 取决于算法；ref 通常 forward-only；reward model 可能是规则或模型。

3. 然后看 sharding 和 wrap:
    FULL_SHARD / SHARD_GRAD_OP / HYBRID_SHARD / reshard_after_forward。

4. 接着看显存相关:
    bf16、gradient checkpointing、activation offload、CPU offload、forward_prefetch。

5. 最后看 batch 和 rollout:
    micro_batch_size_per_gpu、max_token_len_per_gpu、gpu_memory_utilization、max_num_batched_tokens、max_num_seqs。
```

---

## 6. 常见 OOM 排查路线

### 6.1 初始化阶段 OOM

常见原因：

```text
每个 rank 都加载完整 checkpoint；
没有使用 meta device / empty init / cpu_ram_efficient_loading；
sync_module_states 配置不对；
FSDP wrap 之前模型已经完整落到 GPU。
```

Accelerate 文档提到，`fsdp_cpu_ram_efficient_loading=True` 时，只有第一个进程加载 pretrained checkpoint，其他进程为空权重；该设置需要 `fsdp_sync_module_states=True`，否则非主进程可能有随机权重。([Hugging Face](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp "Fully Sharded Data Parallel · Hugging Face"))

### 6.2 forward 前 OOM

常见原因：

```text
FSDP unit 太大，一次 all-gather 参数过大；
wrap 太粗，只包 root；
forward_prefetch 导致当前和下一个 unit 参数同时存在；
rollout/vLLM 占用了过多显存；
micro batch 或 max token length 太大。
```

排查顺序：

```text
1. 确认是否按 Transformer block wrap。
2. 先关 forward_prefetch。
3. 降低 actor/ref/rollout 的 per-GPU micro batch 或 max token length。
4. 如果 actor 与 rollout colocate，降低 rollout.gpu_memory_utilization。
5. 检查是否在 FSDP wrap 前就把完整模型放到了 GPU。
```

### 6.3 backward 开始处 OOM

常见原因：

```text
activation 在 backward 开始达到峰值；
backward all-gather 参数与 activation 峰值叠加；
gradient checkpointing 没开；
sequence length / response length 太长；
entropy/logits 计算保留了大 [bsz * seq_len, vocab] 张量。
```

PyTorch activation checkpointing 文档解释，默认 eager autograd 会在 forward 中不断保存 activation，通常在 backward 开始处达到峰值；checkpointing 可以减少保存的中间张量，但会带来重算成本。([PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/ "Current and New Activation Checkpointing Techniques in PyTorch – PyTorch"))

veRL 文档也专门提到 logits 形状通常为 `[bsz*seq_len, voc]`，熵计算可能造成显存峰值，可以通过 chunked entropy 或 entropy checkpointing 降低峰值。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

### 6.4 rollout OOM

rollout OOM 往往和 FSDP 不是同一个瓶颈。优先检查：

```text
KV cache；
rollout.gpu_memory_utilization；
max_num_batched_tokens；
max_num_seqs；
prompt length；
response length；
sampling 并发；
actor 和 rollout 是否共用同一组 GPU。
```

如果训练端 actor OOM 和 rollout OOM 交替出现，通常要同时降低训练 micro batch 和 rollout 显存占用，不能只调 FSDP sharding。

### 6.5 checkpoint 保存 OOM 或卡住

常见原因：

```text
保存 FULL_STATE_DICT 时触发大规模 all-gather；
rank0 CPU 内存不足；
NCCL timeout；
optimizer state dict 太大；
训练中间 checkpoint 没用 SHARDED_STATE_DICT。
```

Accelerate 文档建议 FSDP 中间 checkpoint 使用 `SHARDED_STATE_DICT`，并指出保存 full state dict 即便 CPU offload 也可能耗时并导致 NCCL timeout。([Hugging Face](https://huggingface.co/docs/transformers/en/fsdp "FullyShardedDataParallel · Hugging Face"))

---

## 7. 性能调优路线

优先级可以这样排：

```text
第一步：确认模型能稳定跑
    FULL_SHARD；
    Transformer block wrap；
    bf16；
    gradient checkpointing；
    合理 micro_batch_size_per_gpu。

第二步：提升吞吐
    增大 micro_batch_size_per_gpu 或 max_token_len_per_gpu；
    使用 dynamic batch；
    打开 remove padding / sequence packing；
    调整 rollout max_num_batched_tokens、max_num_seqs；
    尝试 forward_prefetch；
    检查是否 wrap 过细。

第三步：降低通信瓶颈
    单机内优先 FULL_SHARD；
    多机时考虑 HYBRID_SHARD / HSDP；
    减少跨机 all-gather / reduce-scatter；
    检查网络拓扑、NCCL、IB/NVLink。

第四步：极限省显存
    optimizer offload；
    param offload；
    activation offload；
    更小 micro batch；
    更细 wrap；
    更保守 prefetch；
    sharded checkpoint。
```

veRL 性能文档建议调优 rollout generation throughput、remove padding、batch size、dynamic batch、Ulysses sequence parallel、LigerKernel、FSDP forward prefetch、entropy logits memory 等多个环节。([Verl](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html "Performance Tuning Guide — verl  documentation"))

调优时要避免一个常见误区：只看 FSDP。RL 后训练是训练端和推理端耦合的系统，actor update、ref logprob、rollout generation、reward 计算、critic update、checkpoint 都可能成为瓶颈。

