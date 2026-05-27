
## 0. 快速结论

Ray 在 veRL 里主要解决四件事：**启动多个远程进程、按 CPU/GPU 资源调度这些进程、把 Python 方法包装成远程 RPC、在不同 worker 之间传递对象引用和结果**。它不是模型并行通信库，也不是 GPU 显存管理器。Ray 负责“谁在哪台机器哪张卡上运行”，而 FSDP、Megatron、vLLM 的 TP/PP、NCCL/torch.distributed 负责“模型内部 tensor 怎么通信”。Ray Core 的官方心智模型就是 task、actor、object、placement group 这些基础原语；vLLM 的官方文档也把 Ray 定位为多节点推理默认 distributed runtime，而 tensor parallel / pipeline parallel 是 vLLM 自己的并行策略。([Ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html?utm_source=chatgpt.com "What's Ray Core? — Ray 2.55.1 - Ray Docs"))

必须掌握的 6 个概念是：`Driver / Controller`、`Task`、`Actor`、`ObjectRef / Object Store`、`Resource Scheduler`、`Placement Group`。在 veRL 里再额外记住 4 个概念：`ResourcePool`、`RayWorkerGroup`、`@register`、`dispatch_fn / collect_fn`。veRL 的 single_controller 设计文档明确说，`WorkerGroup` 负责管理一组远程 worker，`ResourcePool` 负责把计算资源绑定到 worker 进程，`ClassWithArgs` 用于延迟实例化远程对象。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

常见排查入口可以按这个顺序看：worker pending 先看 `ray status` 和 placement group；actor 是否启动看 `ray list actors`；placement group 是否泄漏或卡住看 `ray list placement-groups`；object store 爆内存看 `ray memory` / dashboard；actor died 看 `ray logs`；vLLM rollout 卡住但 Ray 正常时，再看 NCCL、torch.distributed、vLLM executor 日志。Ray Dashboard 和 State API 官方支持查看 tasks、actors、placement groups、resource demands 和 logs。([Ray](https://docs.ray.io/en/latest/ray-observability/getting-started.html "Ray Dashboard — Ray 2.55.1"))

---

## 1. Ray Core 模型

### 1.1 Cluster / Head node / Worker node

Ray cluster 由一个 head node 和若干 worker nodes 组成。head node 除了像普通节点一样能运行任务，还会运行一些集群级进程，例如 GCS、autoscaler、driver jobs；worker node 主要负责运行用户的 task / actor，并参与分布式调度和对象存储。Ray 官方也提醒，在大规模集群里，head node 默认也可能被调度 task / actor，这往往不是期望行为。([Ray](https://docs.ray.io/en/latest/cluster/key-concepts.html "Key Concepts — Ray 2.55.1"))

对 veRL 来说，可以粗略理解成：

```text
driver / controller 进程
    |
    | 通过 Ray 创建 actor
    v
Ray cluster
    |
    +-- node 0: actor / rollout / critic / ref / reward workers
    +-- node 1: actor / rollout / critic / ref / reward workers
    +-- node N: ...
```

Ray 是“分布式进程编排层”，不是“训练算法本身”。

### 1.2 Driver / Controller

Ray job 是由同一个脚本产生的一组 tasks、objects、actors；运行 Python 脚本的 worker 被称为 driver。([Ray](https://docs.ray.io/en/latest/cluster/key-concepts.html "Key Concepts — Ray 2.55.1"))

在 veRL 里，你可以把 single controller 理解成 Ray driver 上的一层训练控制逻辑。它负责按 PPO / GRPO / DPO 等算法流程调用各类 worker，例如：

```text
controller:
    generate_sequences()
    compute_log_prob()
    compute_reward()
    compute_advantage()
    update_policy()
```

这些调用在代码上看起来像普通 Python 方法，但背后可能会被 `RayWorkerGroup` 分发成多个 Ray actor method call。

### 1.3 Task

Ray task 是把普通函数变成异步远程函数。使用 `@ray.remote` 装饰函数后，调用 `.remote()` 不会立即返回函数值，而是返回一个 `ObjectRef`。Ray 会在后台把这个函数调度到某个 worker process 上执行。Ray 文档明确说，Ray remote function 的异步调用就是 Ray task。([Ray](https://docs.ray.io/en/latest/ray-core/tasks.html "Tasks — Ray 2.55.1"))

最小例子：

```python
import ray

ray.init()

@ray.remote
def f(x):
    return x * 2

ref = f.remote(10)      # ref 是 ObjectRef，不是 20
value = ray.get(ref)    # value == 20
```

Task 更适合无状态、短生命周期、可并行展开的计算。例如数据预处理、独立 reward 计算、批量文件处理等。

### 1.4 Actor

Ray actor 是把 Python class 变成有状态远程对象。实例化 actor 时，Ray 会创建一个新的 worker process，并把该 actor 的方法调度到这个特定 worker 上运行；这些方法可以访问和修改 actor 内部状态。([Ray](https://docs.ray.io/en/latest/ray-core/actors.html "Actors — Ray 2.55.1"))

最小例子：

```python
import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
        return self.value

counter = Counter.remote()
ref = counter.inc.remote()
print(ray.get(ref))  # 1
```

Actor 更适合有状态、长生命周期、绑定 GPU 上下文的对象。例如：模型 worker、optimizer worker、rollout engine、vLLM engine、parameter server、训练状态管理器等。veRL 主要使用 Actor，而不是 Task，因为 RL 后训练中的 worker 通常持有模型、tokenizer、optimizer、CUDA context、NCCL process group 等状态。

### 1.5 ObjectRef

`ObjectRef` 可以理解为“远程对象的指针”或“future”。它不直接包含对象值，而是引用 Ray object store 中的远程对象。`ObjectRef` 有两种常见来源：remote function / actor method 的返回值，或者 `ray.put()` 的返回值。([Ray](https://docs.ray.io/en/latest/ray-core/objects.html "Objects — Ray 2.55.1"))

```python
x_ref = ray.put({"a": 1})
x = ray.get(x_ref)
```

重要心智模型是：**`.remote()` 提交计算，返回引用；`ray.get()` 拉取结果，可能阻塞。**

### 1.6 Object Store

Ray 会把 remote object 缓存在分布式 shared-memory object store 中，并且每个节点都有一个 object store。对象可以存在于一个或多个节点上，与持有 `ObjectRef` 的进程不必在同一个地方。Ray remote object 是 immutable 的，创建后不能原地修改。([Ray](https://docs.ray.io/en/latest/ray-core/objects.html "Objects — Ray 2.55.1"))

这对 veRL 很重要，因为 `DataProto`、tensor batch、rollout 结果、log probs、reward 等数据在 Ray worker 和 controller 之间传递时，经常会经过序列化、object store、`ray.get()` 这些路径。大对象频繁穿过 controller，会造成 CPU 内存、object store memory、序列化和反序列化开销。

### 1.7 Scheduler

Ray scheduler 看的是**逻辑资源**，不是你机器上的真实使用率。Ray resource 是 key-value pair，例如 `CPU: 8`、`GPU: 4`、`NPU: 8`、`accelerator_type:A100: 0.001`。Ray 会根据 task / actor 声明的资源需求决定它能不能被调度。Ray 官方文档强调，Ray 资源主要用于调度准入控制，不会限制程序真实使用多少 CPU 线程或 GPU 显存。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html "Resources — Ray 2.55.1"))

这解释了一个常见现象：`nvidia-smi` 显示 GPU 空着，但 Ray worker 仍然 pending。原因可能不是 GPU 真的忙，而是 Ray 逻辑资源、placement group、bundle、custom resource、accelerator_type、head/worker node 资源声明不满足。

---

## 2. Task vs Actor

### 2.1 Stateless task

Task 是 stateless 的。每次调用 remote function，本质上是提交一个独立计算。它适合纯函数式计算：

```python
@ray.remote
def score(sample):
    return reward_fn(sample)

refs = [score.remote(x) for x in samples]
scores = ray.get(refs)
```

这种模式的优点是简单、容易扩展、适合大量独立任务。缺点是它不适合长期持有模型权重、GPU context、通信组、缓存等状态。

### 2.2 Stateful actor

Actor 是 stateful 的。一个 actor 创建后会长期存在，方法调用发生在同一个 actor 进程里：

```python
@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, model_path):
        self.model = load_model(model_path).cuda()

    def forward(self, batch):
        return self.model(batch)
```

这正是 veRL worker 的典型形态：worker 不是一次性函数，而是一个长期运行的训练/推理参与者。

### 2.3 Actor lifecycle

创建 actor：

```python
worker = ModelWorker.remote(model_path)
```

调用 actor method：

```python
ref = worker.forward.remote(batch)
```

同步拿结果：

```python
output = ray.get(ref)
```

异步批量调用：

```python
refs = [w.forward.remote(batch_i) for w, batch_i in zip(workers, batches)]
outputs = ray.get(refs)
```

Actor handle 可以传给其他 task / actor / driver。只要 handle 还存在，就可以继续远程调用该 actor。Ray 的 actor 生命周期、命名 actor、detached actor 都可以影响资源释放和泄漏排查；placement group 也有 `detached` 生命周期选项，detached placement group 会独立于创建者而存在。([Ray](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.placement_group.html "ray.util.placement_group — Ray 2.55.1"))

### 2.4 为什么 veRL 主要用 Actor

veRL 的训练角色通常包括 actor、rollout、reference、critic、reward model 等。每个角色都可能持有模型权重、优化器状态、tokenizer、device mesh、NCCL process group、vLLM engine 或 SGLang engine。Task 无法自然表达这种长期状态；Actor 可以。

所以 veRL 里更常见的结构是：

```text
RayWorkerGroup
    |
    +-- Ray actor worker 0
    +-- Ray actor worker 1
    +-- Ray actor worker 2
    +-- Ray actor worker 3
```

然后 controller 调用：

```python
rollout_wg.generate_sequences(prompts)
actor_wg.compute_log_prob(batch)
actor_wg.update_policy(batch)
```

表面是一个方法，背后是多个 actor method calls。

---

## 3. ObjectRef 与 Object Store

### 3.1 `ray.put` / `ray.get`

`ray.put(obj)` 把对象放进 Ray object store，返回 `ObjectRef`。`ray.get(ref)` 根据 `ObjectRef` 拉取对象值。如果当前节点没有这个对象，Ray 会下载它。对于 NumPy array 或 NumPy array collection，Ray 文档说明 `ray.get()` 可以 zero-copy 返回由 shared object store memory 支持的数组；其他对象通常会反序列化成 Python 对象。([Ray](https://docs.ray.io/en/latest/ray-core/objects.html "Objects — Ray 2.55.1"))

```python
x_ref = ray.put({"tokens": [1, 2, 3]})
x = ray.get(x_ref)
```

### 3.2 top-level vs nested ObjectRef

这是读 veRL 数据流时很容易忽略的一点。Ray 对 `ObjectRef` 参数有两种处理方式：

当 `ObjectRef` 作为 top-level 参数传给 task / actor method 时，Ray 会自动 dereference，也就是先取到真实值再执行函数。

```python
actor.method.remote(obj_ref)     # actor 看到的是 obj 的真实值
```

当 `ObjectRef` 被包在 list / dict / tuple 等 nested object 里时，Ray 不会自动 dereference，函数里看到的仍然是引用，需要自己 `ray.get()`。

```python
actor.method.remote([obj_ref])   # actor 看到的是 [ObjectRef(...)]
```

Ray 官方文档明确区分了 top-level argument 和 nested argument 的行为。([Ray](https://docs.ray.io/en/latest/ray-core/objects.html "Objects — Ray 2.55.1"))

### 3.3 object store memory

Object store memory 是 Ray object store 的内存空间，不等于 GPU 显存，也不等于 Python heap。Ray 默认会根据启动时可用内存设置 logical memory 和 object store memory；文档里说明 `object_store_memory` 不是 logical scheduling resource，不能像 `CPU` / `GPU` 那样用于调度。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html "Resources — Ray 2.55.1"))

常见误区：

```text
错误理解：num_gpus=1 意味着 Ray 会限制这个 actor 只能用 1 张卡的显存。
正确理解：num_gpus=1 主要让 Ray 分配 1 个逻辑 GPU，并设置 CUDA_VISIBLE_DEVICES。
```

Ray 会设置 `CUDA_VISIBLE_DEVICES` 来帮助大多数 ML 框架只看到被分配的 GPU，但如果程序绕过或覆盖这个环境变量，Ray 并不能真正阻止它使用其他 GPU。Ray 文档也明确说，Ray 会设置可见设备变量，但用户代码仍有责任不要超用 accelerator memory。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html "Accelerator Support — Ray 2.55.1"))

### 3.4 object spilling

当 object store 满了，Ray 会把对象 spill 到本地文件系统目录；默认目录通常在 Ray session 的临时目录下。你也可以通过 `ray.init(object_spilling_directory=...)` 或 `ray start --object-spilling-directory=...` 指定目录。([Ray](https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html "Object Spilling — Ray 2.55.1"))

排查时要看：

```bash
ray memory
ray memory --stats-only
cat /tmp/ray/session_latest/logs/raylet.out | grep Spilled
```

object spilling 能避免部分 object store OOM，但不等于没有代价。spill / restore 会引入磁盘 IO，训练吞吐可能突然下降。

### 3.5 DataProto 为什么可能造成 controller bottleneck

veRL 的 `DataProto` 是数据交换协议，包含 `batch` 和 `meta_info`；当前源码里还可以看到 `non_tensor_batch`。`batch` 通常是 TensorDict，用于承载 tensor 数据。([verl](https://verl.readthedocs.io/en/latest/api/data.html?utm_source=chatgpt.com "Data interface — verl documentation - Read the Docs"))

在 classic single-controller 路线里，controller 调用 `RayWorkerGroup` 方法，`dispatch_fn` 把输入拆给多个 workers，workers 返回结果，`collect_fn` 再把结果聚合。veRL single_controller 文档明确描述了这个调用链：`dispatch_fn` split 输入，`execute_fn` 做远程调用，`collect_fn` gather 结果。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

所以如果 `DataProto` 里装了很大的 tensor，例如多模态 RL 里的图像、视频、长序列 token、log prob、value、reward、mask 等，就可能出现：

```text
worker -> Ray Object Store -> controller ray.get -> collect_fn -> 新 DataProto
```

这个路径会给 controller 带来 CPU 内存、序列化、反序列化和 object store 压力。这里的瓶颈不是 Ray “错了”，而是同步 single-controller 聚合方式天然容易让 controller 变成数据汇聚点。

---

## 4. 资源调度

### 4.1 `num_cpus` / `num_gpus`

Ray 允许在 task 或 actor 上声明逻辑资源需求：

```python
@ray.remote(num_cpus=2, num_gpus=1)
class Worker:
    ...
```

Ray 只有在某个节点有足够空闲逻辑资源时，才会把这个 actor 调度到该节点。Ray 文档也提醒，默认情况下 Ray task 使用 1 个逻辑 CPU；Ray actor 在调度时默认需要 1 个逻辑 CPU，但运行时默认 0 个逻辑 CPU，因此实践中最好显式声明 actor 的 `num_cpus`，避免意外。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html "Resources — Ray 2.55.1"))

### 4.2 custom resources

custom resource 是给节点打一个数值资源标签，然后 task / actor 通过 `resources={...}` 要求调度到满足该资源的节点。

```bash
ray start --head --resources='{"special_hardware": 1}'
```

```python
@ray.remote(resources={"special_hardware": 1})
def f():
    ...
```

Ray 官方建议，当你需要用数值资源进行调度控制时使用 custom resources。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html "Resources — Ray 2.55.1"))

在 veRL 里，custom resource 可以用于区分特定机器、特定硬件池、特殊网络拓扑、NPU/GPU 类型等。

### 4.3 `accelerator_type`

`accelerator_type` 用于要求 task / actor 运行在特定 accelerator 类型的节点上。Ray 内部把它实现成类似 `"accelerator_type:<type>": 0.001` 的 custom resource requirement。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html "Accelerator Support — Ray 2.55.1"))

```python
from ray.util.accelerators import NVIDIA_TESLA_V100

@ray.remote(num_gpus=1, accelerator_type=NVIDIA_TESLA_V100)
def train():
    ...
```

在混合集群里，例如 A100 + H100 + L40S，如果没有正确声明 accelerator type，就可能出现模型 worker 被调度到不合适的卡上。

### 4.4 Ray 资源 vs GPU 显存

Ray 的 `GPU: 1` 不是“显存预约 80GB”，而是“逻辑上占用一张 GPU”。Ray 会设置 `CUDA_VISIBLE_DEVICES`，但它不理解你的模型需要多少显存，也不会阻止 PyTorch 在可见 GPU 上吃满显存。Ray 文档明确说，资源需求不会限制真实物理资源使用；对于 GPU，Ray 主要通过可见设备变量提供隔离。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html "Resources — Ray 2.55.1"))

因此：

```text
Ray 层 pending：
    多半是逻辑资源 / placement group / bundle 不满足。

CUDA OOM：
    多半是模型、batch size、sequence length、KV cache、optimizer state、activation 等显存不够。

GPU 空着但 Ray 不调度：
    多半是 Ray 资源声明、PG、custom resource、accelerator_type 或节点状态问题。
```

### 4.5 `CUDA_VISIBLE_DEVICES`

Ray 对 GPU actor / task 会设置 `CUDA_VISIBLE_DEVICES`，大多数深度学习框架会尊重它。Ray accelerator 文档给出的例子显示，一个 GPU actor 和一个 GPU task 会看到不同的 `CUDA_VISIBLE_DEVICES`。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html "Accelerator Support — Ray 2.55.1"))

在 worker 里排查时可以打印：

```python
import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))
```

这比单独看 `nvidia-smi` 更接近 Ray 的视角。

### 4.6 worker pending 排查

worker pending 时，优先按以下路径查：

```bash
ray status
ray status -v
ray list actors
ray list placement-groups
ray list placement-groups --detail
```

重点看：

```text
1. Demands 里缺什么资源？
2. placement group 是否 CREATED，还是 PENDING？
3. bundle 是否太大，单节点放不下？
4. STRICT_PACK 是否要求所有 bundle 放到一台机器？
5. custom resources / accelerator_type 是否写错？
6. head node 是否被设成 num-cpus=0 或没有 GPU？
7. worker node 是否已经加入 Ray cluster？
```

Ray 文档说明 `ray status` 可以显示节点状态、资源使用、pending / failed nodes、resource demands；placement group 文档也说明可以用 `ray status`、Dashboard 和 State API 检查 placement group 状态。([Ray](https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html "Monitoring with the CLI or SDK — Ray 2.55.1"))

---

## 5. Placement Group

### 5.1 bundle

Placement group 是 Ray 的 gang scheduling 机制。它可以原子性地预留一组资源，并让多个 task / actor 按指定策略放置。Ray 文档说，placement group 可以跨多个节点原子性预留资源，经常用于 gang scheduling actors 或 tasks。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html "Placement Groups — Ray 2.55.1"))

bundle 是 placement group 的最小预留单位：

```python
bundles = [
    {"CPU": 4, "GPU": 1},
    {"CPU": 4, "GPU": 1},
    {"CPU": 4, "GPU": 1},
    {"CPU": 4, "GPU": 1},
]
```

一个 bundle 必须能放进单个节点。如果你有两台机器，每台 8 GPU，不能创建一个 `{"GPU": 16}` 的 bundle；应该创建 16 个 `{"GPU": 1}` bundle，或者按 node 拆成多个 placement group。

### 5.2 PACK / SPREAD / STRICT_PACK / STRICT_SPREAD

Ray placement group 支持四种常见策略：

```text
PACK:
    尽量把 bundles 放到尽可能少的节点上。

SPREAD:
    尽量把 bundles 均匀分散到不同节点上。

STRICT_PACK:
    必须把整个 placement group 放到一个节点上；放不下就 pending。

STRICT_SPREAD:
    bundles 必须分散到不同节点上。
```

Ray API 文档对这些策略有明确说明。([Ray](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.placement_group.html "ray.util.placement_group — Ray 2.55.1"))

在 LLM 训练/推理里，策略选择通常和通信拓扑有关：

```text
同一个 tensor parallel group：
    倾向 PACK / STRICT_PACK，减少跨节点通信。

多个独立 data parallel replica：
    可以 SPREAD，避免单节点故障影响全部 replica。

多节点 pipeline parallel：
    可能需要按 stage 控制放置，而不是盲目 PACK。
```

### 5.3 gang scheduling

Placement group 的关键价值是“要么资源一起拿到，要么不要部分启动”。分布式训练经常需要所有 rank 同时启动，否则某些 rank 先起来后会等待其他 rank，造成死锁或长时间挂起。Ray 文档把 placement group 和 gang scheduling 直接关联起来，并说明它常用于分布式训练等需要 all-or-nothing scheduling 的场景。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html "Placement Groups — Ray 2.55.1"))

### 5.4 ResourcePool 和 WorkerGroup

veRL 的 `RayResourcePool.get_placement_groups()` 会基于 `process_on_nodes` 构造 placement groups。源码里可以看到，它会为每个 process count 创建 bundles，默认 bundle 包含 CPU，如果 `use_gpu=True`，还会加入 `GPU: 1`；如果有 `accelerator_type`，也会加入对应资源。然后它调用 `placement_group(..., strategy=strategy, lifetime=lifetime)` 并 `ray.get(pg.ready())`。([verl](https://verl.readthedocs.io/en/latest/_modules/verl/single_controller/ray/base.html "verl.single_controller.ray.base — verl  documentation"))

简化理解：

```text
ResourcePool(process_on_nodes=[4, 4])
    表示：希望 node 0 有 4 个 worker，node 1 有 4 个 worker。

RayResourcePool
    -> 创建 placement groups
    -> 每个 worker 对应某个 bundle
    -> RayWorkerGroup 根据 bundle 创建 Ray actors
```

`RayWorkerGroup._init_with_resource_pool()` 里会选择 `PACK` 或 `STRICT_PACK`，然后调用 `resource_pool.get_placement_groups(...)`，再循环创建 worker。([verl](https://verl.readthedocs.io/en/latest/_modules/verl/single_controller/ray/base.html "verl.single_controller.ray.base — verl  documentation"))

### 5.5 colocate actor / rollout / reference / critic

veRL 里的 colocate 不是抽象概念，而是资源放置策略。常见配置会把 actor、rollout、reference 放在同一组资源上，减少权重同步、log prob 计算、rollout engine 调用的跨节点成本。也可能把 critic 或 reward model 放到独立资源池，避免和 rollout 抢显存。

你要区分两层 colocate：

```text
Ray 层 colocate：
    actor 进程是否放在同一个 node / GPU bundle 附近。

模型层 colocate：
    actor model、reference model、rollout engine 是否复用同一份权重、
    是否共享 GPU、是否切换 engine mode。
```

Ray 只能决定进程放置；具体模型权重如何共享、显存如何复用，是 veRL / vLLM / FSDP / Megatron 层的事情。

### 5.6 多机多卡拓扑

假设 2 台机器，每台 8 张 GPU：

```text
node0: GPU 0 1 2 3 4 5 6 7
node1: GPU 0 1 2 3 4 5 6 7
```

如果你要一个 16-worker 的 data parallel worker group，可以是：

```python
RayResourcePool(process_on_nodes=[8, 8])
```

如果每个 TP group 需要 8 张卡，最好让一个 TP group 在单节点内：

```text
TP group 0: node0 的 8 张 GPU
TP group 1: node1 的 8 张 GPU
```

如果 TP 跨节点，NCCL 通信压力会显著上升。Ray 可以把 actors 放到节点上，但 TP/PP group 的具体通信组织仍由训练/推理框架负责。

### 5.7 placement group 生命周期与泄漏

Placement group 默认和创建者 fate-share；如果创建者死了，placement group 会被删除。`lifetime="detached"` 的 placement group 会作为全局对象独立存在。([Ray](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.placement_group.html "ray.util.placement_group — Ray 2.55.1"))

排查泄漏：

```bash
ray list placement-groups
ray list placement-groups --detail
```

如果看到很多历史 placement group 仍然 `CREATED` 或 `PENDING`，就要检查是否有 detached PG 没释放、driver 异常退出、或资源池反复创建但没有清理。

---

## 6. veRL 的 Single Controller 与 WorkerGroup RPC

### 6.1 Single Controller 为什么存在

veRL 的 single_controller 设计目标是：**让分布式多进程调用看起来像单进程方法调用**。官方设计文档说，single_controller 起源于把单进程 RLHF toy script 改造成分布式系统，同时尽量保持可调试性。它把训练循环拆成 `generate_sequences`、`compute_advantages` 等阶段，并用 Ray 作为初始 backend，因为 Ray 可以把 Python class methods 暴露成 RPC endpoints。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

普通单进程写法：

```python
rollout.generate_sequences(batch)
```

veRL 多进程写法仍然类似：

```python
rollout_wg.generate_sequences(batch)
```

但背后发生的是：

```text
controller
    -> dispatch_fn 拆输入
    -> execute_fn 调多个 Ray actor
    -> collect_fn 聚合输出
```

veRL 文档明确描述了这个调用链。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

### 6.2 RayWorkerGroup

`RayWorkerGroup` 是 veRL 对一组 Ray actor workers 的封装。源码说明它扩展了 `WorkerGroup`，用于创建和管理具有资源需求和调度策略的一组 Ray actors。([verl](https://verl.readthedocs.io/en/latest/_modules/verl/single_controller/ray/base.html "verl.single_controller.ray.base — verl  documentation"))

可以把它理解为：

```text
RayWorkerGroup = 一组 Ray ActorHandle + 分发/聚合逻辑 + 方法绑定逻辑
```

当你调用：

```python
rollout_wg.generate_sequences(prompts)
```

实际执行链大致是：

```text
1. generate_sequences 是被 @register 标记过的方法
2. RayWorkerGroup 初始化时把这个方法绑定到自己身上
3. 调用时根据 dispatch_mode 选择 dispatch_fn
4. execute_fn 调用每个 worker actor 的 generate_sequences.remote(...)
5. collect_fn 收集多个 worker 的返回值
```

### 6.3 RayClassWithInitArgs

`RayClassWithInitArgs` 是 veRL 用来延迟创建 Ray actor 的包装类。源码里它保存 class、args、kwargs，并在真正创建 worker 时结合 placement group、bundle index、GPU 数量、device name 等 Ray options 调用 `.remote()`。([verl](https://verl.readthedocs.io/en/latest/_modules/verl/single_controller/ray/base.html "verl.single_controller.ray.base — verl  documentation"))

简化理解：

```python
ray_cls = RayClassWithInitArgs(
    cls=ray.remote(MyWorker),
    config=config,
    role="actor"
)
```

这不是立即创建 worker，而是把“如何创建 worker”保存起来，等 `RayWorkerGroup` 根据 `ResourcePool` 和 placement group 真正实例化。

### 6.4 `@register`

`@register` 是 veRL single_controller 的关键装饰器。它会给方法附加 metadata，例如 `dispatch_mode`、`execute_mode`、`blocking`。初始化 `RayWorkerGroup` 时，`_bind_worker_method` 会扫描 worker class 上被 `@register` 标记的方法，并动态绑定到 `WorkerGroup` 接口上。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

例子：

```python
from verl.single_controller.base.decorator import register, Dispatch

class ActorRolloutRefWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        ...
```

这意味着 `generate_sequences` 不是普通本地方法，而是可以通过 `RayWorkerGroup.generate_sequences(...)` 被 controller 统一调用的远程分布式方法。

### 6.5 `dispatch_fn` / `collect_fn`

`dispatch_fn` 负责把 controller 输入拆成每个 worker 的输入。`collect_fn` 负责把多个 worker 的返回结果合并回 controller 期望的结果。veRL 文档对比了 `ONE_TO_ALL` 和 `DP_COMPUTE_PROTO`：`ONE_TO_ALL` 会把输入复制给 N 个 worker；`DP_COMPUTE_PROTO` 会用 `DataProto.chunk` 把大 DataProto 拆成 N 份。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

粗略对比：

```text
Dispatch.ONE_TO_ALL:
    同一个参数广播给所有 worker。
    适合 init、load checkpoint、set mode 等。

Dispatch.DP_COMPUTE_PROTO:
    把 DataProto 按 data parallel 维度切分。
    适合 generate_sequences、compute_log_prob、reward、advantage 等 batch 计算。
```

### 6.6 `execute_mode`

`execute_mode` 决定方法调用哪些 worker。常见模式：

```text
Execute.ALL:
    所有 worker 都执行。

Execute.RANK_ZERO:
    只在 rank 0 worker 上执行。
```

例如保存 checkpoint、打印状态、获取某些全局 metadata 时，可能只需要 rank zero 执行；而 forward / rollout / update 通常需要所有 worker 执行。

### 6.7 `generate_sequences` 的完整调用链

以 rollout 为例：

```text
controller 调用:
    rollout_wg.generate_sequences(prompts)

RayWorkerGroup:
    读取 generate_sequences 上的 MAGIC_ATTR
    dispatch_mode = DP_COMPUTE_PROTO
    dispatch_fn = dispatch_dp_compute_data_proto
    collect_fn = collect_dp_compute_data_proto
    execute_fn = execute_all

dispatch_fn:
    prompts.chunk(world_size)
    得到每个 worker 的 prompts shard

execute_fn:
    worker_0.generate_sequences.remote(shard_0)
    worker_1.generate_sequences.remote(shard_1)
    ...
    worker_n.generate_sequences.remote(shard_n)

Ray:
    调度 actor method
    返回 ObjectRef 列表

collect_fn:
    ray.get / gather outputs
    merge 成新的 DataProto
```

这就是 veRL single_controller 的核心：**controller 写同步算法逻辑，RayWorkerGroup 负责远程并行执行。**

---

## 7. Ray 与 DataProto / TransferQueue

### 7.1 经典 DataProto 路线

经典路线是：

```text
DataProto
    -> dispatch_fn split
    -> Ray actor method
    -> worker compute
    -> ObjectRef result
    -> collect_fn merge
    -> new DataProto
```

`DataProto` 提供统一数据协议，方便不同 worker method 之间传递 batch、non-tensor data 和 meta info。官方 API 文档说 `DataProto` 是数据交换接口，其 `batch` 是 TensorDict，`meta_info` 是附加信息。([verl](https://verl.readthedocs.io/en/latest/api/data.html?utm_source=chatgpt.com "Data interface — verl documentation - Read the Docs"))

### 7.2 `DataProto.chunk` 与 `DP_COMPUTE_PROTO`

`DP_COMPUTE_PROTO` 适合 data-parallel batch 计算。它的核心思想是把一个大 batch 拆成 `world_size` 份，让每个 worker 处理一份，最后再合并。这和 PyTorch DDP / FSDP 的 SPMD 风格不同：controller 显式组织一次“分发—执行—收集”。

优点：

```text
1. 算法代码可读性强。
2. controller 可以检查中间 DataProto。
3. 很适合 PPO 这种多阶段 DAG。
```

缺点：

```text
1. controller 容易成为数据汇聚点。
2. 大 tensor / 多模态数据会放大 CPU 和 object store 压力。
3. 同步 collect 会形成 barrier。
```

### 7.3 大对象流经 controller 的成本

当 rollout 输出很大时，例如：

```text
responses
attention_mask
position_ids
old_log_probs
values
rewards
advantages
multi-modal tensors
```

如果这些都作为 `DataProto` 在 controller 和 workers 之间反复传递，可能产生：

```text
1. Ray object store memory 压力。
2. Python 序列化 / 反序列化开销。
3. controller Python heap 增长。
4. ray.get 阻塞导致 GPU 等 CPU 数据。
5. object spilling 后吞吐下降。
```

Ray object store 的机制本身没有问题；问题在于大对象频繁走同步控制路径。Ray 文档说明 remote object 缓存在每个节点的 distributed shared-memory object store，`ray.get()` 会在当前节点没有对象时下载对象。([Ray](https://docs.ray.io/en/latest/ray-core/objects.html "Objects — Ray 2.55.1"))

### 7.4 TransferQueue 的动机

TransferQueue 是 veRL 面向后训练工作流的数据系统。官方文档把它描述为高性能数据存储与传输模块，具备全景数据可见性和流式调度能力，用于优化 post-training workflow 的高效数据流。文档还说它提供细粒度、sub-sample-level 的数据管理和负载均衡能力，作为 data gateway 解耦计算任务之间的显式数据依赖。([GitHub](https://github.com/volcengine/verl/blob/main/docs/data/transfer_queue.md "verl/docs/data/transfer_queue.md at main · verl-project/verl · GitHub"))

你可以把它理解为：

```text
旧路线：
    controller 同时管控制流和大 tensor 数据流。

TransferQueue 路线：
    controller 主要管 metadata / 调度 / 控制流；
    tensor 数据尽量通过专门的数据通道流动。
```

### 7.5 metadata control plane vs tensor data plane

这是理解新数据流的关键：

```text
control plane:
    哪个 batch、哪个 sample、哪个 worker、哪个 step、哪个任务状态。
    数据量小，适合 controller 管。

data plane:
    token tensor、image tensor、video tensor、log prob tensor、reward tensor。
    数据量大，应该尽量避免每一步都汇聚到 controller。
```

Ray 本身也有类似边界：它适合做 actor 编排、资源调度、对象引用传递，但超大 tensor 的高频数据搬运不一定应该全部压到 driver/controller 的同步 `ray.get()` 路径上。

### 7.6 多模态 RL 为什么更容易暴露数据流瓶颈

文本 RL 的 token tensor 已经不小；多模态 RL 会叠加 image / video tensor，单个 sample 可能变成 MB 级。这样 `DataProto` 在 worker 间、controller 间反复传输时，CPU memory、object store、序列化、磁盘 spilling 都会更快成为瓶颈。

所以读 veRL 新版本数据系统时，要特别关注：

```text
1. DataProto 是否仍然承载大 tensor？
2. controller 是否 ray.get 了完整 batch？
3. tensor 是否能直接从 producer worker 到 consumer worker？
4. metadata 和 tensor data 是否分离？
5. 是否引入 streaming / micro-batch 级调度？
```

---

## 8. Ray、veRL、vLLM 的关系

### 8.1 veRL：Ray 负责训练流程编排

veRL 使用 Ray 的核心价值是：把 RL 后训练中的多个角色变成可调度、可 RPC 调用、可资源绑定的远程 worker。single_controller 文档明确说，Ray 被选为初始 backend，是因为它可以把 Python class methods 暴露成 RPC endpoints；但 Ray 默认“一次方法调用对应一次 RPC”，而 LLM 训练通常要协调多个进程，所以 veRL 用 `WorkerGroup` 隐藏多 Ray actors 调用。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

可以理解为：

```text
Ray:
    负责 actor 创建、资源调度、placement group、远程方法调用。

veRL:
    负责 PPO/GRPO/DPO 算法流程、worker 角色划分、DataProto 数据协议、
    dispatch/collect、FSDP/Megatron/vLLM/SGLang 集成。
```

### 8.2 vLLM：Ray 负责多节点 runtime / executor

vLLM 官方文档说明，它支持 distributed tensor-parallel 和 pipeline-parallel inference / serving；多节点推理默认 runtime 是 Ray，单节点默认 runtime 是 Python multiprocessing，也可以通过 `distributed_executor_backend` 或命令行参数覆盖。([vLLM](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ "Parallelism and Scaling - vLLM"))

也就是说：

```text
vLLM 使用 Ray：
    多节点 worker 编排、资源可见性、分布式 executor。

vLLM 自己负责：
    KV cache、PagedAttention、TP/PP、batching、scheduler、model execution。
```

### 8.3 vLLM 的 TP / PP 与 Ray 资源调度

vLLM 的 `tensor_parallel_size` 表示模型 tensor parallel 使用多少 GPU；`pipeline_parallel_size` 表示 pipeline stage 数。Ray 的 `num_gpus` / placement group 决定这些 vLLM worker 能不能放到合适的 GPU 上。vLLM 文档给出的例子是 `LLM(..., tensor_parallel_size=4)` 或 `vllm serve ... --tensor-parallel-size 4`。([vLLM](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ "Parallelism and Scaling - vLLM"))

简单说：

```text
Ray 问题：
    worker 起不来、资源 pending、placement group pending、actor died。

vLLM 问题：
    engine 初始化失败、KV cache 不够、TP/PP 配置不合理、NCCL 卡住、吞吐低。

NCCL / torch.distributed 问题：
    多机网络、IB、端口、rank/world_size、CUDA/NCCL 版本、环境变量。
```

vLLM 的 distributed API 文档显示，它的 `GroupCoordinator` 是 PyTorch ProcessGroup wrapper，而 PyTorch ProcessGroup 会绑定到 NCCL、Gloo、MPI 等通信 backend；vLLM 的 StatelessProcessGroup 文档也明确说它只用于 metadata，data-plane communication 需要创建 NCCL 相关对象。([vLLM](https://docs.vllm.ai/en/latest/api/vllm/distributed/ "distributed - vLLM"))

### 8.4 rollout engine、actor worker、reference worker 如何共存

在 veRL 里，rollout 可能调用 vLLM；actor/reference/critic 可能用 FSDP 或 Megatron；reward 可能是模型或规则函数。Ray 统一编排这些 worker，但每个 worker 内部可以是不同 backend。

典型关系：

```text
Ray actor process
    |
    +-- veRL Worker
            |
            +-- FSDP model
            +-- Megatron model
            +-- vLLM rollout engine
            +-- SGLang rollout engine
            +-- reward model / rule-based reward
```

这解释了为什么排查时不能只看 Ray。Ray actor 正常 alive，不代表 vLLM engine 没卡；Ray placement group CREATED，也不代表 NCCL 通信没问题。

### 8.5 Ray 调度问题 vs vLLM 推理性能问题

区分方法：

```text
Ray 层问题：
    actor 没创建、pending、资源 demand 不满足、placement group pending。
    看 ray status / ray list actors / ray list placement-groups。

vLLM 层问题：
    actor 已 alive，但 generate 卡住、吞吐低、KV cache OOM、engine init 报错。
    看 vLLM logs / CUDA OOM / NCCL logs / engine config。

torch.distributed / NCCL 问题：
    多机初始化卡住、allreduce 卡住、rank 不一致、网络不通。
    看 NCCL_DEBUG、端口、IB、容器网络、hostfile、CUDA/NCCL 版本。
```

---

## 9. Observability

### 9.1 Ray Dashboard

Ray Dashboard 可以查看 job、task、actor、placement group、cluster resource、node、worker、GPU assignment、metrics 等。官方文档说明 Jobs view 会显示 Ray Cluster 状态、autoscaling 状态和 resource demands；Task / Actor / Placement Group tables 展示对应实体状态。([Ray](https://docs.ray.io/en/latest/ray-observability/getting-started.html "Ray Dashboard — Ray 2.55.1"))

常见用途：

```text
1. 看 actor 是否 ALIVE。
2. 看 task 是否 pending / running / failed。
3. 看 placement group 是否 CREATED / PENDING。
4. 看 resource demands 缺什么。
5. 看某个 worker 的 pid / node / logs。
```

### 9.2 `ray status`

```bash
ray status
ray status -v
```

重点看：

```text
Node status:
    Healthy / Pending / Recent failures

Resources:
    CPU / GPU / memory / object_store_memory 使用量

Demands:
    当前无法满足的资源需求
```

Ray 官方说明 `ray status` 可以在 head node 上查看节点状态和资源使用，包括 pending nodes、failed nodes、task/actor 请求的 CPU/GPU 等。([Ray](https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html "Monitoring with the CLI or SDK — Ray 2.55.1"))

### 9.3 `ray list actors`

```bash
ray list actors
ray list actors --detail
```

重点看：

```text
STATE:
    ALIVE / DEAD / RESTARTING

CLASS_NAME:
    是否是你期望的 Worker 类。

PID / NODE_ID:
    worker 在哪个节点哪个进程。
```

如果 actor 是 DEAD，再去查对应 pid 的 logs。

### 9.4 `ray list placement-groups`

```bash
ray list placement-groups
ray list placement-groups --detail
```

重点看：

```text
STATE:
    CREATED / PENDING / REMOVED

bundles:
    每个 bundle 要多少 CPU/GPU/custom resource

strategy:
    PACK / SPREAD / STRICT_PACK / STRICT_SPREAD
```

Ray placement group 文档也推荐用 `ray list placement-groups` 和 `ray status` 查看 metadata、scheduling state 和 resource demands。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html "Placement Groups — Ray 2.55.1"))

### 9.5 `ray logs`

```bash
ray logs cluster
ray logs <log-file>
```

也可以从 dashboard 点到具体 actor / task 的日志。Ray State API 文档说明 `ray logs` 可以获取 task、actor 等实体日志。([Ray](https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html "Monitoring with the CLI or SDK — Ray 2.55.1"))

常见日志文件：

```text
raylet.out / raylet.err:
    调度、object spilling、node 级错误。

worker-*.out / worker-*.err:
    Python worker 输出、异常栈、CUDA OOM。

dashboard_agent.log:
    dashboard / state API 相关问题。

gcs_server.out:
    GCS、actor metadata、placement group metadata。
```

### 9.6 object store memory

检查 object store：

```bash
ray memory
ray memory --stats-only
```

重点看：

```text
Plasma memory usage
Spilled bytes
Restored bytes
ObjectRef owners
哪些对象被 pin 住
```

如果 object store 持续上涨，常见原因是：

```text
1. controller 持有大量 ObjectRef。
2. ray.get 后 Python 对象仍被引用。
3. nested ObjectRef 被放进大对象里，生命周期变长。
4. 结果列表 refs 没释放。
5. DataProto 太大、collect 后又复制。
```

### 9.7 配合 `nvidia-smi` / NCCL / vLLM 日志

Ray 只能告诉你 actor 放在哪、逻辑 GPU 分配如何；`nvidia-smi` 告诉你真实 GPU 利用率和显存；NCCL logs 告诉你多机 GPU 通信；vLLM logs 告诉你 engine、KV cache、batching、TP/PP 初始化。

建议排查顺序：

```text
1. Ray actor 是否启动？
2. actor 是否拿到正确 CUDA_VISIBLE_DEVICES？
3. vLLM / FSDP / Megatron 是否初始化成功？
4. NCCL process group 是否建立？
5. GPU 是否有计算利用率？
6. CPU / object store 是否成为瓶颈？
```

---

## 10. 常见问题

### 10.1 worker 一直 pending

优先检查：

```bash
ray status
ray status -v
ray list placement-groups --detail
```

可能原因：

```text
1. bundle 太大，单个节点放不下。
2. STRICT_PACK 要求所有 bundle 在一台机器，实际放不下。
3. custom resource 名字写错。
4. accelerator_type 和节点资源不匹配。
5. Ray 启动时 num-gpus 声明错。
6. head node / worker node 没正常加入集群。
7. placement group 已经占住资源但没有释放。
```

### 10.2 GPU 空着但 Ray 不调度

先记住：Ray 看逻辑资源，不看 `nvidia-smi` 的空闲显存。Ray 文档明确说资源是 logical resources，主要用于 scheduling admission control。([Ray](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html "Resources — Ray 2.55.1"))

检查：

```bash
ray status
ray list nodes
ray list placement-groups --detail
```

确认：

```text
1. Ray 是否识别到了 GPU 资源？
2. actor 是否请求了 GPU？
3. placement group 是否预留了 GPU 但 actor 没用上？
4. 是否使用了 accelerator_type？
5. 是否有 detached placement group 泄漏？
```

### 10.3 object store 爆内存

表现：

```text
ObjectStoreFullError
raylet 日志大量 spilling
训练吞吐突然下降
controller memory 增长
```

处理方向：

```text
1. 减小一次 DataProto 携带的数据量。
2. 避免 controller 持有过多 refs。
3. 尽快释放不需要的 Python 引用。
4. 避免把巨大对象 closure capture 到 remote function。
5. 检查 nested ObjectRef 生命周期。
6. 配置 object spilling 目录到更快、更大的盘。
7. 对多模态大 tensor，考虑 TransferQueue / streaming 数据流。
```

### 10.4 controller 内存越来越高

常见原因：

```text
1. 每个 step 的 DataProto 被日志、metrics、debug list 保存。
2. ray.get 后的结果没有释放。
3. collect_fn 产生新对象，但旧对象仍被引用。
4. validation generation 保存了大量文本或 tensor。
5. Python traceback / exception 持有大对象引用。
```

排查：

```text
1. 看 controller 进程 RSS。
2. 看 ray memory 是否有大量 pinned objects。
3. 暂时关闭详细日志和 sample 保存。
4. 检查训练循环里是否 append 了完整 DataProto。
5. 对大 tensor 使用 del + gc.collect() 做定位，不要当成最终优化方案。
```

### 10.5 actor died

常见原因：

```text
1. CUDA OOM。
2. Python exception。
3. NCCL 初始化失败。
4. 节点被杀。
5. object store / 系统内存 OOM。
6. runtime_env / package 缺失。
7. worker 进程启动后环境变量不对。
```

排查：

```bash
ray list actors --detail
ray logs cluster
```

然后根据 actor 的 pid / node 找 worker 日志。

### 10.6 placement group 泄漏

表现：

```text
ray status 显示资源被 reserved。
ray list placement-groups 里有很多历史 PG。
新任务 pending，但 nvidia-smi 似乎空闲。
```

处理：

```text
1. 检查是否 lifetime="detached"。
2. 检查 driver 是否异常退出。
3. 检查 veRL 是否重复创建 ResourcePool。
4. 手动清理无用 PG。
5. 重启 Ray cluster 是最后手段。
```

### 10.7 vLLM rollout 卡住但 Ray actor 正常

说明 Ray 只完成了 actor 编排，卡点可能在 vLLM engine 内部。检查：

```text
1. vLLM engine 初始化日志。
2. tensor_parallel_size / pipeline_parallel_size。
3. NCCL_DEBUG=INFO。
4. CUDA_VISIBLE_DEVICES。
5. GPU 利用率是否为 0 或某张卡 100% 卡死。
6. 多节点网络、端口、IB、容器 hostname。
7. KV cache 是否 OOM。
```

### 10.8 多机 NCCL 报错但 Ray 状态正常

这很常见。Ray actor 可以正常 alive，但 NCCL allreduce / broadcast / process group init 失败。vLLM distributed 文档显示其通信层会使用 PyTorch ProcessGroup / NCCL / Gloo 等 backend，因此 Ray 状态正常不代表模型通信正常。([vLLM](https://docs.vllm.ai/en/latest/api/vllm/distributed/ "distributed - vLLM"))

检查：

```text
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME
NCCL_IB_DISABLE
NCCL_IB_HCA
CUDA_VISIBLE_DEVICES
torch.cuda.device_count()
hostname / /etc/hosts
容器网络模式
防火墙和端口
```

---

## 11. 源码阅读清单

### 11.1 Ray 官方最小代码

先写三个最小例子：

```text
1. remote task
2. actor
3. placement group + actor
```

目标不是学完 Ray，而是建立 `.remote()`、`ObjectRef`、`ray.get()`、`num_gpus`、`placement_group` 的直觉。

### 11.2 veRL single_controller 设计文档

优先读官方设计文档的这几段：

```text
Origin
A Running Example: generate_sequences
Step 1: Register with a Decorator
Step 2: Binding During Initialization
ONE_TO_ALL vs DP_COMPUTE_PROTO
Step 3: Call Chain
```

这份文档基本就是理解 `RayWorkerGroup` 的路线图。([verl](https://verl.readthedocs.io/en/latest/single_controller.html "The Design of verl.single_controller — verl  documentation"))

### 11.3 `verl/single_controller/base/decorator.py`

重点看：

```text
Dispatch
Execute
register
DISPATCH_MODE_FN_REGISTRY
get_predefined_dispatch_fn
get_predefined_execute_fn
```

理解 `@register` 如何把普通 worker method 变成可被 `WorkerGroup` 统一调用的分布式方法。

### 11.4 `verl/single_controller/base/worker_group.py`

重点看：

```text
ResourcePool
WorkerGroup
_bind_worker_method
dispatch_fn / collect_fn 绑定
```

源码里 `ResourcePool` 管理跨节点资源和 world size / local rank 信息；`WorkerGroup._bind_worker_method` 会扫描被 `@register` 标记的方法，并绑定到 WorkerGroup 实例。([verl](https://verl.readthedocs.io/en/latest/_modules/verl/single_controller/base/worker_group.html "verl.single_controller.base.worker_group — verl  documentation"))

### 11.5 `verl/single_controller/ray/base.py`

重点看：

```text
RayResourcePool.get_placement_groups
RayClassWithInitArgs
RayWorkerGroup.__init__
RayWorkerGroup._init_with_resource_pool
RayWorkerGroup._create_worker
RayWorkerGroup.execute_all_async
```

`get_placement_groups()` 是资源池到 Ray placement group 的桥；`_init_with_resource_pool()` 是 placement group 到 worker actors 的桥；`execute_all_async()` 是 WorkerGroup 方法到多个 actor RPC 的桥。([verl](https://verl.readthedocs.io/en/latest/_modules/verl/single_controller/ray/base.html "verl.single_controller.ray.base — verl  documentation"))

### 11.6 `verl/protocol.py`

重点看：

```text
DataProto
DataProtoItem
__getstate__
__setstate__
chunk / concat / select
non_tensor_batch
meta_info
```

理解 `DataProto` 如何被序列化、切分、合并，对排查 controller memory 和 object store 很重要。([GitHub](https://github.com/volcengine/verl/blob/main/verl/protocol.py "verl/verl/protocol.py at main · verl-project/verl · GitHub"))

### 11.7 vLLM distributed executor / parallelism 文档

重点看：

```text
tensor_parallel_size
pipeline_parallel_size
distributed_executor_backend
Ray vs multiprocessing
vllm.distributed
GroupCoordinator
StatelessProcessGroup
```

vLLM 文档说明多节点推理默认使用 Ray runtime；vLLM distributed API 里还可以看到 PyTorch ProcessGroup / NCCL / Gloo 等通信 backend 的角色。([vLLM](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ "Parallelism and Scaling - vLLM"))

---

## 12. 理解检查表

读完这份笔记后，建议用下面的问题自测。

```text
Ray Core:
[ ] 我能解释 task 和 actor 的区别。
[ ] 我知道 .remote() 返回的是 ObjectRef，不是函数值。
[ ] 我知道 ray.get() 可能阻塞，并可能触发跨节点对象传输。
[ ] 我知道 object store memory 不是 GPU 显存。
[ ] 我知道 top-level ObjectRef 和 nested ObjectRef 的区别。

资源调度:
[ ] 我知道 num_gpus=1 是逻辑资源，不是显存限制。
[ ] 我知道 Ray 会设置 CUDA_VISIBLE_DEVICES。
[ ] 我知道 custom resource 和 accelerator_type 的用途。
[ ] 我知道 GPU 空闲但 Ray pending 时要看 ray status，而不是只看 nvidia-smi。

Placement Group:
[ ] 我知道 bundle 必须能放进单个节点。
[ ] 我知道 PACK / SPREAD / STRICT_PACK / STRICT_SPREAD 的区别。
[ ] 我知道 placement group 是 gang scheduling。
[ ] 我知道 veRL ResourcePool 会创建 placement groups。
[ ] 我知道 placement group 泄漏会导致资源看似空闲但 Ray 不调度。

veRL:
[ ] 我知道 single_controller 为什么存在。
[ ] 我知道 RayWorkerGroup 是一组 Ray actors 的封装。
[ ] 我知道 @register 只是给方法附加 dispatch / execute metadata。
[ ] 我知道 dispatch_fn 拆输入，execute_fn 远程执行，collect_fn 合并结果。
[ ] 我能讲清 generate_sequences 从 controller 到 worker 再回到 controller 的路径。

DataProto / TransferQueue:
[ ] 我知道 DataProto 里 batch 通常是 TensorDict。
[ ] 我知道大 DataProto 反复经过 controller 会造成瓶颈。
[ ] 我知道 TransferQueue 的核心动机是解耦控制流和大 tensor 数据流。
[ ] 我能区分 metadata control plane 和 tensor data plane。

vLLM:
[ ] 我知道 Ray 是 vLLM 多节点 runtime 的一部分。
[ ] 我知道 TP/PP 是 vLLM 的模型并行策略，不是 Ray 自己的通信算法。
[ ] 我知道 Ray actor 正常 alive 不代表 NCCL / vLLM engine 一定正常。
```

---

## 13. 参考资料

本笔记主要依据 Ray、veRL、vLLM 官方文档和源码说明整理。建议优先读这些：

```text
Ray:
- Ray Core Key Concepts
- Ray Tasks
- Ray Actors
- Ray Objects
- Ray Resources
- Ray Accelerator Support
- Ray Placement Groups
- Ray Dashboard / State API / CLI

veRL:
- The Design of verl.single_controller
- Single Controller API
- RayWorkerGroup / RayResourcePool 源码
- DataProto API / protocol.py
- TransferQueue Data System

vLLM:
- Parallelism and Scaling
- vllm.distributed
- vllm.distributed.utils
```

核心学习路线可以压缩成一句话：**先把 Ray 理解成“资源调度 + Actor RPC + ObjectRef 数据引用”，再去看 veRL 如何用 RayWorkerGroup 把多 worker 训练过程伪装成单 controller 方法调用，最后区分 Ray 编排层和 vLLM / FSDP / Megatron / NCCL 的模型计算通信层。**