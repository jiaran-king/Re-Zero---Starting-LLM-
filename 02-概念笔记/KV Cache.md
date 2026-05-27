---
type: concept
domain: 推理与部署
status: active
---
# KV Cache

> [!note]
> KV Cache 是 Transformer 推理阶段的一种缓存机制：把历史 token 的 Key 和 Value 存下来复用，避免自回归生成时每一步都把前文重新算一遍。它节省的是"历史前缀被反复编码"的成本，不是"当前 token 依赖长上下文"的成本。

---

## 为什么需要

大语言模型生成文本时采用自回归方式，一次只生成一个 token：

```
生成第 2 个 token → 参考第 1 个
生成第 3 个 token → 参考前 2 个
生成第 N 个 token → 参考前 N-1 个
```

在 self-attention 中，每个 token 生成三组向量：

| 向量 | 角色 |
|:--:|:--|
| Q (Query) | "我想查什么" |
| K (Key) | "我能被匹配到什么" |
| V (Value) | "我携带的实际信息" |

当前 token 用自己的 Q 与历史 token 的 K 做匹配，再对历史 V 加权求和。历史 K/V 会被后面每一个新 token 反复访问，但如果没有缓存，每生成一个新 token 都要把整段历史重新过一遍模型——这就是纯粹的重复计算。KV Cache 把历史 K/V 保存下来直接复用，只为新 token 计算新的 Q/K/V。

---

## 缓存什么 & 怎么工作

KV Cache 只缓存 K 和 V，不缓存 Q。在自回归 decode 里，生成第 $t$ 个 token 时：

1. 得到新位置的 hidden state
2. 投影出 $Q_t, K_t, V_t$
3. 用 $Q_t$ 和历史所有 $K_1, \dots, K_t$ 做 attention
4. 对对应 $V$ 加权求和

当前步骤真正参与查询的只有 $Q_t$ 这一个向量，用完即弃。历史位置的 K 和 V 会被未来每一步反复读取，值得缓存。可以把 decode 理解成一个检索系统：历史部分提供可查询的"数据库"（K/V Cache），当前 token 提供一次性的"检索请求"（$Q_t$）。

每次 decode 只新增一行。第 $t$ 步不重新计算历史 token 的任何向量，只计算新 token 对应的 $Q_t$、$K_t$、$V_t$，然后把 $K_t, V_t$ 追加进 cache，执行：

$$\text{Attn}(Q_t,\ [K_1, \dots, K_t],\ [V_1, \dots, V_t])$$

### Prefill 与 Decode 两个阶段

| | Prefill（预填充） | Decode（解码） |
|---|---|---|
| 做什么 | 一次性处理用户输入的完整 prompt | 逐 token 生成后续内容 |
| 计算特点 | 一次处理大量 token，计算量大但并行度高 | 每步只处理 1 个 token，不断循环 |
| 对 Cache | 建立初始 KV Cache | 持续追加并复用 Cache |

没有 KV Cache，就像每写一句新话都要把前文重新抄一遍再理解；有了 KV Cache，前文的整理结果已保留，只需补充新内容继续往下写。

---

## 加速了什么 & 没加速什么

KV Cache 让历史 token 的 K/V 不再重复计算，推理延迟显著降低。但当前 token 仍需与全部历史 K/V 做 attention 计算，每步代价仍随序列长度增长。

> [!warning] 常见误区
> - "有了 KV Cache 每步生成代价就固定了"——不对，当前 token 仍要和全部历史 K/V 做 attention
> - "KV Cache 也能加速训练"——训练时整段序列并行计算，没有重复编码问题，KV Cache 几乎无收益
> - "KV Cache 消除了长上下文的代价"——长 prompt 的 prefill 计算量、decode 阶段逐步增长的 attention 开销都无法消除

---

## 存储与显存开销

KV Cache 不是只存一份。Transformer 每一层 attention 都会产生各自的 K 和 V，因此每层都要维护独立缓存。单层缓存的张量形状为 `[B, H, T, D]`：

| 符号 | 含义 | 影响 |
|:--:|:--|:--|
| B | Batch Size | 批量越大，缓存越大 |
| H | Attention Head 数 | 头越多，缓存越大 |
| T | 已缓存的历史序列长度 | 序列越长，缓存越大 |
| D | 每个 Head 的维度 | 维度越高，缓存越大 |

总开销为：

$$\text{KV Cache 大小} \propto 2 \times L \times B \times H \times T \times D \times \text{dtype\_size}$$

$L$ 为模型层数，系数 2 来自 K 和 V 两份缓存。每多生成一个 token，每一层都要多存一份 K 和 V，显存占用随上下文长度近似线性增长。大量推理优化工作都围绕 KV Cache 的显存管理展开。

---

## 与 Causal Mask、RoPE 的关系

Causal Mask 保证当前位置只能看见自己和过去，不能看到未来——负责正确性。KV Cache 保存过去位置的 K/V 供后续复用——负责效率。两者职责独立、互不干扰。

KV Cache 与 RoPE 天然兼容：为当前 token 生成 Q/K 后，对它们应用 RoPE，然后将处理后的 K 存入 cache。缓存中的 K 本身已携带位置信息，后续无需对历史 token 重新做 RoPE，只需对新增 token 按当前位置继续处理即可。

---

## 核心逻辑伪代码

```python
cache_k, cache_v = None, None

for step in range(max_new_tokens):
    # 仅为新增 token 计算 Q/K/V
    q, k, v = model.project(hidden_state_of_new_token)

    # 拼接历史缓存
    if cache_k is None:
        all_k, all_v = k, v          # 第一步，无历史
    else:
        all_k = concat(cache_k, k)   # 沿序列维度拼接
        all_v = concat(cache_v, v)

    # 当前 Q 对全部历史 K/V 做 attention
    out = attention(q, all_k, all_v)

    # 更新缓存
    cache_k, cache_v = all_k, all_v

    # 预测下一个 token
    next_token = lm_head(out)
```

真实工程实现中通常不会每次做 `concat`，而是预分配缓存空间，同时还要处理多层 cache、batch 内不同样本长度不一致、beam search 时的 cache 复制与重排等问题，但核心逻辑一致。

---

## 关联
- 属于：[[推理优化]]
- 相关：[[Attention]] [[Transformer]] [[位置编码： RoPE]] [[Flash Attention]] [[Prefix Caching]]
- 用于：[[vLLM]] [[大模型推理实验]] [[推理加速方案]]

## 相关概念
- [[Transformer]]
- [[Attention]]
- [[位置编码： RoPE]]
- [[vLLM]]
- [[推理优化]]
