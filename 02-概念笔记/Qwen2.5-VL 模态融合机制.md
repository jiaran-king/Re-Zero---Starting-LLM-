---
tags:
  - 概念笔记
  - 多模态
  - VLM
  - Qwen
  - Transformer
aliases:
  - Qwen2.5-VL 融合机制
  - Qwen2.5-VL visual token injection
updated: '2026-05-10'
---
# Qwen2.5-VL 模态融合机制

这篇笔记着重关注这一点：**Qwen2.5-VL 如何把图像 / 视频变成 LLM 可以处理的 token，并让这些视觉 token 与文本 token 在同一个 decoder 中交互。**

先记住一句话：

> Qwen2.5-VL 的核心是 **visual token injection + decoder self-attention fusion**：视觉输入先被 Vision Encoder 编成 visual embeddings，再替换文本序列中的视觉 placeholder，最后和文本 embedding 一起进入 Qwen2.5 decoder，由 decoder self-attention 完成跨模态融合。

换句话说：**视觉 token 不是被放在 LLM 外面供 cross-attention 读取，而是被直接放进 LLM 的上下文序列里。**

---

## 0. 学习路线

阅读这篇笔记时，按下面顺序理解：

1. **先看主链路**：图片 / 视频如何变成 visual embeddings，并被注入文本序列。
2. **再区分三个动作**：视觉编码、placeholder 替换、decoder 融合。
3. **再理解两套位置编码**：Vision Encoder 内部 RoPE 与 LLM Decoder 内部 MRoPE。
4. **最后看输出**：OCR、JSON、bbox、视频时间戳本质上都是条件文本生成。

---

## 1. 总览图

下面这张 Canvas 是按学习顺序整理的主流程：

![qwen25_vl_fusion](../08-%E5%9B%BE%E7%89%87/canvas-preview/qwen25_vl_fusion.svg)

[打开原始 Canvas](../08-%E5%9B%BE%E7%89%87/qwen25_vl_fusion.canvas)

官方结构图如下，重点看三处：

![Pasted image 20260510192333](../08-%E5%9B%BE%E7%89%87/Pasted%20image%2020260510192333.png)

这张图可以这样读：

- 下方的图片和视频先进入 **Vision Encoder**。
- 不同尺寸图片会产生不同数量的视觉 token，例如图中 Picture 1、Picture 2、Picture 3 的 token 数量差异很大。
- 视频部分引入 dynamic FPS sampling，并把采样后的时间位置和真实时间对齐。
- 中间的 visual tokens 被插入到文本 token 序列中。
- 上方的 **Qwen2.5 LM Decoder** 接收的是一个混合序列：`文本 token + 图像 token + 视频 token`。

所以这张图最重要的信息不是“有一个视觉编码器”，而是：

```text
Vision Encoder 只负责视觉内部编码；
真正的图文 / 文视融合发生在 Qwen2.5 LM Decoder 的 self-attention 中。
```

---

## 2. 最短主链路

先把完整流程压缩成一条链：

```text
图片 / 视频 + 文本问题
  ↓
Processor
  - 文本 → input_ids
  - 图像 / 视频 → pixel_values
  - 生成 image_grid_thw / video_grid_thw
  - 展开 image/video placeholder
  ↓
Vision Encoder
  - patch_embed
  - window attention / full attention
  - merger
  ↓
visual embeddings: [N_visual_tokens, D]
  ↓
placeholder replacement
  - 用 visual embeddings 替换 input_ids 中的 image/video placeholder 位置
  ↓
混合 inputs_embeds: [text_embeds + visual_embeds]
  ↓
MRoPE
  - 文本：1D 顺序
  - 图像：height / width
  - 视频：time / height / width
  ↓
Qwen2.5 Decoder self-attention
  ↓
lm_head
  ↓
文本答案 / OCR / JSON / bbox / point / 视频时间戳
```

这条链路里有三个关键分界点：

| 分界点 | 发生了什么 | 是否已经跨模态融合 |
|---|---|---|
| Vision Encoder 输出 visual embeddings | 视觉信息被编码到 LLM hidden size | 还没有 |
| placeholder replacement | visual embeddings 被放进文本序列 | 只是完成注入 |
| Decoder self-attention | 文本 token 和视觉 token 互相 attend | 是，真正融合 |

---

## 3. Processor：先把输入对齐成模型协议

Processor 的任务不是“理解图像”，而是把用户输入整理成模型 forward 可以消费的格式。

输入通常是：

```text
图片 / 视频 + 用户文本
```

输出通常包括：

```text
input_ids
attention_mask
pixel_values
pixel_values_videos
image_grid_thw
video_grid_thw
second_per_grid_ts
```

其中：

| 字段 | 含义 | 学习时怎么记 |
|---|---|---|
| `input_ids` | 文本 token ids，也包含 image/video placeholder token | LLM 的文本骨架 |
| `pixel_values` | 图像预处理后的视觉张量 | 给 Vision Encoder 的图像输入 |
| `pixel_values_videos` | 视频预处理后的视觉张量 | 给 Vision Encoder 的视频输入 |
| `image_grid_thw` | 图像视觉 token 的 temporal / height / width 网格 | 视觉 token 的空间结构说明 |
| `video_grid_thw` | 视频视觉 token 的 temporal / height / width 网格 | 视频 token 的时空结构说明 |
| `second_per_grid_ts` | 视频时间维度上每个 grid 对应的秒数 | 用于把视频位置和真实时间对齐 |

如果用户输入里有：

```text
<|vision_start|><|image_pad|><|vision_end|>
请描述这张图片
```

processor 会把 `<|image_pad|>` 或 `<|video_pad|>` 扩展成多个视觉 placeholder。placeholder 数量必须和后面 Vision Encoder 输出的 visual token 数量一致，否则后面无法正确替换。

这一步的核心作用是：

```text
提前在文本序列中挖好视觉 token 的位置。
```

参考：Hugging Face 文档中说明 Qwen2.5-VL processor 会返回 `pixel_values`、`image_grid_thw`、`video_grid_thw` 等字段，并支持图像 / 视频输入。([Hugging Face](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl "Qwen2.5-VL · Hugging Face"))

---

## 4. 文本路径：先得到一个带坑位的 LLM 输入骨架

文本路径可以单独理解：

```text
input_ids: [B, S]
  ↓ embed_tokens
inputs_embeds: [B, S, D]
```

其中：

- `B` 是 batch size。
- `S` 是序列长度。
- `D` 是 Qwen2.5 decoder 的 hidden size。

注意：这个阶段的 `input_ids` 里既有普通文本 token，也有 image/video placeholder token。

所以刚做完 embedding lookup 时，序列长这样：

```text
[text_embed,
 image_placeholder_embed,
 image_placeholder_embed,
 ...,
 text_embed]
```

此时 placeholder 位置还没有真实视觉信息，只是普通 token embedding。可以把它理解成“待替换的坑位”。

源码层面，Qwen2.5-VL text model 在没有传入 `inputs_embeds` 时，会通过 `embed_tokens(input_ids)` 得到输入 embedding。([GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py "modeling_qwen2_5_vl.py"))

---

## 5. 视觉路径：图像 / 视频如何变成 visual embeddings

视觉路径可以单独理解：

```text
pixel_values / pixel_values_videos
  ↓
patch_embed
  ↓
Vision Encoder RoPE
  ↓
window attention / full attention blocks
  ↓
merger
  ↓
image_embeds / video_embeds: [N_visual_tokens, D]
```

这里最重要的是两个点。

第一，Qwen2.5-VL 使用 native dynamic resolution。不同尺寸的图片不会被简单压成固定 token 数，而是根据分辨率产生不同长度的视觉 token。官方图里给出的例子很直观：

| 输入 | 图中示例 token 数 | 含义 |
|---|---:|---|
| Picture 1 | 11427 | 高分辨率长图，视觉 token 很多 |
| Picture 2 | 8 | 小图或信息量较少，视觉 token 很少 |
| Picture 3 | 1125 | 中等规模图像 |
| Video 1 | 644 / 1288 / 2576 | 视频 token 数随采样设置变化 |

第二，Vision Encoder 的输出维度已经对齐到 LLM hidden size `D`。所以输出的视觉 embedding 可以直接放入 LLM 的 `inputs_embeds` 序列。

```text
image_embeds: [N_img_tokens, D]
video_embeds: [N_video_tokens, D]
```

这一步完成的是：

```text
视觉模态内部编码 + 视觉特征投影到语言模型空间。
```

它还不是最终的图文融合，因为文本 token 还没有和视觉 token 在同一个 attention 空间里交互。

参考：Qwen 官方博客提到 Qwen2.5-VL 使用 dynamic resolution、window attention、dynamic FPS training 和 absolute time encoding。([Qwen](https://qwenlm.github.io/blog/qwen2.5-vl/ "Qwen2.5-VL"))

---

## 6. Placeholder Replacement：视觉信息注入文本序列

这是理解 Qwen2.5-VL 的关键动作。

输入：

```text
inputs_embeds: [B, S, D]
image_embeds: [N_img_tokens, D]
video_embeds: [N_video_tokens, D]
```

模型会找到 `input_ids` 中 image/video placeholder 对应的位置，然后把这些位置原来的 placeholder embedding 替换成真实视觉 embedding。

替换前：

```text
[text_embed,
 image_placeholder_embed,
 image_placeholder_embed,
 text_embed]
```

替换后：

```text
[text_embed,
 image_visual_embed_1,
 image_visual_embed_2,
 text_embed]
```

可以把这一步理解成：

```text
文本序列中的坑位没有移动，
只是坑位里的内容从 placeholder embedding 变成了 visual embedding。
```

所以替换后：

```text
融合前后序列长度 S 不变；
hidden size D 不变；
placeholder 位置的内容变了。
```

这一步是视觉信息进入 LLM 上下文的入口，但严格说它只是 **injection**，不是最终的 **fusion**。真正的融合要等下一步 decoder self-attention。

---

## 7. MRoPE：让混合序列带上正确的位置感

placeholder 替换完成后，模型得到一个混合序列：

```text
[text tokens + image tokens + video tokens]
```

如果只用普通 1D RoPE，模型只能知道“第几个 token”，但不能自然表达图像 token 的二维位置、视频 token 的时间位置。因此 Qwen2.5-VL 使用 MRoPE，也就是 multimodal rotary position embedding。

可以这样记：

| token 类型 | 位置维度 | 直觉 |
|---|---|---|
| 文本 token | temporal = height = width | 退化成普通 1D RoPE |
| 图像 token | height / width，temporal 基本固定 | 知道图像中的行列位置 |
| 视频 token | temporal / height / width | 知道第几段时间、帧内什么位置 |

源码注释中将 MRoPE 解释为 1D RoPE 的多模态 3D 扩展：视觉 embedding 会分别使用 temporal、height、width 三个维度的位置；文本 embedding 的三个位置索引相同，因此等价于普通 1D RoPE。([GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py "modeling_qwen2_5_vl.py"))

这一步的作用是：

```text
让 LLM Decoder 在同一个序列里同时理解：
- 文本 token 的先后顺序；
- 图像 token 的二维空间位置；
- 视频 token 的时间位置和空间位置。
```

### 7.1 MRoPE position_ids 极小例子

下面用一个极小例子说明 MRoPE 的位置内容。注意：这里说的“RoPE 内容”，更准确地说是 **MRoPE 的 `position_ids` 内容**。真正的 RoPE 会进一步把这些 position id 查成 sin / cos，然后作用到 attention 的 Q / K 上。

假设输入是：

```text
文本A：5 个 token
图片：1 张，视觉网格是 T=1, H=2, W=3
文本B：3 个 token
```

整体输入可以抽象成：

```text
[text_0, text_1, text_2, text_3, text_4,
 image_00, image_01, image_02, image_10, image_11, image_12,
 text_5, text_6, text_7]
```

其中图片的网格是：

```text
image_00  image_01  image_02
image_10  image_11  image_12
```

也就是：

```text
T = 1
H = 2
W = 3
```

先看第一段文本A。文本是 1D 的，所以它的位置就是普通顺序：

```text
text_0 -> 0
text_1 -> 1
text_2 -> 2
text_3 -> 3
text_4 -> 4
```

但 MRoPE 统一要求三个维度：`temporal / height / width`。所以文本 token 会把同一个 1D 位置复制到三个维度：

```text
text_0 -> [0, 0, 0]
text_1 -> [1, 1, 1]
text_2 -> [2, 2, 2]
text_3 -> [3, 3, 3]
text_4 -> [4, 4, 4]
```

如果忽略 batch 维度，这段文本的 `position_ids` 可以写成：

```python
# shape: [3, 5]
text_A_position_ids = [
    [0, 1, 2, 3, 4],   # temporal
    [0, 1, 2, 3, 4],   # height
    [0, 1, 2, 3, 4],   # width
]
```

这就是为什么说：文本在 MRoPE 里退化成普通 1D RoPE。三行完全一样。

接着看图片。文本A长度是 5，所以下一个块，也就是图片块的起始 offset 是：

```text
offset = 5
```

图片的局部网格是：

```text
image_00 -> [t=0, h=0, w=0]
image_01 -> [t=0, h=0, w=1]
image_02 -> [t=0, h=0, w=2]

image_10 -> [t=0, h=1, w=0]
image_11 -> [t=0, h=1, w=1]
image_12 -> [t=0, h=1, w=2]
```

加上 offset 后：

```text
image_00 -> [5, 5, 5]
image_01 -> [5, 5, 6]
image_02 -> [5, 5, 7]

image_10 -> [5, 6, 5]
image_11 -> [5, 6, 6]
image_12 -> [5, 6, 7]
```

所以这张图片的 `position_ids` 是：

```python
# token 顺序:
# image_00, image_01, image_02, image_10, image_11, image_12

# shape: [3, 6]
image_position_ids = [
    [5, 5, 5, 5, 5, 5],   # temporal
    [5, 5, 5, 6, 6, 6],   # height
    [5, 6, 7, 5, 6, 7],   # width
]
```

这就是图像和文本不同的地方。

文本是：

```text
[pos, pos, pos]
```

图像是：

```text
[offset + t, offset + h, offset + w]
```

对于静态图片，`t` 通常固定，所以真正起主要作用的是 `h / w` 两个空间维度。

图片之后，下一段文本B从哪里开始？

这张图片的 MRoPE 最大位置是：

```text
max position = 7
```

所以下一段文本B的起始 offset 是：

```text
next offset = 8
```

文本B有 3 个 token，所以：

```text
text_5 -> [8, 8, 8]
text_6 -> [9, 9, 9]
text_7 -> [10, 10, 10]
```

对应：

```python
# shape: [3, 3]
text_B_position_ids = [
    [8, 9, 10],   # temporal
    [8, 9, 10],   # height
    [8, 9, 10],   # width
]
```

把三段拼起来，完整混合序列的 `position_ids` 就是：

```python
# token 顺序:
# text_0, text_1, text_2, text_3, text_4,
# image_00, image_01, image_02, image_10, image_11, image_12,
# text_5, text_6, text_7

position_ids = [
    # temporal
    [0, 1, 2, 3, 4,   5, 5, 5, 5, 5, 5,   8, 9, 10],

    # height
    [0, 1, 2, 3, 4,   5, 5, 5, 6, 6, 6,   8, 9, 10],

    # width
    [0, 1, 2, 3, 4,   5, 6, 7, 5, 6, 7,   8, 9, 10],
]
```

可以观察到几个关键点：

```text
文本 token 的三行 position 完全一样：
  [pos, pos, pos]

图片 token 的三行 position 不一样：
  temporal 行基本固定
  height 行表达第几行
  width 行表达第几列

图片块不是简单按 token 顺序 5, 6, 7, 8, 9, 10 编号；
而是按照图像自己的 H / W 网格来编号。

但图片块整体又加了 offset = 5，
所以它仍然知道自己出现在文本A之后。
```

因此，MRoPE 的直觉就是：

```text
文本：用 1D 位置
图像：用 2D 空间位置，外加全局 offset
视频：用 3D 时间-空间位置，外加全局 offset
```

这也对应前面的核心描述：文本 token 退化为 1D 顺序，图像 token 使用 height / width，视频 token 使用 temporal / height / width；最终它们都在 decoder 的同一条混合序列中参与 self-attention 融合。

---

##  两套 RoPE：不要混在一起

Qwen2.5-VL 中容易混淆的是：模型里其实有两层位置建模。

| 层级                  | 使用位置               | 作用对象                           | 目的               |
| ------------------- | ------------------ | ------------------------------ | ---------------- |
| Vision Encoder RoPE | 视觉编码器内部            | 图像 / 视频 patch tokens           | 建模视觉 patch 的空间关系 |
| LLM Decoder MRoPE   | Qwen2.5 decoder 内部 | 文本 token + 图像 token + 视频 token | 建模统一多模态序列的位置关系   |

更直观地说：

```text
第一套 RoPE：
  让 Vision Encoder 看懂视觉结构。

第二套 MRoPE：
  让 LLM Decoder 看懂多模态混合序列的位置结构。
```

第一套发生在视觉特征进入 LLM 之前；第二套发生在视觉 token 已经和文本 token 放到同一个序列之后。

一个常见误解是：

> 有了 Vision Encoder 内部 RoPE，为什么还需要 Decoder MRoPE？

原因是两者服务的对象不同：

```text
Vision Encoder RoPE 解决的是“视觉 patch 之间怎么排列”；
Decoder MRoPE 解决的是“视觉 token 和文本 token 放在同一条上下文里时，各自的位置关系是什么”。
```

---

## 8. Decoder Self-Attention：真正的跨模态融合

当混合 embedding 和 MRoPE position ids 都准备好后，它们会一起进入 Qwen2.5 decoder。

输入：

```text
融合后的 inputs_embeds: [B, S, D]
position_ids / MRoPE
attention_mask
```

在每一层 decoder self-attention 中，文本 token 和视觉 token 都会产生 Q/K/V：

```text
Q = XWq
K = XWk
V = XWv
```

由于文本 token 和视觉 token 已经在同一个序列里，后续生成 token 可以 attend 到：

```text
用户问题文本；
图像 visual tokens；
视频 visual tokens；
前面已经生成的文本 token。
```

这就是 Qwen2.5-VL 与传统 cross-attention VLM 的重要区别：

| 方案 | 视觉信息怎么进入语言模型 | 融合位置 |
|---|---|---|
| cross-attention VLM | LLM 通过额外 cross-attention 读取视觉特征 | cross-attention 模块 |
| Qwen2.5-VL 这类方案 | visual tokens 直接进入 LLM 输入序列 | decoder self-attention |

因此，Qwen2.5-VL 的融合核心可以写成：

```text
visual tokens as part of the language model context
```

---

## 9. 输出：为什么 OCR / bbox / JSON 都能统一成生成任务

Decoder 输出 hidden states 后，再经过 `lm_head` 映射到词表 logits，然后自回归生成下一个 token。

所以输出形式可以是：

```text
自然语言回答
OCR 结果
结构化 JSON
bbox 坐标
point 坐标
视频时间戳
```

例如 bbox 可以被生成为类似下面的文本结构：

```json
{
  "bbox_2d": [x1, y1, x2, y2],
  "label": "motorcyclist"
}
```

这里的关键理解是：

```text
模型不是先走一个传统 detection head 再输出 bbox，
而是在视觉条件下通过语言模型头生成结构化文本。
```

OCR、表格抽取、JSON 生成、视频时间戳，本质上也都可以看作：

```text
在视觉 token 条件下进行文本生成。
```

---

## 10. 易混点速查

| 易混点 | 更准确的理解 |
|---|---|
| Vision Encoder 是不是已经完成图文融合？ | 不是。它主要完成视觉内部编码，文本还没有参与。 |
| placeholder replacement 是不是融合？ | 只是视觉 token 注入 LLM 序列，是融合的入口。 |
| 真正的跨模态交互在哪里？ | 在 Qwen2.5 decoder 的 self-attention 里。 |
| 为什么 visual embeddings 可以直接替换 placeholder？ | 因为 Vision Encoder + Merger 已经把视觉特征投影到 LLM hidden size `D`。 |
| 为什么需要 MRoPE？ | 因为混合序列里既有 1D 文本，也有 2D 图像和 3D 视频。 |
| 两套 RoPE 有什么区别？ | Vision Encoder RoPE 处理视觉内部位置；Decoder MRoPE 处理多模态混合序列位置。 |
| bbox 是不是检测头输出？ | 在这里更适合理解为视觉条件下生成的结构化文本。 |

---

## 12. 变量速查

| 符号 / 字段 | 含义 |
|---|---|
| `B` | batch size |
| `S` | LLM 输入序列长度，包括文本 token 和视觉 placeholder token |
| `D` | Qwen2.5 decoder hidden size |
| `N_img_tokens` | 图像经过 Vision Encoder / Merger 后得到的视觉 token 数 |
| `N_video_tokens` | 视频经过 Vision Encoder / Merger 后得到的视觉 token 数 |
| `input_ids` | 文本 token ids，包含 image/video placeholder token |
| `inputs_embeds` | LLM 输入 embedding，形状通常是 `[B, S, D]` |
| `pixel_values` | 图像输入张量 |
| `pixel_values_videos` | 视频输入张量 |
| `image_grid_thw` | 图像 visual tokens 的 temporal / height / width 网格 |
| `video_grid_thw` | 视频 visual tokens 的 temporal / height / width 网格 |
| `second_per_grid_ts` | 视频 grid 与真实秒数之间的对应关系 |
| `MRoPE` | 多模态 RoPE，用于文本、图像、视频混合序列 |

---

## 13. 自测问题

学完以后，应该能回答下面这些问题：

1. Qwen2.5-VL 为什么要在文本序列中放 `<|image_pad|>` / `<|video_pad|>`？
2. `image_grid_thw` 和 `video_grid_thw` 为什么不只是辅助信息，而是位置建模所必需的信息？
3. Vision Encoder 输出的 `image_embeds` 为什么能直接写入 `inputs_embeds`？
4. placeholder replacement 改变了序列长度吗？改变了什么？
5. 为什么说 Vision Encoder RoPE 和 Decoder MRoPE 不是同一件事？
6. Qwen2.5-VL 的图文融合为什么不需要额外 cross-attention 模块？
7. bbox、OCR、JSON 为什么都可以看成条件文本生成？

---

## 14. 最终总结

Qwen2.5-VL 的模态融合机制可以用一句话概括：

**它先用 Vision Encoder 将图像 / 视频编码成视觉特征，再通过 Merger 投影成 LLM hidden size 的 visual embeddings，然后用 placeholder replacement 将 visual embeddings 注入文本序列，形成 text + vision 的统一 embedding 序列，最后在带有 MRoPE 的 Qwen2.5 decoder self-attention 中完成跨模态交互与生成。**

最关键的学习结论是：

```text
视觉 token 不是 LLM 外部的附加信息，
而是 LLM 上下文序列的一部分。

因此，Qwen2.5-VL 的融合核心是：
visual tokens as part of the language model context.
```

---

## 参考来源

- [Hugging Face：Qwen2.5-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl "Qwen2.5-VL · Hugging Face")
- [Qwen 官方博客：Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/ "Qwen2.5 VL! Qwen2.5 VL! Qwen2.5 VL! | Qwen")
- [Hugging Face Transformers 源码：modeling_qwen2_5_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py "modeling_qwen2_5_vl.py")
