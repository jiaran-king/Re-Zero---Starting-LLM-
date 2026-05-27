---
type: concept
status: draft
domain: 基础架构
aliases:
  - Byte Pair Encoding
  - BBPE
  - tokenizer
  - control tokens
---

# BPE

> [!note] 摘要
> BPE（Byte Pair Encoding）是大模型 tokenizer 中常见的子词切分方法：它通过反复合并高频相邻符号，把文本压缩成词表中的 token 序列。对 LLM 来说，BPE 不是模型主体，但它决定文本如何变成 token id，从而影响上下文长度、词表规模、罕见词处理、多语言表现和对话格式协议。

## 1. 定义

BPE 最早是一种压缩算法，后来被用于 NLP 的子词分词。它的核心思想是从更细粒度的符号开始，统计相邻符号对的出现频率，每次合并最常见的一对，直到达到目标词表规模。

在大模型里，BPE 通常服务于 tokenizer：

```text
原始文本 -> tokenizer -> token -> token id -> embedding -> Transformer
```

模型真正处理的是 token id 和 embedding，而不是自然语言字符串本身。

## 2. 为什么重要

BPE 位于模型输入链路最前面，影响后续几件事：

- **上下文利用率**：同一句话被切成多少 token，会直接影响 context window 能装下多少内容。
- **词表规模**：词表越大，embedding 和 lm head 参数越多；词表越小，序列可能变长。
- **罕见词和新词处理**：BPE 可以把没见过的词拆成更小的子词，避免纯 word-level tokenizer 的 OOV 问题。
- **多语言和代码表现**：不同语言、符号、空格和代码片段的切分质量，会影响模型学习和生成。
- **对话协议**：chat template、role marker、function call 边界通常依赖特殊 token 或 control tokens。

> [!warning] 常见误区
> Tokenizer 不是“无关紧要的预处理”。同一个模型结构换不同 tokenizer，输入长度、边界信号、特殊符号和训练分布都会变化。

## 3. 核心机制

典型 BPE 训练流程可以压缩成四步：

1. 初始化基础符号表，例如字符、byte 或 unicode 片段。
2. 在语料中统计相邻符号对的频率。
3. 合并最高频的符号对，得到一个新 token。
4. 重复统计和合并，直到词表达到目标大小。

推理或训练时，tokenizer 使用训练好的 merge rules，把输入文本切成词表中的 token 序列。

## 4. BPE、BBPE 与 control tokens

Byte-level BPE（BBPE）把 byte 作为更底层的基础单位。它的好处是覆盖能力强：理论上任意文本都能被表示，不容易遇到无法编码的字符。

**Control tokens（控制符或特殊标记）** 指的是 tokenizer 词表中那些不代表任何实际自然语言文本、专门用于给模型传递结构化信号或指令的保留 token。

普通文本会被 BPE / BBPE 切分成子词；当模型进行指令遵循、多轮对话或特定格式生成时，还需要明确知道上下文边界和逻辑结构。Control tokens 就是用来做这种“物理隔离”和“信号提示”的。

常见用途包括：

| 类型 | 作用 |
|---|---|
| BOS / EOS | 标记序列开始和结束 |
| PAD | batch padding |
| user / assistant / system marker | 区分对话角色 |
| tool / function marker | 标记工具调用或结构化输出边界 |

## 5. 与模型结构的关系

BPE 本身不参与 Transformer 层内计算，但会影响模型看到的序列：

- token id 经过 embedding 查表变成连续向量；
- token 数量决定注意力计算的序列长度；
- 特殊 token 决定模型能否稳定识别对话边界；
- 词表大小影响 embedding 和 lm head 的参数量。

因此，BPE 应归入 [基础架构](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84.md) 下的输入表示层，而不是归入 [预训练](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E9%A2%84%E8%AE%AD%E7%BB%83.md)。预训练使用 tokenizer 产出的 token 序列作为训练对象。

## 6. 关联

- 属于：[基础架构](../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84.md)
- 相关：[Transformer](Transformer.md) [Attention](Attention.md) [位置编码](%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
- 用于：tokenizer 训练、SFT 数据协议、多轮对话模板、结构化输出协议

## 7. 待补充

- [ ] 对比 BPE、WordPiece、Unigram LM 的训练目标差异
- [ ] 补充 GPT-2 byte-level BPE 的空格处理细节
- [ ] 结合 MicroLM tokenizer 训练代码补一个项目侧例子

