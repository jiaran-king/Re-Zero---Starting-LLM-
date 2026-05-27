# index

本文件是 LLM 读取本仓库时的快速索引。它只负责帮助定位关键入口，不替代主题地图和正文笔记。

## LLM 工作规范

- [主页](../%E4%B8%BB%E9%A1%B5.md)：仓库总入口。
- [LLM-WIKI使用指南](../LLM-WIKI%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)：面向日常使用的场景化说明。
- [LLM-WIKI工作流](LLM-WIKI%E5%B7%A5%E4%BD%9C%E6%B5%81.md)：资料处理、知识编译、查询和自检规则。
- [LLM-WIKI处理日志](LLM-WIKI%E5%A4%84%E7%90%86%E6%97%A5%E5%BF%97.md)：记录每次整理动作。
- [LLM-WIKI自检清单](LLM-WIKI%E8%87%AA%E6%A3%80%E6%B8%85%E5%8D%95.md)：图谱健康检查清单。
- `00-首页与索引/LLM工作规范/`：LLM 工作时优先读取的规范目录。

## 笔记规范

- [笔记整理规则](../%E7%AC%94%E8%AE%B0%E8%A7%84%E8%8C%83/%E7%AC%94%E8%AE%B0%E6%95%B4%E7%90%86%E8%A7%84%E5%88%99.md)：Obsidian 笔记风格、图谱分层和双链规则。
- [canvas](../%E7%AC%94%E8%AE%B0%E8%A7%84%E8%8C%83/canvas.md)：Obsidian Canvas 绘制规则。
- [绘图规则](../%E7%AC%94%E8%AE%B0%E8%A7%84%E8%8C%83/%E7%BB%98%E5%9B%BE%E8%A7%84%E5%88%99.md)：图形和流程图相关规范。
- handoff模板：项目交接和长任务恢复模板。
- `00-首页与索引/笔记规范/`：笔记组织、图谱维护、Canvas 和交接相关规则目录。

## 主题地图

- 0：LLM 知识库总图，只连接总览页和主主题域。
- 0a：从表示、预训练、后训练、推理部署到系统工程的整体学习路线。
- [基础架构](../../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84.md)：模型结构与输入表示，包括 Tokenizer、Embedding、Attention、Transformer、位置编码、激活函数等。
- [预训练](../../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E9%A2%84%E8%AE%AD%E7%BB%83.md)：基座能力形成，包括语言建模目标、数据、Scaling Law、优化器和训练闭环。
- [参数高效微调](../../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83.md)：低成本适配方法，包括 LoRA、QLoRA、Adapter、Prefix / Prompt Tuning 等。
- [后训练与对齐](../../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E5%90%8E%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%AF%B9%E9%BD%90.md)：模型行为塑形，包括 SFT、RLHF、PPO、DPO、GRPO、Rollout、Expert Iteration、Reasoning RL。
- [推理优化](../../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md)：训练后高效使用，包括 KV Cache、Flash Attention、MQA/GQA、MLA、稀疏注意力、vLLM、PagedAttention 和服务效率。
- [训练系统工程](../../01-%E4%B8%BB%E9%A2%98%E5%9C%B0%E5%9B%BE/%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F%E5%B7%A5%E7%A8%8B.md)：横切工程透视，包括并行、显存、吞吐、checkpoint、调度、可观测性、veRL、Ray、FSDP。

## 概念笔记入口

- [Transformer](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Transformer.md)
- [Attention](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Attention.md)
- [BPE](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/BPE.md)
- [位置编码](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
- [激活函数](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- [QKV Bias](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/QKV%20Bias.md)
- [KV Cache](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/KV%20Cache.md)
- [Flash Attention](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Flash%20Attention.md)
- [MHA变体：MQA与GQA](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/MHA%E5%8F%98%E4%BD%93%EF%BC%9AMQA%E4%B8%8EGQA.md)
- [MLA](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/MLA.md)
- [稀疏注意力](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/%E7%A8%80%E7%96%8F%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
- [LoRA](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/LoRA.md)
- [RLHF](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/RLHF.md)
- [PPO](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/PPO.md)
- [DPO](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/DPO.md)
- [GRPO](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/GRPO.md)
- [Rollout](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Rollout.md)
- [Expert Iteration](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Expert%20Iteration.md)
- [LLM_后训练 Reward Return Value Advantage](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/LLM_%E5%90%8E%E8%AE%AD%E7%BB%83%20Reward%20Return%20Value%20Advantage.md)
- [DAPO：大规模长链推理强化学习系统](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/DAPO%EF%BC%9A%E5%A4%A7%E8%A7%84%E6%A8%A1%E9%95%BF%E9%93%BE%E6%8E%A8%E7%90%86%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%B3%BB%E7%BB%9F.md)
- [Scaling Law](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Scaling%20Law.md)
- [大模型资源分析](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%B5%84%E6%BA%90%E5%88%86%E6%9E%90.md)
- [Chat-template](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Chat-template.md)
- Qwen2

## 框架与综合资料入口

- [vLLM 学习笔记](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/vllm/VLLM%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)
- [Verl 学习笔记](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Verl/Verl.md)
- [FSDP in veRL](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Verl/FSDP.md)
- [Ray in veRL](../../02-%E6%A6%82%E5%BF%B5%E7%AC%94%E8%AE%B0/Verl/Ray%20in%20veRL%EF%BC%9A%E4%BB%8E%20Ray%20Core%20%E5%88%B0%20RL%20Worker%20%E7%BC%96%E6%8E%92.md)

## 项目

当前文件树中未发现已成体系的项目页。后续如果接入项目资料，优先放入 `03-项目笔记` 或 `04-实验与日志`，并在本节登记项目入口。

## 资料索引

- [资料索引](../../05-%E8%B5%84%E6%96%99%E6%91%98%E5%BD%95/%E8%B5%84%E6%96%99%E7%B4%A2%E5%BC%95.md)：追踪资料处理状态。
- [Qwen2.5 技术报告阅读笔记](../../05-%E8%B5%84%E6%96%99%E6%91%98%E5%BD%95/qwen2.5/Qwen2.5%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0.md)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/v23/21-0998.html)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://openreview.net/forum?id=WE_vluYUL-X)

## 当前待接入资料

当前文件树中未发现待接入的项目或比赛资料目录。后续如果同步这类资料，应先登记到 [资料索引](../../05-%E8%B5%84%E6%96%99%E6%91%98%E5%BD%95/%E8%B5%84%E6%96%99%E7%B4%A2%E5%BC%95.md) 并完成敏感信息检查，再按 [LLM-WIKI工作流](LLM-WIKI%E5%B7%A5%E4%BD%9C%E6%B5%81.md) 做 Wiki 化试点。
