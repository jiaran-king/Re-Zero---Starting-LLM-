# index

本文件是 LLM 读取本仓库时的快速索引。它只负责帮助定位关键入口，不替代主题地图和正文笔记。

## LLM 工作规范

- [[主页]]：仓库总入口。
- [[LLM-WIKI使用指南]]：面向日常使用的场景化说明。
- [[LLM-WIKI工作流]]：资料处理、知识编译、查询和自检规则。
- [[LLM-WIKI处理日志]]：记录每次整理动作。
- [[LLM-WIKI自检清单]]：图谱健康检查清单。
- `00-首页与索引/LLM工作规范/`：LLM 工作时优先读取的规范目录。

## 笔记规范

- [[笔记整理规则]]：Obsidian 笔记风格、图谱分层和双链规则。
- [[canvas]]：Obsidian Canvas 绘制规则。
- [[绘图规则]]：图形和流程图相关规范。
- [[handoff模板]]：项目交接和长任务恢复模板。
- `00-首页与索引/笔记规范/`：笔记组织、图谱维护、Canvas 和交接相关规则目录。

## 主题地图

- [[0.LLM学习地图]]：LLM 知识库总图，只连接总览页和主主题域。
- [[0a.LLM全流程总览]]：从表示、预训练、后训练、推理部署到系统工程的整体学习路线。
- [[基础架构]]：模型结构与输入表示，包括 Tokenizer、Embedding、Attention、Transformer、位置编码、激活函数等。
- [[预训练]]：基座能力形成，包括语言建模目标、数据、Scaling Law、优化器和训练闭环。
- [[参数高效微调]]：低成本适配方法，包括 LoRA、QLoRA、Adapter、Prefix / Prompt Tuning 等。
- [[后训练与对齐]]：模型行为塑形，包括 SFT、RLHF、PPO、DPO、GRPO、Rollout、Expert Iteration、Reasoning RL。
- [[推理优化]]：训练后高效使用，包括 KV Cache、Flash Attention、MQA/GQA、MLA、稀疏注意力、vLLM、PagedAttention 和服务效率。
- [[训练系统工程]]：横切工程透视，包括并行、显存、吞吐、checkpoint、调度、可观测性、veRL、Ray、FSDP。

## 概念笔记入口

- [[Transformer]]
- [[Attention]]
- [[BPE]]
- [[位置编码]]
- [[激活函数]]
- [[QKV Bias]]
- [[KV Cache]]
- [[Flash Attention]]
- [[MHA变体：MQA与GQA]]
- [[MLA]]
- [[稀疏注意力]]
- [[LoRA]]
- [[RLHF]]
- [[PPO]]
- [[DPO]]
- [[GRPO]]
- [[Rollout]]
- [[Expert Iteration]]
- [[LLM_后训练 Reward Return Value Advantage]]
- [[DAPO：大规模长链推理强化学习系统]]
- [[Scaling Law]]
- [[大模型资源分析]]
- [[Chat-template]]
- [[Qwen2.5-VL 模态融合机制]]

## 框架与综合资料入口

- [[02-概念笔记/vllm/VLLM学习笔记|vLLM 学习笔记]]
- [[02-概念笔记/Verl/Verl|Verl 学习笔记]]
- [[02-概念笔记/Verl/FSDP|FSDP in veRL]]
- [[02-概念笔记/Verl/Ray in veRL：从 Ray Core 到 RL Worker 编排|Ray in veRL]]

## 项目

当前文件树中未发现已成体系的项目页。后续如果接入项目资料，优先放入 `03-项目笔记` 或 `04-实验与日志`，并在本节登记项目入口。

## 资料索引

- [[资料索引]]：追踪资料处理状态。
- [[05-资料摘录/qwen2.5/Qwen2.5 技术报告阅读笔记|Qwen2.5 技术报告阅读笔记]]
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/v23/21-0998.html)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://openreview.net/forum?id=WE_vluYUL-X)

## 当前待接入资料

当前文件树中未发现待接入的项目或比赛资料目录。后续如果同步这类资料，应先登记到 [[资料索引]] 并完成敏感信息检查，再按 [[LLM-WIKI工作流]] 做 Wiki 化试点。
