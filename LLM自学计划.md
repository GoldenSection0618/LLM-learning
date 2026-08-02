# LLM 自学项目计划

> *从语言模型的基本机制出发，循序完成预训练、监督微调、偏好优化与模型蒸馏，最终进入 Qwen 正式后训练。*

---

## 目录

1. [项目概览](#一项目概览)
2. [阶段执行计划](#二阶段执行计划)
3. [项目里程碑](#三项目里程碑)

---

## 一、项目概览

### 学习目标

本项目用于系统掌握大语言模型从基础训练到正式后训练的完整技术链路。

> **学习主线**
>
> 语言模型基本机制 → 最小训练循环 → MiniMind Pretrain → SFT → LoRA → DPO → Distillation → Qwen 正式后训练与回归评估 → 架构研究

项目执行遵循以下原则：

> 每个阶段只增加一个主要变量。

正式实验统一采用以下数据原则：

> 优先使用论文和主流开源项目已经采用的公开数据集、官方标签、固定 split 与标准评估工具。

- 不安排大规模人工标注、逐条清洗或手工修正模型输出；
- 教学阶段的极小样本直接取自官方数据子集；
- Teacher 生成数据必须由公开 gold answer 或现成 verifier 自动筛选；
- 无法解析、输出截断或重复的生成结果不进入派生数据；可解析但答案错误的候选可保留在原始结果中，并按后续阶段的确定性规则用于 DPO rejected；
- 人工操作限于少量只读抽查，用于发现流水线错误，不修改标签；
- 正式 test 只用于评估，不进入 SFT、DPO、蒸馏数据生成或超参数选择。
- 正式模型在目标任务之外保留固定的通用回归评估，用于检查指令遵循、通用能力与真实性风险信号。

学习者已具备 Python、PyTorch 和机器学习基础，因此本计划不再重复完整的神经网络基础课程；但在进度安排上，仍按缺少 LLM 完整训练链路经验的起点执行。MoE、GRPO、Tool Use、Qwen、QLoRA 和分布式训练等内容不得在项目初期同时引入。

### MiniMind 的定位与毕业条件

MiniMind 在本计划中定位为一次完整的白盒训练项目，用于连接 TinyGPT 的最小实现与 Qwen 的真实后训练生态。它主要承担三项任务：

- 阅读现代 Decoder-only 结构，理解 RoPE、GQA、RMSNorm、gated FFN、KV Cache 等组件；
- 以较低成本跑通 Pretrain、SFT、LoRA、DPO 和 Distillation 的关键数据流与训练信号；
- 观察过拟合、loss mask、KL、输出长度、能力退化和 checkpoint 恢复等训练行为。

MiniMind 实验能够支持对代码正确性、数据流、loss 计算和优化行为的判断。其任务效果容易受到模型容量限制，因此不直接用于证明某种后训练算法在真实 LLM 上更优，也不默认将 MiniMind 的学习率、batch size、LoRA rank、DPO beta 或蒸馏配置迁移至 Qwen。

完成阶段 3～8 的验收，并在阶段 9 固化必要的配置、记录和评估入口后，即视为完成 MiniMind 主线。生成质量不作为延迟进入 Qwen 的理由；模型已经暴露容量上限时，不继续通过增加数据、延长训练或扩大超参数搜索追求可用能力。

此后重新使用 MiniMind 只限于快速复现训练故障。正式任务效果与算法收益由 Qwen 及后续规模验证提供证据；阶段 12 使用公开 GPT-2 checkpoint 与 OLMo 3 官方预设开展架构迁移和受控研究。

### MiniMind 固定评估基线

MiniMind 阶段的客观评估直接采用 MiniMind 官方横评使用的 `lm-evaluation-harness` 七任务套件：`ceval-valid`、`cmmlu`、`arc_easy`、`piqa`、`openbookqa`、`hellaswag` 和 `social_iqa`。评测框架、task 配置及七个底层 Hugging Face Dataset revision 均按[模型与数据来源](./模型与数据来源.md)冻结。Pretrain checkpoint 按 Base 模型口径评估；SFT、LoRA 与 DPO checkpoint 使用同一任务配置并应用 MiniMind 官方 chat template。原生 PyTorch checkpoint 通过 MiniMind 官方 `scripts/convert_model.py` 转为 Transformers 格式后交给现成 `hf` backend，不编写任务、标签、prompt 模板或评分器。

这组结果只用于检查阶段间变化和明显能力退化。MiniMind 容量较小，接近随机水平的分数仍按原样记录，不通过增加训练、修改评测 prompt 或挑选题目追求榜单表现。

生成行为观察固定复用 MiniMind 官方 README 的三条演示 prompts：“为什么天空是蓝色的”“解释什么是机器学习”“推荐一些中国的美食”。三条 prompts 只用于比较 checkpoint、greedy decoding 与 sampling，不进行人工打分。评测工具、任务版本与来源 revision 统一记录在 [模型与数据来源](./模型与数据来源.md) 中。

### 时间与资源

| 项目 | 安排 |
| --- | --- |
| 每周投入 | 35 小时 |
| 每周新内容 / 第三天复习 | 约 27 小时 / 约 8 小时 |
| MiniMind 基础与后训练链路 | 预计 3 周 |
| 进入 Qwen 正式后训练 | 第 4 周后半开始 |
| 完成 Qwen 正式后训练核心项目 | 预计累计 6 周 |
| 前期硬件 | RTX 4060 8GB |
| Qwen 正式后训练阶段硬件 | 计划租用 RTX 5090 32GB |

以上时间按每周 35 小时的有效学习与实践投入估算，不包含数据下载、模型训练等无需持续操作的等待时间。每天完成新内容后，在第三天（学习日记为 D 时的 D+2）安排约为首次学习时间 30% 的复习；复习时间已经计入各阶段总投入和每周 35 小时，不在计划外追加。复习以脱离笔记解释、重画关键数据流和重新运行一个关键实验为主。

阶段代码和实验完成后可以进入下一阶段，对应第三天复习完成后才关闭里程碑。完成阶段 4、阶段 10 和蒸馏专题 D4 后，使用实际主动投入重新校准尚未开始的阶段；实际时间偏离预算超过 20% 时才调整后续时间，不降低验收标准。

阶段 10 可选旁路的 7～10.5 小时不计入第 4～6 周核心工期。机动时间不足时直接延后，未完成该旁路不影响阶段 10 验收或第 7 周蒸馏专题启动。阶段 12 按既定路线执行时，该旁路默认跳过；只有希望单独观察领域适应与通用能力回退时再启动。

前两周以训练正确性为主要目标，不以模型生成质量为进度判断标准。第 3～4 周完成 MiniMind 的 SFT、LoRA、DPO、蒸馏和实验规范整理，并开始 Qwen 正式后训练；第 5～6 周完成 Qwen 核心项目。

### 阶段总览

| 阶段 | 工作内容 | 主要模型 | 总投入（含复习） | 计划窗口 |
| ---: | --- | --- | ---: | --- |
| 0 | 语言模型张量与 loss | 无 | 约 8 小时（实际 6 + 复习 2） | 第 1 周 |
| 1 | 最小训练循环 | Bigram LM | 约 6 小时（实际 4.5 + 复习 1.5）；拓展另加 2～4 小时 | 第 1 周 |
| 2 | 最小 Transformer | 1M～10M TinyGPT | 约 9 小时（实际 7 + 复习 2） | 第 1 周 |
| 3 | 阅读现代模型结构 | 缩小版 MiniMind | 5～6.5 小时 | 第 2 周 |
| 4 | 从头预训练 | MiniMind 64M | 13～17 小时 | 第 2 周 |
| 5 | 监督微调 | MiniMind 64M | 13～17 小时 | 第 2～3 周 |
| 6 | LoRA | MiniMind 64M | 5～6.5 小时 | 第 3 周 |
| 7 | DPO | MiniMind 64M | 12～15.5 小时 | 第 3～4 周 |
| 8 | 模型蒸馏 | Qwen3-4B / MiniMind Teacher → MiniMind Student | 8.5～12 小时 | 第 4 周 |
| 9 | 实验规范整理 | MiniMind | 5～6.5 小时 | 第 4 周 |
| 10 | 正式后训练 | Qwen3 0.6B / 1.7B / 4B | 44～62.5 小时 | 第 4～6 周 |
| 10（旁路） | 领域继续预训练（可选） | Qwen3 0.6B Base | 7～10.5 小时 | 有领域适应专项兴趣时 |
| 11 | 可选 RL 后训练 | Qwen | 按实验确定 | 蒸馏专题完成后 |
| 12 | 架构迁移与受控研究 | GPT-2 Medium / OLMo 3 100M～370M | 193～255 小时 | 蒸馏后 6～7.5 个研究周 |

### 周进度安排

| 周次 | 计划内容 | 周目标 |
| ---: | --- | --- |
| 第 1 周 | 阶段 0～2 | 完成语言模型基础、最小训练循环和 TinyGPT，实现从 token 到 loss 的完整链路；时间允许时完成 BPE 拓展练习 |
| 第 2 周 | 阶段 3～4，并启动阶段 5 | 理解 MiniMind Dense 结构，完成预训练流程和评估，开始 SFT |
| 第 3 周 | 完成阶段 5，执行阶段 6～7 | 完成 SFT、LoRA 对照和 DPO，打通主要后训练方法 |
| 第 4 周 | 完成阶段 7，执行阶段 8～9，启动阶段 10 | 完成 MiniMind 后训练方法与实验规范，开始 Qwen 环境和开发模型调试 |
| 第 5～6 周 | 阶段 10 | 完成 Qwen 4B QLoRA SFT、完整评估、最小 DPO、回归检查和复现检查 |
| 第 7～9 周 | 蒸馏专题主体 | 完成 Teacher 数据、1.7B 调试、4B 序列蒸馏和 DPO 蒸馏 |
| 第 10 周前半 | 蒸馏专题结项 | 按条件执行 8B 验证，或记录停止理由并完成专题总结 |
| 蒸馏后 6～7.5 个研究周 | 阶段 12 | 执行 GPT-2 checkpoint 迁移与 OLMo 3 GQA 受控研究；如先进行可选 RL，窗口顺延 |

不执行可选 GRPO 时，完整主线预计在第 16～18 周结束。蒸馏专题提前结束时，阶段 12 可以直接进入 R1，不等待预设日历周。

周次是进度基线，不替代阶段验收标准。某阶段未达到验收标准时，应优先使用后续周次中的机动时间补齐，不因日历周结束而直接进入下一阶段。

---

## 二、阶段执行计划

### 阶段 0　补齐语言模型最小知识

> **时间记录与复习预算**　首次学习实际 6 小时；第三天复习约 2 小时；合计约 8 小时
>
> **阶段目标**　建立语言模型训练所需的最基本张量认知和训练认知。本阶段不运行 MiniMind。

#### 0.1　学习内容

##### 自回归语言模型

给定 token：

```text
今天 天气 很 好
```

训练样本实际为：

```text
输入：今天 天气 很
目标：天气 很 好
```

张量关系：

```text
input_ids: [B, T]
logits:    [B, T, V]
labels:    [B, T]
```

其中：

- `B`：batch size
- `T`：sequence length
- `V`：vocabulary size

loss 的本质是每个位置的 next-token cross entropy。

##### 必备概念

- tokenizer
- embedding
- causal mask
- attention
- logits
- softmax
- cross entropy
- teacher forcing
- gradient accumulation
- train / validation split
- checkpoint

本阶段不要求推导完整 Transformer，但需要理清数据从 token 转换为 loss 的完整过程。

#### 0.2　实践任务

编写一个短 notebook，完成以下内容：

1. 使用 tokenizer 编码一句话；
2. 打印 token ID；
3. 构造 `[B, T]` 输入；
4. 构造 shifted labels；
5. 随机生成 `[B, T, V]` logits；
6. 调用 `F.cross_entropy`；
7. 打印每个张量的 shape。

#### 0.3　阶段交付物

- 一个可运行的语言模型张量与 loss 演示 notebook
- notebook 中包含 token ID、shifted labels、logits 和各张量 shape 的输出

#### 0.4　验收标准

完成本阶段时，应能独立说明：

- labels 为什么需要错开一个 token；
- logits 为什么比 labels 多一个 vocabulary 维度；
- padding token 为什么不应参与 loss 计算；
- causal mask 阻止了哪类信息泄漏。

---

### 阶段 1　训练最简单的语言模型

> **时间记录与复习预算**　首次学习实际 4.5 小时；第三天复习约 1.5 小时；合计约 6 小时
>
> **阶段目标**　掌握完整训练循环。先训练 bigram language model 或简单 embedding language model，不直接训练 Transformer。

#### 1.1　训练流程

```text
Dataset
→ DataLoader
→ Forward
→ Loss
→ Backward
→ Optimizer step
→ Validation
→ Checkpoint
```

#### 1.2　实践任务

固定使用 Tiny Shakespeare，并沿用 nanoGPT 的顺序切分——前 90% 为 train、后 10% 为 validation。完成：

- 字符级 tokenizer；
- 训练一个 bigram model；
- 每隔若干 step 输出 train loss 和 validation loss；
- 保存 checkpoint；
- 从 checkpoint 恢复训练；
- 生成一段文本。

模型可缩小至：

```python
nn.Embedding(vocab_size, vocab_size)
```

本阶段不以学习模型架构为目的，重点是隔离并掌握训练工程。

#### 1.3　故障观察实验

至少主动制造并观察一次训练错误，可从以下情况中选择：

- 忘记调用 `optimizer.zero_grad()`；
- validation 时忘记调用 `model.eval()`；
- label 未做 shift；
- padding 未做 mask；
- learning rate 设置过大。

记录错误对 loss 和生成结果造成的变化。

#### 1.4　阶段交付物

- 一个可独立运行的 Bigram LM 训练项目
- 训练与验证 loss 记录
- checkpoint 保存与恢复结果
- 一份故障观察记录
- 一段模型生成文本

#### 1.5　验收标准

能够从空项目开始编写并运行完整训练循环，而非仅修改现成脚本参数。

#### 1.6　拓展练习：BPE 分词器

> **预计总投入**　2～4 小时，已包含第三天复习；本练习在阶段 1 验收后执行，不影响进入阶段 2。

复用 Tiny Shakespeare 的固定 train / validation split，调用 Hugging Face Tokenizers 的 Byte-level BPE 实现，分别训练 vocabulary size 为 512 和 2,000 的两个临时 tokenizer。对固定文本比较字符级、BPE-512 与 BPE-2000 的切分结果、每 100 个字符对应的 token 数和 encode / decode 可逆性，并说明 vocabulary size 与序列长度之间的关系，以及特殊 token 的基本作用。

交付物为一个短 notebook 和一张三种 tokenizer 的对照表。本练习不手写 BPE merge 算法，不新增或清洗语料，不为不同 tokenizer 重复训练模型；产生的 tokenizer 只用于教学观察，不替换 TinyGPT、MiniMind 或后续模型的既定 tokenizer。

---

### 阶段 2　实现最小 Decoder-only Transformer

> **时间记录与复习预算**　首次学习实际 7 小时；第三天复习约 2 小时；合计约 9 小时
>
> **阶段目标**　自主实现最小 Decoder-only Transformer，并理解一次完整 forward。此阶段仍不进入完整 MiniMind。

#### 2.1　实现范围

仅实现以下五个组件：

1. token embedding
2. positional encoding
3. causal self-attention
4. MLP
5. residual connection 和 normalization

模型规模控制在 1M～10M 参数。

本阶段不加入：

- RoPE
- GQA
- FlashAttention
- MoE
- KV Cache
- LoRA

以上功能将在后续阶段逐步引入。

#### 2.2　实践任务

自主实现：

```python
class CausalSelfAttention(nn.Module):
    ...

class TransformerBlock(nn.Module):
    ...

class TinyGPT(nn.Module):
    ...
```

至少打印一次完整张量流：

```text
input_ids          [B,T]
embedding          [B,T,D]
Q/K/V              [B,H,T,head_dim]
attention scores   [B,H,T,T]
hidden states      [B,T,D]
logits             [B,T,V]
```

#### 2.3　阶段交付物

- 一个 1M～10M 参数的 TinyGPT 实现
- 一份完整 forward 张量流记录
- loss 下降的训练记录

#### 2.4　验收标准

能够逐行解释以下 forward 流程：

```text
token
→ embedding
→ attention
→ MLP
→ logits
→ cross entropy
```

本阶段不要求生成质量，只要求代码正确且 loss 能够下降。

---

### 阶段 3　阅读并理解 MiniMind 模型结构

> **预计总投入**　5～6.5 小时，其中首次学习 4～5 小时，第三天复习 1～1.5 小时
>
> **阶段目标**　以 Dense 模型为主线，理解 MiniMind 相比 TinyGPT 增加的现代模型结构。

MiniMind 当前主线提供约 64M 的 Dense 模型，以及约 198M 总参数、64M active parameters 的 MoE 模型。项目实现了 Pretrain、SFT、LoRA、DPO、蒸馏和多种 RL 流程。本阶段第一遍仅阅读 Dense 模型。

#### 3.1　代码阅读顺序

按照代码数据流依次阅读：

1. `MiniMindConfig`
2. `RMSNorm`
3. `Attention`
4. `FeedForward`
5. `MiniMindBlock`
6. `MiniMindModel`
7. `MiniMindForCausalLM`
8. `PretrainDataset`
9. `train_pretrain.py`

不按 README 的展示顺序通读。

MiniMind 已包含：

- RoPE
- QK-Norm
- GQA
- RMSNorm
- gated FFN
- SDPA
- KV Cache

这些实现集中在较短的模型文件中。

#### 3.2　实践任务

将模型缩小，例如：

```text
hidden_size: 256
num_hidden_layers: 4
num_attention_heads: 4
num_key_value_heads: 2
```

完成以下任务：

- 使用随机输入执行 forward；
- 打印参数量；
- 打印每层输出 shape；
- 分别关闭和开启 KV Cache；
- 对比 GQA 与 MHA 的 K/V shape。

#### 3.3　阶段交付物

- 一份 MiniMind Dense 模型代码阅读笔记
- 缩小版 MiniMind 配置
- 参数量和逐层输出 shape 记录
- KV Cache 开关及 GQA/MHA 对比记录

#### 3.4　验收标准

能够说明 MiniMind 相比自主实现的 TinyGPT 增加了哪些现代结构，以及每种结构解决的问题。

---

### 阶段 4　MiniMind 从头预训练

> **预计总投入**　13～17 小时，其中首次学习 10～13 小时，第三天复习 3～4 小时
>
> **阶段目标**　掌握 MiniMind 的完整预训练流程，并建立可复现的训练与评估记录。

第一轮不追求 64M 模型的最佳效果。先使用缩小配置验证训练流程，再恢复至官方 64M Dense 配置。

MiniMind 的预训练脚本包括 mixed precision、gradient accumulation、gradient clipping、checkpoint 和 resume。

#### 4.1　任务一：小配置过拟合测试

从固定 revision 的 `pretrain_t2t_mini.jsonl` 按原始顺序取前 256 条，使缩小模型过拟合，以检查训练流程的正确性。保存 Dataset fingerprint 与原始 row ID，不临时挑选文本。

> 如果模型无法在几百条数据上过拟合，应优先检查代码、数据或 loss；模型能力暂不作为首要原因。

观察并记录：

- loss 是否持续下降；
- learning rate；
- gradient norm；
- GPU memory；
- tokens/s。

#### 4.2　任务二：运行官方 Mini 数据

使用以下配置开始训练：

- MiniMind 64M Dense；
- 官方 mini pretrain 数据；
- sequence length 256～340；
- 本机 RTX 4060 8GB。

#### 4.3　任务三：补充 Evaluation

原始训练脚本不能替代完整评估，需要增加：

- validation loss；
- perplexity；
- 固定生成观察 prompts；
- MiniMind 七任务客观基线；
- checkpoint 间结果比较；
- 训练 token 数统计。

固定生成观察 prompts 必须同时固定 generation config，至少记录：

- greedy decoding 与 sampling 的区别；
- `do_sample`；
- `temperature`；
- `top_k`；
- `top_p`；
- `max_new_tokens`；
- EOS 设置；
- sampling 使用的 random seed。

使用同一个 checkpoint 和上述三条官方 README prompts，对比一次 greedy decoding 与 sampling 的输出差异。比较不同 checkpoint 时，必须使用相同的 generation config，避免把采样随机性误判为训练效果。

#### 4.4　阶段交付物

- 小配置过拟合实验记录
- MiniMind 64M Dense 的 mini 数据训练记录
- validation loss 与 perplexity 结果
- MiniMind 七任务客观基线结果
- 固定生成观察 prompts 的 checkpoint 对比结果
- greedy decoding 与 sampling 的最小对照记录
- 用于 checkpoint 对比的固定 generation config
- checkpoint 恢复验证记录
- 训练 token、effective batch size、显存和吞吐统计

#### 4.5　验收标准

完成本阶段时，应能确认并说明：

- 模型训练的 token 总数；
- effective batch size；
- 每一步显存的主要消耗位置；
- train loss 和 validation loss 是否同时下降；
- generation config 如何影响输出，以及为什么 checkpoint 对比必须固定生成参数；
- checkpoint 是否能够正确恢复。

---

### 阶段 5　监督微调 SFT

> **预计总投入**　13～17 小时，其中首次学习 10～13 小时，第三天复习 3～4 小时
>
> **阶段目标**　在理解预训练的基础上，掌握对话数据构造、response mask 和完整 SFT 流程。

#### 5.1　原理范围

Pretrain 学习：

```text
预测所有文本中的下一个 token
```

SFT 学习：

```text
在对话上下文下预测 assistant response
```

MiniMind 的 `SFTDataset` 根据 chat template 构造完整对话，并只为 assistant 部分生成有效 labels。用户输入和 system prompt 对应位置设置为 `-100`，不参与 loss 计算。

#### 5.2　任务一：单样本检查

训练前先打印一个完整样本：

```text
token
token_id
role
label
是否计算 loss
```

检查以下事项：

- user token 不计算 loss；
- assistant token 计算 loss；
- EOS 是否计算；
- 截断发生的位置。

#### 5.3　任务二：100 条数据过拟合

从固定 revision 的 `sft_t2t_mini.jsonl` 以 seed `42` 打乱后取前 100 条，保存 Dataset fingerprint 与原始 row ID。使用该子集确认模型能够学习固定问答和指定输出格式，不单独编写、标注或人工挑选样本。

#### 5.4　任务三：完整 Mini SFT

完成并对比：

- pretrained checkpoint；
- SFT checkpoint；
- MiniMind 七任务客观基线；
- 三条固定生成观察 prompts；
- 上述固定 100 条子集中的问题；
- 输出长度；
- 输出格式观察。

#### 5.5　阶段交付物

- 单样本 token、role、label 与 loss mask 检查记录
- 100 条数据过拟合结果
- 完整 mini SFT checkpoint
- Pretrain/SFT checkpoint 的生成效果对比
- 输出长度与格式观察记录

#### 5.6　验收标准

能够识别并定位以下常见问题：

- chat template 错误；
- response mask 错位；
- assistant 内容被截断；
- 模型只复述 user；
- loss 很低但生成很差；
- EOS 学习失败。

---

### 阶段 6　LoRA

> **预计总投入**　5～6.5 小时，其中首次学习 4～5 小时，第三天复习 1～1.5 小时
>
> **阶段目标**　在完成 Full SFT 后，理解 LoRA 对全参数微调的替代方式，并完成可控对照实验。

LoRA 安排在 Full SFT 之后执行，以便明确它所替代的训练过程。MiniMind 自行实现了 LoRA 注入、参数冻结和 LoRA 权重保存，没有完全依赖 PEFT。

#### 6.1　学习内容

理解：

$$
W' = W + BA
$$

其中：

- 原始矩阵 \(W\) 冻结；
- 仅训练低秩矩阵 \(A\) 和 \(B\)；
- rank 决定 adapter 容量。

#### 6.2　对照实验

Full SFT 与 LoRA 使用相同的：

- pretrained checkpoint；
- SFT 数据；
- training steps；
- sequence length。

对比以下指标：

| 项目 | Full SFT | LoRA |
| --- | ---: | ---: |
| 可训练参数量 | 高 | 低 |
| GPU memory | 高 | 低 |
| tokens/s | 较低 | 较高 |
| MiniMind 七任务结果 | 基准 | 对照 |
| 相对 Pretrain 的变化 | 记录 | 记录 |

#### 6.3　阶段交付物

- LoRA 训练配置与 adapter 权重
- Full SFT 与 LoRA 的可训练参数量、显存、吞吐和七任务结果对比
- adapter 保存与合并验证记录

#### 6.4　验收标准

能够说明：

- LoRA 节省显存的原因；
- LoRA rank 的作用；
- target modules 的选择方式；
- adapter 的保存与合并方式；
- LoRA 与 QLoRA 的区别。

---

### 阶段 7　DPO

> **预计总投入**　12～15.5 小时，其中首次学习 9～12 小时，第三天复习 3～3.5 小时
>
> **阶段目标**　掌握第一个 preference optimization 流程，并建立 SFT、Reward Model、PPO、DPO、GRPO 和蒸馏在后训练体系中的位置感。本阶段只实际训练 DPO，不同时实现 PPO 或 GRPO。

#### 7.1　后训练算法定位

先建立算法全景：

```text
Base Model → SFT
                 ├→ preference pairs → DPO（offline）
                 ├→ preference pairs → Reward Model → PPO（online）
                 └→ verifier / reward → GRPO（online）

Teacher Model / Checkpoint → Distillation → Student Model
```

完成以下概念对照：

| 方法 | 主要训练信号 | 数据产生方式 | 本项目要求 |
| --- | --- | --- | --- |
| SFT | 标准答案 token | 离线示范数据 | 实际训练 |
| Reward Model | chosen / rejected 偏好 | 离线偏好数据 | 理解输入、输出与用途 |
| PPO | Reward Model 或规则奖励 | 在线采样回答 | 理解 rollout、reward、reference 和 policy update |
| DPO | chosen / rejected 偏好 | 离线偏好数据 | 实际训练 |
| GRPO | 一组采样回答的相对奖励 | 在线采样回答 | 可选实际训练 |
| Distillation | Teacher response 或 logits | Teacher 生成或前向计算 | 实际训练 |

本阶段不实现 Reward Model 和 PPO，但需要理解：

- offline preference optimization 与 online RL 的区别；
- Reward Model 在经典 RLHF 流程中的位置；
- PPO 为什么需要 rollout、reward、reference model 和 policy update；
- DPO 为什么可以绕过单独训练 Reward Model；
- GRPO 与 DPO 的训练数据产生方式为何不同；
- 蒸馏为什么不属于 preference optimization。

#### 7.2　数据与原理

每个样本包含：

```text
prompt
chosen response
rejected response
```

模型需要提高：

$$
\log \pi(y_{\text{chosen}}\mid x)
-
\log \pi(y_{\text{rejected}}\mid x)
$$

同时避免无限偏离 reference model。

MiniMind 的 DPO 实现会直接计算 policy model 和 reference model 对 chosen、rejected 的 sequence log-probability。

#### 7.3　任务一：读取极小 Preference 子集

从固定 revision 的 MiniMind 官方 `dpo.jsonl` 以 seed `42` 打乱后取前 64 条 chosen / rejected 样本，保存 Dataset fingerprint 与原始 row ID，并检查字段、模板、长度和 loss mask。该阶段直接复用公开偏好标签，不编写新的 preference pairs，也不按主观判断人工挑选样本。

#### 7.4　任务二：记录中间量

打印并记录：

- chosen policy log-probability；
- rejected policy log-probability；
- policy margin；
- reference margin；
- DPO logits；
- DPO loss。

#### 7.5　任务三：增加评估

除 DPO loss 外，增加：

- preference accuracy；
- chosen win rate；
- response length；
- KL divergence；
- MiniMind 七任务客观基线。

#### 7.6　阶段交付物

- 一页后训练算法全景与对照笔记
- 固定的 MiniMind preference 子集与检查记录
- DPO 中间量记录
- preference accuracy、chosen win rate、response length 和 KL divergence 评估
- DPO 前后的 MiniMind 七任务结果对比

#### 7.7　验收标准

能够说明：

- SFT、Reward Model、PPO、DPO、GRPO 和蒸馏分别使用什么训练信号；
- offline preference optimization 与 online RL 的区别；
- policy model 与 reference model 的区别；
- `beta` 的作用；
- DPO 可能使输出变长的原因；
- preference 数据质量比 loss 下降更重要的原因；
- DPO 造成能力退化的可能原因。

---

### 阶段 8　模型蒸馏

> **预计总投入**　8.5～12 小时，其中首次学习 6.5～9 小时，第三天复习 2～3 小时
>
> **阶段目标**　在已掌握监督数据和概率分布的基础上，完成 logit distillation 对照，并用极小样本体验 sequence-level distillation 的数据流。

#### 8.1　实现基线与工作边界

本阶段直接复用 MiniMind 官方蒸馏脚本提供的 Teacher / Student 加载、CE + KL、temperature、checkpoint resume 和分布式训练能力。学习重点放在训练信号、关键中间量与对照实验，不重新搭建蒸馏训练基础设施。

Sequence-level Distillation 固定使用 `openai/gsm8k` 的 `main` 配置。阶段 8 开始时生成项目级 `gsm8k_split_v1.json`：为官方 train 添加原始 row ID，以 seed `42` 打乱一次，前 500 条作为 development，其余 6,973 条作为正式训练区，并将正式训练区前 100 条记录为 `smoke_100`。本阶段读取这 100 个固定 row ID；阶段 10、蒸馏专题和可选 GRPO 只读复用同一清单，不重新抽样。`question` 作为 prompt，官方 `answer` 作为自动验证依据。

Teacher 固定使用已完成后训练的 `Qwen/Qwen3-4B`，批量生成 reasoning 与 final answer，再由 Math-Verify 对照 GSM8K gold answer 自动验证。只有验证通过的 Teacher response 进入 MiniMind SFT 数据；其余结果只保留自动状态，不进行人工改写、补标或筛选。

该任务只验收 `公开 prompt → Teacher response → 自动验证 → MiniMind Student SFT` 的数据流，不形成正式蒸馏结论，也不扩大数据量或训练矩阵。

#### 8.2　任务一：Sequence-level Distillation

由 Teacher 生成回答：

```text
prompt → teacher response
```

随后由 Student 对生成结果执行普通 SFT。

该方式是最容易迁移至 Qwen 和 gpt-oss 的蒸馏方式。

#### 8.3　任务二：Logit Distillation

MiniMind 已实现：

$$
L =
\alpha L_{\mathrm{CE}}
+
(1-\alpha)T^2
D_{\mathrm{KL}}\left(p_T^{(T)} \Vert p_S^{(T)}\right)
$$

推荐配置：

- Teacher：更大的 Dense MiniMind，或 MiniMind MoE；
- Student：MiniMind 64M Dense；
- 数据：`pretrain_t2t_mini.jsonl` 或 `sft_t2t_mini.jsonl` 的固定子集；
- tokenizer：Teacher 与 Student 完全相同。

本阶段开始时从上述范围选择一组 Teacher、Student 与数据配置，并在全部 CE / KD 对照中保持不变。数据文件选定后以 seed `42` 生成一次固定子集索引，保存 Dataset revision、fingerprint 与原始 row ID；全部 CE / KD 对照只读复用。该选择只服务于白盒蒸馏教学，不形成模型能力结论。

#### 8.4　对照实验

比较：

1. 纯 CE；
2. 纯 KD；
3. CE + KD；
4. 在 CE + KD 上增加一个 temperature 对照。

Teacher / Student size ratio 只记录当前配置与观察，不在本阶段扩展为完整网格。

#### 8.5　阶段交付物

- `gsm8k_split_v1.json` 及固定 100 条 prompts 的生成、自动验证与 sequence-level distillation 链路记录
- logit distillation 训练结果
- CE、KD、CE + KD 对照记录
- 一个 temperature 对照记录
- MiniMind 官方蒸馏脚本的关键调用链与配置说明

#### 8.6　验收标准

能够说明：

- response distillation 与 logit distillation 的区别；
- temperature 的作用；
- 不同 tokenizer 难以直接执行 token-level KD 的原因；
- teacher 错误传递给 student 的方式；
- student capacity 限制蒸馏上限的原因。

---

### 阶段 9　固化实验规范

> **预计总投入**　5～6.5 小时，其中首次学习 4～5 小时，第三天复习 1～1.5 小时
>
> **阶段目标**　在进入 Qwen 前暂停增加算法，借助现成工具固化轻量、可复现的实验规范。

本阶段不开发通用实验平台，也不为 MiniMind 与 Qwen 维护两套平行封装。MiniMind 沿用项目官方脚本和参数入口；Qwen 阶段使用 TRL CLI 的 YAML 配置以及 Transformers、Accelerate 提供的原生能力。

#### 9.1　配置规范

MiniMind 只在官方参数缺失时增加最薄的配置补充。Qwen 使用 TRL CLI 的 YAML 配置管理：

```text
model
dataset
seed
learning rate
batch size
sequence length
checkpoint
evaluation
```

不再额外开发一套通用 YAML / argparse 配置框架。

#### 9.2　记录规范

训练指标优先写入 Accelerate / Transformers / TRL 原生 tracker；本项目默认使用本地 TensorBoard，不要求注册额外实验平台。每次运行同时保存一份简洁的运行清单，至少记录：

- git commit；
- random seed；
- GPU；
- 数据版本；
- trainable parameters；
- peak VRAM；
- tokens/s；
- total tokens；
- train loss；
- validation loss；
- task metrics。

#### 9.3　评估规范

MiniMind 的 validation loss、perplexity 和固定 generation config 样例生成只保留薄入口；客观任务评估直接调用冻结的 `lm-evaluation-harness` 七任务配置。Qwen 的标准任务评估直接使用 LightEval，不开发覆盖所有模型与任务的通用 `evaluate.py`。

训练与评估使用固定数据 revision、split、seed 和 generation config。评估命令与结果目录写入对应实验的运行清单。

#### 9.4　Checkpoint 规范

MiniMind 沿用官方 checkpoint / resume 实现；Qwen 使用 Transformers Trainer 或 Accelerate 的 `save_state()` / `load_state()`。本阶段不编写独立 checkpoint 管理器。

#### 9.5　阶段交付物

- 一份轻量运行清单模板
- 一份 MiniMind 薄评估入口、七任务配置及固定 generation config
- 一份可供 Qwen 阶段复用的 TRL YAML 配置样例
- TensorBoard 本地实验记录
- 至少一次复现实验结果

#### 9.6　验收标准

能够使用现成工具完成配置加载、指标记录、checkpoint 恢复和独立评估；同一实验重新运行时，主要结果能够复现，并能清楚区分：

- 算法变化；
- 数据变化；
- 模型变化；
- 随机波动；
- 实现 bug。

---

### 阶段 10　Qwen 正式后训练

> **预计总投入**　44～62.5 小时，其中首次学习与执行 34～48 小时，第三天复习 10～14.5 小时
>
> **阶段目标**　将 MiniMind 阶段建立的底层理解迁移至真实大模型训练生态，完成 Qwen SFT 正式项目、训练前后回归评估和一次低成本 DPO 框架验证。

本阶段开始使用租用的 RTX 5090 32GB。

#### 10.1　模型安排

##### 开发模型

优先使用：

- `Qwen/Qwen3-0.6B-Base`；
- `Qwen/Qwen3-1.7B-Base`。

开发模型用于快速调试数据、chat template 和训练配置。

##### 最终模型

使用：

- `Qwen/Qwen3-4B-Base`。

8B 与更大规模 Student 留给后续蒸馏专题按条件验证。主计划不通过扩大模型规模重复相同训练流程。

#### 10.2　工具栈

在本阶段系统学习：

- Transformers
- Datasets
- PEFT
- TRL
- Accelerate
- bitsandbytes
- FlashAttention / SDPA
- vLLM 或 SGLang

MiniMind 阶段用于理解底层实现，Qwen 阶段用于掌握真实生产生态。

#### 10.3　框架最小实践

工具栈中的每个框架至少完成一次可验证的最小操作：

- **Transformers**：加载 tokenizer、模型和 generation config，完成一次前向计算与文本生成；
- **Datasets**：加载数据、划分 train/validation，并以一次确定性的 `Dataset.map` 转换为 TRL 标准数据格式；
- **PEFT**：注入 LoRA，打印可训练参数，完成 adapter 的保存、重新加载与合并；
- **bitsandbytes**：以 4-bit 方式加载基础模型，确认量化参数与 LoRA 可训练参数的区别，并跑通一次 QLoRA；
- **TRL**：使用 `SFTTrainer` 和 `DPOTrainer` 分别跑通一次 SFT 和 DPO，并使用 TRL CLI YAML 保存训练配置；
- **Accelerate**：通过统一启动配置运行一次训练，使用原生 tracker 记录指标，并从 checkpoint 恢复；
- **FlashAttention / SDPA**：选择与当前硬件和模型兼容的 attention backend，确认能够稳定完成训练并记录实际 backend；
- **vLLM 或 SGLang**：任选其一加载训练结果，使用固定 generation config 完成批量生成。

本阶段不要求阅读所有框架源码或掌握全部 API。验收重点是能够说明每个框架负责哪一段流程、接收什么输入、产生什么输出，以及 checkpoint、adapter 和数据如何在框架之间流转。

Qwen 阶段直接采用 TRL 支持的 prompt-completion 或 conversational 数据格式，由训练器应用 chat template 和 completion mask。数据适配层只负责字段映射、类型检查和少量批次抽查，不手写完整模板管线，也不维护第二套 SFT / DPO collator。LoRA / QLoRA 使用 PEFT，checkpoint 与日志使用 Transformers / Accelerate 原生能力。

#### 10.4　项目执行顺序

1. 使用固定数据、generation config 和评估任务建立 Qwen Base baseline；
2. 使用主流框架完成 QLoRA SFT；
3. 完成目标任务与通用回归评估；
4. 在公开 preference baseline 上完成一次最小 DPO 流程；
5. 重新加载 SFT checkpoint，完成固定 development 评估和一段短 resume 检查，再整理蒸馏专题可直接复用的资产；不执行第二次完整 4B SFT。

#### 10.5　固定任务与数据基线

正式任务统一采用公开的 `openai/gsm8k`：

- config：`main`；
- 官方 train：7,473 条；
- split 清单：复用阶段 8 已生成的 `gsm8k_split_v1.json`，核验 Dataset revision、fingerprint 与清单哈希；
- 正式 train：清单中的 6,973 个固定 row ID；
- development：同一清单中的 500 个固定 row ID；
- test：官方 1,319 条，只用于最终评估；
- 普通 SFT：直接使用公开的 question、rationale 与 gold answer；
- 自动判分：Math-Verify；
- 标准评估：LightEval 的现成 GSM8K task。

额外使用 `HuggingFaceH4/MATH-500` 的 500 条题目进行高难度泛化评估。MATH-500 不参与训练、Teacher 数据生成和超参数选择。

该配置用于 Qwen SFT、后续蒸馏专题及可选 GRPO。阶段 10 不重新生成 split，只读取并校验阶段 8 的项目级清单。训练数据转换采用可复现的程序化薄适配，只处理字段映射和格式校验；chat template 与 completion mask 交由 TRL 处理，不修改公开标签。

#### 10.6　通用回归评估基线

在 GSM8K 与 MATH-500 之外，固定采用 [LightEval 的现成任务](https://huggingface.co/docs/lighteval/available-tasks) 建立三项轻量回归检查：

| 维度 | 固定 benchmark | 用途 |
| --- | --- | --- |
| 指令遵循 | IFEval | 检查格式约束与显式指令遵循是否退化 |
| 通用知识与推理 | MMLU | 检查数学专项训练之外的通用能力变化 |
| 真实性风险信号 | TruthfulQA multiple-choice | 观察模型选择常见误解或不实陈述的倾向变化 |

首次运行时使用 `lighteval tasks list` 与 `lighteval tasks inspect` 确认当前版本的准确 task ID、prompt 与 metric，并将 LightEval revision、task ID 和 generation config 一并冻结。项目直接使用官方任务实现，不编写 benchmark、标签、prompt 模板或评分脚本。

完整回归集合只在以下关键 checkpoint 上运行：

1. Qwen3-4B Base；
2. 普通 GSM8K QLoRA SFT；
3. 蒸馏专题选出的唯一胜出 SFT；
4. 最终 Verifier-filtered DPO；
5. 条件式 8B 验证中实际进入结论表的 checkpoint。

阶段 10 只产生前两项结果，其余结果由蒸馏专题继承同一配置按条件补齐。开发模型、失败运行和未入选的中间 adapter 不重复执行完整集合。IFEval、MMLU 和 TruthfulQA 全程只读，不用于选择 checkpoint、调整超参数或生成训练数据；单项 benchmark 结果只作为回归信号，不形成完整安全能力结论。

#### 10.7　DPO 框架验证基线

DPO 只用于熟悉 TRL preference 数据流，不在本阶段构造 GSM8K preference pairs。固定使用 TRL 官方示例采用的公开 `trl-lib/ultrafeedback_binarized`，从 train split 以 seed `42` 打乱后抽取 1,000 条，在官方示例模型 `Qwen/Qwen3-0.6B` 上完成一次短训练。

本任务只验收：

- `prompt / chosen / rejected` 字段流转；
- policy 与 reference model 的加载；
- DPO loss、preference accuracy 和长度变化；
- adapter 保存、加载和独立评估。

该结果不用于推断 GSM8K 能力，也不进入后续蒸馏专题的数学偏好对照。正式数学 preference pairs 统一由蒸馏专题中的 Teacher 候选和 Math-Verify 自动产生。

该短流程与 4B 正式任务使用不同模型和数据，只验收 DPO 框架数据流，不额外运行通用回归集合。

#### 10.8　gpt-oss-20b 的定位

gpt-oss-20b 定位为后续蒸馏专题的固定 Teacher。主计划只确认模型来源、Harmony 格式和推理入口，不在阶段 10 生成正式 Teacher 数据。完成 Qwen 项目后，将其用于：

- 作为 teacher 生成高质量数据；
- sequence-level distillation；
- 理解 MoE reasoning model 的推理格式。

不将 gpt-oss-20b 作为第一个正式训练模型。

#### 10.9　领域继续预训练旁路（可选）

> **预计总投入**　7～10.5 小时，其中首次学习与执行 5.5～8 小时，第三天复习 1.5～2.5 小时
>
> **阶段目标**　体验 Base model 在公开领域语料上继续执行 causal language modeling 的流程，并观察领域 loss 改善与通用能力回退之间的关系。

该旁路只在阶段 10 核心验收完成且对领域适应问题仍有专项兴趣时启动，未执行时不影响进入蒸馏专题。阶段 12 已列入确定路线时，本旁路默认跳过，由 GPT-2 checkpoint 恢复训练承担一次完整的 continued-pretraining 流程实践。需要独立观察数学领域 loss 改善与通用能力回退时，采用 [Alignment Handbook](https://github.com/huggingface/alignment-handbook) commit `1de1fc9` 的 continued pretraining 配方与公开的 [`open-web-math/open-web-math`](https://huggingface.co/datasets/open-web-math/open-web-math)：

- 模型固定为 `Qwen/Qwen3-0.6B-Base`；
- 使用 Hugging Face Datasets streaming，冻结数据 revision，以 seed `42` 确定性抽取并程序化划分 train / validation；
- 训练预算固定为 10M tokenizer tokens，validation 从同一冻结样本池保留 5%；
- 文本只进行 tokenizer 编码、定长 packing 和空文本过滤；公开数据已经完成数学内容过滤、质量过滤与去重，本项目不再次清洗或人工筛选；
- 复用现成 causal language modeling trainer、Accelerate 配置、checkpoint 与 TensorBoard 能力，不实现新的训练器；
- 只运行一个正式配置，不搜索学习率、数据配比、序列长度或训练轮数。

实验前后比较 OpenWebMath validation loss / perplexity、GSM8K development 和阶段 10 的三项通用回归指标。该实验从原始 Base revision 独立启动，产物不替换 4B SFT baseline，也不进入蒸馏专题，以免把领域继续预训练与后续监督形式混为同一变量。

最小交付物包括一份训练 YAML、一条 validation loss 曲线，以及一张 Base / CPT 的领域 loss、数学能力和通用回归对照表。一次对照足以完成本阶段，不因效果有限而扩大数据量或追加 SFT 矩阵。

#### 10.10　阶段交付物

- 固定 revision 与 split 的 GSM8K 数据基线
- GSM8K 到 TRL 标准格式的确定性薄适配脚本
- Qwen Base baseline
- 一份框架职责与数据、checkpoint、adapter 流转记录
- Transformers、Datasets、PEFT、bitsandbytes、TRL 和 Accelerate 的最小实践结果
- 使用 vLLM 或 SGLang 完成的固定参数批量生成结果
- LightEval 的 GSM8K、MATH-500 与三项通用回归评估结果
- QLoRA SFT checkpoint 与完整评估结果
- 公开 preference baseline 上的最小 DPO 流程记录
- 可供蒸馏专题直接继承的数据 split、SFT baseline、评估配置和 checkpoint 索引

#### 10.11　验收标准

- GSM8K train、development、test 和 MATH-500 的用途边界清晰；
- 正式项目未引入人工标注或逐条数据修正；
- 能够说明各框架在后训练流程中的职责，以及数据和模型产物如何流转；
- 能够独立加载基础模型、训练 adapter、恢复 checkpoint，并加载训练结果完成批量生成；
- 开发模型能够稳定完成数据、模板和训练配置调试；
- 4B QLoRA SFT 已完成独立评估、checkpoint 重载和短 resume 复现检查，未重复执行完整训练；
- Qwen3-4B Base 与普通 QLoRA SFT 已按同一冻结配置完成通用回归评估；
- 能够用公开 preference 数据说明 TRL DPO 的完整输入输出；
- 未在主计划中提前构造正式 Teacher 数据或数学 preference pairs；
- 蒸馏专题可以直接读取本阶段的数据 split、baseline 和评估配置。

---

### 阶段 11　GRPO 或 RL 后训练（可选）

> **阶段性质**　可选阶段，不属于基础阶段。
>
> **阶段目标**　仅在训练和评估基础设施成熟后，基于成熟 baseline 对可验证任务开展一次最小 GRPO 实验。

#### 11.1　启动条件

仅当以下条件全部满足时进入本阶段：

- SFT 稳定；
- DPO 稳定；
- 已用 Math-Verify 跑通自动验证；
- evaluation 可靠；
- 能够区分 reward 与真实性能；
- 能够监测 KL、长度和 reward hacking。

#### 11.2　任务范围

继续使用冻结的 GSM8K train prompts 与 gold answers，reward 由 Math-Verify 自动产生。正式 test 和 MATH-500 保持只读评估用途。

初始化模型固定采用数学任务 checkpoint：默认使用阶段 10 的 `Qwen/Qwen3-4B-Base` GSM8K QLoRA SFT 结果；若蒸馏专题 D5 已完成，也可以改用其 4B Verifier-filtered DPO 结果。两者只选一个并在运行前冻结。阶段 10 的 `Qwen/Qwen3-0.6B` UltraFeedback DPO adapter 只用于框架验证，不纳入本阶段的初始化范围。

实现基线固定为 TRL `GRPOTrainer` 与 Hugging Face Open R1 的数学奖励配方。训练器负责 rollout、组内相对优势、policy update、指标记录及 vLLM 接入；奖励函数复用 Open R1 对 Math-Verify 的调用方式。本阶段不自行开发 rollout 调度器、分布式同步、推理服务连接或另一套 GRPO 训练循环。

基础主线（阶段 0～10）只要求理解 PPO 和 GRPO 在算法体系中的位置，不要求实际训练。进入本可选阶段后，优先只选择 GRPO，按以下最小流程完成一次实验：

```text
prompt
→ policy 采样一组 responses
→ verifier / reward function 打分
→ 计算组内相对优势
→ policy update
→ 独立 evaluation
```

至少打印并检查一组 responses、对应 rewards、相对优势、KL 和训练 loss。允许为理解算法读取 TRL / Open R1 的关键实现并打印中间量，不复制整套训练基础设施。暂不同时学习 PPO、CISPO 和 Agentic RL，避免一次引入多套 rollout、reward、reference、sampling、policy update 和环境变量。

自进化和 RSI 不纳入本项目路线。

#### 11.3　阶段交付物

- 一份 TRL GRPO YAML 配置与 Open R1 reward 适配记录
- Math-Verify 自动验证测试结果
- 一组 rollout responses、rewards、相对优势、KL 和 loss 的检查记录
- reward、KL、输出长度和 reward hacking 监测记录
- RL 前后真实性能对比

#### 11.4　验收标准

- 训练任务具有可靠的自动验证机制；
- 能够说明 rollout、reward、相对优势和 policy update 的数据流；
- reward 提升能够与真实任务性能提升区分；
- KL、输出长度和 reward hacking 均处于可监测状态。

---

### 阶段 12　架构迁移与受控研究

> **阶段性质**　主计划最后阶段。
>
> **预计总投入**　193～255 小时，按独立计划的 6～7.5 个相对研究周执行
>
> **阶段目标**　围绕 GQA 完成一次已有 checkpoint 迁移和一次从头预训练消融，建立架构研究所需的实现、控制变量与归因能力。

本阶段执行独立的 [模型架构迁移与研究学习计划](./架构迁移与研究计划.md)，采用一条固定路线：

```text
GPT-2 Medium 官方 checkpoint
→ 基于冻结官方实现的本地薄改、SDPA 与基础动态 KV Cache
→ MHA-to-GQA 权重转换与 MHA / GQA 同预算训练
→ OLMo 3 100M 官方预设冒烟测试
→ OLMo 3 190M 候选筛选
→ OLMo 3 370M baseline / 最佳 GQA 双 seed 验证
```

OLMo 部分统一采用 OLMo 3 的 100M、190M 和 370M 官方预设。该选择省去 OLMo 2 风格小规模配置的结构维护，使核心修改收敛到 `n_kv_heads`、数据 manifest、token budget 和 seed。

阶段 12 使用 Hugging Face 上的公开 checkpoint、nanoGPT 固定 OpenWebText baseline 与 DataDecide 预分词 shards。数据处理限于执行现成预处理脚本、版本冻结、自动完整性检查和少量只读抽查，不安排语料清洗、人工标注或规则改写。

QK-Norm、FFN、MoE、位置编码和长上下文训练均保留为结项后的独立选题。核心阶段不同时引入第二个架构变量。

#### 12.1　阶段交付物

- GPT-2 兼容实现、权重转换、correctness tests 与统一推理 benchmark
- MHA 同预算继续预训练对照，以及 GQA-8 / GQA-4 的即时迁移损失和恢复曲线
- TTFT、TPOT、GQA backend、Q/K/V shape 与 KV Cache layout 审计
- OLMo 3 190M 的 MHA / GQA / MQA 筛选结果
- OLMo 3 370M baseline 与最佳 GQA 的双 seed 正式结果
- 固定配置、数据 manifest、checkpoint 索引、结果 JSON 与分析 notebook
- 质量、训练效率、推理效率和结论边界说明

#### 12.2　验收标准

- 官方 GPT-2 与本地兼容实现通过 logits、loss 和 greedy generation 对齐；
- KV Cache 与非 Cache 的固定 greedy generation 一致；
- GPT-2 GQA 转换与固定 token budget 恢复训练完成；
- MHA 同预算对照能够区分 OpenWebText 继续适应与 GQA 恢复；
- OLMo 3 对照只改变 `n_kv_heads`，其余核心条件保持一致；
- OLMo 初始化配对、实际 attention backend 与 KV Cache layout 已通过审计；
- 370M baseline 与最佳 GQA 完成两个 seed，或具有符合预设规则的停止记录；
- 实验结果能够区分迁移损失、架构差异、seed 波动与系统测量噪声。

---

## 三、项目里程碑

### 里程碑一　最小训练能力建立

> **覆盖阶段**　0～2
>
> **目标完成时间**　第 1 周末

#### 完成标志

- 能够解释 token 到 loss 的完整过程；
- 能够从空项目编写训练循环；
- 能够实现并解释最小 Decoder-only Transformer。

### 里程碑二　MiniMind 训练链路打通

> **覆盖阶段**　3～5
>
> **目标完成时间**　第 3 周前半

#### 完成标志

- 能够解释 MiniMind 的主要现代结构；
- 能够完成从头预训练并恢复 checkpoint；
- 能够正确构造 SFT 数据与 response mask。

### 里程碑三　后训练方法验证

> **覆盖阶段**　6～8
>
> **目标完成时间**　第 4 周前半

#### 完成标志

- 完成 Full SFT 与 LoRA 对照；
- 能够说明 SFT、Reward Model、PPO、DPO、GRPO 和蒸馏在后训练体系中的位置；
- 完成 DPO preference optimization；
- 完成 sequence-level 和 logit distillation 实验。

### 里程碑四　实验规范固化

> **覆盖阶段**　9
>
> **目标完成时间**　第 4 周末

#### 完成标志

- 使用现成工具固化配置、记录、checkpoint 和评估规范；
- 同一实验的主要结果能够复现。

### 里程碑五　Qwen 正式项目完成

> **覆盖阶段**　10
>
> **目标完成时间**　第 6 周末

#### 完成标志

- 在开发模型上完成调试；
- 跑通 Transformers、Datasets、PEFT、TRL、Accelerate 和量化训练的最小框架流程；
- 完成 4B QLoRA SFT、完整评估，以及 checkpoint 重载和短 resume 复现检查；
- 完成 Base 与 SFT 的指令遵循、通用能力和真实性风险回归检查；
- 使用公开 preference baseline 完成最小 DPO 框架验证；
- 使用固定 generation config 完成批量生成与结果对比；
- 已归档蒸馏专题可直接复用的数据 split、配置、baseline 和 checkpoint 索引。

### 里程碑六　最终研究阶段

> **覆盖阶段**　可选阶段 11、最终阶段 12
>
> **目标开始时间**　蒸馏专题完成后

#### 完成标志

- 按 [模型架构迁移与研究学习计划](./架构迁移与研究计划.md) 完成 GPT-2 迁移与 OLMo 3 GQA 受控研究；
- 若选择阶段 11，已在进入阶段 12 前完成一次具备 verifier 的可控 GRPO 实验。
