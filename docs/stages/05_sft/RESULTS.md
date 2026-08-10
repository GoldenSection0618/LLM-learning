# 阶段 5 实验记录

## 当前状态

阶段 5 的数据、训练、固定生成、格式转换与七任务评估已经完成。

| 项目 | 状态 |
| --- | --- |
| 固定数据文件与 data lock | 已完成 |
| 固定 100 条教学子集 | 已完成 |
| 独立 full SFT validation | 已完成 |
| 单样本 response mask 检查 | 已完成 |
| batch 16 capacity check | 已完成 |
| 专项测试 | 已完成，39 passed |
| 100 条过拟合训练 | 已完成，step 180 达到停止条件 |
| 完整 Mini SFT | 已完成一轮，step 56,480 主动停止 |
| Pretrain/SFT 生成对比 | 已完成，8 个固定 prompts |
| Transformers 转换与 logits 验证 | 已完成 |
| SFT 七任务基线 | 已完成 |

## 初始化记录

| 项目 | 结果 |
| --- | --- |
| 数据行数 | 905,718 |
| Dataset fingerprint | `b6c6f1b40b21c792` |
| 教学子集 row ID | seed 42 排列的前 100 条 |
| `overfit100` split SHA-256 | `01460610d048b9cbb7bc8ac143e81b2d1b663ecad51a3556d344e79dedd8d2f7` |
| `mini64` train / validation | 903,670 / 2,048 条 |
| `mini64` split SHA-256 | `33bc790dabe07ec52c38e0749898b6a1a48c108f298f4eb6d6eaa76de4a58dbc` |
| Pretrain 起始权重 SHA-256 | `3b9f475ca67a8fb98824633b9e93fe8577d9faf5bd19d8da250c48bf8889f06e` |

固定 row ID 82,374 的首条教学样本包含一轮 user/assistant 对话。sequence length 768 下共有
570 个非 PAD token，其中 541 个 shifted assistant target 参与 loss；user、assistant role marker
和 PAD 不参与 loss，assistant 结束标记参与 loss。该样本没有发生 truncation。

## Capacity check

在 RTX 4060 Laptop GPU 上，使用阶段 4 最终权重执行一次真实 optimizer update：

| 字段 | 结果 |
| --- | ---: |
| batch size | 16 |
| sequence length | 768 |
| supervised response tokens | 6,389 |
| 初始 batch loss | 2.5267 |
| peak allocated GPU memory | 6,615,620,096 bytes，约 6.16 GiB |

该结果只确认正式配置能够执行一次 forward、backward、gradient clipping 和 AdamW step，不代表
完整训练吞吐或稳定性。初始化不继续搜索更大的 batch size。

## overfit100

固定 100 条训练样本同时用于 evaluation。loss 从 2.5488 降至 0.9834，在第 18 个 epoch、
180 个 optimizer steps 达到 loss 不高于 1.0 的停止条件。最终 perplexity 为 2.6736，累计处理
745,956 个 supervised response tokens，peak allocated GPU memory 为 4.55 GiB。

该结果支持 response mask、loss、backward、optimizer 与 checkpoint 链路能够拟合固定小数据，
不提供未见对话上的泛化证据。

## 正式一轮 SFT

正式训练使用 903,670 条 train sequences 与独立的 2,048 条 validation。配置原计划 2 epochs，
本次通过 `stop_after_step=56480` 在第一个 epoch 结束时停止。

| 字段 | 实测结果 |
| --- | ---: |
| completed optimizer steps | 56,480 |
| completed epochs | 1 |
| trained supervised response tokens | 367,458,576 |
| validation subset loss，step 0 / 56,400 | 2.5504 / 1.5381 |
| final full validation loss | 1.5631 |
| final full validation perplexity | 4.7735 |
| final full validation tokens | 852,275 |
| peak allocated GPU memory | 6.77 GiB |
| 单 step tokens/s 中位数 | 14,501 |
| 纯模型权重 | `checkpoints/minimind/stage5/mini64/weights_step_56480.pth` |
| 纯模型权重 SHA-256 | `0600cc7618f3665a709bc5f439de58e9b124c9ceea4ceaac8a5c82f1dafbccde` |

固定 256 条 validation subset 的 loss 相对下降约 39.7%。后半程仍缓慢改善，没有持续反向上升；
边际改善已经明显小于训练早期。完整 validation 与 subset 样本范围不同，不把两者的差值解释为
训练末尾退化。

`wall_time_seconds` 在每次启动或 resume 时重新计时，并从训练循环开始处统计。最终 summary 中的
11,795.9 秒只属于最后一次运行；本次训练中途经过停止与 resume，因此不把 3.28 小时记录成完整
端到端训练时间。

## 固定生成

Pretrain 使用普通文本 prompt，SFT 使用官方 chat template。两边均对 8 个固定 prompts 运行
greedy 与 sampling：

| weights | mode | stopped on EOS | generated-token range | median tokens/s |
| --- | --- | ---: | ---: | ---: |
| Pretrain | greedy | 2 / 8 | 35-128 | 143.6 |
| Pretrain | sampling | 2 / 8 | 44-128 | 136.5 |
| SFT | greedy | 1 / 8 | 27-128 | 145.5 |
| SFT | sampling | 2 / 8 | 91-128 | 138.4 |

SFT 输出通常直接进入回答正文，段落和列表格式更明显。当前模型仍会重复、产生事实错误并在复杂
指令上失败。四组生成的 median length 都达到 128，多数输出由 `max_new_tokens` 停止；本次结果
没有显示 EOS 稳定改善。8 个 prompts 只用于观察生成行为，不代表全部输入。

生成产物为 `outputs/minimind/stage5/mini64/generation.json`，包含文本、token IDs、生成长度、
tokens/s、EOS、解码配置和 checkpoint，不包含 accuracy 或能力 loss。

## Transformers 转换

同一份 `weights_step_56480.pth` 分别由 MiniMind 原生代码与本地 Transformers 格式加载。这里的
Hugging Face 表示文件格式和加载接口，不是与 Hub 上另一份模型权重比较。

| 检查 | 结果 |
| --- | ---: |
| tokenizer vocabularies match | True |
| maximum logit absolute difference | 0 |
| logits close at 1e-3 | True |

转换目录为 `outputs/minimind/stage5/mini64_transformers/`。验证结果支持当前转换没有改变固定输入上
的模型计算。配置与权重映射不会随机变化；该检查用于发现转换脚本 bug、源码结构变化或选错文件。

## 七任务评估

`prepare_lm_eval` 复制并冻结七项现成任务的 YAML 与 Dataset revision，只准备规则，不加载模型或
计算分数。实际评估用 SFT chat template 包装题目，分别计算候选答案的 log-likelihood，选择得分
最高的答案并汇总 accuracy；过程不执行随机生成、backward 或参数更新。

| task | samples | metric | Pretrain | SFT | SFT - Pretrain |
| --- | ---: | --- | ---: | ---: | ---: |
| C-Eval | 1,346 | acc | 0.2377 | 0.2303 | -0.0074 |
| CMMLU | 11,582 | acc | 0.2506 | 0.2536 | +0.0030 |
| ARC-Easy | 2,376 | acc_norm | 0.2908 | 0.3102 | +0.0194 |
| PIQA | 1,838 | acc_norm | 0.5087 | 0.5141 | +0.0054 |
| OpenBookQA | 500 | acc_norm | 0.2740 | 0.2740 | 0.0000 |
| HellaSwag | 10,042 | acc_norm | 0.2852 | 0.2782 | -0.0070 |
| SocialIQA | 1,954 | acc | 0.3480 | 0.3465 | -0.0015 |

变化有升有降，绝对值均不超过约 0.02，没有形成七任务一致上升的趋势。结果仍大致处在对应
multiple-choice 任务的随机选择水平附近。当前证据不支持“一轮 SFT 普遍提高七任务能力”。

## 阶段结论

一轮正式 SFT 降低了独立 validation response loss，并使固定样本中的回答更直接、更接近
assistant 格式。当前 64M 模型仍存在重复、事实错误、复杂指令失败和 EOS 不稳定；七任务也没有
整体上升。阶段 5 已建立可复现的 Full SFT 与评估闭环，现有证据满足阶段验收，可以进入阶段 6。
