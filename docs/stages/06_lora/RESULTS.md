# 阶段 6 实验记录

## 学习投入

本阶段实际主动学习 7 小时。正式训练、生成和七任务评估在学习时间之外运行，运行时间单独记录。

## 验收状态

阶段 6 已完成 LoRA 核心检查、真实 MiniMind target audit、capacity check、一轮 rank-16 正式训练、
三方固定生成和七任务评估，并通过阶段验收。

| 项目 | 状态 |
| --- | --- |
| rank 与 target modules 配置 | 已固定 |
| 低秩 shape 与参数量检查 | 已完成，小矩阵 CUDA/FP32 检查 |
| B 零初始化与初始输出检查 | 已完成 |
| Base 参数冻结检查 | 已完成 |
| adapter save/load 检查 | 已完成 |
| merge logits 检查 | 已完成，正式权重最大绝对差约 `5.25e-6` |
| 真实 MiniMind target audit | 已完成，8 层共 16 个 `q_proj` / `o_proj` |
| 真实 64M capacity check | 已完成，peak allocated 约 4.46 GiB |
| rank 16 正式训练 | 已完成，56,480 optimizer steps |
| Full SFT / LoRA 系统指标对比 | 已完成 |
| Pretrain / Full SFT / LoRA 固定生成 | 已完成 |
| merged weights 转换 | 已完成，转换前后最大 logit 差为 0 |
| LoRA 七任务评估 | 已完成 |

## 固定实验身份

| 字段 | 值 |
| --- | --- |
| profile | `lora_rank16` |
| Base checkpoint | 阶段 4 `weights_step_9908.pth` |
| Base SHA-256 | `3b9f475ca67a8fb98824633b9e93fe8577d9faf5bd19d8da250c48bf8889f06e` |
| Base SFT config | `docs/stages/05_sft/configs/mini64.json` |
| rank | 16 |
| target modules | `q_proj`、`o_proj` |
| learning rate | `1e-4` |
| 停止 step | 56,480 |

上述训练参数来自固定计划。真实 target audit 得到 393,216 个 trainable LoRA parameters，与结构
推导一致。

LoRA 专项测试结果为 `11 passed`，覆盖真实 MiniMind 注入、参数冻结、adapter 严格加载、merge、
resume 连续性和三方生成口径。正式 shape capacity check 使用 RTX 4060 Laptop GPU 和 BF16，结果如下：

| 项目 | 实测值 |
| --- | ---: |
| batch size / sequence length | 16 / 768 |
| target Linear | 16 |
| trainable parameters | 393,216 |
| loss | 11.5641 |
| gradient norm | 0.5265 |
| peak allocated memory | 4,792,367,616 bytes，约 4.46 GiB |
| 第一个 optimizer step | 1.52 秒，包含首次 kernel 开销 |

## 正式训练结果

LoRA 与阶段 5 Full SFT 都从同一个阶段 4 Pretrain checkpoint 开始，使用相同 train/validation
split、batch size 16、sequence length 768，并各自处理 367,458,576 个 supervised response tokens。

| 指标 | LoRA | Full SFT |
| --- | ---: | ---: |
| optimizer steps | 56,480 | 56,480 |
| trainable parameters | 393,216 | 63,912,192 |
| 初始 validation subset loss | 2.5504 | 2.5504 |
| 最终 validation subset loss | 1.8350 | 1.5381 |
| 完整 validation loss | 1.8640 | 1.5631 |
| 完整 validation perplexity | 6.450 | 4.773 |
| peak allocated GPU memory | 4.48 GiB | 6.77 GiB |
| 单 step tokens/s 中位数 | 17,752 | 14,501 |
| task-specific weights | 0.76 MiB adapter | 131.31 MiB full weights |

LoRA 的完整 validation loss 相对初始值下降约 26.9%，说明 Q/O adapters 已改善同分布未见对话上
的 response prediction。Full SFT 的最终 loss 更低；LoRA 的主要收益体现在资源侧：peak allocated
memory 降低约 33.8%，训练吞吐中位数提高约 22.4%，任务权重文件缩小到约 0.76 MiB。

固定 validation subset 的关键节点为：

| optimizer step | loss | perplexity |
| ---: | ---: | ---: |
| 0 | 2.5504 | 12.81 |
| 1,000 | 1.9920 | 7.33 |
| 10,000 | 1.8769 | 6.53 |
| 20,000 | 1.8527 | 6.38 |
| 40,000 | 1.8379 | 6.28 |
| 56,480 | 1.8350 | 6.27 |

曲线前段下降较快，20,000 steps 后边际改善逐渐缩小，一轮末尾保持缓慢下降。正式运行用时
22,974.1 秒，约 6 小时 23 分钟。

最终产物为：

| 文件 | 大小 | 内容 |
| --- | ---: | --- |
| `adapter_step_56480.pth` | 798,845 bytes | LoRA A、B 与 adapter 身份 |
| `latest.pt` | 267,683,915 bytes | Base、adapter、optimizer 与恢复状态 |
| `merged_weights_step_56480.pth` | 137,686,120 bytes | merge 后的完整 MiniMind 权重 |

merge 前后最大 logit 绝对差为 `5.2452e-6`，低于 `1e-4` 的检查阈值，merged weights 已成功导出。
merged weights 随后转换到 `outputs/minimind/stage6/rank16_transformers/`。转换目录的 tokenizer 与
原 tokenizer 一致，固定输入上的最大 logit 绝对差为 0。

## 三方固定生成

`outputs/minimind/stage6/rank16/generation.json` 复用阶段 5 的 Pretrain 与 Full SFT 结果，并新增
LoRA merged weights 在相同 8 个 prompts 和 decoding 参数下的输出。

| 模型 | greedy EOS | sampling EOS | greedy median length | sampling median length |
| --- | ---: | ---: | ---: | ---: |
| Pretrain | 2 / 8 | 2 / 8 | 128 | 128 |
| Full SFT | 1 / 8 | 2 / 8 | 128 | 128 |
| LoRA | 1 / 8 | 0 / 8 | 128 | 128 |

LoRA 与 Full SFT 都更倾向于直接进入回答正文，并使用段落或列表组织文本。LoRA 已学到 assistant
response format，但输出仍存在重复、事实错误、instruction following 不完整和 EOS 不稳定；多数样本
达到 `max_new_tokens=128` 后停止。Full SFT 的部分长回答更连贯。

## 七任务评估

LoRA merged weights 使用与阶段 5 相同的 chat template、task 配置和 metric 口径完成评估。

| Task | Pretrain | Full SFT | LoRA | LoRA - Pretrain |
| --- | ---: | ---: | ---: | ---: |
| C-Eval | 0.2377 | 0.2303 | 0.2273 | -0.0104 |
| CMMLU | 0.2506 | 0.2536 | 0.2524 | +0.0018 |
| ARC-Easy | 0.2908 | 0.3102 | 0.3072 | +0.0164 |
| PIQA | 0.5087 | 0.5141 | 0.5250 | +0.0163 |
| OpenBookQA | 0.2740 | 0.2740 | 0.2660 | -0.0080 |
| HellaSwag | 0.2852 | 0.2782 | 0.2750 | -0.0102 |
| SocialIQA | 0.3480 | 0.3465 | 0.3511 | +0.0031 |

LoRA 在四项任务上略有提高，在三项任务上下降，变化没有一致方向。多数分数仍接近对应选择题的
随机选择水平，当前 64M 模型没有表现出稳定的知识与推理能力提升。

validation loss 与七任务 accuracy 衡量不同目标。前者衡量 SFT validation 中 assistant response
tokens 的预测能力；后者衡量正确候选能否排在错误候选之前。LoRA 学到回答格式与训练分布中的语言
模式后可以降低 validation loss，但这不会自动转化为选择题 accuracy 提升，因此两组结果并不矛盾。

## 阶段结论

rank-16 LoRA 只训练 Full SFT 约 0.62% 的参数，使 peak allocated GPU memory 降低约 33.8%、训练
吞吐中位数提高约 22.4%，并把 task-specific weights 缩小到约 0.76 MiB。Full SFT 的 validation
loss 和部分固定生成质量仍然更好。此次实验验证了 LoRA 的参数效率与完整工程链路，没有证明它能在
当前模型规模和数据条件下提高知识与推理能力。

阶段 6 的计划学习内容、实验交付物和知识验收均已完成。
