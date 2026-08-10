# 阶段 4 实验记录

## 学习投入

本阶段实际主动学习 4 小时。数据处理、训练、生成和七任务评估主要在休息时间自行运行，等待
时间不计入学习时长。阶段 3 已主动投入 20 小时补齐 Transformer 理论和 MiniMind 代码细节，
因此阶段 4 可以集中完成预训练工程闭环。

阶段 3～4 实际主动投入合计 24 小时；原计划合计 18～23.5 小时，只比计划上限多 0.5 小时，
没有达到 20% 的后续预算调整条件。阶段 5 之后的时间预算保持不变。

## 验收状态

阶段 4 已正式结束。两项训练、固定生成、格式转换、七任务评估和 checkpoint 恢复测试均已完成。

| 交付物 | 状态 |
| --- | --- |
| 256 条小数据记忆测试 | 已完成，达到停止阈值 |
| MiniMind 64M Dense 训练 | 已完成，2 epochs |
| validation loss 与 perplexity | 已完成，使用完整 2,048 条 validation |
| 三个 checkpoint 的固定生成对比 | 已完成 |
| greedy decoding 与 sampling 对照 | 已完成 |
| Transformers 格式转换 | 已完成，logits 一致 |
| 七任务客观基线 | 已完成 |
| checkpoint 恢复验证 | 已完成 |
| Notebook 结果复盘 | 已完成 |

这些结果用于验证训练与评估链路，不用于证明 64M 模型已经具备可靠的知识问答能力。

## 运行身份

| 项目 | 值 |
| --- | --- |
| MiniMind source revision | `89d674b8a517010f5561b6d8ab2dcbb58e2fb91b` |
| 数据 revision | `312afb4f76391145c6902f765bb51691c09a12f5` |
| 数据文件 SHA-256 | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` |
| 数据行数 | 1,270,238 |
| mini64 train / validation | 1,268,190 / 2,048 条序列 |
| split SHA-256 | `e3043e428669744ea0288bc145fdf4062f88c2396e34f3c536e825a3bbb70caf` |
| 训练设备 | NVIDIA GeForce RTX 4060 Laptop GPU |
| 训练 precision | BF16 autocast；模型参数与 AdamW 状态为 FP32 |

Hugging Face Datasets fingerprint 由库版本参与计算。训练时记录的值为
`d8155c461476413a`；同一 JSONL 在 `datasets` 3.6.0 下为 `3f53fed479a316f9`。两次读取的文件
SHA-256、行数和 split 身份一致。fingerprint 保留作环境记录，checkpoint 恢复以原始文件
SHA-256 和 split SHA-256 判断数据身份。

## 自动化验证

执行 `pytest -q tests/minimind`，结果为 `15 passed in 6.46s`；完整项目测试为
`32 passed in 6.50s`。阶段测试覆盖数据编码、split、right padding、token 加权 loss、
optimizer-step learning rate、gradient accumulation 尾部更新、checkpoint 精确恢复、独立
sampling 随机数和本地 task 目录身份。Notebook 的 8 个代码单元已依次执行，execution count
为 1-8，没有 error 或 warning 输出。

## 小数据记忆测试

缩小模型固定使用原始 row ID 0-255。训练和 evaluation 读取同一组数据，因此该结果只检查
实现能否记忆小数据，不衡量泛化能力。

| 指标 | 结果 |
| --- | ---: |
| 参数量 | 4,983,552 |
| 有效 shifted label token | 30,444 |
| 初始 loss / perplexity | 8.8221 / 6,782.36 |
| 最终 loss / perplexity | 0.8178 / 2.27 |
| optimizer steps | 240 / 1,000，达到 loss 不高于 1.0 后停止 |
| 训练 token | 913,320 |
| effective batch size | 32 条序列 |
| peak allocated GPU memory | 433,835,008 bytes，约 413.7 MiB |
| tokens/s | optimizer step 中位数 80,125 |
| gradient norm | 中位数 1.921，裁剪前最大值 6.022 |
| 原始训练 wall time | 17.96 秒 |

loss 从随机初始化水平降到 0.8178，说明当前 tokenization、label shift、反向传播和 AdamW
update 能够共同工作。这个实验不支持对未见文本作结论。

## MiniMind 64M Dense 训练

正式运行从随机参数开始，不继承小数据记忆测试的权重。

| 指标 | 结果 |
| --- | ---: |
| 参数量 | 63,912,192 |
| 训练长度 | 2 epochs，9,908 optimizer steps |
| microbatch / gradient accumulation | 32 / 8 |
| effective batch size | 256 条序列 |
| 训练 token | 528,940,952 |
| 初始 validation subset loss / perplexity | 8.9022 / 7,348.48 |
| 最终 validation subset loss / perplexity | 1.9193 / 6.82 |
| 最终完整 validation loss / perplexity | 1.9292 / 6.88 |
| 完整 validation 有效 token | 428,836 |
| tokens/s | optimizer step 中位数 17,234 |
| gradient norm | 中位数 0.322，裁剪前最大值 2.789 |
| peak allocated GPU memory | 6,576,906,752 bytes，约 6.13 GiB |
| 训练 wall time | 32,584.97 秒，约 9 小时 3 分 |

周期 evaluation 固定使用 validation 的前 256 条，完整 evaluation 使用全部 2,048 条。两者
样本范围不同，不能把最终完整 loss 直接接到周期曲线上。

| optimizer step | validation subset loss | perplexity |
| ---: | ---: | ---: |
| 0 | 8.9022 | 7,348.48 |
| 100 | 5.5962 | 269.40 |
| 1,000 | 2.5784 | 13.18 |
| 2,500 | 2.2632 | 9.61 |
| 5,000 | 2.0702 | 7.93 |
| 7,500 | 1.9686 | 7.16 |
| 9,900 | 1.9191 | 6.81 |
| 9,908 | 1.9193 | 6.82 |

train loss 的 100-step rolling mean 与 validation subset loss 都明显下降。最后两个 validation
subset 点基本持平，当前运行没有出现持续的 validation loss 回升。单个 step 的 train loss 受
当前 batch 影响，不用于判断整体趋势。

### 第一次长训练的观察

运行前担心一次遍历训练不足，因此正式配置使用 2 epochs。训练完成后，用第 1 个 epoch 的
`weights_step_4954.pth` 在相同的固定 256 条 validation subset 上补做了一次 evaluation：

| checkpoint | optimizer step | 训练 token | validation loss | perplexity |
| --- | ---: | ---: | ---: | ---: |
| 第 1 个 epoch 结束 | 4,954 | 264,470,476 | 2.0737 | 7.95 |
| 第 2 个 epoch 结束 | 9,908 | 528,940,952 | 1.9193 | 6.82 |

第二个 epoch 增加了 264,470,476 个训练 token，validation loss 相对下降约 7.4%。step 4,954
的生成已经能够围绕 prompt 组织中文句子；step 9,908 的结构更稳定，但两者仍有重复、事实错误
和未完成列表。第二个 epoch 有可测量的改善，边际收益明显小于第一个 epoch。七任务只在最终
checkpoint 上运行，因此这里不推断第二个 epoch 对七任务分数的影响。

单个 optimizer step 的 train loss 由当前 batch 的内容、长度和预测难度共同影响，正常情况下
会上下波动，不要求单调递减。train loss 的 rolling mean 用于观察拟合趋势；固定口径的
validation loss 用于判断模型在同分布未见数据上的泛化能力是否继续改善。不能用某一个 batch
的 train loss 判断训练是否已经停滞。

本次曲线不属于明显的 grokking。典型 grokking 是训练集已经被充分拟合后，validation 表现
长期较差，随后才突然改善；该现象最初主要在小型 algorithmic datasets 上研究。普通自然语言
LLM pretraining 不应预设会出现这种突变。本次 train 与 validation 从早期开始同步下降，后期
逐渐变平，表现为常规的渐进改善。

这次第一次长训练说明，train loss 最小化不是无限延长训练的充分理由。停止判断还要结合
validation 趋势、目标任务指标、生成观察和额外计算成本。当前结果不支持继续盲目增加 epoch。
step 4,954 使用的是预先按 2 epochs 计算的 cosine learning-rate schedule，因此它不等价于从
一开始就配置 1 epoch 的独立训练。

## Checkpoint 与生成对比

完整 checkpoint `latest.pt` 位于 optimizer step 9,908，包含模型、AdamW、GradScaler、训练
位置、配置、Tokenizer、数据 manifest、指标历史和随机数状态。自动化测试比较了不中断训练和
save/load 后的下一次 AdamW update，参数逐项完全相同。真实 `overfit256/latest.pt` 也已完成
重载，step、累计 token 和停止原因保持不变。

生成对比固定使用三个 prompts、相同模型配置和以下 generation config：

| 模式 | `do_sample` | `temperature` | `top_k` | `top_p` | `max_new_tokens` | seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | false | 1.0 | 0 | 1.0 | 128 | 2026 |
| sampling | true | 0.85 | 50 | 0.95 | 128 | 2026 |

| checkpoint | 观察结果 |
| --- | --- |
| step 0 | 输出是无意义的中英文片段和 token 重复。 |
| step 4,954 | 已能围绕问题生成中文句子，但存在重复、事实错误和不完整列表。 |
| step 9,908 | 结构与主题相关性更稳定，仍有重复和事实矛盾，不具备可靠问答能力。 |

最终 checkpoint 的 greedy decoding 更容易反复选择高概率表达；sampling 的措辞更多样，也会
引入新的事实错误。固定 seed 只让本次 sampling 对照可复现。checkpoint 对比还必须固定权重、
prompt、generation config、Tokenizer 和运行实现，才能把主要差异归因于训练进度。

完整生成文本和 token ID 位于本地
`outputs/minimind/stage4/mini64/generation.json`。

## 格式转换与七任务基线

最终纯权重已通过 MiniMind 官方转换脚本导出为 Transformers 格式。固定输入上的核验结果为：

| 检查项 | 结果 |
| --- | ---: |
| Tokenizer vocabulary 一致 | 是 |
| logits 最大绝对误差 | 0.0 |
| `atol=1e-3, rtol=1e-3` 下相等 | 是 |

七任务使用 `lm-evaluation-harness` 0.4.12 的标准 `hf` backend。CEVAL 和 CMMLU 报告 group
aggregate；其余任务使用 harness 提供的 primary normalized accuracy，SocialIQA 使用 accuracy。

| task | metric | samples | score | standard error |
| --- | --- | ---: | ---: | ---: |
| CEVAL valid | accuracy | 1,346 | 0.2377 | 0.0117 |
| CMMLU | accuracy | 11,582 | 0.2506 | 0.0040 |
| ARC Easy | normalized accuracy | 2,376 | 0.2908 | 0.0093 |
| PIQA | normalized accuracy | 1,838 | 0.5087 | 0.0117 |
| OpenBookQA | normalized accuracy | 500 | 0.2740 | 0.0200 |
| HellaSwag | normalized accuracy | 10,042 | 0.2852 | 0.0045 |
| SocialIQA | accuracy | 1,954 | 0.3480 | 0.0108 |

评估用时 486.71 秒。七项结果大致处在相应选择题的随机选择水平附近，例如四选一约为 0.25、
PIQA 二选一约为 0.50、SocialIQA 三选一约为 0.33。这组结果没有显示出稳定的 zero-shot
multiple-choice 能力，与 64M base model、有限训练 token 和未做 instruction tuning 的条件一致。
这里不计算跨任务平均分，因为任务、样本数和 metric 不同。

任务 Dataset revision 已固定。本地 task 目录不包含运行生成的 Python cache，内容 SHA-256 为：

```text
59e5189f22a0af7d56c65dc85c039651b299c85b1f92a6f29ebdb84f904416a5
```

原始 harness 结果和逐样本记录位于本地
`outputs/minimind/stage4/lm_eval_results/`。

## 复现命令

以下命令记录本次生成、转换和评估使用的入口。转换命令要求输出目录为空。

```bash
python -m llm_learning.minimind.evaluate \
  --checkpoint checkpoints/minimind/stage4/mini64/weights_step_0.pth \
  --checkpoint checkpoints/minimind/stage4/mini64/weights_step_4954.pth \
  --checkpoint checkpoints/minimind/stage4/mini64/weights_step_9908.pth \
  --output outputs/minimind/stage4/mini64/generation.json

python -m llm_learning.minimind.convert \
  --weights checkpoints/minimind/stage4/mini64/weights_step_9908.pth \
  --output outputs/minimind/stage4/mini64_transformers

python -m llm_learning.minimind.prepare_lm_eval \
  --output outputs/minimind/stage4/lm_eval_tasks

python -m llm_learning.minimind.lm_eval \
  --model outputs/minimind/stage4/mini64_transformers \
  --task-dir outputs/minimind/stage4/lm_eval_tasks \
  --output outputs/minimind/stage4/lm_eval_results
```

## 阶段结论

阶段计划要求的训练、记录、生成对照、七任务基线和 checkpoint 恢复证据均已覆盖。训练链路
通过小数据记忆测试；64M run 的 train loss 与 validation loss 同时下降；最终模型能够生成与
prompt 相关的中文段落，但七任务结果仍接近随机选择水平。阶段 4 的目标是建立可复现的完整
预训练与评估闭环，这一目标已经达到。生成质量不作为继续延长训练的理由。阶段 4 正式结束，
下一步进入阶段 5 的 SFT。
