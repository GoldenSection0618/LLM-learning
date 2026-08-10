# 阶段 5：MiniMind 监督微调

本阶段从阶段 4 的最终 Pretrain 权重开始训练 MiniMind SFT。学习重点是 chat template、
response mask、EOS 与 truncation；模型结构、mixed precision、gradient accumulation、
checkpoint 和 validation 复用阶段 4 的实现。

主要学习入口是：

```text
notebooks/05_minimind_sft.ipynb
```

## 固定输入

MiniMind source revision 保持为：

```text
89d674b8a517010f5561b6d8ab2dcbb58e2fb91b
```

SFT 数据锁位于 `configs/data_lock.json`：

| 字段 | 值 |
| --- | --- |
| repo | `jingyaogong/minimind_dataset` |
| revision | `312afb4f76391145c6902f765bb51691c09a12f5` |
| file | `sft_t2t_mini.jsonl` |
| size | 1,739,201,170 bytes |
| SHA-256 | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518` |

数据共有 905,718 条 conversations 记录。部分早期记录只有基础字段，后期记录还包含
`tools` 与 `tool_calls`。`sft_data.py` 使用完整固定 schema 读取，避免自动 schema inference
在文件中途失败。

两项训练都从以下纯模型权重初始化：

```text
checkpoints/minimind/stage4/mini64/weights_step_9908.pth
SHA-256: 3b9f475ca67a8fb98824633b9e93fe8577d9faf5bd19d8da250c48bf8889f06e
```

SFT 会重新创建 AdamW 和 learning-rate schedule，不恢复阶段 4 的 optimizer。

## 从 conversations 到 response mask

Dataset 先用 MiniMind 官方 chat template 渲染完整对话，再执行 tokenization。system 和 user
位置的 labels 为 `-100`；每个 assistant response 的内容及 `<|im_end|>` 参与 loss；right
padding 仍为 `-100`。

模型内部执行 causal label shift。检查单样本时，每一行同时显示当前位置的 input token 和下一
位置的 label，避免把“当前 token 是否参与 loss”与“当前 logits 预测哪个 token”混为一谈。

官方 Dataset 会随机添加 system prompt，并随机处理空 `<think>`。本项目保留相同概率，但随机
选择由 seed 和原始 row ID 决定，使相同样本不受 DataLoader worker 数量或 resume 影响。

truncation 直接作用于 chat template 的完整 token 序列。Dataset 额外返回 `truncated` 和
`supervised_tokens`，用于定位 assistant 内容或 EOS 被截断，以及样本没有有效 response label
的问题。

## 固定划分

seed 42 只生成一次确定性 row ID 排列：

- 前 100 个 row ID 固定为教学子集；
- `overfit100` 的 train 和 evaluation 都使用这 100 条；
- 完整 SFT 使用接下来的 2,048 条作为 validation；
- 完整 train 排除 validation，但保留教学子集。

前五个教学 row ID 为：

```text
82374, 232959, 319150, 780082, 156736
```

`overfit100` evaluation 重新测量训练集 response loss，不代表泛化能力。完整 SFT 的 validation
与 train 相互独立。

## 两项训练

| 字段 | `overfit100` | `mini64` |
| --- | ---: | ---: |
| 参数量 | 63,912,192 | 63,912,192 |
| sequence length | 768 | 768 |
| microbatch | 10 | 16 |
| gradient accumulation | 1 | 1 |
| epochs | 最多 100 | 2 |
| optimizer-step 上限 | 1,000 | 112,960 |
| learning rate | `1e-5` | `1e-5` |
| precision | BF16 | BF16 |

`overfit100` 每个 epoch 包含 10 个 optimizer steps，因此 100 epochs 与 1,000 steps 的上限
一致；完整训练集 response loss 不高于 1.0 时提前停止。正式配置保存第 1 个 epoch
和第 2 个 epoch 的权重，用 validation、固定生成和额外计算成本判断第二个 epoch 的边际收益，
不以 train loss 最小化为唯一目标。

本次正式运行通过 `stop_after_step=56480` 在第 1 个 epoch 结束时停止，没有执行第 2 个 epoch。
learning-rate schedule 仍按原配置的 112,960 steps 计算。实际结果见 [`RESULTS.md`](./RESULTS.md)。

一次真实 capacity check 使用 batch 16、sequence length 768 完成 forward、backward、gradient
clipping 和 AdamW step，peak allocated GPU memory 为 6,615,620,096 bytes，约 6.16 GiB。
因此保留 batch 16，不继续做 batch-size sweep。

## Evaluation

生成继续使用阶段 4 的三条官方 prompts、greedy decoding 与 sampling 参数，并增加固定教学子集
中的前五个 user 问题。Pretrain 使用普通文本 prompt；SFT 使用官方 chat template 和 assistant
generation prompt。该对比衡量两个 checkpoint 的实际使用方式，不是只改变权重的消融实验。

每条生成记录文本、token ID、长度、tokens/s 和是否由 EOS 正常结束。100 条教学子集整体只
计算 teacher-forced response loss，不批量生成 100 份长文本。

SFT 权重通过官方脚本转换并验证 logits 后，继续运行相同七任务。SFT 配置启用官方 chat
template；Pretrain 基线直接读取阶段 4 结果，不重复运行。七任务只作为阶段间回归记录，不参与
数据选择或超参数调整，也不要求全部分数上升。

## 执行顺序

先准备数据与固定教学子集：

```bash
python -m llm_learning.minimind.sft_data
```

再阅读并完整执行 Notebook。确认 response mask 后依次运行：

```bash
python -m llm_learning.minimind.sft_train \
  --config docs/stages/05_sft/configs/overfit100.json

python -m llm_learning.minimind.sft_train \
  --config docs/stages/05_sft/configs/mini64.json
```

中断后使用相同配置和完整 checkpoint 恢复：

```bash
python -m llm_learning.minimind.sft_train \
  --config docs/stages/05_sft/configs/mini64.json \
  --resume checkpoints/minimind/stage5/mini64/latest.pt
```

如需恢复后只完成第一个 epoch，同时保持原定两个 epoch 的 LR schedule，可指定停止 step：

```bash
python -m llm_learning.minimind.sft_train \
  --config docs/stages/05_sft/configs/mini64.json \
  --resume checkpoints/minimind/stage5/mini64/latest.pt \
  --stop-after-step 56480
```

训练完成后执行固定生成、转换与七任务评估：

```bash
python -m llm_learning.minimind.sft_evaluate \
  --config docs/stages/05_sft/configs/mini64.json \
  --generation-config docs/stages/05_sft/configs/generation.json \
  --pretrain-checkpoint checkpoints/minimind/stage4/mini64/weights_step_9908.pth \
  --sft-checkpoint checkpoints/minimind/stage5/mini64/weights_step_56480.pth \
  --manifest outputs/minimind/stage5/mini64/data_manifest.json \
  --output outputs/minimind/stage5/mini64/generation.json

python -m llm_learning.minimind.convert \
  --config docs/stages/05_sft/configs/mini64.json \
  --weights checkpoints/minimind/stage5/mini64/weights_step_56480.pth \
  --output outputs/minimind/stage5/mini64_transformers

python -m llm_learning.minimind.prepare_lm_eval \
  --config docs/stages/05_sft/configs/lm_eval.json \
  --output outputs/minimind/stage5/lm_eval_tasks

python -m llm_learning.minimind.lm_eval \
  --config docs/stages/05_sft/configs/lm_eval.json \
  --model outputs/minimind/stage5/mini64_transformers \
  --task-dir outputs/minimind/stage5/lm_eval_tasks \
  --output outputs/minimind/stage5/lm_eval_results
```

## 测试与完成条件

```bash
pytest -q tests/minimind
```

专项测试覆盖完整数据 schema、固定 row ID、chat template、response mask、causal label shift、
EOS、truncation、padding、SFT 配置加载和 lm-eval chat template。

完成阶段时应能定位：chat template 错误、response mask 错位、assistant 被截断、模型复述
user、loss 很低但生成很差，以及 EOS 学习失败。实验状态见 [`RESULTS.md`](./RESULTS.md)。
