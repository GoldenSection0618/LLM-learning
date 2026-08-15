# 阶段 6：MiniMind LoRA

本阶段把 LoRA（Low-Rank Adaptation）作为 Full SFT 的参数高效替代方式。LoRA 在选定的原
Linear 旁增加两块低秩矩阵 A、B；两者合称 adapter。rank 是 A、B 之间的中间维度，target
modules 是接收 adapter 的原 Linear。学习重点还包括参数冻结、adapter 保存与 merge。SFT 数据
格式、response mask、validation 和生成规则复用阶段 5。

主要学习入口是：

```text
notebooks/06_minimind_lora.ipynb
```

## 当前实现

Core 2 已合入本分支。`lora_train.py` 复用阶段 5 的 SFT Dataset、split、validation 和 checkpoint
循环，在 Base checkpoint 加载后注入 LoRA，并让 optimizer 只接收 adapter parameters。训练过程
导出 adapter，完整 resume checkpoint 保存中断位置；训练结束后自动验证 merge 前后 logits 并
导出 merged weights。

## 固定对照

LoRA 与阶段 5 Full SFT 使用相同的 pretrained checkpoint、SFT 数据、train/validation split、
sequence length、batch size 和实际 optimizer steps。这样可以直接比较两种参数更新方式。

| 字段 | 固定值 |
| --- | --- |
| Base checkpoint | `checkpoints/minimind/stage4/mini64/weights_step_9908.pth` |
| Base SHA-256 | `3b9f475ca67a8fb98824633b9e93fe8577d9faf5bd19d8da250c48bf8889f06e` |
| SFT 数据 | `sft_t2t_mini.jsonl` |
| train / validation | 903,670 / 2,048 条 |
| sequence length | 768 |
| batch size | 16 |
| precision | BF16 |
| 实际 optimizer steps | 56,480 |
| rank | 16 |
| target modules | 每层的 `q_proj`、`o_proj` |
| learning rate | `1e-4` |

阶段 5 Full SFT 使用 `1e-5`，本阶段按 MiniMind LoRA 默认值使用 `1e-4`。因此该实验是固定实用
配置的对照，不是只改变参数更新方式的单变量消融实验。

LoRA 从阶段 4 Pretrain 权重开始。阶段 5 Full SFT checkpoint 只作为对照结果，不作为 LoRA 的
初始权重。

## 从 Linear 到 LoRA

设原 Linear 权重的 shape 为 `[D_out, D_in]`。LoRA 增加两个不含 bias 的 Linear：

```text
A.weight: [rank, D_in]
B.weight: [D_out, rank]

output = W(x) + B(A(x))
```

原权重 `W` 冻结，只训练 A 和 B。当前 MiniMind 官方实现不额外使用 `alpha / rank` scaling；本
阶段保持相同定义，不把其他 LoRA 库的默认值混入实验。

A 使用小随机数初始化，B 初始化为零。因此训练开始前 `B(A(x))` 为零，注入 LoRA 不改变 Base
模型输出。第一次 backward 通常先给 B 产生非零 gradient；B 更新后，A 才开始获得有效 gradient。

## Target modules

MiniMind 官方实现会为输入维度和输出维度相同的 Linear 注入 LoRA。在当前 64M Dense 配置中，
这条规则选中每层 attention 的 `q_proj` 和 `o_proj`，共 16 个 Linear。

本项目在配置中显式记录这两个名称，并在训练前输出完整模块列表。模型结构变化导致实际列表不
一致时直接报错，避免 target modules 静默变化。

rank 16 时，每个 `[768, 768]` Linear 增加：

```text
16 × 768 + 768 × 16 = 24,576 parameters
```

16 个目标 Linear 共增加 393,216 个 LoRA parameters。真实 64M capacity check 已重新枚举完整
target names，并从模型对象得到相同计数。

## 显存边界

LoRA 不为冻结的 Base weights 保存 parameter gradients 和 AdamW optimizer states，因此通常降低
训练显存。Transformer forward、activation 保存和传播到更早 adapter 的 backward 仍然存在，
所以显存和吞吐变化必须实测，不按 trainable parameter 比例推算。

## Adapter、merge 与 checkpoint

三个文件用途不同：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| adapter weights | 只含 A、B | 与相同 Base checkpoint 组合推理 |
| merged weights | `W + BA` 后的完整模型 | 按普通 Base 模型加载和转换 |
| resume checkpoint | 模型、optimizer、step、配置、数据状态和 RNG | 中断恢复训练 |

adapter 元数据需要记录 Base SHA-256、rank 和完整 target module 列表。加载时逐项验证。merge 完成
后使用固定输入比较 merge 前后 logits；通过容差检查后再转换 Hugging Face checkpoint。

## 实验与评估

阶段 6 只运行一个 rank 16 正式实验，不增加 rank sweep 或新领域数据。对比以下实测指标：

- trainable parameters；
- peak allocated GPU memory；
- tokens/s 与 wall time；
- 固定 full validation loss；
- adapter 文件大小；
- 固定 prompts 的输出长度、EOS 和文本；
- MiniMind 七任务结果；
- 相对 Pretrain 与 Full SFT 的变化。

阶段 4 与阶段 5 已有结果直接复用，不重复训练或评估。单次 MiniMind 实验只支持当前模型、数据、
rank 和训练预算下的结论，不扩展成 LoRA 的普遍效果保证。

## 执行顺序

先运行专项测试：

```bash
PYTHONPATH=src pytest -q tests/minimind/test_minimind_lora.py
```

完整运行 Notebook：

```text
notebooks/06_minimind_lora.ipynb
```

正式训练前可重复执行一次 capacity check。它使用真实 64M Base weights、batch size 16、sequence
length 768，完成一次 forward、backward、gradient clipping 和 AdamW step，不写训练产物：

```bash
PYTHONPATH=src python -m llm_learning.minimind.lora_train \
  --config docs/stages/06_lora/configs/rank16.json \
  --capacity-check
```

正式 rank-16 LoRA 只运行一轮：

```bash
PYTHONPATH=src python -m llm_learning.minimind.lora_train \
  --config docs/stages/06_lora/configs/rank16.json
```

中断后从完整 checkpoint 恢复：

```bash
PYTHONPATH=src python -m llm_learning.minimind.lora_train \
  --config docs/stages/06_lora/configs/rank16.json \
  --resume checkpoints/minimind/stage6/rank16/latest.pt
```

训练入口会生成：

- `adapter_step_*.pth`：只含 LoRA A、B 和 adapter 身份；
- `latest.pt`：包含 Base、adapter、optimizer、step、数据状态和 RNG 的 resume checkpoint；
- `merged_weights_step_*.pth`：merge 后可按普通 MiniMind 权重加载的完整模型。

本机 capacity check 的 peak allocated GPU memory 为 4,792,367,616 bytes，约 4.46 GiB。正式
56,480-step 训练实际用时约 6 小时 23 分钟。

三方固定生成复用阶段 5 已保存的 Pretrain 与 Full SFT 文本，只对 LoRA merged weights 运行同一组
8 个 prompts、greedy 和固定 seed sampling：

```bash
PYTHONPATH=src python -m llm_learning.minimind.lora_evaluate
```

结果写入 `outputs/minimind/stage6/rank16/generation.json`，顶层分别保存 `pretrain`、`full_sft` 和
`lora`。

将 merged weights 转成 `lm-evaluation-harness` 可加载的 Transformers 目录，并自动验证转换前后
logits：

```bash
PYTHONPATH=src python -m llm_learning.minimind.convert \
  --config docs/stages/05_sft/configs/mini64.json \
  --weights checkpoints/minimind/stage6/rank16/merged_weights_step_56480.pth \
  --output outputs/minimind/stage6/rank16_transformers
```

这里复用阶段 5 的模型结构配置；输入权重是阶段 6 merge 后的 LoRA 权重。

训练完成后再按以下顺序收尾：

1. 运行 LoRA 七任务评估；
2. 将生成与七任务结果写入 Notebook 和 `RESULTS.md`。

## 完成条件

专项测试最终覆盖真实 MiniMind 注入、参数冻结、optimizer 参数范围、adapter 严格加载、merge 数值
一致性和 resume 连续性。

完成阶段时应能说明：LoRA 节省显存的来源、rank 的作用、target modules 的选择方式、adapter 与
merged weights 的区别，以及 LoRA 与 QLoRA 的区别。实验状态见 [`RESULTS.md`](./RESULTS.md)。
