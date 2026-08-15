# 阶段 8：MiniMind 模型蒸馏

本阶段学习两种信号层级不同的蒸馏。Sequence-level Distillation 让 Student 学习
Teacher 生成的文本。Logit Distillation 让 Student 在同一个 token 位置上学习 Teacher
的完整词表概率分布。

主要学习入口是：

```text
notebooks/08_minimind_distillation.ipynb
```

## 两条实验链路

| 项目 | Sequence-level Distillation | Logit Distillation |
| --- | --- | --- |
| Teacher | 本地 Qwen3.5-9B Q4_K_M | MiniMind 198M MoE |
| Student | MiniMind 64M Dense | MiniMind 64M Dense |
| Student 看到的 Teacher 信号 | 生成文本 | 每个有效位置的 6,400 维 logits |
| tokenizer 需要相同 | 否 | 是 |
| 学习方式 | 对 Teacher response 执行 SFT | CE 与 KL 组合 |

Qwen GGUF 只进入第一条链路。GSM8K 是文本任务，不加载 `mmproj` 文件。

## Sequence-level Distillation

### 固定数据

`openai/gsm8k` 的 train split 有 7,473 条。划分程序先为每条样本保留原始 row ID，
再用 seed 42 打乱一次：

```text
前 500 条                       reserved development（本阶段不使用）
后 6,973 条                     candidate training area
candidate training area 前 100 条   本阶段实际 Teacher prompts
```

清单写入 `configs/gsm8k_split_v1.json`。后续阶段只读复用这份 row ID 清单。

### 本地 Teacher

LM Studio 加载 `Qwen3.5-9B-Q4_K_M.gguf` 并开启 Local Server。项目访问
`http://localhost:1234/v1`，通过 `/models` 确认已加载的 model ID。配置只保存
model ID 的名称特征，不保存 Windows 或 WSL 绝对路径。

先检查 OpenAI-compatible 接口：

```bash
curl http://localhost:1234/v1/models

curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3.5-9b",
    "messages": [
      {"role": "system", "content": "Solve the math problem briefly. End with the final numerical answer inside \\boxed{}."},
      {"role": "user", "content": "There are 12 apples shared equally by 3 people. How many apples does each person get?"}
    ],
    "temperature": 0,
    "max_tokens": 256,
    "stream": false
  }'
```

WSL 需要使用另一个可访问地址时，通过环境变量覆盖：

```bash
export LM_STUDIO_BASE_URL=http://<windows-host>:1234/v1
```

Teacher 对每个 question 生成 reasoning 和最终 `content`。原始记录分别保存两个字段。
Math-Verify 解析 `content` 中的 `\boxed{...}`、数值或数学表达式，再与 GSM8K `####`
后的 gold answer 判断数学等价；它只验证最终答案，不验证中间推理。
`finish_reason` 为 `stop`、`content` 非空且验证通过的 response 转换为：

```text
user: GSM8K question
assistant: Teacher content
```

生成程序每完成一条就追加到 JSONL。中断后重新执行会跳过已完成 row ID。

### 执行命令

更新环境：

```bash
mamba env update -n llm -f environment.yml
```

在 LM Studio 加载模型并开启 Local Server，然后生成固定划分：

```bash
PYTHONPATH=src python -m llm_learning.minimind.distill_data prepare
```

生成、验证并导出 SFT 数据：

```bash
PYTHONPATH=src python -m llm_learning.minimind.distill_data generate
```

使用阶段 5 Full SFT checkpoint 作为 Student 起点，训练最多 100 个 optimizer steps：

```bash
PYTHONPATH=src python -m llm_learning.minimind.sft_train \
  --config docs/stages/08_distillation/configs/sequence_sft.json
```

实际生成中有 87 条 response 通过 Math-Verify，随后划分为 77 条 Student train 和 10 条
Student validation。500 条 reserved development 没有进入本次 SFT 或 validation loss。
这 100 条 prompts 只用来走通数据闭环，不用于判断 Qwen3.5-9B 或 MiniMind 的整体数学能力。

## Logit Distillation

### 固定 Teacher、Student 与数据

| 字段 | 固定值 |
| --- | --- |
| Student 初始权重 | 阶段 5 自训练 MiniMind 64M Full SFT |
| Teacher | 官方 MiniMind 198M MoE Full SFT |
| 数据 | `sft_t2t_mini.jsonl` |
| train / validation | 4,096 / 256 条 |
| sequence length | 340 |
| effective batch size | 16 |
| optimizer steps | 256 / 组 |
| learning rate | `5e-6` |

Teacher 和 Student 使用同一个 MiniMind tokenizer，因此第 `v` 个 logit 都表示同一个
token ID。Teacher 运行在 `eval()` 和 `no_grad()` 下；Student 接收 gradient。

下载官方 MoE Teacher：

```bash
hf download jingyaogong/minimind-3-pytorch full_sft_768_moe.pth \
  --revision edba70e \
  --local-dir data/minimind
```

### Loss 与对照

对于一个有效 response token，Student 同时可以接收两种信号：

- CE 只指定 gold token；
- KD 使用 Teacher 在 6,400 个 token 上的概率分布。

Forward KL 与 Reverse KL 使用相同的两个分布，交换 KL 两个参数的位置：

- Forward KL：`KL(Teacher || Student)`，Teacher 概率较高的 token 都会推动 Student
  分配概率，倾向于覆盖 Teacher 的分布；
- Reverse KL：`KL(Student || Teacher)`，Student 会重点压低 Teacher 概率很小而自己概率
  较高的 token，倾向于集中到 Teacher 的高概率区域。

两者都只在 response 的有效 label 位置计算，并乘以 `temperature` 的平方。本阶段增加一组
纯 Reverse KD，使 KL 方向成为唯一变量。

本阶段共运行五组：

| 配置 | CE weight | KD weight | KL direction | temperature | 用途 |
| --- | ---: | ---: | --- | ---: | --- |
| `ce.json` | 1.0 | 0.0 | Forward | 1.5 | CE baseline；训练时跳过 Teacher forward |
| `kd.json` | 0.0 | 1.0 | Forward | 1.5 | 纯 Forward KD |
| `reverse_kd.json` | 0.0 | 1.0 | Reverse | 1.5 | 只改变 KL direction |
| `ce_kd.json` | 0.5 | 0.5 | Forward | 1.5 | 混合 CE 与 Forward KD |
| `ce_kd_t2.json` | 0.5 | 0.5 | Forward | 2.0 | 只改变 temperature |

`temperature` 越高，softmax 分布越平缓，Teacher 对非最高概率 token 的相对偏好更容易
被 Student 看到。KD 乘以 `temperature` 的平方，用来补偿高温下 gradient 缩小。

依次运行：

```bash
for profile in ce kd reverse_kd ce_kd ce_kd_t2; do
  PYTHONPATH=src python -m llm_learning.minimind.distill_train \
    --config "docs/stages/08_distillation/configs/${profile}.json"
done
```

中断后单独恢复某组：

```bash
PYTHONPATH=src python -m llm_learning.minimind.distill_train \
  --config docs/stages/08_distillation/configs/ce_kd.json \
  --resume checkpoints/minimind/stage8/logit/ce_kd/latest.pt
```

## 产物

```text
docs/stages/08_distillation/configs/gsm8k_split_v1.json
outputs/minimind/stage8/sequence/teacher_generations.jsonl
data/minimind/gsm8k_teacher_verified.jsonl
checkpoints/minimind/stage8/sequence_sft/
outputs/minimind/stage8/logit/<profile>/
checkpoints/minimind/stage8/logit/<profile>/
```

## 完成条件

阶段完成时需要：

1. 完成 100 条 GSM8K prompts 的 Teacher 生成与自动验证；
2. 用验证通过的 response 完成一次 MiniMind SFT；
3. 完成 CE、Forward KD、Reverse KD、CE + KD 和 temperature 对照；
4. 对比固定 validation 的 CE loss、Forward KL、Reverse KL、显存与用时；
5. 能说明两种蒸馏、KL direction、temperature、tokenizer 对齐、Teacher 错误传递和 Student
   capacity 上限。

实验结果写入 [`RESULTS.md`](./RESULTS.md)。
