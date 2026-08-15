# 阶段 8 结果

## 当前状态

两条蒸馏链路与五组 Logit Distillation 对照均已完成。

## Sequence-level Distillation

| 项目 | 结果 |
| --- | --- |
| Teacher model ID | `qwen/qwen3.5-9b` |
| 请求数 | 100 |
| 正常结束 | 90 |
| 因长度上限结束 | 10 |
| Math-Verify 通过 | 87 |
| 验证通过率 | 87% |
| Student 训练 / validation | 77 / 10 条 |
| optimizer steps | 100 |
| validation loss | 1.0869 → 0.8498 |
| validation perplexity | 2.96 → 2.34 |
| peak GPU memory | 2.69 GiB |
| wall time | 50.2 s |

## Logit Distillation

每组训练目标均包含其配置指定的 CE/KD 权重；KD 使用 `T² × raw KL`。

| 配置 | T | 初始训练目标 | 最终训练目标 |
| --- | ---: | ---: | ---: |
| CE | 1.5 | 1.5974 | 1.5963 |
| Forward KD | 1.5 | 0.8934 | 0.8537 |
| Reverse KD | 1.5 | 1.0890 | 0.9262 |
| CE + Forward KD | 1.5 | 1.2454 | 1.2318 |
| CE + Forward KD, T=2.0 | 2.0 | 1.5814 | 1.5268 |

下表中的 KL 是除去 `T²` 后的 raw KL。

| 配置 | validation CE | raw Forward KL | raw Reverse KL | peak GPU memory | wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE | 1.5963 | 0.3970 | 0.4843 | 2.55 GiB | 78.3 s |
| Forward KD | 1.6079 | 0.3794 | 0.4401 | 2.73 GiB | 214.8 s |
| Reverse KD | 1.6333 | 0.3963 | 0.4117 | 2.73 GiB | 124.3 s |
| CE + Forward KD | 1.6001 | 0.3838 | 0.4526 | 2.79 GiB | 126.7 s |
| CE + Forward KD, T=2.0 | 1.6080 | 0.3614 | 0.4002 | 2.79 GiB | 127.1 s |

## 阶段结论

- Sequence-level 链路完成了 100 条 Teacher 生成、自动验证和 Student SFT。87% 通过率
  描述这批固定 prompts，不代表完整 GSM8K 能力。
- CE baseline 基本保持初始 CE。纯 Forward KD 将 scaled Forward KD loss 从 0.8934
  降至 0.8537，同时 CE 升至 1.6079，说明贴近 Teacher distribution 与贴近 gold token
  并非同一个目标。
- 纯 Reverse KD 将 scaled Reverse KD loss 从 1.0890 降至 0.9262，但 CE 升至 1.6333；在本次短实验
  中，它对 gold-token 指标的影响最大。
- CE + Forward KD 的最终 raw Forward KL 为 0.3838，CE 为 1.6001，体现混合信号
  在两个目标之间的折中。
- 纯 KD 跳过 CE 后峰值显存约为 2.73 GiB，混合训练约为 2.79 GiB，CE baseline 为
  2.55 GiB。Forward KD 重跑用时 214.8 秒，偏离其余相同规模运行，只记录为本次 wall time。
