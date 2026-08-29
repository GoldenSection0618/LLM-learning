# 阶段 9 结果

## 当前状态

阶段 9 的固定复现检查和 TensorBoard 指标导入已完成。

## 固定评估

| 指标 | Run 1 | Run 2 | 差值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| validation loss | 1.5630904553 | 1.5630904553 | 0.0 | 通过 |
| perplexity | 4.7735509186 | 4.7735509186 | 0.0 | 通过 |
| greedy token IDs | 两个 prompts 均一致 | 两个 prompts 均一致 | — | 通过 |

固定输入为阶段 5 的 MiniMind 64M Full SFT checkpoint、2,048 条固定 validation rows、
greedy generation config、seed `42`、`cuda:0` 与 `bfloat16`。每次评估处理 852,275 个
validation tokens。两次比较的 `reproduced` 字段为 `true`。

## 运行记录

| 项目 | 结果 |
| --- | --- |
| manifest | `outputs/minimind/stage9/reproduction/manifest_1.json` 与 `manifest_2.json` |
| TensorBoard event | `outputs/tensorboard/stage8_ce_kd/`，由阶段 8 `ce_kd` 的 `metrics.jsonl` 导入 |
| checkpoint resume 状态检查 | 阶段 5 `latest.pt` 包含 model、optimizer、scaler、RNG state、data manifest 与 training state |
| 七任务配置 | 复用阶段 5 冻结配置 |

## 阶段结论

在当前 MiniMind checkpoint、固定 validation rows、greedy generation config、`cuda:0` 与
`bfloat16` 条件下，两次独立固定评估的数值和生成 token IDs 完全一致。该结果说明本阶段的
运行清单、固定评估入口、结果比较和 TensorBoard 指标查看已经形成闭环。
