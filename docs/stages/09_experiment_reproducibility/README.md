# 阶段 9：固化实验规范

本阶段暂停增加训练算法，整理阶段 4～8 已经使用的配置、记录、评估和 checkpoint
规则。主要学习入口是：

```text
notebooks/09_experiment_reproducibility.ipynb
```

## 固定范围

- MiniMind 继续调用现有训练、生成和七任务入口；
- 指标保留 JSONL，同时转换为本地 TensorBoard event；
- 每次正式运行保存一份 `run_manifest.json`；
- Qwen 阶段直接使用 TRL CLI YAML，不新增通用训练框架；
- 复现检查只重复固定 validation 与 greedy generation，不重新正式训练。

## 运行清单

模板位于 `templates/run_manifest.json`，记录以下四类信息：

| 类别 | 内容 |
| --- | --- |
| 输入身份 | commit、模型 checkpoint、数据 revision、split、seed |
| 训练条件 | batch、token 数、trainable parameters、dtype |
| 运行环境 | GPU、peak VRAM、wall time、tokens/s |
| 输出结果 | loss、perplexity、task metrics、generation 与产物路径 |

wall time 和 tokens/s 属于性能观测，不作为复现成功的精确相等条件。

## 复现检查

阶段 9 固定复用阶段 5 MiniMind Full SFT checkpoint。两次运行使用相同 validation row
IDs、generation config 和 seed：

```text
run 1 ─┐
       ├→ 比较 validation loss、perplexity 和 greedy token IDs
run 2 ─┘
```

数值指标使用 `1e-6` absolute tolerance，greedy token IDs 要求完全一致。sampling 结果可以
用于观察，但不作为本次复现验收字段。

依次执行：

```bash
PYTHONPATH=src python -m llm_learning.minimind.reproduce run \
  --config docs/stages/09_experiment_reproducibility/configs/minimind_reproduction.json \
  --run-index 1

PYTHONPATH=src python -m llm_learning.minimind.reproduce run \
  --config docs/stages/09_experiment_reproducibility/configs/minimind_reproduction.json \
  --run-index 2

PYTHONPATH=src python -m llm_learning.minimind.reproduce compare \
  --config docs/stages/09_experiment_reproducibility/configs/minimind_reproduction.json \
  --output outputs/minimind/stage9/reproduction/comparison.json
```

## TensorBoard

将阶段 8 的一组已有指标转换为 TensorBoard event：

```bash
PYTHONPATH=src python -m llm_learning.experiment_record \
  --metrics outputs/minimind/stage8/logit/ce_kd/metrics.jsonl \
  --log-dir outputs/tensorboard/stage8_ce_kd

tensorboard --logdir outputs/tensorboard
```

TensorBoard 用于查看曲线，不替代 JSON summary 与运行清单。

## Qwen 配置样例

`configs/qwen_sft_template.yaml` 只展示下一阶段 TRL CLI 的配置边界。数据名称、模型 revision
和正式 batch 参数在阶段 10 冻结后填写，本阶段不启动 Qwen 训练。

## 完成条件

1. 生成一份实际 MiniMind 运行清单；
2. 将一次现有 JSONL 指标导入 TensorBoard；
3. 连续执行两次固定 validation 与 greedy generation；
4. 对比数值容差、greedy token IDs 和允许波动的性能字段；
5. 能区分算法、数据、模型、随机波动与实现 bug。

结果写入 [`RESULTS.md`](./RESULTS.md)。
