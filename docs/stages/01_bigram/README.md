# 阶段 1：Bigram 语言模型

本目录使用字符级 tokenizer 和 `nn.Embedding(vocab_size, vocab_size)`，实现阶段 1 的完整训练流程。

## 数据

训练命令会下载 Karpathy char-rnn commit `6f9487a` 中固定版本的 Tiny Shakespeare，
并检查完整 SHA-256。程序先编码全文，再将前 90% 用于训练，
最后 10% 用于验证。

## 训练

在仓库根目录运行：

```bash
mamba activate llm
python -m llm_learning.bigram.train
```

用于快速验证的短训练：

```bash
python -m llm_learning.bigram.train \
  --max-steps 100 \
  --eval-interval 25 \
  --eval-batches 10 \
  --generate-tokens 200
```

命令会生成：

```text
outputs/bigram/config.json
outputs/bigram/data_manifest.json
outputs/bigram/tokenizer.json
outputs/bigram/metrics.jsonl
outputs/bigram/generated.txt
checkpoints/bigram/latest.pt
```

这些运行产物已被 Git 忽略。

## 恢复训练

提高目标步数并加载最近的 checkpoint：

```bash
python -m llm_learning.bigram.train \
  --max-steps 2200 \
  --resume checkpoints/bigram/latest.pt
```

checkpoint 会恢复模型权重、优化器状态、训练步数、历史指标和 PyTorch 随机数状态。
阶段 1 尚未保存 DataLoader 在打乱后 epoch 中的精确位置，因此恢复训练可以正常继续，
但无法保证逐位一致。精确恢复数据顺序将在后续的实验可复现阶段处理。

## 错误实验

比较默认学习率和一个故意设置的过大学习率：

```bash
python -m llm_learning.bigram.fault_experiment --steps 200
```

实验结果写入：

```text
outputs/bigram_fault/fault_experiment.json
outputs/bigram_fault/fault_report.md
```

固定实验的观察记录保存在 [`FAULT_OBSERVATION.md`](./FAULT_OBSERVATION.md)。

## 测试

```bash
python -m pytest -q tests/bigram/test_bigram.py
```
