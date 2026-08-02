# 阶段 2：最小 Decoder-only Transformer

本阶段在 Tiny Shakespeare 字符数据上实现一个 3.2M 参数的 TinyGPT。数据切分、
字符 tokenizer 和 DataLoader 沿用阶段 1，新增代码集中于 Transformer forward。

## 实现范围

模型包含：

1. token embedding
2. learned absolute position embedding
3. 手写 multi-head causal self-attention
4. 两层 GELU MLP
5. Pre-LayerNorm 与 residual connection
6. linear language-model head 与 cross entropy loss

本阶段不包含 RoPE、GQA、FlashAttention、MoE、KV Cache 和 LoRA。
当前 Dataset 返回等长文本片段，batch 中没有 padding，因此 attention 只使用 causal mask。

默认结构如下：

```text
vocab_size: 65
block_size: 128
d_model: 256
num_heads: 4
head_dim: 64
num_layers: 4
parameters: 3,225,600
```

## 代码入口

核心文件：

```text
src/llm_learning/tinygpt/config.py
src/llm_learning/tinygpt/model.py
src/llm_learning/tinygpt/train.py
```

建议先按顺序阅读：

1. `ModelConfig`
2. `CausalSelfAttention`
3. `FeedForward`
4. `TransformerBlock`
5. `TinyGPT`
6. `train`

Notebook 会调用这些实现并逐步观察张量：

```text
notebooks/02_tinygpt_forward_and_training.ipynb
```

## 训练

在仓库根目录执行完整默认训练：

```bash
mamba activate llm
python -m llm_learning.tinygpt.train
```

用于检查全流程的 200 步短训练：

```bash
python -m llm_learning.tinygpt.train \
  --max-steps 200 \
  --eval-interval 50 \
  --eval-batches 10 \
  --generate-tokens 100
```

运行产物位于 `outputs/tinygpt/`：

```text
config.json
data_manifest.json
tokenizer.json
forward_trace.json
metrics.jsonl
final_metrics.json
model.pt
generated.txt
```

这些训练产物已由 `.gitignore` 忽略。阶段 2 只要求 loss 下降，生成质量不属于验收条件。

## 测试

```bash
python -m pytest -q tests/tinygpt/test_tinygpt.py
```

测试覆盖 shape、因果遮罩、未来信息隔离、超长生成上下文裁剪、参数规模、单批次过拟合
和 validation loss 的 token 加权。
