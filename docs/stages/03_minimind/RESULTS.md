# MiniMind 阶段 3 初始检查结果

## 学习投入

本阶段实际主动投入 20 小时，高于计划中的 5～6.5 小时。新增时间主要用于补习 Transformer
相关理论并逐行理解代码细节，包括 Embedding 与 LM Head、Pre-Norm 残差路径、RMSNorm、
GQA 与 KV Cache、attention mask，以及训练循环中的学习率和 checkpoint 语义。这部分投入补齐了
进入 MiniMind 训练前所需的结构基础。

## 实验条件

```text
MiniMind revision: 89d674b8a517010f5561b6d8ab2dcbb58e2fb91b
PyTorch: 2.9.1
设备: NVIDIA GeForce RTX 4060 Laptop GPU（cuda:0）
参数 dtype: torch.float32
随机种子: 2026
input_shape: [2, 8]
hidden_size: 256
num_hidden_layers: 4
num_attention_heads: 4
head_dim: 64
vocab_size: 6,400
dropout: 0
模型模式: eval
```

`model.eval()` 关闭 dropout 等训练模式行为，使两条 forward 路径处于相同的确定性条件。
`torch.no_grad()` 不记录梯度和计算图，减少本次只读 forward 的内存开销。两者作用不同。

## 参数量与张量流

| 配置 | KV heads | 参数量 | Q shape | K/V shape |
| --- | ---: | ---: | --- | --- |
| GQA | 2 | 4,983,552 | $2\times8\times4\times64$ | $2\times8\times2\times64$ |
| MHA | 4 | 5,245,696 | $2\times8\times4\times64$ | $2\times8\times4\times64$ |

两种配置的 embedding、四层 block 输出和 logits shape 相同：

| Tensor | 形状 |
| --- | --- |
| input IDs | $2\times8$ |
| embedding | $2\times8\times256$ |
| block 0～3 | $2\times8\times256$ |
| logits | $2\times8\times6{,}400$ |

GQA 比 MHA 少 262,144 个参数。这一差异来自四层 attention 中更小的 K/V 投影；
Q 投影、输出投影和 block 外部的 hidden state shape 不变。

## KV Cache

关闭 `use_cache` 时，四层返回的 cache 项均为 `None`。开启 cache 后，第一层原生保存的
K/V shape 如下：

| 配置 | 7-token prefix | 增加 1 token 后 |
| --- | --- | --- |
| GQA | $2\times7\times2\times64$ | $2\times8\times2\times64$ |
| MHA | $2\times7\times4\times64$ | $2\times8\times4\times64$ |

cache 保存的是调用 `repeat_kv` 之前的 K/V，因此在相同 batch、序列长度和 head dim 下，
GQA/MHA 的 cache 比例等于 $H_{\mathrm{kv}}/H_{\mathrm q}=2/4=0.5$，所以 GQA 的 K/V
元素数是 MHA 的一半。这里的 $H_{\mathrm{kv}}$ 是 KV head 数；若每 $g$ 个 query heads
共享一组 K/V，也可以把比例写成 $1/g$。attention 计算前，GQA 再把两个 KV heads
映射到四个 query heads。

使用相同输入比较完整 forward 与“7-token prefix cache + 1 个增量 token”，最后位置
logits 的最大绝对差为：

$$
\Delta_{\mathrm{GQA}}=5.364418029785156\times10^{-7},\qquad
\Delta_{\mathrm{MHA}}=4.76837158203125\times10^{-7}.
$$

在当前 RTX 4060、float32、PyTorch、随机权重、输入和配置下，两条计算路径在浮点误差范围内
一致。这项检查为本次样例中的 cache 拼接提供一致性证据，不扩展为对全部输入和设备的
完整证明。
