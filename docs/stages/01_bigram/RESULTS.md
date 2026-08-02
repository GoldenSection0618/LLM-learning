# Bigram 基线结果

## 固定实验配置

```text
随机种子: 1337
模型: nn.Embedding(65, 65)
上下文长度: 128
批量大小: 64
学习率: 0.01
训练步数: 2,000
设备: NVIDIA GeForce RTX 4060 Laptop GPU
```

Tiny Shakespeare 来自固定的 `6f9487a` 版本。完整 SHA-256 为：

```text
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

按顺序切分后，训练集包含 1,003,854 个字符，验证集包含 111,540 个字符。
字符词表包含 65 个 token。

## 损失

| 训练步数 | 训练损失 | 验证损失 |
| ---: | ---: | ---: |
| 1 | 4.7098 | 4.7105 |
| 200 | 2.9149 | 2.9272 |
| 400 | 2.5586 | 2.5804 |
| 600 | 2.4984 | 2.5239 |
| 800 | 2.4800 | 2.5064 |
| 1,000 | 2.4716 | 2.4985 |
| 1,200 | 2.4668 | 2.4938 |
| 1,400 | 2.4639 | 2.4908 |
| 1,600 | 2.4617 | 2.4903 |
| 1,800 | 2.4601 | 2.4892 |
| 2,000 | 2.4591 | 2.4859 |

两项损失持续下降，随后逐渐接近 Bigram 模型的能力上限。

## 生成样本

```text
ThabotNCHu hit OPr dith inge n! vehaiefthes me braimand Hor n'd rNGLAR:
wh ncat t n. LLO t g t fals mavear CLInge d f thanguraneis ufoorsorureareszy
te nope.
```

输出已经学到字符转移、近似单词的片段、标点和换行。Bigram 模型无法维持更长的上下文和连贯语义。

## Checkpoint 恢复

一次独立的冒烟测试在第 20 步保存状态，加载 `latest.pt` 后从第 20 步继续，
并完成至第 25 步。恢复后的验证损失从 4.4518 继续下降到 4.3878。
