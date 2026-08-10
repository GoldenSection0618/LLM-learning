# TinyGPT 阶段 2 结果

## 4,000 步训练

实验使用默认 3,225,600 参数模型、Tiny Shakespeare 固定切分、seed 2026 和 RTX 4060
Laptop GPU。周期性 loss 使用一次选定后保持不变的评估子集。

| 训练步数 | 训练损失 | 验证损失 |
| ---: | ---: | ---: |
| 1 | 3.6890 | 3.6996 |
| 200 | 2.3502 | 2.3598 |
| 400 | 1.9721 | 2.0550 |
| 600 | 1.7443 | 1.9050 |
| 800 | 1.6108 | 1.7949 |
| 1,000 | 1.5272 | 1.7257 |
| 1,200 | 1.4644 | 1.6710 |
| 1,400 | 1.4176 | 1.6273 |
| 1,600 | 1.3797 | 1.5967 |
| 1,800 | 1.3561 | 1.5692 |
| 2,000 | 1.3297 | 1.5604 |
| 2,200 | 1.3070 | 1.5415 |
| 2,400 | 1.2879 | 1.5331 |
| 2,600 | 1.2722 | 1.5253 |
| 2,800 | 1.2604 | 1.5173 |
| 3,000 | 1.2458 | 1.5069 |
| 3,200 | 1.2318 | 1.5154 |
| 3,400 | 1.2185 | 1.5059 |
| 3,600 | 1.2086 | 1.5005 |
| 3,800 | 1.1967 | 1.5044 |
| 4,000 | 1.1843 | 1.4983 |

第 4,000 步使用当前模型完整遍历 validation Dataset，得到 loss `1.4882`，低于
2,000 步时的 `1.5517`。周期性 validation loss 在 3,200 和 3,800 步短暂上升，随后
继续下降。当前仍未出现持续反弹，因此没有明确的过拟合拐点。Train 与 validation 的
差距继续扩大，说明泛化差距正在增加；继续训练时需要用完整 validation loss 判断最低点。

## 生成样本

```text
The povertize of us, Lord of Hereford, England,
As thy house is the quiest of English caest,
Yet coldly we consuls our ignorance did emperate
Of our friends. I acquaint you, gentle Bridkendon,
That death he, or else you shall be thus gone.
Go obedience, the pretty of faults,
Did tell you, a royal m
```

模型已经学到角色名、换行、标点和局部句式。阶段 2 不使用生成质量作为验收条件。

## 完整 forward 张量流

记录使用 $B=2$、$T=128$、$D=256$、$H=4$、$d_{\mathrm{head}}=64$ 和 $V=65$：

| Tensor | 形状 |
| --- | --- |
| input IDs | $2\times128$ |
| token embedding | $2\times128\times256$ |
| position embedding | $128\times256$ |
| embedding | $2\times128\times256$ |
| block 0 attention Q/K/V | $2\times4\times128\times64$ |
| block 0 attention scores/weights | $2\times4\times128\times128$ |
| block 0 attention output | $2\times128\times256$ |
| block 0～3 hidden states | $2\times128\times256$ |
| final hidden states | $2\times128\times256$ |
| logits | $2\times128\times65$ |
| loss | scalar |

四个 block 的 Q、K、V 和 attention score 形状相同。完整逐层记录由训练命令写入
`outputs/tinygpt/forward_trace.json`。
