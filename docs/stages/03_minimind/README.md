# 阶段 3：阅读 MiniMind Dense 模型

本阶段对照 TinyGPT 阅读 MiniMind 的 Dense 模型。第一遍不进入 MoE、SFT、DPO 和
RL 代码，只跟踪一次 Dense forward 以及自回归推理时的 KV Cache。

主要学习入口是：

```text
notebooks/03_minimind_dense_architecture.ipynb
```

Notebook 按本页列出的顺序展示和解释固定版本的官方源码，并调用项目 `src` 中的检查
工具运行实验。`src/llm_learning/minimind/inspect.py` 是 Notebook 复用的实验工具，
不作为第一阅读入口。

## 固定源码

官方仓库与完整 revision：

```text
https://github.com/jingyaogong/minimind.git
89d674b8a517010f5561b6d8ab2dcbb58e2fb91b
```

官方源码以 Git submodule 固定在 `third_party/minimind/`。首次克隆本项目时使用：

```bash
git clone --recurse-submodules <本项目地址>
```

已有工作目录使用：

```bash
git submodule update --init third_party/minimind
```

检查工具会验证完整 revision，不接受其他版本。

## 阅读顺序

按一次训练数据流阅读：

1. `MiniMindConfig`
2. `RMSNorm`
3. `Attention`
4. `FeedForward`
5. `MiniMindBlock`
6. `MiniMindModel`
7. `MiniMindForCausalLM`
8. `PretrainDataset`
9. `trainer/train_pretrain.py`

主干数据流为：

$$
\mathbf{X}\in\mathbb{N}^{B\times T}
\rightarrow \mathbf{H}^{(0)}\in\mathbb{R}^{B\times T\times D}
\xrightarrow{\mathrm{Dense\ blocks}}
\mathbf{H}^{(L)}\in\mathbb{R}^{B\times T\times D}
\rightarrow \mathbf{Z}\in\mathbb{R}^{B\times T\times V}
\rightarrow \mathcal{L}\in\mathbb{R}.
$$

每个 Dense block 使用 Pre-Norm。令 block 输入为 $\mathbf{x}$，其两条残差路径为：

$$
\mathbf{x}_{\mathrm{attn}}
=\mathbf{x}+\operatorname{Attention}(\operatorname{RMSNorm}(\mathbf{x})),
$$

$$
\mathbf{x}_{\mathrm{out}}
=\mathbf{x}_{\mathrm{attn}}
+\operatorname{MLP}(\operatorname{RMSNorm}(\mathbf{x}_{\mathrm{attn}})).
$$

第二次 RMSNorm 位于第一次 residual 之后、MLP 之前；归一化结果只进入 MLP 分支，
不会覆盖残差主干。最后经过 final RMSNorm 与 tied LM head。

## 缩小模型检查

检查配置固定为 $D=256$、$L=4$、$H_{\mathrm q}=4$。GQA 使用
$H_{\mathrm{kv}}=2$，MHA 使用 $H_{\mathrm{kv}}=4$。随机输入取 $B=2$、$T=8$。

```bash
python -m llm_learning.minimind.inspect \
  --source-dir third_party/minimind \
  --device cuda \
  --output outputs/minimind/stage3_inspection.json
```

脚本通过 forward hook 记录参数量、逐层输出、Q/K/V shape、关闭 cache 时的返回值、cache 长度增长，
以及完整 forward 和增量 forward 最后一个位置 logits 的最大绝对差。

MiniMind 模型本身已经实现 KV Cache，本阶段不需要部署 vLLM 等推理服务。相同 batch、
序列长度、head dim 和 query head 数下，GQA/MHA 的 KV Cache 比例为
$H_{\mathrm{kv}}/H_{\mathrm q}$；当前配置为 $2/4=0.5$。

## 与 TinyGPT 的结构差异

| 结构 | TinyGPT | MiniMind Dense | 主要作用 |
| --- | --- | --- | --- |
| 位置编码 | learned absolute embedding | RoPE | 将相对位置信息作用在 Q/K 上，并支持更灵活的上下文长度 |
| normalization | LayerNorm | RMSNorm | 稳定子层输入尺度，并省略均值中心化 |
| attention heads | MHA | GQA | 多个 query heads 共享较少的 K/V heads，缩小 KV Cache |
| Q/K 处理 | 无额外归一化 | QK-Norm | 控制 attention logits 的数值尺度 |
| attention kernel | 手写 score/softmax | SDPA | 交给 PyTorch 选择更高效的 attention kernel |
| MLP | GELU 两层 MLP | SwiGLU 风格 gated FFN | 使用 SiLU 门控分支调节送入降维投影的信息 |
| 推理状态 | 每步重算上下文 | KV Cache | 复用历史 token 的 K/V，增量解码只计算新 token |

阶段实测结果见 [`RESULTS.md`](./RESULTS.md)。

## 训练入口的阶段边界

本阶段只阅读并记录官方训练入口的准确行为。`AdamW` 负责参数更新，`get_lr` 负责手写余弦
学习率；梯度累积期间只有 `optimizer.step()` 边界上的学习率真正参与更新。checkpoint 默认每
1,000 个数据 batch 和每个 epoch 末保存，固定文件名会覆盖旧文件。

当前源码可能在非梯度累积边界保存，并在 epoch 末剩余参数更新之前保存；resume 文件也没有
覆盖根 README 要求的全部 config、tokenizer、指标历史和随机数状态。这些问题登记为阶段 4
的实现与恢复测试任务，本阶段不修改固定 revision 的第三方源码。

## 测试

```bash
python -m pytest -q tests/minimind/test_minimind_inspect.py
```

专项测试使用最小假模型验证 forward hook、逐层 shape、KV Cache 长度增长和完整/增量
forward 对比的记录逻辑。固定 revision 的官方模型由完整执行的 Notebook 提供集成检查。
