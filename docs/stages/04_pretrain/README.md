# 阶段 4：MiniMind 从头预训练

本阶段从随机参数开始训练 MiniMind Dense，目标是建立可检查、可恢复、可比较的预训练与
评估链路。第一项实验使用 256 条固定数据验证实现，第二项实验使用官方 64M Dense 配置和
mini pretrain 数据。生成质量不作为延长训练的理由。

主要学习入口是：

```text
notebooks/04_minimind_pretraining.ipynb
```

Notebook 按数据进入模型的顺序解释训练流程，并调用 `src/llm_learning/minimind/` 中的实现。
训练产物写入 Git 忽略的 `outputs/` 与 `checkpoints/`；版本化配置和实验结论保留在仓库中。

## 固定输入

MiniMind 源码与 Tokenizer 继续使用阶段 3 的完整 revision：

```text
89d674b8a517010f5561b6d8ab2dcbb58e2fb91b
```

训练数据锁位于 `docs/stages/04_pretrain/configs/data_lock.json`：

```text
repo: jingyaogong/minimind_dataset
revision: 312afb4f76391145c6902f765bb51691c09a12f5
file: pretrain_t2t_mini.jsonl
size: 1241043656 bytes
sha256: 6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c
```

下载器先取得固定 revision，再把文件复制到临时路径，严格校验大小和完整 SHA-256 后才替换
目标文件。已有目标文件不匹配时直接报错，不自动覆盖。

## 两项训练

### 小数据记忆测试

`docs/stages/04_pretrain/configs/overfit256.json` 固定使用原始 row ID 0-255：

| 字段                    | 值                                                      |
| --------------------- | ------------------------------------------------------ |
| 模型                    | Dense，hidden size 256，4 层，4 个 query heads，2 个 KV heads |
| sequence length       | 128                                                    |
| microbatch            | 16 条序列                                                 |
| gradient accumulation | 2                                                      |
| effective batch size  | 32 条序列                                                 |
| precision             | BF16                                                   |
| 最大更新                  | 1,000 optimizer steps                                  |
| 提前停止条件                | 固定 256 条数据上的 loss 不高于 1.0                              |

`overfit256` 只是 profile 名称。这项实验检查 loss 是否能在固定小数据上明显下降。训练和
evaluation 使用同一组 row ID，因此 evaluation 只重新测量完整训练集 loss，不属于 validation
或 test，也不用于估计模型的泛化能力。

### 64M mini pretrain

`docs/stages/04_pretrain/configs/mini64.json` 恢复官方 64M Dense 结构：

| 字段                    | 值                                                      |
| --------------------- | ------------------------------------------------------ |
| 模型                    | Dense，hidden size 768，8 层，8 个 query heads，4 个 KV heads |
| 参数量                   | 63,912,192                                             |
| sequence length       | 340                                                    |
| microbatch            | 32 条序列                                                 |
| gradient accumulation | 8                                                      |
| effective batch size  | 256 条序列                                                |
| precision             | BF16                                                   |
| 训练长度                  | 2 epochs                                               |

seed 42 的确定性排列先取 2,048 行作为 validation，其余行作为 train。周期评估固定读取
validation 的前 256 行，最终评估读取完整 2,048 行。正式运行重新随机初始化，不继承小数据
记忆测试模型。

## 训练实现的关键边界

每条文本独立编码，在两端加入 BOS 与 EOS，并在右侧补 PAD。PAD 对应的 labels 设为 `-100`。
当前 causal LM、独立样本和标准 right padding 条件下，右侧 PAD 不会影响它左侧的有效 token，因此
训练 forward 不传 padding mask。测试比较了独立序列和 right-padding batch 的有效位置 logits；改变
填充方向、拼接方式或 attention 类型后需要重新检查。

训练循环按 optimizer step 计算 cosine learning-rate schedule。梯度累积窗口完成后，依次执行 unscale、gradient
clipping、设置 learning rate、AdamW step 和清空梯度。epoch 末不足一个完整窗口时仍执行尾部
更新，loss 按实际 microbatch 数缩放。

模型参数和 AdamW 状态保持 FP32；CUDA autocast 仅让适合的 forward 算子使用 BF16，减少
激活显存与计算带宽。GradScaler 主要用于 FP16：它放大 loss 以降低小梯度下溢风险，并在
optimizer step 前通过 unscale 恢复梯度尺度。本阶段使用 BF16，因此关闭 GradScaler，unscale
步骤不会实际执行。gradient clipping 的输入是全部参数的梯度；全局 L2 norm 超过 1.0 时，
所有梯度按同一比例缩小，未超过时保持原值，以限制偶发的异常更新。

train loss、validation loss 与 perplexity 均按有效 shifted label token 数加权。训练日志另记录
learning rate、gradient norm、累计训练 token、tokens/s 和 peak allocated GPU memory。

## Checkpoint 与恢复

完整 checkpoint 只在 optimizer update 边界保存，包含：

- 模型、AdamW 与 GradScaler 状态；
- epoch、下一 batch、optimizer step 和累计 token；
- 完整运行配置；
- Tokenizer 文件 hash、词表大小与特殊 token ID；
- 原始数据 hash、Dataset fingerprint、split 规则与 row ID；
- 指标历史；
- Python、CPU 和 CUDA 随机数状态。

恢复前会校验训练超参数、Tokenizer、原始数据 hash 和 split 身份。Dataset fingerprint 由
`datasets` 版本参与计算，只用于记录读取环境，不作为跨版本恢复条件。本地源码、数据与输出目录可以
迁移；源码 revision、Tokenizer hash 与数据 hash 仍必须一致。专项测试比较 uninterrupted 训练
与 save/load 后下一次 AdamW update 的参数，结果必须完全相同。`weights_step_*.pth` 只包含
模型权重，供生成和格式转换使用，不能恢复训练。

## Evaluation

`docs/stages/04_pretrain/configs/generation.json` 固定三条 prompts，以及 greedy decoding 和 sampling
的完整参数。sampling 使用独立 `torch.Generator`。不同 checkpoint 使用相同配置，避免把采样
随机性误判为训练变化。

客观基线通过 MiniMind 官方 `scripts/convert_model.py` 转为 Qwen3 兼容 Transformers 格式。
转换后先比较固定输入的原生模型与转换模型 logits，再使用 `lm-evaluation-harness` 0.4.12 的
标准 `hf` backend 运行固定七任务。任务与底层 Dataset revision 记录在
`docs/stages/04_pretrain/configs/lm_eval.json`。准备脚本从安装的 0.4.12 task YAML 复制本地目录，注入
`dataset_kwargs.revision` 与 `trust_remote_code: true` 并记录目录 hash。授权只用于执行完整 commit
固定的旧式 Dataset scripts。评估命令强制使用该本地路径。本项目不实现任务 prompt、标签和
评分器。

评估环境固定 `datasets` 3.6.0。冻结的 CMMLU 等 revision 仍使用 Dataset scripts；`datasets`
4.0 及以上版本不再支持这类加载方式。训练数据本身是 JSONL，不依赖 remote Dataset script。

## 执行顺序

在项目 `llm` 环境中依次运行：

```bash
python -m llm_learning.minimind.data

python -m llm_learning.minimind.train \
  --config docs/stages/04_pretrain/configs/overfit256.json

python -m llm_learning.minimind.train \
  --config docs/stages/04_pretrain/configs/mini64.json

python -m llm_learning.minimind.prepare_lm_eval \
  --output outputs/minimind/stage4/lm_eval_tasks
```

task 目录已经存在时，脚本会校验配置身份和目录 hash；完全匹配则直接复用，内容不一致才报错。

中断正式训练后使用同一配置恢复：

```bash
python -m llm_learning.minimind.train \
  --config docs/stages/04_pretrain/configs/mini64.json \
  --resume checkpoints/minimind/stage4/mini64/latest.pt
```

生成、转换与七任务评估的完整命令随具体 checkpoint 写入 `RESULTS.md`，避免在训练完成前放入
不存在的文件名。

## 测试

```bash
pytest -q tests/minimind
```

专项测试覆盖配置、数据编码、split、epoch 顺序重建、right padding、token 加权 loss、optimizer-step
学习率、checkpoint 身份检查、精确恢复和独立采样随机数。

实验结果与阶段验收状态见 [`RESULTS.md`](./RESULTS.md)。
