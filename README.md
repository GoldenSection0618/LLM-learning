# LLM Learning

本仓库用于按计划学习语言模型训练、后训练和架构研究。项目强调理解数据流、亲手检查关键张量，并为每个阶段留下可以复现的代码和实验记录。

主计划见 [LLM 自学计划](./LLM自学计划.md)。模型、数据和工具版本见 [模型与数据来源](./模型与数据来源.md)。

## 学习原则

1. 按计划顺序推进，不提前同时引入多套模型、算法或框架。
2. 每个阶段先满足验收标准，再决定是否执行拓展练习。
3. 生成质量不属于验收条件时，不通过延长训练追求更好的样例。
4. 当前阶段需要理解的核心代码放在仓库中实现。后续阶段按计划复用成熟框架，不重复开发训练基础设施。
5. Notebook 用于驱动项目库、观察中间结果和记录结论。可复用实现放在 `src/`，不在 Notebook 中复制整套模型。
6. 解释概念时先说明输入、输出和作用，再解释实现细节。涉及 Tensor 时写清 shape，并给出小例子。
7. 新术语在第一次出现时解释清楚。避免只描述代码做了什么而省略设计原因。

## 分支规则

项目包含三套里程碑：

| 命名空间 | 里程碑 |
| --- | --- |
| `core` | 1：阶段 0～2；2：阶段 3～5；3：阶段 6 与阶段 8；4：阶段 9；5：阶段 10；6：阶段 12 |
| `distill` | 1：Teacher 数据闭环；2：序列蒸馏主结论；3：验证与结项 |
| `architecture` | 1：GPT-2 基线可信；2：Checkpoint 迁移闭环；3：OLMo 3 候选冻结；4：正式架构结论 |

- `main` 只标记已经完成验收的里程碑，不承载日常开发。
- 分支格式为 `milestone/<namespace>-<number>-<slug>`。
- Tag 格式为 `<namespace>-milestone-<number>`。
- 三套分支分别使用 `milestone/core-*`、`milestone/distill-*` 和 `milestone/architecture-*`。
- 每个里程碑分支从最新的 `main` 创建。达到验收标准并完成复习后，分支合入 `main`，随后创建对应 tag。
- 新里程碑在前一个里程碑进入 `main` 后创建。
- 同一里程碑内的连续阶段保留在同一个分支，不为每个小修改建立分支。
- 阶段 12 使用四个 `architecture` 分支推进。架构里程碑四完成后，在同一个 `main` 提交上标记 `core-milestone-6`。
- 延期的独立学习专题使用 `study/<topic>` 长期分支。专题完成前不合入 `main`，也不阻塞里程碑验收。

当前里程碑的关闭流程：

```text
milestone/core-1-minimal-training
→ 合入 main
→ 创建 core-milestone-1
→ 从 main 创建 milestone/core-2-minimind-training
→ 开始阶段 3
```

## Commit 规则

- Commit message 使用简短英文文本。
- 不使用 `chore:`、`feat:`、`fix:` 等 Conventional Commits 前缀。
- 一次 commit 只表达一个清楚的改动主题。
- 提交前检查 `git status`、暂存区范围和 `git diff --check`。
- 代码改动需要运行与风险相匹配的测试。Notebook 需要检查 JSON、执行状态和错误输出。
- 用户明确要求提交后再 commit，不自动提交工作区改动。
- 用户指定文件范围时，只提交该范围内的文件，不顺带提交其他改动。
- 需要修改上一条提交时使用 amend，并再次确认最终 commit 内容。

## 目录规则

```text
.
├── notebooks/                 # 分阶段的学习 Notebook
├── src/llm_learning/          # 可复用的 Python 实现
├── tests/                     # 与 src 模块对应的测试
├── docs/stages/               # 阶段说明、结果和故障记录
├── third_party/minimind/       # 固定 revision 的 MiniMind submodule
├── data/                      # 下载的数据，不提交
├── checkpoints/               # 训练状态，不提交
├── outputs/                   # 指标、生成文本和模型产物，不提交
├── environment.yml            # Conda 环境
└── pyproject.toml             # Python 包配置
```

- Python 包使用 `src` layout。新模块放在 `src/llm_learning/<topic>/`。
- 测试放在 `tests/<topic>/`，目录名称与实现模块对应。
- Notebook 文件名使用 `<两位阶段编号>_<topic>.ipynb` 模板。
- 阶段文档放在 `docs/stages/<stage>_<topic>/`，说明和结果分别写入 `README.md` 与 `RESULTS.md`。
- 项目计划、跨阶段约束和数据来源文档保留在仓库根目录。
- 需要直接阅读和调用的第三方源码通过 Git submodule 固定 revision，不复制进项目源码包。
- `data/`、`checkpoints/` 和 `outputs/` 只保存本地产物，由 `.gitignore` 排除。
- `__pycache__`、`.pytest_cache`、Notebook checkpoint 和模型权重不进入 Git。

## Python 代码风格

- 代码优先保证清楚和易读，不压缩多行表达式。
- 函数调用、参数列表和复杂表达式按逻辑换行，保持统一缩进。
- 使用有含义的变量名。单字母名称只用于通用数学维度。
- 公共函数和类使用类型标注。
- 项目内 docstring 使用简短中文。
- 注释使用中文，只解释关键目的、边界或容易误解的行为。
- 注释保持稀疏。代码已经清楚表达的内容不重复解释。
- 已有关键注释默认保留。修改逻辑时同步更新失效注释。
- 随机种子写入配置。种子只用于复现，不代表行业固定值。
- 生成文本使用独立的 `torch.Generator`，避免消耗训练使用的随机数序列。

## Notebook 规则

- Notebook 调用 `src/llm_learning` 中的实现，并展示关键输入、输出和 Tensor shape。
- 每个关键代码单元通常保留一条简短中文注释。复杂训练循环可以按阶段保留少量注释。
- Notebook 提交前应完整执行。所有代码单元需要有执行记录，且不能包含错误输出。
- 数学变量、公式和抽象 Tensor shape 使用 `$...$` 或 `$$...$$` 渲染；反引号只用于真实代码标识、字段值和命令。
- 固定随机种子只能支持特定条件下的复现。描述结果时同时说明模型权重、输入、生成参数和运行环境等条件。
- 实验只能支持当前输入、当前配置和当前实现时，结论需要明确限定范围。
- 一个测试样例用于提供检查证据，不写成对全部输入的完整证明。
- 问题与答案分行排版，答案放在问题下方。
- Notebook 以陈述句为主，禁用先否定再转折的固定句式。
- 不使用夸张结论。结论需要说明适用的输入、配置、实现或运行条件。

## 写作风格

- README、阶段文档、Notebook 说明和代码注释使用中文。
- Commit message 使用英文。
- 除硬规则和模板规则外，README 不写具体例子。教学示例放在对应的阶段文档或 Notebook。
- 语言贴近教材：短句、直接、准确。
- 避免文言化表达、翻译腔、绕口句和含义不清的指代。
- 一个长段落包含多个独立概念时，拆成短段落或有顺序的列表。
- 先定义符号，再使用符号。相同字母表示不同概念时增加具有语义的下标。
- 含义相近的术语必须分别定义，并保持用法一致。
- 专有名词优先使用社区通用英文；只有存在稳定、常用的中文译名时才使用中文。
- 结论强度必须与证据一致。单次实验不扩展成普遍规律，趋势不写成保证。
- 技术事实发生变化时同步更新 Notebook、阶段文档和结果记录。

## 数据与实验规则

- 数据和模型来源使用固定 revision，并记录可验证的版本标识。
- 下载文件使用完整 SHA-256 精确比较。替换异常文件前先下载并验证临时文件。
- Train 和 validation 在创建滑动窗口 Dataset 前切分，避免样本跨越边界。
- 训练 Loader 可以打乱样本索引。观察用 Loader 和完整 validation Loader 保持固定顺序。
- 周期评估使用固定验证子集，正式结论使用完整 validation 或冻结的客观评估。
- 同一图中的指标计算方式不同时，需要分别说明样本范围、模型模式和统计窗口。
- 需要恢复训练的 checkpoint 应保存模型、optimizer、step、配置、tokenizer、指标历史和随机数状态。
- 评估使用 `model.eval()` 和 `torch.no_grad()`。两者用途不同，文档中分别说明。
- 故障实验只改变一个目标变量，并保留正常对照。

## 环境与常用命令

项目使用名为 `llm` 的 Conda 环境，并通过 Mamba 管理：

```bash
mamba env update -n llm -f environment.yml
mamba run -n llm pytest -q
```

`environment.yml` 中的 `pip: -e .` 会以 editable 模式安装本项目。修改 `src/llm_learning` 后无需重复安装。

开始新的模型或数据阶段前，先确认计划要求、磁盘空间和运行环境。未进入对应阶段时，不提前下载大型模型或数据集。
