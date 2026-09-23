# AGENTS.md

## 1. 项目概述

* 项目名称：holypoly
* 项目目标：从 Oxford Nanopore cDNA 原始信号中定位 Poly 区域，并估计其长度。
* Poly 类型：poly(A) 与反向互补方向的 poly(T)。
* 测序平台：Oxford Nanopore
* 测序类型：cDNA
* 当前代码范围：外部数据诊断、训练 manifest、Bonito v5 辅助 head、冻结
  encoder 训练和小规模 pipeline 验证。
* 项目流程：数据整理 -> 标签审计 -> 算法开发 -> 模型训练 -> 独立验证。

当前只支持 Bonito v5 Transformer。不要在没有明确任务的情况下扩展到旧版
Bonito RNN、CTC 模型、多 GPU 或 distributed training。

### 1.1 科学问题

本项目希望解决以下问题：

1. 从原始电流信号中定位 Poly 区域。
2. 估计 Poly 区域长度，单位为 nt 或 base。
3. 将预测结果与已知构建长度、Dorado 等其他工具的结果比较。
4. 分析不同样本、物种和测序批次对识别结果的影响。

### 1.2 标签与长度定义

cDNA read 可能以两个方向出现，因此目标区域可能表现为 poly(A) 或其反向互补
形式 poly(T)。以下字段不得混为一谈：

* `nominal_length_nt`：构建设计或参考名称给出的标称长度。
* `observed_sequence_length_nt`：Dorado 或其他 basecaller 在序列中观察到的长度。
* `training_length_nt`：明确选择用于当前 loss 的训练目标。
* read-level true length：经独立方法验证的单条 read 真实长度。
* signal-level interval：原始 signal sample 上的 Poly 区间。

除非存在可靠的逐 read 证据，否则不得把标称构建长度或 basecaller 观察长度描述为
read-level ground truth。弱监督标签必须记录 `label_source` 和限制。

长期定义：

* Poly 区域允许包含少量非目标碱基。
* 目标最大连续中断长度为 2 个非目标碱基。
* 最短有效 Poly 长度为 0 nt。
* `0 nt` 是合法负标签，不等同于检测失败。
* 必须区分完整、截断、无 Poly、无标签和检测失败。
* Poly(A) 与 Poly(T) 使用一致的坐标和状态定义。

---

## 2. 输入与输出

### 2.1 输入数据与路径

项目输入可能包括 POD5/FAST5、BAM/FASTQ、参考序列、move table、构建元数据和
read-level manifest。

实际路径只能通过以下方式提供，优先级从高到低为：

1. 命令行参数；
2. 环境变量；
3. `configs/local.yaml`；
4. `configs/default.yaml`。

不得在源代码、单元测试或长期文档中写死机器绝对路径。常用环境变量包括：

```text
HOLYPOLY_DATA_ROOT
HOLYPOLY_WORK_ROOT
HOLYPOLY_M3_MANIFEST
HOLYPOLY_BONITO_MODEL
HOLYPOLY_M3_CHECKPOINT
```

后三个名称用于 integration test；测试代码不得为它们提供机器相关默认值。

### 2.2 训练 Manifest

训练 manifest 使用 CSV 或 TSV，每行表示一个 read chunk。schema 的权威实现位于
`src/holypoly/training_data.py`，至少包含：

```text
read_id
pod5_path
pod5_file_id
split
orientation
raw_signal_length
chunk_start_sample
chunk_end_sample
poly_start_sample
poly_end_sample
poly_type
true_length_nt
label_status
poly_is_complete
failure_reason
manifest_version
```

可选审计字段包括：

```text
bam_path
nominal_length_nt
observed_sequence_length_nt
label_source
flank_source
flank_status
selection_reason
```

`true_length_nt` 是当前代码中的训练目标字段。创建 manifest 时必须同时记录其来源；
不得仅根据字段名宣称其为逐 read ground truth。

坐标统一采用原始 signal sample、0-based、左闭右开：

```text
[start_sample, end_sample)
```

raw sample 到 encoder timestep 的转换必须来自实际 Bonito 模型结构。当前支持的 v5
模型实测 stride 为 6；不得对其他模型沿用该值。

### 2.3 输出结果

长期目标输出包括逐 read TSV/CSV，以及保留原 basecall 内容的 BAM/FASTQ 注释。
当前辅助 head 和训练代码不负责改写 basecalled sequence。

逐 read 结果至少应包含：

| 字段 | 含义 |
| --- | --- |
| `read_id` | Read 标识符 |
| `predicted_length_nt` | 预测长度，单位为 nt |
| `start_sample` | Poly 区域 raw-sample 起点 |
| `end_sample` | Poly 区域 raw-sample 终点 |
| `confidence` | 预测置信度或质量分数 |
| `status` | 成功、失败或无法判断 |
| `failure_reason` | 失败原因 |

如果某条 read 无法识别，不要静默删除；必须在结果中记录状态和原因。

---

## 3. 项目目录与外部数据

本项目采用代码仓库与测序数据分离的结构。Git 仓库中不保存真实的 POD5、FAST5、
FASTQ、BAM、参考序列、处理结果或模型权重。

### 3.1 代码仓库结构

```text
holypoly/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── default.yaml
│   └── local.example.yaml
├── src/
│   ├── bonito/                 # Git submodule 和独立运行环境
│   └── holypoly/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── doctor.py
│       ├── poly_heads.py
│       ├── model.py
│       ├── losses.py
│       ├── training_data.py
│       └── training.py
└── tests/
    ├── test_config.py
    ├── test_doctor.py
    ├── test_poly_heads_subprocess.py
    └── torch/
```

目录用途：

* `configs/`：保存配置模板和非机器相关的默认参数。
* `src/holypoly/`：保存项目核心代码。
* `src/bonito/`：上游 Bonito Git submodule，不直接修改。
* `tests/`：父项目轻量测试。
* `tests/torch/`：在 Bonito 独立环境中运行的 Torch、GPU 和 integration 测试。
* 项目仓库中不创建用于保存真实测序数据的 `data/` 目录。
* 项目仓库中不保存大型运行结果、缓存或模型权重。
* Notebook 只能用于探索性分析，不得保存核心算法的唯一实现。

### 3.2 外部数据和工作目录

原始数据目录和工作目录必须分开：

```text
<data_root>/                  # 只读原始数据
<work_root>/                  # 可写工作目录
├── processed/
├── metadata/
├── results/
├── models/
├── logs/
└── cache/
```

相对路径解释规则：

* `pod5_dir`、`basecalls_dir` 和 `reference_path` 相对于 `data_root`。
* `processed_dir`、`metadata_dir`、`results_dir`、`models_dir`、`logs_dir` 和
  `cache_dir` 相对于 `work_root`。

### 3.3 数据访问规则

Codex 修改代码和运行命令时必须遵守：

* 不得将真实 POD5、FAST5、FASTQ、BAM 或参考序列复制到项目仓库。
* 不得修改、移动、重命名或覆盖原始数据目录中的任何文件。
* 所有生成文件只能写入 `HOLYPOLY_WORK_ROOT`。
* 输出文件不得与输入文件使用相同路径。
* 写入前必须确认目标路径不属于原始数据目录。
* 不得在没有确认的情况下扫描、索引或加载完整数据集。
* 开发和测试时优先使用 read ID 清单、synthetic fixtures 或小型测试子集。
* 大规模处理前必须报告预计读取的数据量、输出大小、运行时间和计算资源。
* POD5 与 BAM/FASTQ 之间通过 `read_id` 建立关联。
* 路径不存在或权限不足时，应明确报错，不得自动改用其他目录。
* 单元测试不得依赖服务器上的真实数据。
* 真实数据测试必须标记为 `integration`，通过环境变量显式启用；数据、模型或
  GPU 不可用时必须明确跳过或报错。
* 真实 read ID、manifest、模型权重和 checkpoint 只能写入外部 work root。

### 3.4 Git 忽略规则

`.gitignore` 至少应包含：

```gitignore
configs/local.yaml

*.pod5
*.fast5
*.fastq
*.fastq.gz
*.bam
*.sam
*.cram

*.pt
*.pth
*.ckpt

results/
models/
logs/
cache/
```

允许提交的内容包括不含真实样本信息的配置模板、程序生成的 synthetic fixtures、
数据格式说明、manifest schema、可复现脚本和不含敏感信息的小型汇总结果。

---

## 4. 模型与训练接口

### 4.1 Bonito 子模块

* `src/bonito` 是 Git submodule。
* 不得直接修改 Bonito 源代码或其 site-packages。
* Holypoly 的扩展必须位于 `src/holypoly/`。
* 更新 Bonito 前必须记录旧、新 commit，并单独验证回归。
* 完成工作前检查：

```bash
git submodule status
git -C src/bonito status --short
```

### 4.2 模型接口契约

`HolypolyModel` 必须：

* 复用 Bonito v5 的共享 signal encoder。
* 保留原 CRF/basecalling head。
* 不改变原 basecalling 输出接口或解码逻辑。
* 同一次 forward 共享计算 encoder features。
* 同时返回 basecall、location 和 length 输出。
* 默认冻结 encoder 和 basecalling head。
* 默认只优化 `poly_location_head` 和 `poly_length_head`。

实际 shape 必须从模型代码或运行结果确认，不得凭空假设。当前 v5 接口约定为：

```text
signal:                  [batch, 1, raw_samples]
encoder_features:        [batch, time, channels]
basecall_scores:         [time, batch, states]
location_logits:         [batch, time, 3]
location_status_logits:  [batch, 3]
length_nt:               [batch]
length_status_logits:    [batch, 3]
```

location 类别为 `background / poly(A) / poly(T)`；read 状态为
`poly_present / zero_nt / failure`。padding mask 使用 PyTorch 约定，`True` 表示
padding。

### 4.3 辅助 Loss

冻结 encoder 阶段的默认优化目标为：

```text
total_loss =
    lambda_location * location_loss
    + lambda_length * length_loss
    + lambda_status * shared_status_loss
```

* basecalling loss 不加入当前辅助 head 优化目标。
* `poly_present` 参与 location、length 和 status loss。
* `zero_nt` 的 location 为 background，长度目标为 0 nt。
* `failure` 的 location 使用 ignore index，length 不参与回归。
* padding 区域不得参与 location loss。
* NaN 或空长度目标不得进入普通回归 loss。
* loss 权重必须由配置传入并分别记录。

### 4.4 GPU 约束

训练和 integration test 只使用一个显式选择的 GPU。通过 `CUDA_VISIBLE_DEVICES` 或
配置选择物理设备，模型内部使用映射后的显式 `cuda:0`。不得在生产代码中写死物理
设备编号，也不得实现多 GPU 或 distributed training，除非另有明确任务。

---

## 5. 开发环境与常用命令

* 操作系统：Ubuntu 24.04
* Python 版本：Python 3.11
* 环境管理：uv
* Basecaller：Bonito v5 Transformer

### 5.1 依赖隔离

父项目环境由根目录 `pyproject.toml` 和 `uv.lock` 管理：

```bash
uv sync --frozen
```

父项目不得安装 Torch、Bonito 或 POD5 训练依赖。相关依赖仅安装在：

```text
src/bonito/.venv
```

不得把 Bonito 环境依赖加入父项目以规避 subprocess 测试。如果 Bonito 环境缺失，
应明确报告，不得静默安装到错误环境。

### 5.2 常用检查

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/holypoly
uv run pytest -q
```

根目录 pytest 会通过 subprocess 调用 `src/bonito/.venv` 运行 `tests/torch/`，避免
污染父项目依赖。

需要直接运行 Torch 测试时：

```bash
PYTHONPATH=src \
src/bonito/.venv/bin/python -m pytest -q tests/torch
```

真实数据 integration test 必须显式设置外部路径和 GPU，例如：

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> \
HOLYPOLY_M3_MANIFEST=<external-manifest> \
HOLYPOLY_BONITO_MODEL=<external-model-directory> \
HOLYPOLY_M3_CHECKPOINT=<external-work-checkpoint> \
PYTHONPATH=src \
src/bonito/.venv/bin/python -m pytest -q tests/torch -m integration
```

不要把示例占位路径替换为机器路径后提交。

---

## 6. 测试要求

### 6.1 必须覆盖的行为

* manifest 正常、空文件、缺字段、重复 read/chunk 和非法状态。
* raw-sample 到 encoder-timestep 的边界转换。
* Poly(A)、Poly(T)、zero_nt、failure 和 padding。
* 不同 batch size 和 signal length。
* forward、有限 loss 和 backward。
* 冻结参数无梯度、可训练参数有梯度。
* state_dict/checkpoint 保存和恢复。
* 固定随机种子下重复运行一致。
* basecalling 输出接口回归。
* 单元测试不依赖真实 POD5。
* integration test 不修改外部 POD5/BAM。

CPU 可用于确定性单元测试。训练循环、真实模型 backward 和真实数据 integration
必须在明确指定的单 GPU 环境中运行。

完整数据规模较大时，默认只运行 synthetic 和小型测试。运行完整流程或长时间训练前，
必须先说明预计时间、读取量、输出大小和计算资源。

### 6.2 测试结果报告

如果命令不存在、环境缺失、GPU 不可用或测试被跳过，必须报告实际原因，不得伪造
通过结果。integration smoke test 只证明数据链路可以运行，不得解释为泛化能力。

---

## 7. 评价指标

优先使用以下指标：

* MAE：预测长度与真实长度的平均绝对误差。
* Median AE：绝对误差的中位数。
* RMSE：对较大误差更敏感。
* Bias：平均预测偏差，用于判断系统性高估或低估。
* Within-tolerance accuracy：分别报告误差在 1 nt、5 nt 和 10 nt 内的比例。
* 相关性：Pearson 和 Spearman。
* 无法识别率：无法输出有效长度的 read 比例。

主要评价指标为 Within-tolerance accuracy 和 MAE。需要分别报告：

* 不同真实长度区间的结果。
* 不同 read 质量区间的结果。
* 成功识别与识别失败的数量。

只有在存在可靠 read-level 标签时，才允许将这些指标描述为正式准确率。使用构建级
弱监督或 provisional 边界的 smoke test 不得报告正式 MAE、泛化能力或最终准确率。

训练集、验证集和测试集必须按 POD5 文件、run 或其他独立采集单元划分，不得仅随机
拆分 read，从而避免数据泄漏。

---

## 8. 编码要求

Codex 修改代码时应遵守：

* 修改前先阅读相关代码、配置文件和测试。
* 优先进行小范围、可验证的修改。
* 不修改或覆盖原始数据。
* 不在代码或测试中写死数据绝对路径。
* 路径、阈值和模型参数应放入配置文件、manifest、环境变量或命令行参数。
* 新增函数应包含清晰的 docstring。
* 对关键数组说明 shape、坐标单位、坐标系统和 dtype。
* 固定随机种子，并记录会影响结果的随机过程。
* 不要仅在 Notebook 中实现核心算法。
* 新增依赖前说明用途，并更新对应环境的依赖文件。
* 修改算法后必须增加或更新测试。
* 不确定生物学定义或数据含义时，先提出问题，不要自行假设。
* 不得把标称长度、Dorado 观察值和真实逐 read 标签互相替代。
* 不得为通过测试而修改 Bonito 原输出、放宽标签定义或读取完整数据集。
* 生成文件必须写入 work root，并在写入前确认目标不位于 data root。

代码风格：

* 格式化工具：`ruff format`。
* 静态检查：`ruff check`；核心模块额外使用 `mypy`。
* 最大行宽：88。
* Python 文件、模块、函数和变量使用 `snake_case`。
* 类使用 `PascalCase`。
* 常量使用 `UPPER_SNAKE_CASE`。
* 内部函数以单下划线开头。
* 包含单位的数值应在名称中注明单位，例如 `poly_length_nt`、
  `duration_ms`、`current_pa`。
* 除循环索引外，避免使用含义不清的单字母变量。

---

## 9. 长期开发边界

除非用户明确要求，不要：

* 修改 Bonito 子模块、原 CRF/CTC head 或解码逻辑。
* 改变原 basecalling 输出格式。
* 将 basecalling loss 自动加入冻结 encoder 的辅助 head 训练。
* 实现多 GPU、distributed training 或完整数据训练。
* 自动选择 label truth、flank 或坐标定义。
* 根据模型预测改写原始 POD5、BAM、FASTQ 或参考序列。
* 把 smoke-test 指标解释为正式泛化性能。

如任务需要重新定义标签、坐标系统、Poly 中断规则或 loss 目标，必须暂停并向用户
确认。

---

## 10. 已知问题与限制

* 短 Poly 区域容易与背景信号混淆。
* 部分 read 的方向、完整性和 Poly 边界不确定。
* 构建标称长度不一定等于单条 read 的真实 Poly 长度。
* Dorado/basecaller 观察长度包含 basecalling 误差。
* provisional flank 或弱监督边界不能直接作为最终 ground truth。
* `0 nt` 必须与 failure、无标签和漏检分别处理。
* 跨 chunk Poly 区域及 chunk 拼接仍需独立验证。

对于这些问题，可以提出解决方案，但在未经确认前不要大规模修改分析流程。

---

## 11. 结果可复现性

每次实验应记录：

* Git commit ID。
* Bonito submodule commit。
* 配置文件和运行命令。
* 输入数据与 manifest 版本。
* Bonito model config 和模型来源。
* 标签来源和坐标系统。
* manifest schema version。
* loss 配置和权重。
* 随机种子。
* Python、PyTorch、CUDA 和 GPU 版本。
* 开始与结束时间。
* 输出目录。
* 主要指标及其是否仅为 smoke-test 数值。

checkpoint 应至少保存 model state、optimizer state、epoch、global step、random seed、
版本信息、manifest schema 和 loss 配置。

实验记录和 checkpoint 只能写入外部 work root，不得提交到 Git。

---

## 12. Codex 的交付格式

完成任务后，请说明：

1. 修改了哪些文件。
2. 实现了什么功能。
3. 做了哪些假设。
4. 运行了哪些测试。
5. 测试是否通过。
6. 尚未解决的问题。
7. 建议的下一步工作。
8. 是否读取或写入了外部数据。
9. GPU、模型和 manifest 是否参与测试。
