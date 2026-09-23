# AGENTS.md

## 1. 项目概述

* 项目名称：holypoly
* 项目目标：从纳米孔测序数据中识别并估计 Poly 区域的长度。
* Poly 类型：poly(A)和poly(T)的集合
* 测序平台：Oxford Nanopore
* 测序类型：cDNA
* 当前阶段：数据整理
* 项目流程：数据整理 -> 算法开发 -> 模型训练 -> 结果验证

### 1.1 科学问题

本项目希望解决以下问题：

1. 从原始电流信号中定位 Poly 区域。
2. 估计 Poly 区域长度，单位为 nt或base。
3. 将预测结果与已知构建长度、dorado等其他工具的结果结果比较。
4. 分析不同样本、物种 对识别结果的影响。

### 1.2 “长度”的定义

本项目中的 Poly 长度定义为：

cDNA read 可能以两个方向出现，因此目标区域可能表现为 poly(A)或其反向互补形式 poly(T)。

需要特别说明：

* 是否允许 Poly 区域中存在非目标碱基：这是最终目标，如有非目标碱基，也能准确识别
* 允许的最大中断长度：最多允许2个非目标碱基
* 最短有效 Poly 长度：0nt
* 是否需要区分完整与截断的 Poly 区域：需要

---

## 2. 输入与输出

### 2.1 输入数据

项目可能使用以下数据：

| 数据类型            | 路径        | 用途        |
| --------------- | --------- | --------- |
| POD5/FAST5 原始信号 | /data/biolab-backup-hdd3/zhaoxy/holypoly_data/fixed_len_data/pod5 | 信号级长度识别   |
| BAM/SAM         | /data/biolab-backup-hdd3/zhaoxy/holypoly_data/fixed_len_data/basecalls | 比对和方向判断   |
| 参考序列            | /data/biolab-backup-hdd3/zhaoxy/holypoly_data/fixed_len_data/fixed_len_ref.fa | 定位目标序列    |
| 真实长度标签          | /data/biolab-backup-hdd3/zhaoxy/holypoly_data/fixed_len_data/fixed_len_ref.fa的id | 训练和评估     |

数据文件较大时，不要将真实数据提交到 Git。

参考序列中只有poly区域前的特异性序列，实际的read结构应该是 特异性序列-poly区域-统一的flank序列

### 2.2 数据表格式

长度标签文件预计至少包含以下字段：

| 字段            | 含义           | 示例          |
| ------------- | ------------ | ----------- |
| read_id     | 纳米孔 read 标识符 | abc123    |
| sample_id   | 样本标识符        | sample_01 |
| true_length | 已知 Poly 长度   | 50        |
| poly_type   | Poly 类型      | polyA     |
| batch       | 测序批次         | run_01    |

实际字段定义：

### 2.3 输出结果

主要输出文件为bam/fastq/fasta，以及一个csv/tsv

bam/fastq/fasta除正常的数据外要将特定长度的polyA/T体现在序列中

csv/tsv至少包含：

| 字段                 | 含义         |
| ------------------ | ---------- |
| read_id          | Read 标识符   |
| predicted_length | 预测长度       |
| start_position   | Poly 区域起点  |
| end_position     | Poly 区域终点  |
| confidence       | 预测置信度或质量分数 |
| status           | 成功、失败或无法判断 |
| failure_reason   | 失败原因       |

如果某条 read 无法识别，不要静默删除；必须在结果中记录状态和原因。

---

## 3. 项目目录与外部数据

本项目采用代码仓库与测序数据分离的结构。Git 仓库中不保存真实的 POD5、FAST5、FASTQ、BAM、参考序列、处理结果或模型权重。

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
│   ├── bonito/
│   └── holypoly/
│       ├── __init__.py
│       ├── preprocess.py
│       ├── detect_poly.py
│       ├── estimate_length.py
│       └── evaluate.py
├── scripts/
├── tests/
└── notebooks/
```

目录用途：

* `configs/`：保存配置模板和非机器相关的默认参数。
* `src/holypoly/`：保存可复用的核心代码。
* `scripts/`：保存数据整理、训练、预测和评估的命令行入口。
* `tests/`：保存自动化测试。
* `notebooks/`：仅用于探索性分析，不保存唯一版本的核心算法。
* 项目仓库中不创建用于保存真实测序数据的 `data/` 目录。
* 项目仓库中不保存大型运行结果、缓存或模型权重。

### 3.2 外部数据目录

当前原始数据位于：

```text
/data/biolab-backup-hdd3/zhaoxy/holypoly_data/fixed_len_data/
├── pod5/
├── basecalls/
└── fixed_len_ref.fa
```

路径含义：

* `pod5/`：POD5 原始电流信号，只读。
* `basecalls/`：已有的 FASTQ、SAM 或 BAM basecall 结果，只读。
* `fixed_len_ref.fa`：固定 Poly 长度构建的参考序列，只读。

建议为处理中间文件和结果单独设置外部工作目录：

```text
/data/biolab-backup-hdd3/zhaoxy/holypoly_work/
├── processed/
├── metadata/
├── results/
├── models/
├── logs/
└── cache/
```

目录用途：

* `processed/`：预处理后的数据和中间文件。
* `metadata/`：样本表、构建信息、manifest 和 read-level 标签。
* `results/`：模型预测和评估结果。
* `models/`：模型检查点和最终权重。
* `logs/`：运行日志和实验记录。
* `cache/`：允许重新生成的临时缓存。

原始数据目录和工作目录必须分开。任何程序均不得修改、移动、重命名或覆盖原始数据。

### 3.3 路径配置

代码中不得写死服务器绝对路径。

外部路径应通过以下方式传入，优先级从高到低为：

1. 命令行参数；
2. 环境变量；
3. 本地配置文件；
4. 默认配置。

建议使用以下环境变量：

```bash
export HOLYPOLY_DATA_ROOT=/data/biolab-backup-hdd3/zhaoxy/holypoly_data/fixed_len_data
export HOLYPOLY_WORK_ROOT=/data/biolab-backup-hdd3/zhaoxy/holypoly_work
```

路径含义：

* `HOLYPOLY_DATA_ROOT`：只读原始数据根目录。
* `HOLYPOLY_WORK_ROOT`：允许程序写入的工作目录。

`configs/default.yaml` 只保存通用参数和相对路径，不保存特定服务器的绝对路径。

机器相关的路径可以写入：

```text
configs/local.yaml
```

但 `configs/local.yaml` 必须加入 `.gitignore`，不得提交到 Git。仓库中只提交不包含真实路径的：

```text
configs/local.example.yaml
```

配置示例：

```yaml
paths:
  data_root: "/path/to/read-only/data"
  work_root: "/path/to/writable/work"

  pod5_dir: "pod5"
  basecalls_dir: "basecalls"
  reference_path: "fixed_len_ref.fa"

  processed_dir: "processed"
  metadata_dir: "metadata"
  results_dir: "results"
  models_dir: "models"
  logs_dir: "logs"
  cache_dir: "cache"
```

相对路径的解释规则：

* `pod5_dir`、`basecalls_dir` 和 `reference_path` 相对于 `data_root`。
* `processed_dir`、`metadata_dir`、`results_dir`、`models_dir`、`logs_dir` 和 `cache_dir` 相对于 `work_root`。

### 3.4 数据访问规则

Codex 修改代码时必须遵守：

* 不得将真实 POD5、FAST5、FASTQ、BAM 或参考序列复制到项目仓库。
* 不得修改原始数据目录中的任何文件。
* 所有生成文件只能写入 `HOLYPOLY_WORK_ROOT`。
* 输出文件不得与输入文件使用相同路径。
* 在执行写入操作前，必须检查目标路径不属于原始数据目录。
* 不得在没有确认的情况下扫描或加载完整数据集。
* 开发和测试时优先使用 read ID 清单或小型测试子集。
* 大规模处理前必须报告预计读取的数据量、输出大小、运行时间和计算资源。
* POD5 与 BAM/FASTQ 之间通过 `read_id` 建立关联。
* 路径不存在或权限不足时，应明确报错，不得自动改用其他目录。
* 单元测试不依赖服务器上的真实数据；优先使用程序生成的模拟数据。
* 需要真实数据的测试应标记为 integration test，并在数据不可用时明确跳过。

### 3.5 Git 忽略规则

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

允许提交的内容包括：

* 不包含真实样本信息的配置模板；
* 小型人工合成测试数据；
* 数据格式说明；
* manifest 字段定义；
* 可复现的处理脚本；
* 汇总统计和不包含敏感信息的小型图表。

---

## 4. 分析流程

当前计划处理流程为：

1. 扫描 POD5/FAST5、BAM/FASTQ、参考序列和样本信息。
2. 通过 `read_id` 检查 POD5 与已有 basecall 结果是否能够关联。
3. 如果没有 basecall：
   - 使用 Bonito 对 POD5/FAST5 进行 basecalling；
   - 输出未比对 BAM；
   - 尽可能保留 sequence-to-signal 对应信息。
4. 使用 minimap2 将 basecalled sequence 比对到参考序列。
5. 结合比对位置、引物位置和信号映射，生成 Poly 区域候选标签。
6. 运行 Dorado poly(A/T) estimation，作为基线结果。
7. 在 Bonito 基础上开发：
   - `poly_location_head`：预测 Poly 区域边界；
   - `poly_length_head`：预测 Poly 长度。
8. 对低质量或无法估计的结果进行分类标记。
9. 输出带注释的 BAM 和逐 read 的 TSV。
10. 对 Holypoly、Dorado 和真实/标称长度进行比较。

最终形态流程说明：

1. 读取POD5/FAST5/FASTQ/BAM
2. 如有FASTQ/BAM，进入下一步，否则进行basecall
3. 使用bonito为基础的深度学习模型定位、估计poly长度
4. 对低质量或异常结果进行标记
5. 输出每条 read 的预测结果和汇总报告，要有FASTQ/BAM和TSV/CSV两种格式。FASTQ/BAM中直接注明poly区域位置和长度。TSV/CSV做总结。

---

## 5. 开发环境

* 操作系统：Ubuntu 24.04.5 LTS
* Python 版本：Python 3.11
* 环境管理：uv
* GPU：Device 0 - NVIDIA GeForce RTX 3080 12G & Device 2 - NVIDIA GeForce RTX 3080 10G
* Basecaller：bonito
* 主要依赖：numpy、pandas、pysam、pod5、torch等

- 依赖声明：`pyproject.toml`
- 锁定文件：`uv.lock`
- 安装环境：`uv sync --frozen`
- 运行测试：`uv run pytest -q`
- 检查格式：`uv run ruff format --check .`
- 静态检查：`uv run ruff check .`
- 类型检查：`uv run mypy src/holypoly`
- GPU 设备必须通过配置文件或命令行指定，不得在代码中写死设备编号。

---

## 6. 评价指标

优先使用以下指标：

* MAE：预测长度与真实长度的平均绝对误差。
* Median AE：绝对误差的中位数。
* RMSE：对较大误差更敏感。
* Bias：平均预测偏差，用于判断系统性高估或低估。
* Within-tolerance accuracy：误差在 10nt 5nt 1nt 内的比例。
* 相关性：Pearson、Spearman二者。
* 无法识别率：无法输出有效长度的 read 比例。

主要评价指标：Within-tolerance accuracy和MAE

需要分别报告：

* 不同真实长度区间的结果；
* 不同 read 质量区间的结果；
* 成功识别与识别失败的数量；

训练集、验证集和测试集可以按照pod5文件 进行划分，避免数据泄漏。

---

## 7. 编码要求

Codex 修改代码时应遵守以下规则：

* 在修改前先阅读相关代码、配置文件和测试。
* 优先进行小范围、可验证的修改。
* 不要修改或覆盖原始数据。
* 不要在代码中写死数据绝对路径。
* 路径、阈值和模型参数应放入配置文件或命令行参数。
* 新增函数应包含清晰的 docstring。
* 对关键数组说明维度、单位和数据类型。
* 固定随机种子，并记录会影响结果的随机过程。
* 不要仅在 Notebook 中实现核心算法。
* 新增依赖前说明用途，并更新 uv的环境文件。
* 修改算法后必须增加或更新测试。
* 不确定生物学定义或数据含义时，先提出问题，不要自行假设。

代码风格：

* 格式化工具：`ruff format`
* 静态检查：`ruff check`；核心模块额外使用 `mypy`
* 最大行宽：`88`
* 命名规则：
  * Python 文件和模块使用 `snake_case`，例如 `estimate_poly_length.py`
  * 函数和变量使用 `snake_case`，例如 `detect_poly_region`
  * 类使用 `PascalCase`，例如 `PolyLengthEstimator`
  * 常量使用 `UPPER_SNAKE_CASE`，例如 `MIN_POLY_LENGTH`
  * 内部使用的函数以单下划线开头，例如 `_normalize_signal`
  * 包含单位的数值应在名称中注明单位，例如 `poly_length_nt`、
    `duration_ms`、`current_pa`
  * 除循环索引外，避免使用含义不清的单字母变量

---

## 8. 测试要求

至少覆盖以下情况：

1. 已知长度的正常 Poly 区域。
2. 非常短和非常长的 Poly 区域。
3. Poly 区域中存在少量非目标碱基。
4. 输入信号或序列不完整。
5. 找不到 Poly 区域。
6. 输入文件为空或格式错误。
7. 重复运行获得一致结果。

用于快速测试的小型数据位于：

目前没有小型数据，可以从现有数据中切一部分

完整数据规模较大时，Codex 默认只运行小型测试；运行完整流程前应先说明预计的时间和资源需求。

---

## 9. 当前任务

## 9. 当前任务

当前阶段仅完成数据整理和基线建立：

1. 扫描现有 POD5、BAM 和参考序列。
2. 生成样本、构建和文件级 manifest。
3. 检查 POD5 与 BAM 的 `read_id` 匹配情况。
4. 提取每条 read 的基础元数据。
5. 创建 100～1000 条 read 的小型测试数据集。
6. 运行 Dorado poly(A/T) estimation，生成基线结果。
7. 明确模型训练所需的 signal-level 和 sequence-level 标签格式。

本阶段不包含：

- 训练最终深度学习模型；
- 修改 Bonito 网络结构；
- 在野生型数据上进行正式评估；
- 根据预测结果修改原始 basecall 序列。

完成标准：

* [ ] 功能已经实现。
* [ ] 不破坏现有接口。
* [ ] 对正常输入和异常输入都有测试。
* [ ] 已运行 测试命令。
* [ ] 输出结果包含必要字段。
* [ ] 更新相关文档或使用示例。

---

## 10. 已知问题与限制

* 短 Poly 区域容易与背景信号混淆
* 部分 read 的测序方向未知
* 真实长度标签并不一定对应read真实的poly长度
* 目前工具对0nt poly识别效果差

对于这些问题，Codex 可以提出解决方案，但在未经确认前不要大规模修改分析流程。

---

## 11. 结果可复现性

每次实验应记录：

* Git commit ID；
* 配置文件；
* 输入数据版本；
* 软件和依赖版本；
* 随机种子；
* 运行命令；
* 开始与结束时间；
* 输出目录；
* 主要评价指标。

实验记录位置：灵活安排，符合项目目录结构

---

## 12. Codex 的交付格式

完成任务后，请说明：

1. 修改了哪些文件；
2. 实现了什么功能；
3. 做了哪些假设；
4. 运行了哪些测试；
5. 测试是否通过；
6. 尚未解决的问题；
7. 建议的下一步工作。
