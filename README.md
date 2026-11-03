# holypoly

`holypoly` 用于整理 Oxford Nanopore cDNA 数据，并为后续 poly(A/T) 区域定位和
长度估计提供工具。当前支持弱监督数据制作、Bonito v5 辅助 head、单 GPU
联合训练和小规模验证。

当前默认流程见 [数据制作与三项联合训练](docs/default_workflow.md)：
先制作 v4 长度/状态 manifest，再生成匹配的 reference-head 序列 manifest；
新训练使用 sum pooling + Huber，以及 length、status、basecalling 三项 loss。

## 代码与测试组织

```text
src/holypoly/
├── cli.py                 # 公共 CLI，仅负责参数和分发
├── core/                  # 配置、外部环境体检
├── datasets/              # 离线数据准备，不包含模型或训练循环
├── models/                # Bonito 适配器、Poly heads
└── training/              # manifest 消费、loss、训练循环和启动器

tests/
├── unit/
│   ├── core/
│   └── training/
├── isolated/              # 在 Bonito 独立环境中执行
│   ├── datasets/
│   ├── models/
│   └── training/
└── test_isolated_environment.py
```

`datasets/` 负责制作数据；`training/data.py` 只负责读取已有训练 manifest、
signal chunk 和 batch，不负责制作数据集。各包的 `__init__.py` 不主动加载重型依赖，
CLI 与训练启动器仍可在父项目环境运行。原平铺模块的 Python 导入路径已迁移到对应
子包，`holypoly doctor` 和 `holypoly train` 命令保持不变。

训练后导出 test split 预测、分组指标及调用训练汇总绘图脚本，见
[训练、测试与绘图流程](docs/evaluate_model.md)。
常用命令速查见 [常用命令](docs/common_commands.md)。

新结果按用途分目录：训练验证/测试位于 `results/training_evaluation/<run>/`，
实际 POD5 预测位于 `results/cdna_prediction/<experiment>/`，
多工具对比位于 `results/tool_comparison/<comparison>/`。历史结果默认保留原位，
目录与命名规则见 [结果目录约定](docs/result_layout.md)。

参考 head 序列监督、sum/Huber 与原生 CTC-CRF 联合训练，见
[联合训练说明](docs/reference_joint_training.md)。`scripts/run_reference_joint.sh`
及 `joint_command` 默认联合训练 length、弱 status 和 basecalling，location 不监督。
历史冻结训练入口保留用于旧实验和明确指定的对照。
针对 0 nt 假阳性的独立验证，见[冻结状态分类训练](docs/status_calibration.md)：
冻结 encoder、投影和长度回归，只对现有 status 分类器做类别平衡训练。

根目录 pytest 运行 `unit/`，并通过 subprocess 在 `src/bonito/.venv` 中运行
`isolated/`。也可单独执行：

```bash
uv run pytest -q tests/unit
PYTHONPATH=src src/bonito/.venv/bin/python -m pytest -q tests/isolated
```

## 安装

项目要求 Python 3.11，并使用 [uv](https://docs.astral.sh/uv/) 管理环境：

```bash
uv sync
```

## 配置

默认配置位于 `configs/default.yaml`。复制示例文件创建仅供本机使用的配置：

```bash
cp configs/local.example.yaml configs/local.yaml
```

在 `configs/local.yaml` 中填写外部只读数据目录和可写工作目录。该文件已被
`.gitignore` 忽略。也可以使用环境变量：

```bash
export HOLYPOLY_DATA_ROOT=/path/to/read-only/data
export HOLYPOLY_WORK_ROOT=/path/to/writable/work
```

路径优先级为：命令行参数 > 环境变量 > 本地配置 > 默认配置。

## 数据体检

```bash
uv run holypoly --help
uv run holypoly doctor --config configs/local.yaml
```

根路径也可以在命令行中覆盖：

```bash
uv run holypoly doctor \
  --config configs/local.yaml \
  --data-root /path/to/read-only/data \
  --work-root /path/to/writable/work
```

`doctor` 只执行以下操作：

- 检查 `data_root` 是否为可读目录；
- 检查 `work_root` 是否为可写目录；
- 查找并计数 POD5、BAM/FASTQ 和参考序列；
- 每类最多抽样少量文件，只读取文件头以验证可读性。

该命令不会创建工作目录，不会修改、移动或复制外部数据，也不会读取完整 POD5
信号。检查失败时命令返回非零退出码，并逐项打印原因。

## 开发检查

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/holypoly
uv run pytest -q
```

## 多任务 head 原型

`src/holypoly/models/poly_heads.py` 提供 Bonito v5 Transformer 专用的
`BonitoV5MultiTaskModel`。它复用 v5 的 `conv -> transformer_encoder -> upsample`
共享特征，再将同一特征分别送入原始 CRF basecalling head、
`PolyLocationHead` 和 `PolyLengthHead`。原 basecalling scores 的 shape 和数值保持
不变。

v5 CRF 前特征为 `[batch, encoder_time, 512]`。padding mask 使用 PyTorch 约定，
`True` 表示 padding。外部 Poly 边界使用 raw signal sample 单位、0-based、左闭右开
`[start_sample, end_sample)`；原型以固定 stride 6 在 raw sample 与 encoder
timestep 间近似映射。Poly(A/T) 标签应在进入模型前统一到 canonical 方向，非目标
碱基标记为 `BACKGROUND`，当前 decoder 不跨 background 合并区间。

辅助 loss 为：

```text
lambda_location * location_cross_entropy
+ lambda_length * smooth_l1(log1p(predicted_nt), log1p(true_nt))
+ lambda_status * shared_status_cross_entropy
```

三个权重位于 `configs/default.yaml`。`ZERO_NT` 参与长度回归且目标为 0；
`FAILURE` 不参与长度回归并继续解码为 `NaN`。Bonito 原 basecalling loss 通过
`basecall_loss()` 委托给原 `model.loss()`，但不加入当前 auxiliary loss。

Torch 依赖仅安装在 `src/bonito/.venv`，因此原型测试使用该独立环境运行：

```bash
uv run pytest -q
```

父项目 pytest 会通过 subprocess 在 `src/bonito/.venv` 中执行 Torch 测试，避免将
Torch 和 Bonito 依赖安装到 holypoly 的 `.venv`。

## 辅助 head 训练数据接口

从 Bonito save-ctc BAM、Dorado move table 和 POD5 生成均衡训练 manifest：
已确认无 flank 的 head–Poly 构建使用 [v4 制作方法](docs/prepare_dataset_v4.md)：
取消 flank 推定，正样本和 0 nt 均使用标称长度弱监督，定位/状态保持无监督。
联合均衡制作可显式使用 `--poly-types polyA polyT --strict-balance`：
正长度 × 类型与 0 nt 两方向均取相同数量，按最稀缺组缩减；
小型 200 条/组示例及状态比例说明见 [v4 制作方法](docs/prepare_dataset_v4.md)。
旧 polyA、0 nt 和配额暂停方案见 [v3 历史制作方法](docs/current_dataset_preparation.md)。
历史 v3 参见 [数据准备脚本与运行说明](docs/prepare_dataset.md)：使用构建标称长度
弱监督，从 0A 推定 provisional flank，保留失败审计，不复制原始信号。
边界不确定的正构建可使用 `supervision_mode=length_only` 训练标称长度，
同时跳过定位和状态 loss；旧 manifest 没有该字段时默认完整监督。

`src/holypoly/training/data.py` 提供只读 manifest、固定长度 Dataset 和 collate。
manifest 使用 CSV 或 TSV，每行对应一个 read chunk，必需字段为：

```text
read_id, pod5_path, pod5_file_id, split, orientation,
raw_signal_length, chunk_start_sample, chunk_end_sample,
poly_start_sample, poly_end_sample, poly_type, training_length_nt,
label_status, poly_is_complete, failure_reason
```

所有 signal 和 Poly 坐标均使用原始 signal sample、0-based、左闭右开。训练 chunk
完整监督 chunk 必须完全包含 `poly_present` 区域；所有 chunk 长度固定且能被 12
整除，从而使 Bonito v5 的
卷积和上采样输出严格对应 `chunk_size / 6` 个 encoder timestep。`zero_nt` 使用全
background location 标签和 0 nt 长度；`failure` 使用 `-100` location 标签、空长度
和明确的 `failure_reason`。

Dataset 只读取 manifest 指定的 POD5 read，并由调用方提供 Bonito 模型配置中的
signal shift/scale。它不裁剪、修改或写入 POD5。Torch 相关测试继续在 Bonito 的
独立环境中运行。

## Milestone 3 冻结 encoder 训练

`HolypolyModel` 位于 `src/holypoly/models/model.py`，默认冻结 Bonito v5 的 convolution、
Transformer、upsample 和原 CRF head，仅让 location/length 两个 Poly head 参与
优化。`src/holypoly/training/losses.py` 提供独立的 masked auxiliary loss，
`src/holypoly/training/engine.py` 提供单 GPU epoch、验证、指标、日志和 checkpoint 恢复。

训练长度 loss 可通过 `--length-loss log_smooth_l1|mae|huber` 选择，默认保持原定义；
Huber 的 `--huber-delta-nt` 默认 10。配置、公式和对照命令见
[长度 loss 说明](docs/length_losses.md)。不同 loss 的对照应新建实验目录。

平均池化与求和池化的独立对照使用 `scripts/run_length_pooling_ablation.sh`，
默认固定 Huber、batch 8、20 轮，见[池化对照说明](docs/length_pooling_ablation.md)。

真实 smoke manifest 必须放在外部 work root，并分别记录构建标称长度、Dorado
观察长度和实际训练目标。当前 2p5per smoke 使用构建级 20 nt 弱监督与 provisional
flank，只验证 POD5→Dataset→model→loss→backward→checkpoint 链路，不用于报告
正式 MAE 或泛化能力。

### 训练命令

训练依赖继续保留在 `src/bonito/.venv`。父项目的 `holypoly train` 只负责读取配置、
验证路径并在该独立环境中启动训练，不会把 Torch、POD5 或 Bonito 安装到父环境。

先配置外部路径并执行只读预检：

```bash
export HOLYPOLY_DATA_ROOT=/path/to/read-only/data
export HOLYPOLY_WORK_ROOT=/path/to/writable/work
export HOLYPOLY_M3_MANIFEST=/path/to/writable/work/metadata/train.tsv
export HOLYPOLY_BONITO_MODEL=/path/to/writable/work/models/bonito-v5
export HOLYPOLY_M3_CHECKPOINT=/path/to/writable/work/models/holypoly-heads.pt
export CUDA_VISIBLE_DEVICES=<physical-gpu-id>

uv run holypoly train --dry-run
```

`--dry-run` 只读取 manifest 元数据和 Bonito `config.toml` 的存在性，不读取 POD5
signal、不加载模型权重，也不创建 checkpoint。预检通过后运行：

```bash
uv run holypoly train --epochs 10 --batch-size 4
```

训练固定使用映射后的 `cuda:0`。继续已有 checkpoint 使用 `--resume`；重新开始并
允许覆盖已有输出必须显式使用 `--overwrite`。相对的 manifest、模型、checkpoint 和
日志路径均相对于 `work_root` 解释，checkpoint 和日志强制写在 `work_root` 内。
train、validation 和 test 可以引用同一 POD5 文件，但不同 split 之间不得包含重复的
`read_id`。

训练在每轮 train 和 validation 完成后保存完整快照，例如 `heads.epoch_0001.pt`、
`heads.epoch_0002.pt`，同时更新 `heads.pt` 供 `--resume` 继续最新进度。
各快照包含模型、优化器、随机状态和版本元数据，可用现有评估命令选择任意一轮。
历史快照增加磁盘占用；命名、覆盖规则和评估示例见
[逐 epoch checkpoint](docs/checkpoints.md)。

## 按参考名称整理标称长度数据

`src/holypoly/datasets/length_groups.py` 从 Dorado BAM 的比对参考名称中解析 `<number>A`，
并将 primary read 整理为按 `nominal_length_nt` 分组的压缩 TSV。该步骤只引用原
BAM/POD5 路径，不复制或修改原始文件，也不会把构建标称长度描述为逐 read 真值。

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.length_groups \
  --data-root /path/to/fixed_len_data \
  --reference /path/to/fixed_len_data/fixed_len_ref.fa \
  --output-dir /path/to/fixed_len_data/derived/nominal_length_groups_v1
```

输出包含每个长度的 `reads.tsv.gz`、未比对 read 的 `unassigned/reads.tsv.gz`、参考
序列目录和汇总 JSON。该结果仅完成 read-level 标称长度分类，尚不包含 location head
训练需要的 signal-level Poly 区间。
