# holypoly

`holypoly` 用于整理 Oxford Nanopore cDNA 数据，并为后续 poly(A/T) 区域定位和
长度估计提供工具。本阶段只包含项目初始化与外部数据环境体检，不执行 basecalling、
GPU 任务、模型训练或完整原始信号读取。

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

`src/holypoly/poly_heads.py` 提供 Bonito v5 Transformer 专用的
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
