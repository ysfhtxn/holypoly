# holypoly

`holypoly` 从 Oxford Nanopore cDNA 原始信号估计 poly(A/T) 候选区域的长度。
当前使用 Bonito v5 Transformer，常规训练入口是
`scripts/run_reference_joint.sh`：sum pooling、Huber(delta=10 nt)，联合优化
length、弱 status 和 reference-head 原生 CTC-CRF；location 不监督。

## 主流程

```text
BAM + move table + POD5 + reference head
  → prepare：v4 长度/状态 manifest
  → prepare_sequence：同源、同 split 的 reference-head 序列 manifest
  → run_reference_joint.sh：训练及 validation 选轮
  → training.evaluate：test 预测和弱标签一致性指标
  → plot_training.py：训练曲线和按实际 nt 间距绘制的结果图
```

完整命令见[默认流程](docs/default_workflow.md)，日常入口见
[命令速查](docs/common_commands.md)，文档导航见 [docs/README.md](docs/README.md)。
数据制作与训练分开执行；训练脚本消费已有的两份 manifest，不重建数据。

## 环境与配置

父环境使用 Python 3.11 和 uv：

```bash
uv sync --frozen
cp configs/local.example.yaml configs/local.yaml
```

在本地配置中填写只读 `paths.data_root`、外部可写 `paths.work_root` 和
`training.bonito_model`。根路径也可通过 `HOLYPOLY_DATA_ROOT`、
`HOLYPOLY_WORK_ROOT` 覆盖，模型可通过 `HOLYPOLY_BONITO_MODEL` 提供。
本地配置不提交；路径优先级为 CLI > 环境变量 > 本地配置 > 默认配置。
Torch、POD5 和 Bonito 仅安装在 `src/bonito/.venv`，父环境不安装训练依赖。

两份已审计 manifest 准备好后，先执行只读预检：

```bash
GPU_ID=0 EPOCHS=5 \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv --dry-run
```

确认输入、规模和资源后，去掉 `--dry-run` 开始训练。物理 GPU 由 `GPU_ID`
选择，内部使用 `cuda:0`；只使用一个 GPU。每次新实验使用新目录。
续训使用原 `EXPERIMENT_ID`、目标总轮数和 `--resume`，详见
[联合训练与恢复](docs/reference_joint_training.md)。

`holypoly doctor` 提供只读环境检查。`holypoly train` 是保留的冻结 encoder
历史入口，仅供明确的基线、消融或旧实验恢复；其默认值不代表联合训练配置。

## 数据与结果的含义

构建标称长度用于弱监督，不是独立验证的逐 read 真值。`nominal_length_nt`、
`observed_sequence_length_nt` 和 `training_length_nt` 分开记录。
`0 nt` 是合法负标签，失败与无标签分别处理。坐标统一为原始 signal sample、
0-based、左闭右开 `[start_sample, end_sample)`。

新数据采用无设计 flank 的 `head–Poly / raw-edge` 候选定义；候选边缘不代表
已确认的 Poly 终点。联合 A/T 数据显式使用 `--poly-types polyA polyT --strict-balance`。
序列 manifest 必须来自同一长度 manifest，并保留 split 和源 SHA256。
标签规则见 [v4 数据准备](docs/prepare_dataset_v4.md)。

manifest、审计、信号导出、模型、日志和结果均位于外部 work root。
原始数据不修改，不复制进仓库。结果按用途存放：

| 目录 | 用途 |
| --- | --- |
| `results/training_evaluation/<run>/` | validation/test、训练曲线和弱标签一致性 |
| `results/cdna_prediction/<experiment>/` | 实际 POD5 候选预测 |
| `results/tool_comparison/<comparison>/` | 多工具分布、覆盖率和定长弱标签比较 |

命名和历史恢复规则见[结果目录约定](docs/result_layout.md)。

## 训练后的预测与比较

[普通 cDNA 预测](docs/predict_cdna.md)使用已有 checkpoint 和 basecall 候选，
不依赖固定参考 head；迁移到真实样本的效果仍需独立验证。
[独立 holdout](docs/final_holdout.md)用于固定未参与训练或调参的比较 read 清单。
随后可进行[定长多工具比较](docs/compare_fixed_len_tools.md)或
[普通 cDNA 分布比较](docs/compare_cdna_tools.md)。

这些分析保留失败状态和覆盖率，不把缺失输出补成零，不宣称逐 read 真值准确率。
可选的冻结消融和状态校准见[文档索引](docs/README.md)；历史 v3 与迁移审计见
[归档索引](docs/archive/README.md)。

## 代码与检查

```text
src/holypoly/
├── cli.py                 # 只读体检及历史冻结训练 CLI
├── core/                  # 配置、体检和结果路径
├── datasets/              # v4、reference-head、subset、holdout 准备
├── models/                # Bonito v5 包装、Poly heads 和联合模型
├── training/              # 数据消费、共享校验、联合训练、评估及历史入口
├── inference/             # cDNA 候选和长度预测
├── evaluation/            # 多工具结果与来源数据整理
└── legacy/                # 显式调用的历史迁移诊断

tests/unit/                # 父环境轻量测试
tests/isolated/            # Bonito 环境中的模型、数据和联合训练测试
tests/legacy/              # 历史审计和旧数据 integration，默认不收集
```

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/holypoly
uv run pytest -q
```

根 pytest 通过 subprocess 在 `src/bonito/.venv` 执行 `tests/isolated/`。
默认开发使用合成数据；真实 integration 需显式配置外部路径和单 GPU。
历史测试的单独运行方式见[归档索引](docs/archive/README.md)。
