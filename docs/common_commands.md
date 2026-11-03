# 常用命令

> 当前数据为无 flank 的 head–Poly 构建，使用 v4。
> 规则、两份数据的制作命令及审计方法见 [v4 制作方法](prepare_dataset_v4.md)。
> 新数据和新训练的默认完整流程见 [数据制作与三项联合训练](default_workflow.md)。
> 参考 head 序列 manifest 与 CTC-CRF 联合训练见
> [联合训练说明](reference_joint_training.md)，一键脚本为 `scripts/run_reference_joint.sh`。

从项目根目录运行；数据根目录、工作目录和 Bonito 模型由 `configs/local.yaml` 提供。
新数据集同时准备 v4 长度/状态 manifest 与匹配的 reference-head 序列 manifest。
联合 A/T 集使用严格均衡；不推断 flank。新训练默认 length + status + basecalling，
location 不监督。下面保留的单独长度训练命令仅用于历史实验或明确的对照。
数据制作和测试输出目录须尚不存在。已生成的 smoke 目录不能重复覆盖。

## Length + Status + Basecalling 联合训练

当前新 polyA 长度 manifest 和配套 reference-head sequence manifest 已完成。
下面先 dry-run，确认配置后去掉末尾开关开始训练：

```bash
GPU_ID=0 EPOCHS=5 \
LAMBDA_LENGTH=1 LAMBDA_STATUS=1 LAMBDA_BASECALL=0.1 \
bash scripts/run_reference_joint.sh \
  metadata/poly2p5per-head-poly-v4-polyA-5000pergroup-20261028-164859/manifest.tsv \
  metadata/poly2p5per-head-poly-v4-polyA-5000pergroup-20261028-164859-reference-head/sequence_manifest.tsv \
  --dry-run
```

先 dry-run 核对配置，去掉 `--dry-run` 才运行。默认启用三项 loss。
保持 sum pooling + Huber(delta=10 nt)，
status 使用构建来源的弱 present/zero 标签，location 不监督；不重定义 0 nt。
脚本新建实验目录，完成训练、最佳 validation 轮次测试和绘图。
默认从原 Bonito 权重和新 Poly heads 开始；不能用旧两项 loss 的 joint run
直接按三项 loss resume。旧实验恢复需显式 `--no-supervise-status` 并保持原配置。
状态分类及按标称长度分组的状态指标见测试 `summary.json`；原始回归指标不做状态判零。

## 独立对照：冻结 encoder/长度回归，单独验证 status

只用 train/validation，对原两个 status 分类器做类别平衡训练；
按 validation 的 0 nt 与 20 nt 召回共同选择 checkpoint，test 不参与。
使用刚结束的三项 loss 实验第 5 轮初始化：

```bash
GPU_ID=0 EPOCHS=20 BATCH_SIZE=32 LEARNING_RATE=0.001 \
bash scripts/run_status_calibration.sh \
  metadata/poly2p5per-v4-length-head-ablation-20261026-141955-subset/manifest.tsv \
  models/v4-reference-status-joint-20261028-140005/heads.epoch_0005.pt
```

可在末尾加 `--dry-run` 只检查 metadata。新目录为 `v4-status-calibration-时间戳`，
训练后自动绘制 0 nt 召回、20 nt 误判等曲线。原始长度预测保持不变。
完整设置、输出和续训见[冻结状态分类训练](status_calibration.md)。

## 安装父项目依赖

```bash
uv sync --frozen
```

## 生成小型数据集

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --structure head-poly --signal-end-policy raw-edge \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-poly-v4-polyA-smoke-new \
  --max-records 10000 \
  --per-group 3 \
  --poly-types polyA \
  --stop-when-full
```

## 生成 PolyA + PolyT 严格均衡数据集

正长度 × 类型各 200 条，0 nt 两方向各 200 条；六个正长度时共 2800 条。
不足时全部组缩减到共同可用数量；任一必需组少于 3 条则报错。
旧单类型命令保持原行为，严格均衡需要显式加 `--strict-balance`。

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --structure head-poly --signal-end-policy raw-edge \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-poly-v4-polyAT-balanced-200pergroup \
  --max-records 0 --per-group 200 --poly-types polyA polyT \
  --strict-balance --stop-when-full --seed 20260923
```

输出目录必须不存在。查看 `summary.json` 的 `balancing`、`selected_stratum_counts`
和 `quota_deficits` 确认实际数量。两份 BAM 各扫描一次，仅读取 POD5 元数据，
无 GPU；HDD 可能数分钟到数小时，全量 audit 可能每百万 reads 数百 MB。
该方案状态比例仍为 present:zero=6:1，status 类别权重需在训练时单独处理。
完整规则和限制见 [v4 联合均衡说明](prepare_dataset_v4.md#polya--polyt-严格均衡的小型联合数据集)。

## 生成训练数据集

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --structure head-poly --signal-end-policy raw-edge \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-poly-v4-polyA-6000pergroup \
  --max-records 0 \
  --per-group 6000 \
  --poly-types polyA \
  --stop-when-full
```

所有请求的正长度组与 0 nt 都达到 6000 条后，暂停边界和 POD5
元数据检查；BAM 剩余部分仍做轻量 UUID 检查，发现重复导致配额不足时恢复收集。
不足时继续到扫描上限或 EOF，数量以 summary 为准。该模式从已检查的候选中采样，
可能受 BAM 顺序影响；去掉 `--stop-when-full` 可恢复扫描范围内的全候选采样。

后续命令使用上述训练集和新的模型、日志、结果目录；不要续训旧 v3 checkpoint。
小型流程测试将目录统一换为 `poly2p5per-head-poly-v4-polyA-smoke-new`，并使用
`--epochs 1 --batch-size 1`。制作结果和选项说明见 [prepare_dataset_v4.md](prepare_dataset_v4.md)。

## 检查数据集和训练配置

```bash
uv run holypoly train \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-poly-v4-polyA-6000pergroup/manifest.tsv \
  --checkpoint models/poly2p5per-head-poly-v4-polyA-6000pergroup/heads.pt \
  --loss-log logs/poly2p5per-head-poly-v4-polyA-6000pergroup/train.jsonl \
  --epochs 5 --batch-size 32 --dry-run
```

## 从头训练 5 轮

```bash
CUDA_VISIBLE_DEVICES=0 uv run holypoly train \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-poly-v4-polyA-6000pergroup/manifest.tsv \
  --checkpoint models/poly2p5per-head-poly-v4-polyA-6000pergroup/heads.pt \
  --loss-log logs/poly2p5per-head-poly-v4-polyA-6000pergroup/train.jsonl \
  --epochs 5 --batch-size 32
```

训练支持 `--length-loss log_smooth_l1`（默认）、`--length-loss mae` 或
`--length-loss huber --huber-delta-nt 10`，详见[长度 loss 对照](length_losses.md)。
切换 loss 时新建 checkpoint 和日志目录，从头训练；续训要求 loss 配置一致。
一键比较三种 loss（固定同一个 head，各 20 轮，并自动测试和画图）可运行：

```bash
bash scripts/run_length_loss_ablation.sh "metadata/<small-v4-dataset>/manifest.tsv"
```

该 loss 对照默认使用 `mean_duration`；添加 `--dry-run` 可先检查读取量和输出路径。

比较平均池化与求和池化（相同层和初始化，默认 Huber、batch 8、20 轮、物理 GPU 0）：

```bash
bash scripts/run_length_pooling_ablation.sh \
  "metadata/<small-v4-dataset>/manifest.tsv" --dry-run
bash scripts/run_length_pooling_ablation.sh "metadata/<small-v4-dataset>/manifest.tsv"
```

自动选择最佳验证轮、测试、画图并生成对照表，见[池化对照说明](length_pooling_ablation.md)。

## 继续训练至累计 10 轮

仅续训本页同一 polyA 数据集的 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0 uv run holypoly train \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-poly-v4-polyA-6000pergroup/manifest.tsv \
  --checkpoint models/poly2p5per-head-poly-v4-polyA-6000pergroup/heads.pt \
  --loss-log logs/poly2p5per-head-poly-v4-polyA-6000pergroup/train.jsonl \
  --epochs 10 --batch-size 32 --resume
```

## 测试预检查

以下测试、画图命令按第 5 轮组织；续训后将 `test-epoch5` 换为对应轮数的新目录。

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-poly-v4-polyA-6000pergroup/manifest.tsv \
  --checkpoint models/poly2p5per-head-poly-v4-polyA-6000pergroup/heads.pt \
  --output-dir results/training_evaluation/poly2p5per-head-poly-v4-polyA-6000pergroup/test-epoch5 \
  --split test --batch-size 32 --dry-run
```

## 开启测试

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-poly-v4-polyA-6000pergroup/manifest.tsv \
  --checkpoint models/poly2p5per-head-poly-v4-polyA-6000pergroup/heads.pt \
  --output-dir results/training_evaluation/poly2p5per-head-poly-v4-polyA-6000pergroup/test-epoch5 \
  --split test --batch-size 32 --max-reads 0
```

## 画训练日志

导出逐 epoch 曲线 `training_curves.*`、分项 loss 曲线 `loss_components.*`
和最新一轮汇总；逐轮数值保存在 `epoch_metrics.tsv`。

```bash
uv run --no-sync python scripts/plot_training.py \
  --config configs/local.yaml \
  --log logs/poly2p5per-head-poly-v4-polyA-6000pergroup/train.jsonl \
  --output results/training_evaluation/poly2p5per-head-poly-v4-polyA-6000pergroup/training-plots
```

## 画测试结果

```bash
uv run --no-sync python scripts/plot_training.py \
  --config configs/local.yaml \
  --predictions results/training_evaluation/poly2p5per-head-poly-v4-polyA-6000pergroup/test-epoch5/predictions.tsv \
  --output results/training_evaluation/poly2p5per-head-poly-v4-polyA-6000pergroup/test-epoch5/plots
```

v4 全部为 `length_only`，无需绘制 full 子集。长度图使用原始回归输出；
未训练的状态头门控输出不用于评价。图表横轴为构建标称弱标签，不能解释为逐 read 真值。
