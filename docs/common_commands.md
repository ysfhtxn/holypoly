# 常用命令

从项目根目录运行；数据根目录、工作目录和 Bonito 模型由 `configs/local.yaml` 提供。
以下使用无 head 的 v3 数据集；数据制作和测试输出目录须尚不存在。

## 安装父项目依赖

```bash
uv sync --frozen
```

## 生成小型数据集

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-free-v3-smoke \
  --max-records 10000 \
  --per-group 3
```

## 生成训练数据集

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-free-v3 \
  --max-records 0 \
  --per-group 1000
```

后续命令使用训练集；小型流程测试将目录统一换为 `poly2p5per-head-free-v3-smoke`，
并使用 `--epochs 1 --batch-size 1`。

## 检查数据集和训练配置

```bash
uv run holypoly train \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-free-v3/manifest.tsv \
  --checkpoint models/poly2p5per-head-free-v3/heads.pt \
  --loss-log logs/poly2p5per-head-free-v3/train.jsonl \
  --epochs 5 --batch-size 32 --dry-run
```

## 从头训练 5 轮

```bash
CUDA_VISIBLE_DEVICES=0 uv run holypoly train \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-free-v3/manifest.tsv \
  --checkpoint models/poly2p5per-head-free-v3/heads.pt \
  --loss-log logs/poly2p5per-head-free-v3/train.jsonl \
  --epochs 5 --batch-size 32
```

## 继续训练至累计 10 轮

仅续训同一 v3 数据集的 checkpoint；不接着训练旧的含 head 模型。

```bash
CUDA_VISIBLE_DEVICES=0 uv run holypoly train \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-free-v3/manifest.tsv \
  --checkpoint models/poly2p5per-head-free-v3/heads.pt \
  --loss-log logs/poly2p5per-head-free-v3/train.jsonl \
  --epochs 10 --batch-size 32 --resume
```

## 测试预检查

以下测试、画图命令按第 5 轮组织；续训后将 `test-epoch5` 换为对应轮数的新目录。

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-free-v3/manifest.tsv \
  --checkpoint models/poly2p5per-head-free-v3/heads.pt \
  --output-dir results/poly2p5per-head-free-v3/test-epoch5 \
  --split test --batch-size 32 --dry-run
```

## 开启测试

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-free-v3/manifest.tsv \
  --checkpoint models/poly2p5per-head-free-v3/heads.pt \
  --output-dir results/poly2p5per-head-free-v3/test-epoch5 \
  --split test --batch-size 32
```

## 画训练日志

```bash
uv run python scripts/plot_training.py \
  --config configs/local.yaml \
  --log logs/poly2p5per-head-free-v3/train.jsonl \
  --output results/poly2p5per-head-free-v3/training-plots
```

## 画测试结果

```bash
uv run python scripts/plot_training.py \
  --config configs/local.yaml \
  --predictions results/poly2p5per-head-free-v3/test-epoch5/predictions.tsv \
  --output results/poly2p5per-head-free-v3/test-epoch5/plots
```

## 按监督方式分别画图

```bash
uv run python scripts/plot_training.py \
  --config configs/local.yaml \
  --predictions results/poly2p5per-head-free-v3/test-epoch5/predictions.tsv \
  --supervision-mode full \
  --output results/poly2p5per-head-free-v3/test-epoch5/plots-full
```

```bash
uv run python scripts/plot_training.py \
  --config configs/local.yaml \
  --predictions results/poly2p5per-head-free-v3/test-epoch5/predictions.tsv \
  --supervision-mode length_only \
  --output results/poly2p5per-head-free-v3/test-epoch5/plots-length-only
```

图表横轴为构建标称长度，属于弱监督标签。
