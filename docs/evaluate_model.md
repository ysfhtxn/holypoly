# 从训练到测试和绘图

训练完成后，用 `src/holypoly/training/evaluate.py` 加载 checkpoint，对 manifest
中的 `test` split 推理。它不创建优化器、不更新模型、不写回 checkpoint，也不修改
POD5。运行环境是 `src/bonito/.venv`，父项目不需要新增 Torch、POD5 或绘图依赖。

## 1. 训练

以下命令从项目根目录运行。模型目录、data root 和 work root 由本地配置提供，
也可以使用 CLI 或已有环境变量。机器路径只保存在本地配置，不写入本说明。

```bash
CUDA_VISIBLE_DEVICES=0 uv run holypoly train \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-train-v1/manifest.tsv \
  --checkpoint models/poly2p5per-train-v1/heads.pt \
  --loss-log logs/poly2p5per-train-v1/train.jsonl \
  --epochs 1 --batch-size 32
```

已有 checkpoint 时不要重复运行新训练；如需继续，用同样路径加
`--epochs 10 --resume`，表示训练至总计第 10 轮。先用 validation 选择参数和轮数，
再用 test 评估最终模型；若根据 test 结果继续调参，该集合不再是独立的最终评估集。

## 2. 测试预检查

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-train-v1/manifest.tsv \
  --checkpoint models/poly2p5per-train-v1/heads.pt \
  --output-dir results/poly2p5per-train-v1/test-epoch1 \
  --batch-size 32 --dry-run
```

预检查只读取 manifest 和路径元数据，不加载权重或电流信号，也不写输出。
它打印 split 数量、POD5 文件数、预计读取的完整 raw samples 和 checkpoint 大小。
同一 UUID 出现在多个 chunk 或 split 时直接报错。

## 3. 导出测试结果

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-train-v1/manifest.tsv \
  --checkpoint models/poly2p5per-train-v1/heads.pt \
  --output-dir results/poly2p5per-train-v1/test-epoch1 \
  --batch-size 32
```

只使用一个显式选择的 GPU，内部为 `cuda:0`。缓存默认放在 work root 的 `cache/`
中；如相关缓存环境变量已经设置，也必须指向 work root 中的路径。
输出目录必须尚不存在；再次评估时使用新的目录，例如 `test-epoch10`。
显存不足时减小 `--batch-size`，不会改变测试集。

当前 741 条 test read，按此前 validation 的 24 个 batch、约 5 秒/batch 估计，
batch size 32 的测试可能约 2–4 分钟，加上模型初始化；该时间尚未实际测量。
只读取这些 read 的 POD5 信号及一个 checkpoint，不扫描全体 BAM/POD5。
输出表预计小于数 MB。

先做少量 read 的链路检查，可另选输出目录并加 `--max-reads 32`。
它按 manifest 顺序取前 32 条，汇总明确记录 `partial_evaluation=true`，不代表
随机抽样或完整 test 指标。`--max-reads 0` 默认处理整个 split。
如需分析开发数据，可用 `--split validation`；不提供 train split 推理选项。

输出位于 `<work_root>/results/poly2p5per-train-v1/test-epoch1/`：

- `predictions.tsv`：每个 read 一行，包含标称长度、训练目标、观察序列长度、方向、
  poly 类型、监督方式、标签来源、原始回归长度、按状态解码的长度、状态置信度和
  可解码的定位区间。输入读取失败或非有限预测也保留行和原因，不静默丢弃。
- `metrics_by_group.tsv`：按“标称长度 × poly 类型 × 监督方式”分组，报告与训练目标
  的 MAE、Median AE、RMSE、Bias、1/5/10 nt 内比例及相关性。常数长度组的相关性
  不定义，留空；分母是有合法标签且成功得到回归预测的 read，失败数另列。
- `summary.json`：全体弱标签一致性指标、失败数量、是否部分测试、输入路径、
  checkpoint epoch/step、随机种子、模型/loss 配置、版本和时间。

长度指标使用 `regression_length_nt`，与训练日志一致。
`decoded_length_nt` 使用 length head 状态：zero_nt 变为 0，failure 留空；
`shared_status` 和定位解码使用两侧 head 的平均状态 logits，含义分别保留。
`prediction_status=ok` 表示推理成功，不表示模型一定认为该 read 有 Poly。
`shared_status_failure_reads` 是模型输出 failure 状态的数量，与输入读取失败不同。

定位区间是当前 decoder 得到的最长连续 Poly run，统一为原始未裁剪信号上的
0-based、左闭右开 `[predicted_start_sample, predicted_end_sample)`；找不到可解码
区间时留空。只监督长度的标签不能用于评价这些定位或状态预测。

标称构建长度和 provisional 边界不等于可靠逐 read ground truth；这些指标只能
解释为弱标签一致性。当前 split 来自同一批次，不证明跨批次泛化。

## 4. 绘制测试结果

绘图脚本直接读取测试导出的 TSV，无需重新推理或加载 POD5/checkpoint。
Matplotlib 已记录在根项目依赖中，使用 uv 环境运行，不依赖 Conda base 的版本。
首次运行可先执行 `uv sync --frozen` 安装锁定依赖：

```bash
uv run python scripts/plot_training.py \
  --config configs/local.yaml \
  --predictions results/poly2p5per-train-v1/test-epoch1/predictions.tsv \
  --output results/poly2p5per-train-v1/test-epoch1/plots
```

输出 `test_summary.png/.svg/.pdf` 四个面板：按标称长度的样本数量、分方向的预测
长度箱线分布、标称与预测长度散点、预测减标称的有符号误差分布。
样本数量统计选中行（含输入失败）；其他面板仅绘制有限预测和有限标称标签，
过滤数量记录于 `plot_summary.json`。箱体为中位数和 IQR，须为 1.5 IQR；
全部点包含离群点，不抽样。箱线图中的水平抖动仅用于显示，种子默认 20260923。
曲线/虚线只作标称目标或零偏差参考，不是置信区间。
`source_predictions.tsv` 保留原始输入，脚本副本也保存在输出目录。

默认合并两种监督方式。分别检查时加 `--supervision-mode full` 或
`--supervision-mode length_only`，同时使用不同输出目录。
横轴均注明标称长度，长度预测使用 `regression_length_nt`，与测试指标一致。
该命令可以重复运行，会更新指定目录中的绘图文件；不会修改原测试结果。

## 5. 绘制训练日志

原临时绘图脚本已移到 `scripts/plot_training.py`，使用根项目的 uv 环境。
绘图不会加载 Bonito 或 Torch，也不修改 Bonito 独立环境：

```bash
uv run python scripts/plot_training.py \
  --config configs/local.yaml \
  --log logs/poly2p5per-train-v1/train.jsonl \
  --output results/poly2p5per-train-v1/training-plots
```

输出最新一轮的 loss/长度误差汇总 PNG、SVG、PDF，以及源指标和绘图脚本副本。
`--log` 和 `--predictions` 二选一，分别用于训练汇总和逐 read 测试图。

本次开发验证使用 synthetic 信号和 CPU，另对已有外部 manifest/checkpoint 做
dry-run；没有启动真实 test split 推理，因此还没有本批完整测试指标。
