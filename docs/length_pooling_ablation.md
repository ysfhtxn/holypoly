# 平均池化与求和池化对照

> 可选对照实验。常规新训练使用 [run_reference_joint 主流程](default_workflow.md)。

`--length-head-mode sum` 是独立可选模式；默认仍为原 `mean`，旧网络和旧结果保持原样。
两者均为 LayerNorm → Linear(H) → GELU → Dropout → masked pooling → Linear → softplus。
所有层、参数名、参数量和相同种子下的初始化一致，唯一差别是 pooling：

- `mean`：有效 encoder timestep 的投影特征之和 / 有效 timestep 数。
- `sum`：有效 encoder timestep 的投影特征之和，不除以有效 timestep 数。

两者都排除 padding。求和后不额外归一化，否则会削弱本次要比较的累计信息。
时长是候选信号的时长，可能包含非 Poly 信号；求和不能自动识别纯 Poly 边界。
两组均不拼接 `mean_duration` 使用的显式 raw sample 数，因此不是之前三组 loss
实验中 `mean_duration` 的直接复跑。定位 head、冻结 encoder、原 CRF 和标签不变。
v4 为 `length_only`，比较原始 `regression_length_nt`；定位和状态没有监督，
不要用状态解码结果判断识别能力。

## 一键运行

脚本复用现有 manifest，默认不抽样、不扫描 BAM、不重建数据：

```bash
# 只检查 metadata、配置、输出路径和资源估算，不加载信号/模型、不写输出。
bash scripts/run_length_pooling_ablation.sh \
  "metadata/<existing-small-v4-dataset>/manifest.tsv" --dry-run

# 顺序训练两组 → 按验证集 MAE 选择最佳轮 → 测试 → 画图 → 汇总。
bash scripts/run_length_pooling_ablation.sh \
  "metadata/<existing-small-v4-dataset>/manifest.tsv"
```

第二个位置参数可指定配置文件，默认 `configs/local.yaml`。
默认参数为 `GPU_ID=0`、`EPOCHS=20`、`BATCH_SIZE=8`、`LENGTH_LOSS=huber`、
`HUBER_DELTA_NT=10`、`LEARNING_RATE=0.001`、`SEED=20260923`。
物理 GPU 由 `GPU_ID` 指定，模型内部为映射后的 `cuda:0`。
这些参数均可用同名环境变量覆盖，例如：

```bash
GPU_ID=0 LENGTH_LOSS=mae SEED=17 \
  bash scripts/run_length_pooling_ablation.sh "metadata/<small-v4-dataset>/manifest.tsv"
```

两组从相同预训练 Bonito 和相同辅助层初始化开始，不从旧实验继续。
独立 data-order seed 保证相同的逐轮 batch 顺序；lr、梯度裁剪和优化器配置一致。
求和会改变数值/梯度尺度，因此结果只比较当前共同超参数下的效果，
不能代表每种池化充分调参后的上限。若求和不稳定，应先检查训练曲线再讨论学习率。

`SUBSET_PER_GROUP=200` 可选地从较大 manifest 生成一份两组共享的分层子集；
默认 0 直接使用已有全部行。标签和 split 不变。不要对已有 200/group 小集重复抽样。
`EXPERIMENT_ID` 默认带时间戳；任何目标目录已存在即报错，不覆盖旧结果。

## 输出和检查

所有运行产物位于外部 work root：

- `models/<experiment>-<mean|sum>/heads.epoch_*.pt`：逐轮完整 checkpoint。
- `logs/<experiment>-<mean|sum>/train.jsonl`：训练/验证曲线原始记录。
- `results/training_evaluation/<experiment>-<mean|sum>/best_checkpoint.json`：最佳验证轮及选择规则。
- `results/training_evaluation/<experiment>-<mean|sum>/test-best-validation/`：测试汇总、逐条预测和图。
- `results/training_evaluation/<experiment>-<mean|sum>/training-plots/`：训练曲线。
- `results/training_evaluation/<experiment>/comparison.tsv` 和 `comparison_by_length.tsv`：整体及分长度对照。

脚本检查完整训练轮次、被测试的权重与最佳轮一致、两组测试 read/目标一致，
以及 loss、初始化种子、batch size 和模型来源等设置一致。
评价优先看 MAE、±10 nt 比例，以及 0/100/150 nt 的 Bias。
单个种子、小测试集只用于初筛；构建标称弱标签一致性不等于逐 read 真实准确率。

已有 1,400 条小集预计两组共约 2–3 小时，20 轮的完整权重约 13 GiB；
实际取决于 GPU、信号长度及占用情况。运行前以 dry-run 估算为准。
开发验证不会启动真实训练。
