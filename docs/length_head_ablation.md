# 长度头三组对照

> 可选对照实验。常规新训练使用 [run_reference_joint 主流程](default_workflow.md)。

三组复用同一份已有 manifest，不重新制作数据，不修改标签、原有实验或 Bonito。
默认仍为 `mean`，原 `PolyLengthHead` 的结构、参数名和计算保持不变。

另有可选的 `sum` 池化及独立的两组对照脚本，见[池化对照说明](length_pooling_ablation.md)。
本页的原三组脚本继续只比较下表中的模式。

| `--length-head-mode` | 长度回归输入 | 回归结构 |
| --- | --- | --- |
| `mean` | 原有投影特征的 masked mean | 原有 Linear → softplus |
| `duration_only` | `log1p(有效原始 sample 数)` | Linear(1, H) → GELU → Dropout → Linear(H, 1) → softplus |
| `mean_duration` | 原有 masked mean 与 `log1p(有效原始 sample 数)` 拼接 | Linear(H+1, H) → GELU → Dropout → Linear(H, 1) → softplus |

`H` 默认 128。有效 sample 数来自 raw padding mask，未提供 mask 时为实际输入长度，
不通过 encoder timestep 或固定 stride 反推。没有用验证集或测试集拟合标准化参数。
这里使用 sample 数；不同采样率条件下不能直接解释为相同物理时长。

这是 head 后候选区间的时长，可能包含接头和噪声，不是经独立验证的纯 Poly 时长。
`duration_only` 的长度和状态输出不使用波形特征，但为保持完整模型输出契约，
仍运行冻结 encoder、原 basecalling head 和定位 head，因此不会明显节省整体计算。

定位 head、原 CRF、训练 loss、长度标签和冻结策略保持原样。
v4 的定位和状态没有监督；比较时使用 `regression_length_nt`，不使用状态门控长度。
新模式的 MLP 容量与旧线性回归器不同，因此提升只能先支持“新输入及回归器组合有用”，
不能单独证明平均池化是唯一原因；严格分离容量影响还需后续容量匹配对照。

## 运行

用外部 work root 下的现有 v4 manifest，例如：

```bash
bash scripts/run_length_head_ablation.sh \
  metadata/<existing-v4-dataset>/manifest.tsv
```

脚本固定物理 GPU 0（模型内部 `cuda:0`），按三种模式顺序执行训练、测试、画图。
默认读取 `configs/local.yaml`，训练 5 轮、batch 32、lr 0.001、seed 20260923；
三个 loss 权重均来自同一配置。可通过第二个参数选择其他配置，通过环境变量
`EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE`、`SEED`、`EXPERIMENT_ID` 调整共同参数。

每组从同一个预训练模型新建辅助 head，不从之前训练权重继续。
相同训练划分、batch size、轮数意味着相同优化步数；三个模型分别重新设置相同种子。
脚本显式使用 `--data-order-seed`：每轮以该值加 epoch 重设独立 shuffle generator，
保证不同网络初始化不会改变训练 read 顺序，续训也使用对应轮次顺序。
不提供这个选项时，原有训练的数据打乱方式保持不变。
所有输出写入新的外部 `logs/models/results` 子目录，已有同名目录会报错，不自动覆盖。
真实训练不会由开发测试自动启动。

每组产出 `training-plots/`、`test-final/summary.json`、`metrics_by_group.tsv`、
`predictions.tsv` 和 `test-final/plots/`。对照目录额外产出：

* `comparison.tsv`：总体指标和单独的正样本 MAE。
* `comparison_by_length.tsv`：各长度指标，优先检查 100/150 nt 的 Bias 和 ±10 nt 比例。

比较固定末轮；若改为选择最佳轮，必须统一依据验证集规则，不能按测试结果选权重。
同批次随机 read 划分只支持构建弱标签一致性结论，不证明跨批次泛化或逐 read 准确率。

## 小样本初筛

小样本初筛可在运行命令前设置 `SUBSET_PER_GROUP=200`：

```bash
SUBSET_PER_GROUP=200 bash scripts/run_length_head_ablation.sh \
  metadata/<existing-v4-dataset>/manifest.tsv
```

脚本只从已有 manifest 抽样，每个标称长度/type 200 条，保留原 split 和全部标签字段。
当前 7 组共 1,400 条：训练 1,120、验证 140、测试 140，各组为 160/20/20。
三个模型共享生成的新 manifest。固定种子；不会扫描 BAM/POD5 元数据或复制原始信号。
新 manifest 和来源 SHA256、抽样种子、数量汇总写入新的外部 metadata 子目录。
任何源分层不足配额会明确报错，不重新分配 split，也不静默缩小其他组。
`SUBSET_PER_GROUP` 默认 0（使用完整源 manifest），非零值要求至少 10 且为 10 的倍数。

默认 5 轮、batch 32 时每个模型只有 175 次更新，原完整集为 5,250 次。
这适合链路和趋势初筛；若验证误差仍下降，可以用新的实验 ID 和更多轮数继续对照，
小样本训练效果差不能单独否定时长特征。缩小数据不会缩小完整 checkpoint 的大小，
三组各 5 轮仍约占 5.5 GiB，增加轮数还会增加逐轮权重数量。

也可单独制作 manifest，先通过 `--dry-run` 检查数量：

```bash
PYTHONPATH=src .venv/bin/python -m holypoly.datasets.subset_manifest \
  --config configs/local.yaml \
  --manifest metadata/<existing-v4-dataset>/manifest.tsv \
  --output-dir metadata/<new-small-dataset> --per-group 200 --seed 20260923 \
  --dry-run
```

去掉 `--dry-run` 才写入文件。之后可把小 manifest 直接传给对照脚本。

## 单独运行和兼容

已完成默认 5 轮的三组实验，可以复制第 5 轮起点到新目录并继续训练至总计 20 轮：

```bash
bash scripts/legacy/resume_length_head_ablation.sh <source-experiment-id>
```

原 manifest、权重、日志、测试和图表全部保留。新目录包含原 1–5 轮日志和新 6–20 轮记录；
新权重保存第 6–20 轮及 latest，第 1–5 轮快照仍在原目录。
脚本根据 1–20 轮的验证集 MAE 选择最佳权重，再测试并画图，记录 `best_checkpoint.json`。
如果最佳轮次落在前 5 轮，将只读引用原目录中的对应权重。
新结果路径为 `results/training_evaluation/<new-experiment-id>-<mode>/test-best-validation/`。
此脚本只适用于默认参数且完整完成 5 轮的对照；不会重新抽样，也不会从零训练。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/holypoly train \
  --config configs/local.yaml \
  --manifest metadata/<existing-v4-dataset>/manifest.tsv \
  --length-head-mode mean_duration \
  --checkpoint models/<new-run>/heads.pt \
  --loss-log logs/<new-run>/train.jsonl \
  --epochs 5 --batch-size 32 --device cuda:0
```

也可配置 `training.length_head_mode`。新 checkpoint 保存 `head_config`（模式、隐藏层宽度、
dropout），评估自动按其重建网络；旧 checkpoint 无该字段时仍按原 mean 网络恢复。
续训需选择相同模式和隐藏层宽度；新 checkpoint 的 dropout 也必须一致，错误配置会拒绝。
