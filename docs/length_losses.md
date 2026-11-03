# 可选择的长度回归 loss

`holypoly train --length-loss` 支持三种方式，默认继续使用 `log_smooth_l1`。
只更改训练循环中的长度项；网络结构、原 CRF、定位/状态项、样本筛选及训练目标均不改变。
所有 length head（`mean`、`sum`、`duration_only`、`mean_duration`）都支持这些 loss。
平均与求和池化的单独比较见[池化对照说明](length_pooling_ablation.md)。

设预测长度为 `p`，manifest 训练目标为 `y`，两者单位为 nt：

| 参数值 | 每条 read 的误差项 | 说明 |
| --- | --- | --- |
| `log_smooth_l1` | `SmoothL1(log1p(p), log1p(y), beta=1)` | 原有默认，保持数值一致 |
| `mae` | `abs(p - y)` | 原始 nt 空间的绝对误差 |
| `huber` | `0.5 * e²`，当 `abs(e) <= delta`；否则 `delta * (abs(e) - 0.5 * delta)` | `e=p-y`，使用 PyTorch Huber 定义 |

三者都在有效长度目标上取平均。`--huber-delta-nt` 默认 10，必须为有限正数，
仅在 `huber` 模式下影响计算；不是 SmoothL1 的 `beta`，不自动除以 delta。
非失败的 `zero_nt` 和 `length_only` 目标参与回归；failure 的 NaN 目标被排除。
非失败目标为 NaN、无穷或负数时明确报错。padding 不参与定位 loss。
所有长度目标均被排除时，长度 loss 为可反向传播的零。

总 loss 仍为三个分项按 `lambda_location`、`lambda_length`、`lambda_status` 加权求和，
三个权重保持配置值。不同定义的 loss 数值尺度不同，尤其 Huber 的线性区梯度大小
为 delta；不能直接通过 total loss 的大小比较实验优劣。
v4 的定位和状态项仍不受监督。

## 参数与配置

在完整训练命令中添加相应参数：

```bash
--length-loss log_smooth_l1
--length-loss mae
--length-loss huber --huber-delta-nt 10
```

也可以在配置文件中设置，CLI 优先于配置：

```yaml
model:
  auxiliary_loss:
    lambda_location: 1.0
    lambda_length: 1.0
    lambda_status: 1.0
    length_loss: huber
    huber_delta_nt: 10.0
```

## 一键训练、测试和画图

`scripts/run_length_loss_ablation.sh` 默认固定 `mean_duration`，在 GPU 0 上分别从头
训练三种 loss，各 20 轮。三组共用 manifest、初始化种子和每轮 batch 顺序；
每组按验证集 nt MAE 选择最佳轮次，测试并画图，最后生成两个对照表。
默认不重新抽样；直接复用已有小数据集可与前面的 head 对照保持相同 read 划分。

```bash
# 先验证参数、manifest、输出路径和预计读取量，不写文件。
bash scripts/run_length_loss_ablation.sh \
  "metadata/<small-v4-dataset>/manifest.tsv" --dry-run

# 执行完整对照，所有输出使用新的时间戳实验目录。
bash scripts/run_length_loss_ablation.sh \
  "metadata/<small-v4-dataset>/manifest.tsv"
```

第二个位置参数可指定配置文件。环境变量支持 `EPOCHS`（20）、`BATCH_SIZE`（32）、
`LEARNING_RATE`（0.001）、`SEED`（20260923）、`LENGTH_HEAD_MODE`（mean_duration）、
`HUBER_DELTA_NT`（10）、`EXPERIMENT_ID`（自动时间戳）。
若输入完整 manifest，可设置 `SUBSET_PER_GROUP=200` 先制作一个三组共享的小 manifest；
默认 0 使用现有全部行。选样只读取 manifest，保留原 split 和全部标签。

例如只改 head 和 Huber 阈值：

```bash
LENGTH_HEAD_MODE=mean HUBER_DELTA_NT=5 \
  bash scripts/run_length_loss_ablation.sh "metadata/<small-v4-dataset>/manifest.tsv"
```

每组产出 `models/<experiment>-<loss>/heads.epoch_*.pt`、日志、
`results/training_evaluation/<experiment>-<loss>/best_checkpoint.json`、`training-plots/` 和
`test-best-validation/`（含测试汇总、分长度指标、预测和图表）。
对照表位于 `results/training_evaluation/<experiment>/comparison.tsv`、`comparison_by_length.tsv`。
脚本检查三组测试 read ID 和目标一致，不按测试结果选权重。
小集 1,400 条三组各 20 轮约 3 小时，完整逐轮权重约 19 GiB，实际依赖 GPU/模型。
完整 manifest 的时间和读取量会明显增加，运行前先检查 dry-run 的估算。

## 仅运行训练

先用小 manifest，新建实验目录，从相同随机种子和同一预训练 Bonito 初始化。
例如下列命令比较同一个 `mean_duration` head 的三种 loss；需先把 manifest
占位路径改为已有小数据集，不重新制作或抽样：

```bash
EXPERIMENT_ID="length-loss-$(date +%Y%m%d-%H%M%S)"
for loss in log_smooth_l1 mae huber; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/holypoly train \
    --config configs/local.yaml \
    --manifest "metadata/<small-v4-dataset>/manifest.tsv" \
    --length-head-mode mean_duration \
    --length-loss "$loss" --huber-delta-nt 10 \
    --checkpoint "models/${EXPERIMENT_ID}-${loss}/heads.pt" \
    --loss-log "logs/${EXPERIMENT_ID}-${loss}/train.jsonl" \
    --epochs 20 --batch-size 32 --learning-rate 0.001 \
    --random-seed 20260923 --data-order-seed 20260923 \
    --chunk-size-samples 12000 --hidden-features 128 --dropout 0 \
    --weight-decay 0 --gradient-clip-norm 2 --num-workers 0 --device cuda:0
done
```

这仅启动三次训练，测试和画图命令与现有流程相同。
每组按验证集原始 nt MAE 选择权重；关注 0/100/150 nt 的 Bias 和 ±10 nt 比例。
评价仍为构建标称弱标签一致性，不能作为逐 read 真值准确率。

## 记录与旧 checkpoint

训练 dry-run、每轮 train/validation JSONL 和 checkpoint 的 `loss_config` 都记录
三个权重、`length_loss`、`huber_delta_nt`；评估汇总继承 checkpoint 的配置。
旧 checkpoint 缺少新字段时按 `log_smooth_l1`、delta=10 解释，不改写旧文件。
`--resume` 要求 loss 类型、三个权重以及 Huber 模式下的 delta 与 checkpoint 一致；
不一致时在恢复模型/优化器状态前报错。比较不同 loss 应新建实验，从头训练。
原型 head 的独立 `.loss()` 辅助接口仍保留原 log loss，CLI 训练使用
`training/losses.py` 的 `PolyAuxiliaryLoss`。
