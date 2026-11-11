# 冻结特征的状态分类验证

> 可选对照实验。常规新训练使用 [run_reference_joint 主流程](default_workflow.md)。

本实验用于判断已有 encoder/投影特征能否区分 0 nt 背景和 Poly。
沿用 v4 的构建弱标签，不改 0 nt 定义、网络层、pooling、原 manifest 或旧结果。

## 冻结与优化目标

从指定完整 checkpoint 加载所有权重，然后冻结所有参数和缓冲区，
只训练原 location head 和 length head 的两个 `status_classifier` 线性层。
length head 的 projection 与回归共享，因此它也必须冻结；location projection
同样冻结。保持全模型 eval，避免 dropout 或其他缓冲区导致原输出变化。

```text
shared_logits = (location_status_logits + length_status_logits) / 2
loss = mean(weight[label] × CrossEntropy(shared_logits, label))
```

类别权重仅根据 train 数量计算：`N / (2 × N_class)`。
当前小数据集 train 为 960 present、160 zero，因此权重为
`present=0.583333, zero=3.5`，相对权重 1:6。使用固定单条样本权重，
不在每个 batch 内重新估计类别比例，不改变采样分布。
原三分类 softmax 保留，failure 无训练例，目标权重为 0；这不验证 failure 检测。
length、location、basecalling loss 均不参与本阶段优化。

## 数据与 checkpoint 选择

- 只读取指定 manifest 的 train/validation read，test 不读取、不参与选择。
- 通过原模型 status 分类器的输入 hook 捕获实际 pooled features，保持原 pooling。
- 每条 read 的冻结特征提取一次，只保留在内存，不缓存真实 signal 或 encoder 时序特征。
- 默认 20 轮、batch=32、AdamW LR=1e-3、weight decay=0、clip=2、seed=20260923。
- 每轮用固定的 shared argmax 评估 train/validation，不根据 test 调阈值。
- 主选择分数为 validation 的 **0 nt 召回与 20 nt present 召回的调和平均**。
  分数相同时依次选全部正样本召回更高、平衡 CE 更低、轮次更早的模型。
  这避免“全部有 Poly”或“全部零”获得较高主分数；不保证某项达到预设指标。

同时记录 20 nt 被判零的比例、全部正样本召回、failure 判定及混淆矩阵。
原始长度和 basecalling 参数完全冻结，结束时逐项核对所有冻结参数/缓冲区
与初始化 checkpoint 完全一致。**原始回归的 0 nt 高估不会在本阶段下降**；
本阶段成功的证据是状态区分改善且 20 nt 不被过度判零。
状态输出与原始回归值分开报告，尚不强制改写长度。

## 运行

路径相对于配置的 work root：

```bash
GPU_ID=0 EPOCHS=20 BATCH_SIZE=32 LEARNING_RATE=0.001 \
bash scripts/run_status_calibration.sh \
  metadata/experiment/manifest.tsv \
  models/source-joint/heads.epoch_0005.pt --dry-run
```

去掉 `--dry-run` 开始训练和绘图。dry-run 不读真实信号、权重或 GPU，不创建输出。
脚本创建全新的 `v4-status-calibration-时间戳` 目录，拒绝覆盖旧目录。
如 source 为 joint checkpoint，会校验其记录的长度 manifest 哈希；
旧冻结 checkpoint 缺少此信息时，调用者须确认 split 一致且未使用 validation/test 训练。

当前 1400-read 小数据集只提取 1260 个 train/validation read，
原信号解压 float32 约 0.079 GiB，另读约 0.33 GiB 初始化 checkpoint 和 Bonito 权重。
pooled features 约 1.3 MiB，模型和原 checkpoint 在内存中另占空间。
单卡 GPU 0，首次特征提取预计数分钟，实际依磁盘/GPU 而定。
完整 latest/best 合计约 0.65 GiB；每轮小状态快照通常不到 0.1 MiB。
latest 每轮更新，20 轮完整权重累计写入约 6.5 GiB，另有 best 更新和日志/绘图。

## 结果与恢复

输出在外部 work root：

- `logs/<run>/train.jsonl`：每轮 train/validation 的 epoch-end 指标。
- `results/training_evaluation/<run>/validation_baseline.json`：初始化时的 validation 状态表现。
- `results/training_evaluation/<run>/best_checkpoint.json`：选择规则、最佳轮次及 validation 指标。
- `results/training_evaluation/<run>/completion.json`：结束时间、冻结一致性检查、配置和最佳结果。
- `models/<run>/heads.pt`、`heads.best.pt`：完整 latest/best，可由现有 evaluate 消费。
- `models/<run>/status.epoch_XXXX.pt`：只含状态参数、optimizer 和实验元数据的
  小型每轮快照，依赖原初始化权重；不能直接作为完整模型交给 evaluate。
- `results/training_evaluation/<run>/training-plots/training_curves.{png,svg,pdf}`：六项状态诊断曲线。

续训用独立入口；`--epochs` 为目标总轮数，其余设置必须保持一致：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
src/bonito/.venv/bin/python -m holypoly.training.status_command \
  --config configs/local.yaml \
  --manifest metadata/experiment/manifest.tsv \
  --initialize-from models/source-joint/heads.epoch_0005.pt \
  --run-id <existing-calibration-run> --epochs 40 --resume
```

恢复从完整 latest 加载模型、optimizer、epoch、step 和随机状态，重新提取冻结特征，
检查日志与配置；拒绝覆盖已有每轮快照。断电若导致 checkpoint/日志不一致，
需人工核对，不自动删除或回退结果。旧 joint/冻结训练入口不能恢复此实验。

这是构建弱标签下的小规模分类验证。验证集改善仍需一次未参与调参的 test 检查，
再讨论联合训练和状态门控；不是逐 read 真值准确率或跨批次泛化结论。
