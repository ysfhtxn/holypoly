# 逐 epoch checkpoint

常规联合训练在每轮训练和 validation 完成后保存完整 epoch 快照。
`EXPERIMENT_ID=experiment` 时，模型目录为 `models/experiment/`：

```text
models/experiment/
├── heads.pt                # 最新完成的 epoch，原 --resume 入口
├── heads.epoch_0001.pt
├── heads.epoch_0002.pt
└── heads.epoch_0003.pt
```

快照与最新 checkpoint 使用相同格式，包含完整模型（含共享 Bonito）、优化器、
epoch、global step、随机状态、manifest schema、模型/loss 配置和版本信息。
只序列化一次，再复制到独立快照；每个文件均通过临时文件原子替换。
快照与最新文件不是硬链接，后续更新不会改变以前的 epoch。

## 联合训练恢复

```bash
EXPERIMENT_ID=experiment EPOCHS=10 GPU_ID=0 \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv --resume --dry-run
```

检查后去掉 `--dry-run`。`EPOCHS` 是目标总轮数，模型、loss 和输入来源必须一致；
已完成的轮次不覆盖。详细恢复检查和可选 batch 调整见
[联合训练说明](reference_joint_training.md)。联合入口不提供 `--overwrite`。

## 历史冻结入口的恢复与覆盖

下面的 `--checkpoint` / `--overwrite` 规则属于 `holypoly train`，不用于恢复 joint。

- `--resume --checkpoint models/experiment/heads.pt --epochs 10` 继续最新进度，
  只新增尚未完成的轮次；旧快照保持不变。
- 新训练若发现最新文件、日志或同名 epoch 快照已存在，默认报错。
- `--overwrite` 从头训练时清除该 checkpoint 对应的旧编号快照，并更新最新文件
  和日志，以免旧运行的高轮次快照混入新运行。不删除其他 checkpoint 的快照。
- 续训若未来轮次的快照已存在，训练前报错，避免覆盖历史。两份文件不是一起
  原子发布；若异常中断后存在比最新 checkpoint 更新的快照，先核查再继续。
- 快照文件用于选择历史轮次评估；续训仍使用最新文件。不要将历史快照自身路径
  当作持续更新的 `--checkpoint` 路径。

所有输出必须位于外部 work root，且不属于 data root。`--dry-run` 不生成快照。
升级前已经覆盖的旧 epoch 权重无法从训练日志补回。

`loss_config` 记录三个权重、`length_loss` 和 `huber_delta_nt`。续训检查这些设置，
delta 仅在 Huber 模式比较；旧 checkpoint 缺失新字段时按原 log loss 兼容。
更换 loss 的实验应使用新输出目录，参见[长度 loss 说明](length_losses.md)。

常规 joint checkpoint 额外记录 `joint_config`（两份 manifest 及参考哈希、encoder
解冻层数、学习率、CRF 权重等）。恢复必须使用对应的 joint 入口且设置一致；
旧冻结入口会拒绝 joint checkpoint。评估可沿用现有 `training.evaluate`。
见[联合训练说明](reference_joint_training.md)。

## 选择某轮评估

例如评估第 3 轮，模型目录和外部根路径仍由配置、环境变量或 CLI 提供：

```bash
CUDA_VISIBLE_DEVICES=<physical-gpu-id> PYTHONPATH=src \
  src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml \
  --manifest metadata/experiment/manifest.tsv \
  --checkpoint models/experiment/heads.epoch_0003.pt \
  --output-dir results/training_evaluation/experiment/validation-epoch3 \
  --split validation --batch-size 32
```

先用 validation 选择轮次。标称长度和 provisional 边界仍是弱标签，不能将指标
解释为可靠逐 read 真值准确率。

## 磁盘占用

每轮保留完整 checkpoint，并额外保留一个最新文件。所需空间约为
`(epoch 数 + 1) × 单个 checkpoint 大小`，另需临时文件空间。
当前完整 checkpoint 约 0.3 GB，20 轮约需 6 GB 以上；应在长训练前检查 work root
可用空间。未自动删除旧轮次或仅保存辅助 head 权重。
