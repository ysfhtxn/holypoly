# 默认数据准备与联合训练流程

常规实验以 `scripts/run_reference_joint.sh` 为主入口。数据准备在训练前独立完成；
已有匹配 manifest 时从预检开始，不重复扫描或重新划分 read。

## 1. 配置与规模检查

在 `configs/local.yaml` 设置 `paths.data_root`、外部 `paths.work_root` 和
`training.bonito_model`。原始目录只读，输出目录必须位于 work root 且尚不存在。
示例中的 `<sample>` 和 `experiment` 换成实际输入及新实验名称。

准备过程读取 BAM 和所需 POD5 元数据，不加载整批 signal 或使用 GPU。
完整扫描前报告 BAM 大小、预计时间、CPU/内存及 audit 输出大小；先用
`--max-records 10000 --per-group 20` 和新 smoke 目录验证。
受限 BAM 前缀缺少共同 read 或长度组不能代表全数据缺组。

## 2. 制作 v4 长度/状态 manifest

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --structure head-poly --signal-end-policy raw-edge \
  --bonito-bam basecalls/<sample>/basecalls.bam \
  --dorado-bam dorado_calls/<sample>/dorado_basecalls.bam \
  --pod5-dir pod5/<sample> --reference fixed_len_ref.fa \
  --output-dir metadata/experiment \
  --max-records 10000 --per-group 20 \
  --poly-types polyA polyT --strict-balance --seed 20260923
```

仅 polyA 时使用 `--poly-types polyA` 并省略 `--strict-balance`。
两方向联合时，正长度×类型及 0 nt 两方向严格均衡；稀缺组会缩减全部组。
检查 `summary.json` 的版本、split、实际配额、拒绝原因和标签来源。
准备命令仍保留历史 v3 默认，日常命令必须明确写出上面的 structure 和 end policy。
详细规则及 `--stop-when-full` 的采样限制见 [v4 制作方法](prepare_dataset_v4.md)。

## 3. 从同一长度 manifest 制作 reference-head 序列 manifest

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare_sequence \
  --config configs/local.yaml \
  --manifest metadata/experiment/manifest.tsv \
  --dorado-bam dorado_calls/<sample>/dorado_basecalls.bam \
  --reference fixed_len_ref.fa \
  --output-dir metadata/experiment-reference-head \
  --max-records 10000 --dry-run
```

`--dry-run` 仅读取 manifest、参考和文件元数据，不扫描 BAM 或写文件。
确认规模后去掉它；需要覆盖分散的 read 时，经规模确认后使用 `--max-records 0`。
序列标签来自参考 head 内部的可靠窗口，不生成标称长度的重复 A/T 序列。
继承原 split 并记录源 manifest SHA256，不能复用其他长度集的旧配套 manifest。

检查 accepted 数量、split 和失败原因，至少 train/validation 各有 accepted 窗口。
head 筛选失败不删除原长度样本，也不保证配套数据仍完全均衡。
筛选与坐标规则见[参考 head 序列监督](reference_joint_training.md)。

## 4. 预检并启动三项联合训练

```bash
GPU_ID=0 EPOCHS=5 BATCH_SIZE=32 EVAL_BATCH_SIZE=4 \
LAMBDA_LENGTH=1 LAMBDA_STATUS=1 LAMBDA_BASECALL=0.1 \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv --dry-run
```

去掉 `--dry-run` 才训练并生成测试及图表。使用单个显式 GPU，内部为 `cuda:0`。
真实训练读取所选 POD5 signal 和模型，可能为小时级；逐轮 checkpoint 通常为 GiB 级。
以预检报告为准，空间或显存不足时减少训练 batch 或实验轮数。
`EVAL_BATCH_SIZE` 只控制训练后 test 推理；未设置时继承显式的 `BATCH_SIZE`，
二者均未设置时训练 batch=32、test batch=4，保持历史调用行为。

```text
total_loss = 1.0 × Huber(length, nominal_length; delta=10 nt)
           + 1.0 × CE(shared_status_logits, weak_status)
           + 0.1 × Bonito_native_CTC_CRF(head_signal, reference_head)
```

sum pooling；仅解冻 encoder 最后一个 Transformer 层和 Poly heads，原 CRF head 冻结。
status 为普通 CE，系数不是类别权重。location 不监督；无 failure 训练例时不验证
failure 检测。原始 v4 `length_only` manifest 保持原样，联合入口显式开启弱 status。

默认从原 Bonito 和新 Poly heads 开始。只有确认旧 checkpoint 未见过当前
validation/test read 后才使用 `INITIALIZE_FROM`。重新划分 split 不能消除历史泄漏。

## 5. 选轮、测试、绘图与恢复

脚本按最佳 validation 长度指标选择 checkpoint，再运行 test 和绘图。
输出在 `results/training_evaluation/<run>/`，训练日志及模型在 `logs/<run>/`、
`models/<run>/`。图使用实际 nt 横轴间距，仅显示存在的 Poly 类型，0 nt 单独标注。
评估解释和手动命令见 [evaluate_model.md](evaluate_model.md)。

```bash
EXPERIMENT_ID=<existing-run> EPOCHS=10 GPU_ID=0 \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv --resume --dry-run
```

恢复使用原 manifest、模型及 loss 设置；`EPOCHS` 为目标总轮数。
旧两项 loss 恢复显式设置 `STATUS_SUPERVISION=0`，不把旧实验自动改成三项。
恢复会保留旧 epoch 和图表，新测试图表另建 resume 子目录。
详见[联合训练说明](reference_joint_training.md)和[checkpoint](checkpoints.md)。

所有长度结果均为构建弱标签一致性，不是独立逐 read 真值准确率或跨批次泛化。
训练后的普通 cDNA 预测、独立 holdout 和工具比较见[文档索引](README.md)。
