# 默认数据准备与联合训练流程

常规实验以 `scripts/run_joint_training.sh` 为统一入口。数据准备先独立完成，训练从
成品数据的预检开始。

## 1. 配置与规模检查

在 `configs/local.yaml` 设置 `paths.data_root`、外部 `paths.work_root` 和
`training.bonito_model`。原始目录只读，输出目录必须位于 work root 且尚不存在。
示例中的 `<sample>` 和 `experiment` 换成实际输入及新实验名称。

准备过程读取 BAM 和所需 POD5 元数据，不加载整批 signal 或使用 GPU。
完整扫描前报告 BAM 大小、预计时间、CPU/内存及 audit 输出大小；先用
`--max-records 10000 --per-group 20` 和新 smoke 目录验证。
受限 BAM 前缀缺少共同 read 或长度组不能代表全数据缺组。

## 2. 统一制作数据集

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
src/bonito/.venv/bin/python -m holypoly.datasets.build \
  --config configs/local.yaml \
  --bonito-bam basecalls/<sample>/basecalls.bam \
  --dorado-bam dorado_calls/<sample>/dorado_basecalls.bam \
  --pod5-dir pod5/<sample> --reference fixed_len_ref.fa \
  --output-dir metadata/experiment \
  --max-records 10000 --per-group 20 \
  --poly-types polyA polyT --seed 20260923
```

默认仅预检，确认后加 `--run` 制作长度和配套序列两份 manifest。
仅 polyA 使用 `--poly-types polyA`；联合 A/T 自动严格均衡，稀缺组会缩减全部组。
可加 `--with-timing-reference` 同时制作可选时间参照，默认不制作。
完整数据另用新目录，预检后改为 `--max-records 0` 和所需 `--per-group`。

输出按 `length/`、`sequence/`、可选 `timing/` 收纳于同一目录。
序列 manifest 自动继承长度集的 split 和 SHA256，不能复用其他数据集的配套文件。
检查 `dataset.json` 完成状态、实际配额、各步骤接受数量和拒绝原因。
定义、资源报告与验收见[统一制作方法](prepare_dataset.md)。

## 3. 预检并启动三项联合训练

```bash
GPU_ID=0 EPOCHS=5 BATCH_SIZE=32 EVAL_BATCH_SIZE=4 \
LAMBDA_LENGTH=1 LAMBDA_STATUS=1 LAMBDA_BASECALL=0.1 \
bash scripts/run_joint_training.sh --dataset-dir metadata/experiment --dry-run
```

`--dataset-dir` 接受外部 work root 下的常规 v4、strict-poly-candidate-v1 或
head-free-platform-qc-v1 成品目录。预检根据制作摘要定位最终的两张表，核验完成状态、
SHA256、split 和配对关系，并使用摘要记录的输入样本数。常规 v4 的 head 默认是
`legacy`，两个 QC 方法默认是 `tcn`。目录模式显式指定物理设备 `GPU_ID`。

已有配套表可以按原顺序直接传入长度和序列 manifest。`run_reference_joint.sh` 和两个
原 QC 训练脚本仍可作为兼容入口。

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

常规 v4 使用 sum pooling。默认仅解冻 encoder 最后一个 Transformer 层和 Poly heads，
原 CRF head 冻结。
status 为普通 CE，系数不是类别权重。location 不监督；无 failure 训练例时不验证
failure 检测。原始 `length_only` manifest 保持原样，联合入口显式开启弱 status。

默认从原 Bonito 和新 Poly heads 开始。常规 v4 使用 `INITIALIZE_FROM` 前须确认旧
checkpoint 未见过当前 validation/test read。QC 目录仅支持新训或同一实验的 `--resume`。

## 4. 选轮、测试、绘图与恢复

脚本按最佳 validation 长度指标选择 checkpoint，再运行 test 和绘图。
输出在 `results/training_evaluation/<run>/`，训练日志及模型在 `logs/<run>/`、
`models/<run>/`。图使用实际 nt 横轴间距，仅显示存在的 Poly 类型，0 nt 单独标注。
评估解释和手动命令见 [evaluate_model.md](evaluate_model.md)。

```bash
EXPERIMENT_ID=<existing-run> EPOCHS=10 GPU_ID=0 \
bash scripts/run_joint_training.sh --dataset-dir metadata/experiment --resume --dry-run
```

恢复使用原 manifest、模型及 loss 设置；`EPOCHS` 为目标总轮数。
旧两项 loss 恢复显式设置 `STATUS_SUPERVISION=0`，不把旧实验自动改成三项。
恢复会保留旧 epoch 和图表，新测试图表另建 resume 子目录。
详见[联合训练说明](reference_joint_training.md)和[checkpoint](checkpoints.md)。

所有长度结果均为构建弱标签一致性，不是独立逐 read 真值准确率或跨批次泛化。
训练后的普通 cDNA 预测、独立 holdout 和工具比较见[文档索引](README.md)。
