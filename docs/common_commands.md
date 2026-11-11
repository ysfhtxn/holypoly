# 常用命令

从项目根目录运行。常规实验按[默认流程](default_workflow.md)准备 v4 长度 manifest
和同源 reference-head 序列 manifest；所有运行产物写入外部 work root。
下列 `experiment`、`<run>` 为占位名称，必须换成实际配套输入和新输出目录。

## 联合训练预检

```bash
GPU_ID=0 EPOCHS=5 BATCH_SIZE=32 EVAL_BATCH_SIZE=4 \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv --dry-run
```

确认配置和资源后去掉 `--dry-run`。默认 length/status/basecalling 系数为 1/1/0.1，
sum pooling + Huber(delta=10 nt)，location 不监督。
完整参数、读取优化及结果解释见[联合训练](reference_joint_training.md)。

## 从已保存的完整 epoch 恢复

```bash
EXPERIMENT_ID=<run> EPOCHS=10 GPU_ID=0 \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv --resume --dry-run
```

`EPOCHS` 是目标总轮数，去掉 `--dry-run` 才继续训练。
必须保留原输入和 loss 设置；旧两项实验显式使用 `STATUS_SUPERVISION=0`。

## 手动测试与绘图

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
src/bonito/.venv/bin/python -m holypoly.training.evaluate \
  --config configs/local.yaml --manifest metadata/experiment/manifest.tsv \
  --checkpoint models/<run>/heads.epoch_0005.pt \
  --output-dir results/training_evaluation/<run>/test-epoch5-new \
  --split test --batch-size 4 --dry-run
```

去掉 `--dry-run` 才读取 signal 和模型。先按 validation 选轮，不用 test 调参。

```bash
uv run python scripts/plot_training.py --config configs/local.yaml \
  --log logs/<run>/train.jsonl \
  --output results/training_evaluation/<run>/training-plots-new

uv run python scripts/plot_training.py --config configs/local.yaml \
  --predictions results/training_evaluation/<run>/test-epoch5-new/predictions.tsv \
  --supervision-mode length_only \
  --output results/training_evaluation/<run>/test-epoch5-new/plots
```

## 其他入口

| 用途 | 说明 |
| --- | --- |
| 长度及 reference-head 数据准备 | [默认流程](default_workflow.md) |
| 普通 POD5/cDNA 预测 | [predict_cdna.md](predict_cdna.md) |
| 独立 holdout 和定长比较 | [final_holdout.md](final_holdout.md)、[compare_fixed_len_tools.md](compare_fixed_len_tools.md) |
| cDNA 输出分布比较 | [compare_cdna_tools.md](compare_cdna_tools.md) |
| 冻结 head/loss/pooling 消融和状态校准 | [可选实验索引](README.md#可选实验) |
| v3、迁移审计和旧 integration | [历史归档](archive/README.md) |

`holypoly train` 保留为历史冻结训练命令；不用于恢复 joint checkpoint。
