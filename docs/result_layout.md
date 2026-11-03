# 结果目录与命名

新结果在外部 work root 中按用途区分，脚本仍放在原 `scripts/` 目录：

```text
<work_root>/results/
├── training_evaluation/
│   └── <run-id>/
│       ├── best_checkpoint.json
│       ├── training-plots/
│       └── test-best-validation/
│           ├── predictions.tsv
│           ├── summary.json
│           ├── metrics_by_group.tsv
│           └── plots/
├── cdna_prediction/
│   └── holypoly-predict-<dataset>-cdna-<YYYYMMDD-HHMMSS>-<PID>/
│       ├── pipeline.log
│       ├── basecalls.bam
│       ├── basecalling.log
│       ├── prediction/
│       │   ├── predictions.tsv
│       │   └── summary.json
│       └── plots/
└── tool_comparison/
    └── <dataset>-tool-comparison-<YYYYMMDD-HHMMSS>/
        ├── fixed_len_tool_comparison.png / .pdf / .svg
        ├── fixed_len_group_distributions.png / .pdf / .svg
        ├── per_read_comparison.tsv
        ├── summary.json
        └── report.md
```

`training_evaluation`：使用训练配套 manifest 的 validation/test split，包含
弱标签一致性指标、状态诊断和训练曲线。联合训练、状态校准、长度 head/loss/
pooling 消融及其结果汇总均使用此目录。状态校准仅使用 validation；
`test-best-validation` 表示测试的是 validation 选择的 checkpoint，
`test-final` 表示测试最终 epoch。手动评估可用
`validation-epoch0005-<YYYYMMDD-HHMMSS>` 或
`test-epoch0005-<YYYYMMDD-HHMMSS>`，重跑时使用新目录。

`cdna_prediction`：对输入 POD5 进行实际候选预测和分布统计，由
`scripts/holypoly_predict_cdna.sh` 创建。通过 `DATASET_NAME=fixed-len` 或
`DATASET_NAME=hek293t` 设置数据集名，时间和 PID 自动追加，例如
`holypoly-predict-fixed-len-cdna-20261102-120000-12345`。
数据集名以字母或数字开头，仅含字母、数字、点、下划线和连字符。
未设置时保留原 `holypoly-predict-cdna-<YYYYMMDD-HHMMSS>-<PID>` 命名。
`EXPERIMENT_ID` 优先级更高，可覆盖完整任务名。
复用已有 BAM 时不生成 `basecalls.bam` 或 `basecalling.log`。
“实际预测”只描述用途；迁移到生物样本的效果尚未验证，候选分布不等同于
已确认的生物学尾分布，不自动产生准确率或 MAE。

`tool_comparison`：对同一数据集的 Holypoly、Dorado、PolyAnalysis 等工具进行
跨工具对比，保存对比图、共同 read 子集、覆盖率、汇总表和比较口径。
定长独立比较集和实际生物样本的多工具汇总均使用此目录；
不归入训练验证目录。原始单工具预测产物仍保留在 `cdna_prediction`。
定长对比入口和统计限制见 [compare_fixed_len_tools.md](compare_fixed_len_tools.md)。

模型、训练日志和缓存仍分别使用 `models/<run-id>/`、`logs/<run-id>/` 和
`cache/<run-id>/`。数据审计等其他结果保留独立用途目录。

默认不移动、重命名或覆盖现有结果；用户明确要求的目录归类迁移应保持内容，
并核对迁移前后的文件哈希。旧联合训练/状态校准通过原命令显式
`--resume` 时，优先继续其已有的旧 `results/<run-id>/`；若新旧路径同时存在，
报错以免混用。`resume_length_head_ablation.sh` 可读取旧或新布局中的源结果，
续训分叉的新结果写入 `training_evaluation`。

底层 `holypoly.training.evaluate`、`holypoly.inference.predict` 及独立绘图
仍接受显式 `--output-dir` / `--output`，保留旧命令兼容性，不自动改写用户路径。
手动命令请遵守上述目录约定。

同一 cDNA 预测流程用于原 held-out read 清单时，可设置
`READ_IDS=metadata/<dataset>/test_read_ids.txt`、`MAX_READS=0` 和
`RESULT_CATEGORY=training_evaluation`。仅处理清单 read，并将该次结果归入
训练测试目录；默认实际样本预测仍归入 `cdna_prediction`。
