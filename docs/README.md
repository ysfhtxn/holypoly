# 文档导航

常规实验沿 `scripts/run_reference_joint.sh` 展开。首次使用从
[默认流程](default_workflow.md)开始，重复操作使用[命令速查](common_commands.md)。

## 数据、训练与结果

| 阶段 | 文档 |
| --- | --- |
| v4 长度/状态 manifest | [prepare_dataset_v4.md](prepare_dataset_v4.md) |
| 配套 reference-head、联合训练及恢复 | [reference_joint_training.md](reference_joint_training.md) |
| checkpoint 保存与历史兼容 | [checkpoints.md](checkpoints.md) |
| validation/test 与绘图 | [evaluate_model.md](evaluate_model.md) |
| 输出目录与命名 | [result_layout.md](result_layout.md) |

## 预测与独立比较

| 用途 | 文档或入口 |
| --- | --- |
| 普通 cDNA 候选预测 | [predict_cdna.md](predict_cdna.md) |
| 独立完整 read holdout | [final_holdout.md](final_holdout.md) |
| 定长三工具比较 | [compare_fixed_len_tools.md](compare_fixed_len_tools.md) |
| 普通 cDNA 两工具比较 | [compare_cdna_tools.md](compare_cdna_tools.md) |
| 普通 cDNA 三工具比较 | `scripts/compare_three_cdna_tools.py --help` |
| 已发表 HeLa 与本地三工具分布背景 | `scripts/compare_four_cdna_tools.py --help`；不同样本不作配对准确率比较 |
| 已发表标准品背景 | [nano3p_source_comparison.md](nano3p_source_comparison.md) |
| PolyAnalysis 原生检测器适配 | `scripts/polyanalysis_distribution.py --help`，使用其独立环境 |

## 可选实验

这些入口用于明确指定的对照，保留历史冻结策略及 checkpoint 兼容性。

| 对照 | 文档 |
| --- | --- |
| mean / duration_only / mean_duration | [length_head_ablation.md](length_head_ablation.md) |
| log_smooth_l1 / MAE / Huber | [length_losses.md](length_losses.md) |
| mean / sum pooling | [length_pooling_ablation.md](length_pooling_ablation.md) |
| 冻结特征的状态校准 | [status_calibration.md](status_calibration.md) |
| 通用窗口时长参照（默认关闭） | [timing_reference.md](timing_reference.md) |

## 历史归档与共享代码

[v3、迁移审计和旧测试](archive/README.md)已与日常入口分开。
`training/run_validation.py` 是联合、状态校准和冻结入口共用的校验模块；
`engine.py`、`launcher.py`、`model.py` 仍提供当前训练或评估需要的共享功能。
旧 manifest、模型、结果和 loss 定义保持兼容，归档不改变任何标签语义。
