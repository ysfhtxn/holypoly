# 历史脚本

从项目根目录调用本目录脚本。新训练使用 `scripts/run_reference_joint.sh`。

- `resume_length_head_ablation.sh`：仅恢复指定默认五轮 head 对照至二十轮。
- `plot_head_poly_audit.py`：绘制已完成的 v3→v4 信号范围审计。

数据、模型、日志和图表仍写入配置的外部 work root；归档不移动已有实验产物。
完整说明见 [历史归档](../../docs/archive/README.md)。
