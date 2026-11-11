# 历史实验与迁移记录

新实验使用[默认流程](../default_workflow.md)。以下内容仅用于复现旧实验，
不参与 `run_reference_joint.sh` 的日常流程，也不会自动启动真实数据处理。

| 内容 | 入口 |
| --- | --- |
| 合并后的 v3 flank、配额和 manifest 说明 | [prepare_dataset_v3.md](prepare_dataset_v3.md) |
| v3→v4 raw-edge 信号范围审计及当时结果 | [head_poly_migration_audit.md](head_poly_migration_audit.md) |
| 特定五轮 head 对照恢复至二十轮 | `scripts/legacy/resume_length_head_ablation.sh --help` |
| 信号审计绘图 | `scripts/legacy/plot_head_poly_audit.py --help` |

历史审计模块入口为 `python -m holypoly.legacy.audit_head_poly`。
原样本专用的 `run_poly2p5per_v4.sh` 已删除；冻结基线仍可通过 `holypoly train`
和[可选消融](../README.md#可选实验)显式运行。

## 历史测试

`tests/legacy/` 默认不收集；共享的冻结模型、loss 和 checkpoint 合成回归仍在
`tests/isolated/training/test_frozen_training.py` 中执行。
历史审计可用合成 fixtures 单独验证：

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m pytest -q \
  tests/legacy/datasets/test_audit_head_poly.py
```

旧真实数据 integration 必须显式提供外部输入和一个 GPU：

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> PYTHONPATH=src \
src/bonito/.venv/bin/python -m pytest -q \
  tests/legacy/training/test_prepared_pipeline.py -m integration
```

该用例只支持冻结 mean-head checkpoint，不能传入 joint checkpoint。
外部路径通过 `HOLYPOLY_DATA_ROOT`、`HOLYPOLY_WORK_ROOT`、
`HOLYPOLY_M3_MANIFEST`、`HOLYPOLY_BONITO_MODEL`、`HOLYPOLY_M3_CHECKPOINT`
提供，测试不内置机器路径。

`tests/legacy/training/test_milestone3_smoke.py` 仅适用于最初的 16-read、20 nt
数据集，要求 checkpoint 输出路径尚不存在；它会执行真实训练和写入 checkpoint。
该用例与读取既有 checkpoint 的 prepared-pipeline 用例应分别配置和调用。
任何真实测试前均需确认规模、外部路径和单 GPU；本目录不提供正式准确率结论。
