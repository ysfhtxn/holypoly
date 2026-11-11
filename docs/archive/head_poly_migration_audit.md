# 历史 v3→v4 信号范围审计

本文保留 2026-10-19 的迁移审计方法和当时结果。新数据按
[v4 制作方法](../prepare_dataset_v4.md) 与[默认流程](../default_workflow.md)准备。

从已有 manifest 每个标称长度抽 20 条，直接读取所选 POD5 信号，比较 v3 输入与
原始 read 外侧范围。先加 `--dry-run` 可查看读取量且不写文件。

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.legacy.audit_head_poly \
  --config configs/local.yaml \
  --manifests metadata/poly0per-head-free-v3/manifest.tsv \
    metadata/poly2p5per-head-free-v3-polyA-6000pergroup/manifest.tsv \
  --per-group 20 --seed 20260923 \
  --output-dir results/head-poly-v4-audit

uv run --no-sync python scripts/legacy/plot_head_poly_audit.py \
  --config configs/local.yaml --audit-dir results/head-poly-v4-audit
```

报告是 `summary.json`、`samples.json` 和每个长度的信号图。
真实 read ID 只存在外部报告中。已有 manifest 的抽样继承 v3 选择偏差，
尤其不能据此估计所有 0 nt 的可用率；head 坐标也还没有独立验证。

2026-10-19 已审计两份 v3 各 140 条，共 280 条、4,734,513 samples，全部读取成功。
各组“原始候选范围比旧输入多出的 samples”中位数为 1–4.5。
这批已选样本没有显示大量末端信号被截掉，不能把长 Poly 低估直接归因于此。
已查看部分 0/150 nt 信号总览，尚未逐 read 验证终点、接头身份或完整性。

两份数据各受限扫描 Bonito/Dorado 前 10,000 records 后，均成功制作 140 条 v4
smoke manifest：每组 20 条（train/validation/test 为 16/2/2），配额缺口为零，
全部为 length_only，未使用 flank。对应目录为
`metadata/poly0per-head-poly-v4-polyA-smoke` 与
`metadata/poly2p5per-head-poly-v4-polyA-smoke`。它们不是完整训练集，
重复 primary 检查仅覆盖这次扫描范围。

两份 smoke manifest 均通过训练 dry-run；每份另取各长度一条、共 14 条真实 POD5
信号，经现有 Dataset/collate 验证得到 `[7, 1, 12000]`，值有限，定位/状态标签
全部忽略。仅读取 Bonito 的标准化配置，未加载预训练权重、未使用 GPU 或训练。
相关合成测试 69 项通过；2 项 GPU 用例因当前测试环境 CUDA 不可用而跳过。
Ruff 和 mypy 检查通过。这些检查不代表长度预测效果已改善。

