# v4：head–Poly 末端构建的弱监督数据集

适用结构由用户确认：`head–Poly`，没有设计的下游 flank。
旧 v3 的 flank 推定规则不适合用来确认这批构建的完整性。
v4 版本为 `construct-nominal-head-poly-v4`，不覆盖 v3 或旧模型。

本文是当前默认流程的第一步。后续新数据集还须以生成的 manifest 为源制作
匹配的 reference-head sequence manifest，再做 length + status + basecalling
联合训练，命令见 [当前默认流程](default_workflow.md)。

## 制作规则

1. 保留 Bonito UUID/构建关联、Dorado 构建一致性和 head 末端覆盖检查。
2. 不从 0 nt 推断 flank，不搜索 flank，也不要求跨长度 flank 支持。
3. 用实际 `mv/ts` 将 head 末端映射到原始 sample；共用 move block 时拒绝。
4. `--signal-end-policy raw-edge` 明确选择候选范围：forward 为
   `[head_boundary_sample, raw_signal_length)`，reverse 为
   `[0, head_boundary_sample)`。保持原始信号方向，不向 head 一侧补真实信号。
   **原始 read 边缘只是候选输入边界，不是 Poly 终点标签**；可包含接头、噪声。
5. 正样本和 0 nt 都是 `length_only`，训练目标为构建标称长度。
   Poly 起止、完整性、观察长度保持未知，定位和状态 loss 全部忽略。
   0 nt 仍使用 `label_status=zero_nt`，其长度目标为 0，不是 failure。
6. 0 nt 不要求有 head 后的已叫出碱基，但必须有真实候选信号。
   默认最少 12 samples，仅为技术下限，不代表生物学质控通过；可通过
   `--min-candidate-samples` 调整。空、过短、超过输入上限的样本明确记录原因。
   不用全零 tensor 代替缺失的负样本。有效短输入在训练时右侧补零并提供 mask。
7. `--poly-types polyA` 只保留 forward 正样本，0 nt 同样只保留 forward，
   避免把测序方向变成区分正负样本的线索；请求两方向时才保留两方向负样本。
8. 默认每组独立随机保留最多 `--per-group` 条，不因一个稀缺组缩减其他组。
   固定种子，按 read ID 分为约 80%/10%/10%。不足三条的组不用于三 split，
   不足配额或缺失的组记录在 `quota_deficits` / `missing_positive_strata`。
9. `--stop-when-full` 只要求请求组配额满足；暂停昂贵检查后仍检查后续 UUID，
   重复使配额失效则恢复。候选来自 BAM 顺序子集，不是全体均匀抽样。
   去掉此选项可从扫描范围内全部候选随机抽样。

v4 不需要新依赖、模型或 GPU。构建时扫描 BAM 和 POD5 元数据，不读取整批原始信号。
不能按标称长度反推截取终点，也不能把缺失标签改成完整定位标签。

旧 v3 输入与 raw-edge 候选的比较记录见
[历史迁移审计](archive/head_poly_migration_audit.md)，不属于新数据的常规制作步骤。

## 构建命令

先做受限运行：将下方 `--max-records 0` 改成 `--max-records 10000`，
`--per-group 6000` 改成 `--per-group 20`，并使用新的 smoke 输出目录。
两份 BAM 各自的扫描前缀未必覆盖相同 read 或全部长度，缺组不表示全数据缺组。

完整制作命令由用户运行；目标为各组 6000 条。输出目录必须不存在。

```bash
for sample in poly0per poly2p5per
do
  PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
    --config configs/local.yaml \
    --structure head-poly --signal-end-policy raw-edge \
    --bonito-bam "basecalls/${sample}/basecalls.bam" \
    --dorado-bam "dorado_calls/${sample}/dorado_basecalls.bam" \
    --pod5-dir "pod5/${sample}" \
    --reference fixed_len_ref.fa \
    --output-dir "metadata/${sample}-head-poly-v4-polyA-6000pergroup" \
    --max-records 0 --per-group 6000 --poly-types polyA --stop-when-full
done
```

检查 summary 的结构、版本、`supervision_counts`（全部 length_only）、
`selected_counts`、`quota_deficits`、拒绝原因和采样范围。
达到配额时每组 train/validation/test 分别为 4800/600/600。

## PolyA + PolyT 严格均衡的小型联合数据集

新增显式选项 `--strict-balance`，仅用于 `head-poly` 且同时请求 polyA/polyT。
保留上述标签、坐标、head 排除和质控规则，改变的只是分组抽样：

* 每个参考正长度 × polyA/polyT 分别一组。
* 0 nt 按 `forward` / `reverse` 分为两组，manifest 的 `poly_type` 仍为 `none`，
  `label_status` 仍为 `zero_nt`。0 nt 不被改称 polyA 或 polyT。
* 所有组都取相同数量：`min(--per-group, 所有必需组的可用数量)`。
  稀缺组会缩减全部组；任何必需组不足 3 条直接报错，不生成 manifest。
  必须包含 0 nt 参考构建；缺失正长度/类型组也报错。失败目录可能已有 audit，
  修正输入或扫描上限后须换一个输出目录。
* 每组独立按同一规则分 train/validation/test，固定种子并保证 read ID 不跨 split。
  每组 validation/test 各 `max(1, count // 10)`，其余为 train。
* `--stop-when-full` 必须等两方向的 0 nt 都达到配额才暂停昂贵检查。
  后续重复 primary 会按其实际分组扣数，必要时恢复收集。

以 20/40/60/80/100/150 nt 六个正长度、每组 200 条为例：

| 分组 | 组数 | 每组 train / validation / test | 合计 |
| --- | ---: | --- | ---: |
| 正长度 × polyA/polyT | 12 | 160 / 20 / 20 | 2400 |
| 0 nt × forward/reverse | 2 | 160 / 20 / 20 | 400 |
| 总计 | 14 | 2240 / 280 / 280（全体） | 2800 |

以下由用户运行，使用新目录，不覆盖已有数据：

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --structure head-poly --signal-end-policy raw-edge \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-poly-v4-polyAT-balanced-200pergroup \
  --max-records 0 --per-group 200 --poly-types polyA polyT \
  --strict-balance --stop-when-full --seed 20260923
```

完整扫描 Bonito 一次、Dorado 一次和所需 POD5 元数据，不加载整批信号或使用 GPU。
读取量由命令开始时按 BAM 大小报告，HDD 可能数分钟到数小时；manifest 通常数 MB，
全量 audit 可能每百万 reads 数百 MB。受限检查可改为 `--max-records 10000`、
`--per-group 20` 和新目录；前缀缺组或组不足 3 条会报错，不能据此判断全数据缺组。

制作后检查 `summary.json` 中：

* `balancing.mode=strict_length_type_zero_orientation` 和 `effective_per_group`；
* `selected_stratum_counts`：逐 split、长度、类型/0 nt 方向的配额；
* `quota_deficits`：相对请求 200 条的缺口，全部组缩减时各组均有缺口；
* `metadata_collection`：配额是否达到、是否暂停及采样范围。

这是类型、长度和方向均衡，仍有 `poly_present:zero_nt=6:1` 的状态数量差。
status 类别权重由训练步骤单独设置，制作选项不会自动修改 loss；
现有 status calibration 命令会从 train 统计类别权重。
训练后要分 polyA/polyT、各长度以及 0 nt 两方向评估，
尤其检查 validation 的 0 nt 召回与 20 nt 误判。
数量均衡不能消除 BAM 顺序、读长/质量差异或弱标签误差；保留 `--stop-when-full`
时仍只对已检查候选随机抽样。测试结果只能说明构建弱标签的一致性。

## 训练与判断效果

当前常规新训练先补齐配套 head 序列监督，再按[默认流程](default_workflow.md)
训练三项 loss。固定随机种子，以新 checkpoint 开始并保留独立对照。
原始 v4 `length_only` 默认不启用定位和状态监督；联合入口默认显式启用弱 status，
定位仍忽略。只有实际训练过的分支才能解释为获得相应监督。
评价使用原始 `regression_length_nt`，不要使用未训练状态头门控的长度。
优先比较 100/150 nt 偏差、±10 nt 内比例，以及训练集拟合的时长基线。
同批次测试只衡量弱标签一致性，交叉数据条件测试和逐 read 真值验证仍未完成。

v4 修正的是结构假设和选样逻辑，不保证单次重训提升，也不能保证各组都有足够
真实可用的负样本或完整正样本。信号终点判定与定位监督需要另行验证。
