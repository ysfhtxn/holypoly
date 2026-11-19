# 统一制作 head–Poly 数据集

常规新数据只使用 `holypoly.datasets.build`。不需要选择版本、flank 模式，
也不需要手动把两次制作的 manifest 配在一起。已有数据、checkpoint 和结果不转换、不覆盖。
适用前提仍为已确认的无设计下游 flank 的 `head–Poly` 构建。

```text
Bonito BAM + Dorado BAM/moves + POD5 + reference head
    → UUID/构建关联、head 排除、raw-edge 候选、分组抽样和 split
    → length/manifest.tsv（构建标称长度与弱状态）
    → sequence/sequence_manifest.tsv（同源、同 split 的 head 内部序列）
    → timing/timing_reference.tsv（可选，普通窗口时长）
    → dataset.json（完成状态、参数、实际数量、文件路径与 SHA256）
```

## 一个制作命令

在项目根目录运行。配置文件中的 data root 和 work root 必须真实存在，
不能保留 `/path/to/...` 占位值。输入路径相对 data root，输出相对外部 work root。
输出目录必须不存在；`<sample>` 换成实际样本名称。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
src/bonito/.venv/bin/python -m holypoly.datasets.build \
  --config configs/local.yaml \
  --bonito-bam basecalls/<sample>/basecalls.bam \
  --dorado-bam dorado_calls/<sample>/dorado_basecalls.bam \
  --pod5-dir pod5/<sample> --reference fixed_len_ref.fa \
  --output-dir metadata/<sample>-head-poly-smoke \
  --poly-types polyA --per-group 20 --max-records 10000 \
  --seed 20260923
```

默认只预检：显示生效的根路径、文件大小、扫描遍数、输出估计及资源要求，
不打开 BAM/POD5，不写文件。`--dry-run` 与此相同。
确认报告后，在同一命令末尾加 `--run` 才制作。

参数调整仍用同一个入口：

| 需求 | 参数 |
| --- | --- |
| 只制作 polyA | `--poly-types polyA` |
| 只制作 polyT | `--poly-types polyT` |
| 联合 A/T | `--poly-types polyA polyT`，自动严格均衡，包括 0 nt 两方向 |
| 附带时间参照 | 加 `--with-timing-reference`，默认关闭 |
| 完整扫描 | 经预检后改为 `--max-records 0`，例如 `--per-group 5000`，另用新的输出目录 |
| 避开历史训练或独立保留 read | `--exclude-read-ids metadata/<inventory>/read_ids.txt` |
| 减少配额达到后的昂贵检查 | `--stop-when-full`；保留 BAM 顺序采样偏差，默认不启用 |

`--max-records` 对所有阶段使用同一上限，避免长度只扫前缀、配套步骤却意外全扫。
两份 BAM 的顺序可能不同，小范围扫描缺组不能证明全数据缺组。
联合 A/T 任一必需组不足 3 条时失败；稀缺组会把所有组缩减到共同可用数量。
单类型各长度独立取最多 `--per-group` 条，不会补造缺组。
目录名不承诺数量，实际数量以 `length/summary.json` 为准。

## 固定的数据定义与质控

1. 保留 Bonito UUID/构建关联、Dorado 构建一致性和 head 末端覆盖检查。
   默认检查 head 最后 24 bp，错误比例最多 0.1，不推断或搜索下游 flank。
2. 用实际 Dorado `mv/ts` 映射原始 sample；拒绝跨越共享 move block 的 head 切口。
   POD5 与 Dorado `ns` 只允许不足一个实际 move stride 的末尾样本差；
   长度、序列和时间参照统一使用此检查，不把 stride 写死为某个数值。
3. forward 候选为 `[head_boundary, raw_signal_length)`，reverse 为
   `[0, head_boundary)`，不反转原始信号。raw-edge 只是候选输入边界，
   可包含接头或噪声，不能作为已经确认的 Poly 终点。
4. 长度目标仍是构建标称弱标签，记录 `label_source`，不等于逐 read 真值。
   0 nt 是合法负标签，方向跟随请求类型；failure、无标签与 0 nt 不混淆。
   Poly 区间、完整性和观察序列长度保持未知，location 不监督。
   manifest 保留 `length_only`；联合训练入口显式使用弱 status。
5. 候选默认最少 12、最多 12000 个真实 samples；这是输入技术限制，
   不是生物学完整性判据。无真实候选信号的 read 不用全零 tensor 代替。
6. 固定随机种子，每组按 read ID 分配约 80%/10%/10%，单数据集不跨 split 重复。
   联合 A/T 的 0 nt 按 forward/reverse 分层；`poly_type` 仍为 `none`。
7. 配套序列只取原长度集中的 read，继承 split 和源 manifest SHA256。
   默认使用 head 内部 96 bp、两端留 32 bp、MAPQ≥20、每碱基 Q≥15、
   连续精确匹配且唯一的窗口，最长同聚物≤4 bp，最多 4096 samples。
   窗口不得与 Poly 候选重叠；不生成标称长度的重复 A/T 作为序列标签。
8. 时间参照可选，从同一 Dorado BAM 的普通窗口提取；QC 不通过或找不到的 read
   保留缺失状态，不删除、不补成有效零值，也不改变长度标签或 split。

## 输出与验收

```text
<work_root>/metadata/<dataset>/
├── dataset.json
├── length/    manifest.tsv, audit.tsv, summary.json
├── sequence/  sequence_manifest.tsv, audit.tsv, summary.json
└── timing/    timing_reference.tsv, window_audit.tsv, plan.json, summary.json（可选）
```

`dataset.json` 仅在所请求步骤完成且输入大小/mtime 未变时标记 `complete`；
这是制作完成标志，不代表质量或准确率认证。失败标为 `failed` 并记录阶段，
中断可能保留 `building`。部分结果及拒绝原因保留供检查，重新运行须换新目录。
序列 train 或 validation 无合格窗口会报错，不把空序列集标成制作完成。

制作后检查：

* `length/summary.json` 的 `selected_counts`、`selected_stratum_counts`、
  `quota_deficits`、`missing_positive_strata`、拒绝原因和采样范围。
* `sequence/summary.json` 的接受数量、各 split 覆盖率与拒绝原因；
  序列筛选不会删掉长度样本，也不保证序列集仍均衡。
* 如启用时间参照，检查 `timing/summary.json` 的各 split 可用率；
  train/validation 必须有可用参照才能启动对应训练。

默认扫描 Bonito 一遍、Dorado 两遍；有时间参照时 Dorado 三遍。
仅读取所需 POD5 元数据，不复制 signal、不加载模型、不使用 GPU。
完整 HDD 扫描可能数分钟至数小时；内存随 UUID/审计表增长，百万 reads 可达 GB 级。
除所选 read 表格外，全量 audit 可能每百万 reads 数百 MB。
实际扫描与耗时受 `--max-records`、文件顺序、候选数量及磁盘速度影响。

## 接到训练

```bash
GPU_ID=0 EPOCHS=5 \
bash scripts/run_joint_training.sh --dataset-dir metadata/experiment --dry-run
```

如制作了时间参照且本次要使用它，在训练命令前设置
`USE_TIMING_REFERENCE=1 TIMING_REFERENCE_MANIFEST=metadata/<dataset>/timing/timing_reference.tsv`。
仅制作表格不会自动打开训练功能。完整说明见[默认流程](default_workflow.md)和
[时间参照](timing_reference.md)。

跨数据集重新划分 split 仍可能与历史训练 read 重叠；沿用 checkpoint 前核对历史 read，
或在制作前显式排除。构建弱标签、同批次 split 和均衡抽样不能证明逐 read 真值准确率
或跨批次泛化，也不能保证数据数量足够。

底层 `prepare`、`prepare_sequence`、`prepare_timing_reference` 保留给组件审计和已有数据补配。
`prepare` 默认也改为 head–Poly/raw-edge；历史 flank 实验必须显式指定
`--structure head-poly-flank`。内部 schema 字符串保留历史标识，供读取旧数据和恢复
checkpoint；不作为新数据集名称，也不重写已有 manifest。
