# Nano3P-seq 已发表数据与当前定长结果的分布对照

只使用来源 Excel 中的 `Fig3c`；其他工作表不导出、不用于新图。
输入工作簿保留原样，CSV 与图均放在外部 work root 的
`results/tool_comparison/<comparison-id>/`，不会修改已有三工具比较结果。

```bash
PYTHONPATH=src .venv/bin/python scripts/export_source_xlsx.py \
  --config configs/local.yaml \
  --input <published-source.xlsx> \
  --sheets Fig3c \
  --output results/tool_comparison/<comparison-id>/source_data

PYTHONPATH=src .venv/bin/python scripts/plot_nano3p_context.py \
  --config configs/local.yaml \
  --local-comparison results/tool_comparison/<fixed-len-comparison>/per_read_comparison.tsv \
  --fig3c-csv results/tool_comparison/<comparison-id>/source_data/02_Fig3c.csv \
  --output results/tool_comparison/<comparison-id>/length_distribution_comparison
```

两个输出子目录都必须尚不存在。转换使用 Python 标准库读取 OOXML，无需新增 Excel
依赖。CSV 保留 `Group,Std_Type,tail_length` 字段、原数值文本、零值和空单元格；
表格前的单格说明存入 `conversion.json`。源表名称与其说明文字可能不同，元数据会
原样记录，不会据此改名。源 Excel 与导出 CSV 都记录 SHA256。

图分三部分：

1. 当前定长 holdout 的 Holypoly、Dorado、PolyAnalysis 正长度输出。
2. Fig3c 的 cDNA 标准品，保留 `cDNA_0/15/30/60/90/120` 原分组。
3. Fig3c 的 RNA 标准品，保留 `R1/R2` 原分组。

所有箱线图只展示正数值；零值仍完整保留于 CSV 和分组统计。箱体为中位数/IQR，
须为 1.5 IQR，显示所有离群点，各面板 y 范围不同。cDNA 组名的数字后缀只用于
显示横轴间距，不映射到当前定长标签，也不宣称为已验证的逐 read 真值。

新图提供跨数据来源的描述性背景，不合并四种方法的 read，不报告跨队列准确率排名，
不将 Excel 中正数值占比作为正式 recall。表中缺少完整尝试 read 的检测分母与
失败状态，不能沿用当前 holdout 的 500 条分母。

输出包括 PNG/PDF/可编辑文字 SVG、`group_distribution_summary.csv` 和
`figure_contract_and_qa.json`。只读既有结果，不读取 POD5/BAM，不使用模型或 GPU。

## 增加 tailfindr 和 Shuyi 独立来源

`scripts/refresh_tool_comparison.py` 可对已有比较目录先做归类 dry-run，
明确传入 `--apply` 才移动目录并生成新图。它读取现有来源表，逐文件核对迁移前后
SHA256，不创建旧目录的符号链接，并在新的 `comparisons/<comparison-id>/`
子目录下写入结果，不覆盖旧图。支持已归类的实体目录结构，无需创建符号链接。
需有且仅有一套对应的定长、本地 HEK293T、Nano3P-seq 比较，以及一个
`rna-*.csv` tailfindr RNA 来源和一个 Shuyi XLSX 来源。

```bash
PYTHONPATH=src .venv/bin/python scripts/refresh_tool_comparison.py \
  --config configs/local.yaml \
  --root "$HOLYPOLY_WORK_ROOT/results/tool_comparison" \
  --run-name <new-comparison-id>
```

确认 dry-run 后，使用同一命令加 `--apply`。来源完整性检查默认遇到残缺列表就报错；
只有在已核实 Excel 截断且明确选择保留完整前缀时，才额外指定
`--shuyi-truncated-policy prefix`。其处理仅限原始长度恰为 32,767 字符的残缺列表，
只接受末尾逗号之前的完整数值，绝不猜测最后一个可能被截断的数字。
逐单元格原计数、保留数量和策略记录在 `source_audit.json`。

第一组图沿用原定长 + Nano3P-seq 合并图，在右侧增添 tailfindr 的独立 RNA 重复
类别；不把来源文件中的名称推定为已验证标称长度。第二组图保留原三工具共同正值
read 和 Nano3P-seq HeLa 背景，增加 Shuyi `raw` 工作表的
`polyA_length_293T_total_rep1/rep2` 长度列表合并分布，不使用转录本中位数。
来源表中无 read ID 时，样本量表述为长度条目数，不推断唯一 read 数。
当列表字段为标量 0 且对应计数为 0 时，记录为空列表占位，而非一次 0 nt 测量。

图和报告明确注明独立数据集、来源、截断限制及未知工具版本；保留全部正长度
统计和完整范围 ECDF，放大视图不重新计算统计。原有分组和四方法统计逐项回归核对。

## 移除 PolyAnalysis 并增加 tailfindr DNA

在已归类的目录上使用新的运行名称，并额外传入：

```bash
PYTHONPATH=src .venv/bin/python scripts/refresh_tool_comparison.py \
  --config configs/local.yaml \
  --root "$HOLYPOLY_WORK_ROOT/results/tool_comparison" \
  --run-name <new-comparison-id> \
  --exclude-polyanalysis \
  --tailfindr-dna <published-dna-result.csv> \
  --shuyi-truncated-policy prefix
```

检查 dry-run 后加 `--apply`。新图、图例和分组汇总不再展示 PolyAnalysis；本地
HEK293T 保留原图的固定共同正值子集，不因移除曲线而重新筛选 read。原图剩余方法
以及原定长组统计均逐项核对。历史输入与图件不更改。

DNA 来源表必须具有 `read_type`、`tail_is_valid`、`replicate` 和 `tail_length`。
只展示原生 `polyA/polyT`、`tail_is_valid=TRUE` 的有限正长度；两个方向分开，
不把缺失、失败或无效结果填为零。定长合并图中按方向和重复分别展示 DNA 类别，
不指定标称 nt；含 Shuyi 的直方图和 ECDF 中按方向合并两个重复，保留重复分组审计。
DNA 和 RNA 仍是独立来源，不与本地 cDNA read 匹配，不据此排名准确率或召回率。

## DNA barcode 六种预期尾长

在上面的命令中再加 `--tailfindr-dna-barcodes <decoded-barcode.csv>`，
用 `replicate + read_id` 将 standard-basecalling DNA 长度表与 flipflop-basecalling
barcode 表连接。两个来源的复合键均须唯一；不按行序、文件路径或预测尾长分组。
只接受 `barcode_passed_threshold=TRUE`、`barcode_tie=FALSE`、barcode 为
10/30/40/60/100/150 的条目；两来源的原生 A/T 方向须一致，尾长须原生有效且为有限正值。
来源的 read_too_long/read_too_short 标记计入审计，不额外筛选。
所有排除状态、未匹配键、方向不一致和缺失/零值分别记录，不填零。

合并主图把 DNA 放在六种预期 nt 的数值位置，A/T 分开，同一长度同一方向内合并
两次重复。局部源偏移依次为 Holypoly −2、Dorado −1、Nano3P-seq 0、DNA A +1、
DNA T +2 nt，组间保留实际不等间距。`tailfindr_dna_six_lengths/` 中另给出
逐重复箱线图（完整范围和 0–220 nt）、统计表及 barcode 匹配条目 TSV。
尾长分组只说明构建标称值，不是逐 read 真值。HEK293T/Shuyi 混合分布图继续使用
全部原生有效正 DNA 长度，不因 barcode 筛选改变之前的队列。

用户确认 RNA 也有相同六种预期尾长，但目前 RNA 结果表只有 replicate 而没有
逐 read barcode/构建分组。因此右侧保留重复混合分布，并注明六种预期长度与映射缺失；
获得 RNA 对应表后才能拆为六组，不根据估计值反推分组，也不借用 DNA 的 read 标签。
