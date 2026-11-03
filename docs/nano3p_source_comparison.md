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
