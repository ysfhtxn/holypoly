# 普通 cDNA 的 Holypoly / Dorado 输出分布比较

入口：`scripts/plot_cdna_tool_comparison.sh`。等 Holypoly 完成全部预测并生成
`prediction/predictions.tsv` 和 `prediction/summary.json` 后再运行。
该脚本只处理既有结果，不重新 basecall、不预测、不读取原始信号、不使用 GPU。

```bash
bash scripts/plot_cdna_tool_comparison.sh \
  results/cdna_prediction/<holypoly-run> \
  results/cdna_prediction/<dorado-run> \
  <dataset-name>
```

输入目录相对于 work root，也支持 work root 内的绝对路径；第四个可选参数为配置。
`BIN_WIDTH_NT=5` 可改变直方图 bin 宽度。目录、样本名和路径均不在脚本中写死。

输出自动使用新目录：

```text
<work_root>/results/tool_comparison/<dataset>-holypoly-dorado-<时间>-<PID>/
├── holypoly_polya_distribution.png/.pdf/.svg
├── holypoly_dorado_polya_comparison.png/.pdf/.svg
├── holypoly_lengths.tsv
├── common_read_comparison.tsv
├── summary.json
└── plot_qa.json
```

第一张图为 Holypoly 全部选定 primary BAM read 中正长度输出的直方图和 CDF。
第二张图为：共同 read ID 队列内的两工具叠加直方图、各工具正输出 CDF、两者均有
正输出的同 read 子集 CDF，以及全部共同 read 的原生输出状态堆叠比例。
直方图使用相同 nt bins，并分别归一为 100%，避免检出数量不同直接造成高度差。

缺失/无候选不补成零长度；Dorado pt=0 与负值、缺标签分开保留。
导出表保留原生数值和状态，`pa` 是否有有效区间另行保留。Holypoly 的模型状态
概率不用于清零回归长度。比较以共同 read 为分母，同时报告双方独有 read 数量。
无真值生物样本只讨论分布和输出覆盖，不报告正式 recall、长度准确率或工具排名。

执行时需要只读扫描一遍 Dorado BAM。约 700k read 的示例预计分钟级 CPU 处理、
约 1–2 GiB 内存、50–200 MiB 表格与图片；实际取决于 read 数和分布范围。
不安装依赖：父环境绘图，已有 Bonito 环境只负责读取 BAM 标签。
源 TSV 记录 SHA256，BAM 检查扫描前后大小和修改时间，避免额外全文件哈希扫描。
