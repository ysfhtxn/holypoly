# 比较现有定长 POD5 推理结果

入口：`scripts/compare_fixed_len_tools.py`。默认使用 Python/matplotlib，BAM tag
读取由已有 Bonito 环境的 subprocess 执行；不安装依赖、不读取 POD5、不运行模型。

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_fixed_len_tools.py \
  --config configs/local.yaml \
  --labels processed/<final-holdout>/labels.tsv \
  --holypoly results/cdna_prediction/<holypoly-run>/prediction/predictions.tsv \
  --dorado-bam results/cdna_prediction/<dorado-run>/basecalls.bam \
  --polyanalysis results/cdna_prediction/<polyanalysis-run>/full/per_read_results.tsv \
  --output results/tool_comparison/<new-fixed-len-comparison>
```

相对路径均相对 work root；输出目录必须是全新的外部 work-root 子目录。
图、逐 read 源数据、分组统计、输入 SHA256、说明和 QA 写在同一个新目录中。
已有结果不修改。

跨工具比较统一放在 `results/tool_comparison/<comparison-id>/`；
原始单工具预测结果继续保留在 `results/cdna_prediction/`。

比较口径：

* labels.tsv 确定完整分母，按 read_id 联结，拒绝重复 ID 和队列外 read；缺失 read
  明确保留为 missing_read。PolyAnalysis 自带标称长度需与 labels 一致。
* 主长度图仅使用原生正长度。Dorado pt=0、pt<0、缺标签、未检出和其他候选
  分别保留，绝不把失败补成 0 nt；pt=0 有无有效 pa 区间另行记录。
* PolyAnalysis 使用其原生 interval/30 换算，不在最终队列上重新标定。
* 标称长度用实际数值横轴，100 至 150 nt 的间距大于 80 至 100 nt；0 nt 是
  单独的负构建组。仅展示 polyA，不混入其他方向的长度。
* 共同子集要求三种工具都给出正长度且标称 >0，报告各组交集数量。这个交集
  不再长度均衡；偏差统计只用于描述共同 read 的输出，不能替代全体覆盖率。
* 分组 CDF 的分母是每组所有 read，曲线终点等于该工具正长度输出比例；所有
  极值保留，symlog 横轴在线性区后使用对数刻度。
* 构建标称长度为弱标签，不是逐 read 真值；图和表不宣称正式准确率。
  不进行显著性检验，也不将 read 当作独立生物学重复。

主图：输出长度中位数/IQR、各标称组正长度覆盖、共同 read 子集的绝对标称偏差、
全部 read 的输出状态。补图：七个长度组的累积分布。
另提供 `fixed_len_box_scatter_combined.png/.pdf/.svg`：a、b、c 三联图，
每张子图同时展示三种工具，用统一颜色区分。a 为分组长度箱线/散点，b 为与标称
长度的一致性散点，c 为带符号偏差箱线/散点；下方表格列出每工具每组正输出数量。
箱体为中位数/IQR，须为 1.5 IQR，散点包含全部正长度输出及极值。
线性 y 轴保留完整数值范围，a/b 共用长度范围；标称横轴保留真实 nt 间距。
a/c 中三种工具箱体偏移 -4、0、+4 nt，另加固定种子的水平抖动；b 使用精确标称
坐标。这些显示偏移不影响任何长度值或汇总指标。旧版分工具图保留原位。
此入口要求定长标签，不用于无标称长度的普通 cDNA 样本。

额外提供 `fixed_len_output_coverage_stacked.png/.pdf/.svg`：堆叠柱状图，
a 汇总所有正标称长度组，b 按标称长度展示三工具；每根柱子以该组全部 read 为
分母，保留正长度、原生零输出、无候选、其他/不确定、错误/缺失五类并合计 100%。
0 nt 负构建单独标明，不计入 a；其正输出比例不能称为 recall。
正长度输出率只表示原生工具的输出覆盖，不代表生物学真阳性召回或长度准确率。
分组分母与计数保存在 `output_coverage_stacked.tsv`，定义与图契约保存在同名
`_summary.json` 中。该额外图不修改已有比较图和统计。
