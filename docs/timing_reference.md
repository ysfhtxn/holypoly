# 可选的通用窗口时长参照

此功能默认关闭。原训练和预测命令、旧 checkpoint 的结构保持兼容。
启用后，用 basecall 普通窗口的 provisional move 时长辅助长度回归，
不需要已知 head 或定位标签。length/status/basecalling 三项 loss 和标签定义不变。

## 如何使用

先设置 `HOLYPOLY_DATA_ROOT`、`HOLYPOLY_WORK_ROOT`，并使用匹配的数据集。
以下占位路径需替换；manifest 和输出相对 work root，BAM 相对 data root。
BAM 必须包含完整 read 的序列、质量值和 `mv/ts/ns` 标签。

小规模验证可直接使用 `scripts/run_timing_reference_smoke.sh`：

```bash
GPU_ID=0 PER_GROUP=100 EPOCHS=3 BATCH_SIZE=8 \
bash scripts/run_timing_reference_smoke.sh \
  "metadata/<dataset>/manifest.tsv" \
  "metadata/<dataset>-reference-head/sequence_manifest.tsv" \
  "dorado_calls/<sample>/dorado_basecalls.bam" configs/local.yaml --dry-run
```

查看报告后去掉 `--dry-run` 执行。7 个长度组时取 700 条（560/70/70），
保留原 split 和弱标签。脚本先验证已有完整长度/序列 manifest 的配对关系，
再从中派生小子集及新的来源 SHA256；不会拿旧子集的序列表直接配给新数据。
只为时长参照扫描一次 BAM，不重新生成序列标签。
训练默认开启参照及三项 loss，完成后测试并画图；时长参照缺失的 read 保留，
但 train 或 validation 完全没有可用参照时会停止并提示检查审计表。

默认只跑启用组；加 `RUN_BASELINE=1` 顺序训练关闭/开启两组，
两组共享子集、种子和训练参数。结果位于
`results/training_evaluation/<实验名>-baseline/`（可选）和 `<实验名>-timing/`，
子集和审计表位于 `metadata/<实验名>-inputs/`。
可用 `PER_GROUP=50` 进一步缩到 350 条。小实验用于检查链路与初步趋势，
几轮训练不足以确认新方法有效。

先查看处理范围；此命令不扫描 BAM、不读 POD5、不创建输出：

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
src/bonito/.venv/bin/python -m holypoly.datasets.prepare_timing_reference \
  --config configs/local.yaml \
  --manifest "metadata/<dataset>/manifest.tsv" \
  --bam "dorado_calls/<sample>/dorado_basecalls.bam" \
  --output-dir "metadata/<dataset>-timing"
```

确认报告的读取量、输出估计和资源后，在同一命令末尾加 `--scan-bam` 执行。
需要顺序扫描一次 BAM；只提取 manifest 对应 read，不加载 POD5 或使用 GPU。
输出目录必须新建，含 `timing_reference.tsv`、`window_audit.tsv`、
`plan.json` 和 `summary.json`。缺失或不合格的 read 也会保留并记录原因。

训练使用新实验名，配套 sequence manifest 必须来自同一长度 manifest：

```bash
USE_TIMING_REFERENCE=1 \
TIMING_REFERENCE_MANIFEST="metadata/<dataset>-timing/timing_reference.tsv" \
GPU_ID=0 EPOCHS=5 INITIALIZE_FROM="" \
EXPERIMENT_ID="v4-reference-status-joint-timing-$(date +%Y%m%d-%H%M%S)" \
bash scripts/run_reference_joint.sh \
  "metadata/<dataset>/manifest.tsv" \
  "metadata/<dataset>-reference-head/sequence_manifest.tsv"
```

建议先在该命令末尾加 `--dry-run`，查看覆盖率和训练规模。脚本自动将同一参照表
传给最终测试评估，之后按原流程画图。直接调用 `joint_command` 时，同时添加
`--use-timing-reference --timing-reference-manifest <table.tsv>`；
直接调用 `training.evaluate` 时添加 `--timing-reference-manifest <table.tsv>`。

普通 cDNA 预测使用新训练得到的 checkpoint：

```bash
USE_TIMING_REFERENCE=1 GPU_ID=0 MAX_READS=1000 \
CHECKPOINT="models/<timing-run>/heads.epoch_0005.pt" \
DATASET_NAME="<sample>" \
bash scripts/holypoly_predict_cdna.sh "pod5/<sample>" configs/local.yaml
```

预测在原有 BAM 读取过程中提取参照，无需训练 manifest、已知 head 或参考序列。
逐 read 输出包含参照可用性、samples/base、离散程度和缺失原因，summary 记录覆盖情况。
此时不要传训练参照表。开启标志必须与 checkpoint 架构匹配；
旧 checkpoint 不能仅靠切换参数获得这个功能，应另开实验训练。
已启用模型的续训继续提供原参照表，不可在 `--resume` 时切换结构。

## 输入与网络

在完整 basecall 上最多均匀探查 32 个不重叠的 96-base 窗口，要求平均 Q≥15、
最长同聚物≤6、单碱基比例≤0.7、序列熵≥1.2 bit。
从合格窗口中均匀选最多 4 个，至少 2 个才视为参照可用；
按 move table 得到原始 sample 区间，以窗口 samples/base 中位数代表时长参照，
以 log(samples/base) 的标准差表示离散程度。正反方向均转回信号采集方向。
这些规则固定记录在 summary；训练、测试和实际预测使用相同提取实现。

仅支持现有 sum pooling。原池化特征与参照速率、离散程度、可用性、
候选 sample 时长/参照速率共同输入一个小型残差分支，在 softplus 前修正长度值。
新分支末层初始化为 0；参照不可用时严格绕过残差，走当前原 sum head。
这不表示其训练后输出必然等于历史 checkpoint，原长度参数仍会参与联合训练。
参照不参与状态分类和候选定位，不增加 loss，也不修改原 CRF 或解码。

表格通过 read ID、split、raw signal 长度、源 manifest SHA256、表格 SHA256
及提取配置核对，覆盖全部源 read。训练会记录参照覆盖率与来源。
表格及其 summary 应一起保存，不要手工编辑或拿其他子集的表替代。

## 证据与下一步

普通复杂序列窗口只是“非 Poly 参照”的候选，不是经独立证实的非 Poly 区间；
依然依赖 basecaller 序列质量与 provisional moves。短 read、质量不足和缺少 move
会降低覆盖率，缺失参照不等于 0 nt、无候选或预测失败。
训练若用 Dorado moves，实际预测若用 Bonito moves，还存在参照分布差异；
比较时应尽量统一 basecaller，或单独验证这种差异。

旧诊断中约 10.4% 的标称偏差改善来自候选附近窗口。
当前实现为避免依赖候选是否检出，改用完整序列的普通窗口抽样；
旧数字不能当作本实现的预期增益。也没有证据表明它已解决 0/150 nt 问题。

建议在相同 read split、seed、训练预算下新建“关闭/开启”两组实验，先比较 validation
覆盖率、分长度标称偏差及 0 nt 召回，再冻结选择并做独立测试。
使用构建标称弱标签时，结果属于标称一致性，不能称作逐 read 真实准确率。
