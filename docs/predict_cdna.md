# 普通 cDNA 的探索性预测（不改变网络）

入口 `scripts/holypoly_predict_cdna.sh` 先用原 Bonito v5 模型生成未比对 BAM，再用
`holypoly.inference.predict` 搜索末端 A/T 候选、通过 move table 映射到信号，
调用已有 holypoly checkpoint 的长度回归分支。无需参考序列、固定 head、
建库试剂盒名称、标称长度或定位标签，不修改训练数据、模型结构、checkpoint。

这不是直接从信号识别生物学尾的端到端模型。候选位置来自 basecall，不使用
未受定位监督的 location 输出。现有构建数据训练的长度模型对普通 cDNA 的
迁移效果尚未验证；不同化学、采样条件及超出训练范围的长度需要独立检查。

## 运行

在项目根目录执行，先设置实际 POD5 路径和已有 checkpoint：

```bash
CDNA_POD5_DIR="/path/to/cdna_pod5"

HOLYPOLY_DATA_ROOT="$CDNA_POD5_DIR" \
GPU_ID=0 MAX_READS=1000 BATCH_SIZE=8 \
DATASET_NAME=hek293t \
CHECKPOINT="models/<joint-run>/heads.epoch_0005.pt" \
bash scripts/holypoly_predict_cdna.sh "$CDNA_POD5_DIR"
```

work root 和 Bonito model 从 `configs/local.yaml` 读取，可用环境变量或
`MODEL_DIR` 覆盖。模型必须是已存在、与 checkpoint 匹配的 Bonito v5 模型；
脚本不会安装依赖或下载模型。父环境执行绘图，Bonito 环境执行 basecall 和预测。
脚本会同时设置项目与 Bonito 子模块的导入路径，并在创建结果目录前检查
Bonito basecaller 启动入口。若旧运行因导入失败而中止，修复后重新运行命令即可；
默认生成新的时间/PID 目录，旧失败目录保留。若曾显式设置 `EXPERIMENT_ID`，
重跑前需取消该设置或使用新名称。

默认使用一个显式 GPU（内部 `cuda:0`），先 basecall 最多 1000 条 read，
随后预测最多 1000 条 primary BAM read。basecall 需要读取所选 read 的完整
信号，POD5 元数据查询可能遍历多个文件的 UUID 索引。耗时随 read 长度和 GPU
而变；BAM 通常 MB 级，图与 TSV 通常 MB 级，长 read 可明显超过这些估计。
全量运行前先用小样本测量时间、输出大小和失败比例；设置 `MAX_READS=0`
才会取消数量限制。脚本本身不会训练模型。

已有带 `mv/ts/ns` 标签的 BAM 时可跳过 basecall：

```bash
HOLYPOLY_DATA_ROOT="$CDNA_POD5_DIR" \
BASECALL_BAM="/path/to/existing_basecalls.bam" \
CHECKPOINT="models/<joint-run>/heads.epoch_0005.pt" \
bash scripts/holypoly_predict_cdna.sh "$CDNA_POD5_DIR"
```

相对 POD5 路径相对于 data root；相对 BAM、checkpoint 和 model 路径相对于
work root。使用 `DATASET_NAME=hek293t` 时，输出位于新建的
`results/cdna_prediction/holypoly-predict-hek293t-cdna-<time>-<pid>/`。
定长数据可设置 `DATASET_NAME=fixed-len`，得到
`holypoly-predict-fixed-len-cdna-<time>-<pid>/`。名称只允许字母、数字、点、
下划线和连字符，且以字母或数字开头；未设置时保留
`holypoly-predict-cdna-<time>-<pid>/`。`EXPERIMENT_ID` 可覆盖完整任务名。
输出保留日志、
`prediction/predictions.tsv`、`prediction/summary.json` 及
`plots/length_distribution.png/.svg/.pdf`。不改写 POD5 或输入 BAM。

单独预检查推理命令可用 `--dry-run`：

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.inference.predict \
  --config configs/local.yaml --pod5-dir /path/to/cdna_pod5 \
  --bam /path/to/basecalls.bam \
  --checkpoint models/<joint-run>/heads.epoch_0005.pt \
  --output-dir results/cdna_prediction/holypoly-predict-cdna-preflight --dry-run
```

预检查只检查路径、POD5 文件清单和大小元数据，不加载 BAM 记录、模型或信号。

## 用原 test read 清单运行

先从 checkpoint 配套的原长度 manifest 提取 `split=test` 的 `read_id`，
保存成 work root 内的纯文本文件，每行一个 UUID，无表头。不得重新划分 split，
也不要把另一数据集的 test 清单配给当前 checkpoint。

```bash
READ_IDS=metadata/<original-dataset>/test_read_ids.txt \
BASECALL_BAM="" EXPERIMENT_ID="" RESULT_CATEGORY=training_evaluation \
DATASET_NAME=fixed-len-test GPU_ID=0 MAX_READS=0 BATCH_SIZE=8 POLY_TYPES=polyA \
CHECKPOINT="models/<joint-run>/heads.epoch_0005.pt" \
bash scripts/holypoly_predict_cdna.sh "pod5/<sample>"
```

脚本在 basecall 前按清单筛选 POD5 read，将清单副本保存到新结果目录的
`read_ids.txt`。`MAX_READS=0` 保留清单中全部可读取的 read；默认 1000 会在
筛选后限制数量。`READ_IDS` 与 `BASECALL_BAM` 不可同时使用，避免复用包含
train/validation 的 BAM。`RESULT_CATEGORY` 只接受 `training_evaluation` 或
`cdna_prediction`，默认仍为后者。目录名为
`results/training_evaluation/holypoly-predict-fixed-len-test-cdna-<time>-<pid>/`。

原 35000 条 polyA manifest 的 test 有 3500 条，需要读取约 21 MB manifest、
POD5 UUID 元数据及所选 3500 条 read 的完整信号，不对全部 read basecall。
按此前 1000 条约 3–4 分钟的记录粗估 10–20 分钟，单 GPU；BAM、预测表和图
预计数至数十 MB，运行缓存另计。实际用时取决于 read 长度和硬件。
预测/绘图仍是候选分布，不会自动生成按标称长度分组的弱标签评价。
BAM 中缺失的清单 read 仍需单独核对，不能视为已成功预测。

## 候选和输出解释

默认搜索靠近序列两端（距离不超过 100 bases）、至少含 10 个目标 A/T 的区域，
允许最多连续 2 个非目标碱基，目标比例至少 0.8。这些仅是候选筛选阈值，不是
Poly 长度真值或 0 nt 的新定义。10 个目标碱基的门槛会漏掉短尾或 basecall
严重低估的尾；100 bases 的末端范围仍可能包含内部 A-rich 区域。
对应参数为 `--min-target-bases`、`--max-interruption-bases`、
`--min-target-fraction` 和 `--end-margin-bases`。

默认只对 polyA 候选回归长度，polyT 候选保留为 `type_not_selected`，适合当前
polyA-only checkpoint。显式设置 `POLY_TYPES="polyA polyT"` 可以尝试两个方向，
但这不会让 polyA-only 权重获得已验证的 polyT 能力。分布始终分类型报告。

反向比对 BAM 会先恢复 acquisition-oriented 序列，再进行候选检测；信号坐标
统一为原始样本上的 0-based、左闭右开 `[start_sample, end_sample)`。
不额外翻转原始电流信号。记录的序列 span 和目标碱基数是观察值，不是训练标签。

每条被选中的 primary BAM read 都保留一行：

- `candidate_predicted`：候选已完成回归，不能解释为已确认的生物学尾。
- `no_candidate`：未检出满足阈值的候选，长度留空，不记成 0 nt。
- `ambiguous_candidate`：多个候选，不擅自选一个给出长度。
- `type_not_selected`：检测到的方向未在本次长度预测范围中。
- `candidate_mapping_failure`：缺 move 标签、边界共享 move block 或候选超过
  模型输入上限；不截断候选、不估算一个替代长度。
- `unsupported_read`：split、duplex 或 hard-clipped read，当前不支持。
- `input_failure` / `prediction_failure`：信号缺失/不匹配或非有限输出。

长度使用 `predicted_length_nt` 原始回归值，不受当前 status 分数清零。
`model_status` 和对应概率单独保留，分数尚未校准。候选完整性统一标记 unknown。
图展示有回归结果的候选长度直方图和 CDF，不依赖标称长度、不把失败补成零。
分母是选中的 primary BAM read，不是全部 POD5 read：没有进入 BAM 的
basecall 失败 read 不在统计中。不得将候选检出率解释为生物学 PolyA 存在率，
也不输出 MAE、准确率或泛化能力结论。
