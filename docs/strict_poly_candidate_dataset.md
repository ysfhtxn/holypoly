# 新方法：严格 Poly 候选数据制作（strict-poly-candidate-v1）

从原始数据制作新 manifest：`python -m holypoly.datasets.build_strict_poly_from_raw`。
对已有配套表重新筛选：`python -m holypoly.datasets.build_strict_poly_dataset`。
算法：`src/holypoly/datasets/strict_poly_candidates.py`。
规则：`configs/strict_poly_candidate_v1.json`，全部是尚未标定的初始值。
联合训练使用 `scripts/run_joint_training.sh --dataset-dir`。

这是独立的探索性入口，不替换现有默认数据流程。`build_strict_poly_from_raw` 从
Bonito BAM、Dorado BAM、POD5 和参考 FASTA 开始，不要求已有 manifest，不导入旧数据集
的 read 选择。它复用 head/move/序列元数据工具，在新目录建立新的元数据池、分配
固定种子的 split，然后进行严格信号筛选，生成本次配套训练表。

如需重新筛选已有配套表，仍可使用 `build_strict_poly_dataset`；该模式只能检查源表
中的 UUID。两个入口使用相同严格规则和训练接口。

## 从原始数据开始（建议使用）

在项目根目录执行。所有数据路径相对 data_root，输出相对 work_root；示例需填写
实际原始数据路径。此入口没有 `--source-manifest` 参数。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STRICT_DATASET="metadata/strict-poly-from-raw-$(date +%Y%m%d-%H%M%S)"
STRICT_RAW_ARGS=(
  --config configs/local.yaml
  --bonito-bam basecalls/condition/basecalls.bam
  --dorado-bam dorado_calls/condition/dorado_basecalls.bam
  --pod5-dir pod5/condition
  --reference fixed_len_ref.fa
  --output-dir "$STRICT_DATASET"
  --poly-types polyA
  --per-group 300
  --max-records 100000
  --max-signal-reads 2100
)
src/bonito/.venv/bin/python -m holypoly.datasets.build_strict_poly_from_raw \
  "${STRICT_RAW_ARGS[@]}" --dry-run
```

查看计划后，将 `--dry-run` 换成 `--run` 执行。`--per-group` 是信号筛选前每个分层
的审核池上限，不是保证通过数量；长度组从参考名称自动解析，必须包含 0 nt。
联合 A/T 使用 `--poly-types polyA polyT`，审核预算应相应增大。

```bash
src/bonito/.venv/bin/python -u -m holypoly.datasets.build_strict_poly_from_raw \
  "${STRICT_RAW_ARGS[@]}" --run
```

重新制作必须使用新的 `STRICT_DATASET`；程序不读取上一轮 manifest、候选审计或
缓存，也不在旧目录续跑。修改代码不会改变已经运行中的旧 Python 进程；若旧任务
仍在当前终端前台，可按 Ctrl+C 停止后再启动新命令，旧输出保留用于排查。

默认每遍最多读 100000 条记录，共 Bonito 一遍、Dorado 两遍：构建关联与 head/move
元数据，然后在 reference-head 序列扫描时一并收集本次审核池的严格 moves。
信号阶段直接使用本次内存中的 moves，不再增加第三遍 Dorado 扫描。
全量时读盘预算为 `Bonito BAM 大小 + 2 × Dorado BAM 大小`；前缀扫描不代表完整数据集。
需全量时显式设置 `--max-records 0 --allow-full-bam-scan` 并重新检查资源计划。
完整扫描前不得据此自动启动全量任务。

新目录布局：

```text
<STRICT_DATASET>/
  raw_strict_dataset.json                  # 整体状态、来源与资源计划
  metadata_pool/                          # 本次从原始数据新建的元数据池与审计
    length/manifest.tsv
    sequence/sequence_manifest.tsv
  strict/                                 # 最终严格筛选结果
    strict_dataset.json
    read_audit.tsv
    candidate_audit.jsonl
    length/manifest.tsv                    # 最终训练用长度/弱状态表
    sequence/sequence_manifest.tsv          # 同一批 read 的配套序列表
```

只有整体状态为 `complete` 才能使用最终表。任何阶段失败或严格分层不足都会保留
审计，训练脚本拒绝不完整目录。继续试制需选新输出目录，不覆盖旧结果。
`--exclude-read-ids` 可显式提供外部 work_root 内的 UUID 清单用于排除；不提供旧表
时不会继承历史 split，因此不能加载曾接触新 validation/test read 的旧 checkpoint。

信号阶段显示 `Strict signal QC` 进度、速度/预计剩余时间，以及 `eligible`（均衡前
通过）、`deferred`（复核）、`excluded`（失败或无法读取）计数。每完成一个入池 read，
`strict/strict_dataset.json` 的 `progress` 都更新已处理数、总数、读取尝试数和状态分布；
`candidate_audit.jsonl` 也立即刷新。`--no-progress` 只关闭终端进度条，不关闭磁盘状态。
外层 `raw_strict_dataset.json.current_stage` 区分元数据阶段与严格筛选阶段。
最终训练表在全部审核、分层均衡和接口验证后写出，运行中看不到它不代表卡死。

## 按最终合格规模早停（可选）

`--per-group` 是**筛选前审核池上限**，`--target-per-group` 是**严格筛选后每个
标称长度 × 方向的最终目标**。两者独立：例如在上述参数组中使用
`--per-group 500 --max-signal-reads 3500`，再增加：

```bash
src/bonito/.venv/bin/python -u -m holypoly.datasets.build_strict_poly_from_raw \
  "${STRICT_RAW_ARGS[@]}" --target-per-group 100 --stop-when-full --dry-run

src/bonito/.venv/bin/python -u -m holypoly.datasets.build_strict_poly_from_raw \
  "${STRICT_RAW_ARGS[@]}" --target-per-group 100 --stop-when-full --run
```

目标 100 分为 train/validation/test 的 80/10/10；一般值按 validation/test 各
`max(1, target // 10)`，其余为 train。每个长度 × 方向 × split 都达到目标后，
立即结束后续 POD5 信号读取、完成均衡并导出恰好目标数量。以 7 个标称长度、
仅 polyA 为例，最终 700 条（560/70/70）；联合 A/T 时为 1400 条（1120/140/140）。
长度分层包含 0 nt，联合 A/T 的 0 nt 两个方向分别计数，绝不只按总数停止。
某个分层先满时，也跳过该分层后续的信号读取，把读取预算留给尚未足额的分层；
保留每个分层按原审核顺序最先通过的目标数量，不改变 read 的 split。

未评估的池内 read 仍写入 `read_audit.tsv`，状态为
`not_evaluated_target_quota_met`（全局足额后）或
`not_evaluated_stratum_quota_met`（单个分层先满），不当作失败或负样本；详细候选审计只包含实际完成
检测的 read。`strict_dataset.json.early_stop` 记录是否达标、是否提前结束、已处理与
未评估数量，`eligible_stratum_counts` 记录各分层通过数。

目标至少 3，不能超过审核池上限；信号读取预算不能低于目标总样本数。审核池耗尽
或预算耗尽仍未达标时，即使每个分层非空，也标为 `incomplete_no_training_manifest`，
保留审计，不导出不足目标数量的训练表，不自动扩池或放宽规则。
不传 `--target-per-group` 时保持原行为：完成审核池，再在各 split 内按最少合格组均衡。

这只提前结束**信号筛选阶段**，不跳过构建关联、配套序列制作和选定 read 的 BAM
重复主比对检查。现有 `--stop-when-full` 是另一个开关：元数据池足额后暂停昂贵的
head/POD5 元数据收集，但仍扫描设定范围内的剩余 BAM 做 UUID 检查，不能理解为
全部 BAM 读取提前结束。两种早停都属于按预设顺序取样，不代表完整数据集的均匀样本。
只能在启动新命令前设置，不能改变已运行进程。

## 提速实现

新启动的原始数据入口自动使用以下优化，无需增加 `--fast` 开关，仍从原始数据新建
manifest，不读取历史 manifest 或历史 moves 缓存：

- 固定种子扩展只检查实际走到的窗口；短于最低时长的区间不再做多余的分位数运算。
  用覆盖掩码跳过已接受区间内的种子，保留右端点不包含的原定义。
- 邻域证据用有序窗口坐标二分定位；IoU 去重和多尺度支持使用数组运算。
  阈值、排序、固定种子电平、方向、候选坐标及质量指标定义不变。
- Dorado 由三遍减少为两遍。严格 moves 观察器遍历序列阶段的整个设定前缀，
  包括首次找到所选 UUID 后的记录；重复主比对检查仍保留。内存结果绑定本次
  源表 SHA256、BAM 指纹、记录上限和 MAPQ 参数，不能替换成其他构建的缓存。
- POD5 只读 Reader 默认最多复用 4 个，按 LRU 淘汰；不缓存已解码信号、不复制原始数据。
  `--pod5-reader-cache-size 0` 可关闭复用；上调会增加文件句柄和元数据内存开销，
  不保证在 HDD 上更快。缓存淘汰、正常结束、早停和异常退出均关闭句柄。
- 配置最终配额后，已满分层跳过解码，全部分层达标后结束信号阶段；审计保留未评估状态。

合成基准只反映信号检测 CPU 时间，不包含实际 HDD、BAM 或 POD5 读盘。
平台密集的信号收益较大；纯噪声和单长平台仍主要花在窗口中位数计算上。
真实完整流程耗时和内存需在新运行中观察，不能用合成检测倍数推算总耗时。

2026-11-17 合成基准：每条 60000 samples、5000 Hz，随机种子 20261117；
旧/新实现各运行 3 次，报告中位数；Python 3.11.14、NumPy 2.4.6，顺序 CPU 运行。
阶梯背景为每 30 samples 一个随机电平叠加低噪声；单长平台包含短尖峰。

| 合成信号 | 优化前（秒） | 优化后（秒） | 比值 |
| --- | ---: | ---: | ---: |
| 波动噪声 | 0.1496 | 0.1471 | 1.02× |
| 波动背景中的单长平台 | 0.1492 | 0.1406 | 1.06× |
| 短平台阶梯背景 | 7.0076 | 0.4408 | 15.90× |

三类基准及另 144 组合成检测（采样率、方向、端点、漂移、常量信号、gap 设置）
的新旧完整结果字典均完全相等；单元测试另覆盖局部扩展、窗口邻域和数组 IoU 的
定义一致性。这些结果不证明真实数据通过率或检测准确率。

## 多 CPU 信号筛选

两个严格制作入口均支持 `--workers N`，默认 1，HDD 建议先用 4。
新增实现为 `src/holypoly/datasets/strict_poly_parallel.py`，仅使用 Python 标准库。

主进程顺序读取选定 UUID 的 POD5，将单条已解码信号交给 spawned 子进程计算。
主进程按原审核池顺序汇总候选、更新配额并写入所有审计与 manifest。
POD5 Reader 仅在主进程复用；子进程原生数学库线程数限制为 1，避免线程数叠加。
BAM 元数据阶段仍采用原顺序扫描，不加载模型，不需要 GPU。

同时在途任务不超过 `workers`，还受每个长度 × 方向 × split 的剩余配额限制。
成功与失败任务都占用配额预留，直到主进程按顺序处理该任务；满组和全局早停不会
额外预读后续信号。接近达标时并发数会自然下降。固定输入、seed、规则和审核预算下，
单进程和多进程的最终 manifest、read 审计及候选审计一致，状态文件记录实际 worker 数
和耗时。普通 read 错误保留原状态，worker 进程崩溃使本轮生成失败并保留已写审计。

进度中的 `signal_reads_attempted` 包括已经预读、尚在计算的任务，可能领先于
`processed_reads`（已按顺序汇总的池内 read 数）；最大预读窗口为 worker 数。
内存包含原有 read-ID/move 元数据及 O(workers) 条信号的检测数组，不缓存整批信号。
先比较 4 和 8 的吞吐与内存，不能将 CPU 进程数直接当作整套流程的加速倍数。

2026-11-18 合成吞吐检查：48 条 × 60000 samples、5000 Hz，噪声、单平台及
阶梯背景各 16 条，随机种子 20261118；包含检测、外部速度、选择、进程启动和 IPC。
1 worker 为 12.598 秒，4 workers 为 4.611 秒（2.73×），完整有序结果的 SHA256 一致。
该检查无真实 POD5/BAM 读盘，不能直接推算完整制作任务的耗时。

从原始数据制作最终 5000 条/组的示例，所有路径仍相对配置的 data_root/work_root：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STRICT_DATASET="metadata/strict-poly-final5000-cpu4-$(date +%Y%m%d-%H%M%S)"
STRICT_RAW_ARGS=(
  --config configs/local.yaml
  --bonito-bam basecalls/condition/basecalls.bam
  --dorado-bam dorado_calls/condition/dorado_basecalls.bam
  --pod5-dir pod5/condition
  --reference fixed_len_ref.fa
  --output-dir "$STRICT_DATASET"
  --poly-types polyA
  --per-group 50000
  --target-per-group 5000
  --max-records 0
  --allow-full-bam-scan
  --max-signal-reads 350000
  --workers 4
  --pod5-reader-cache-size 4
  --seed 20261117
  --stop-when-full
)
src/bonito/.venv/bin/python -u -m holypoly.datasets.build_strict_poly_from_raw \
  "${STRICT_RAW_ARGS[@]}" --dry-run
```

查看扫描规模、内存和输出空间计划后，使用同一参数组加 `--run` 执行。新进程才能
使用 `--workers`；已运行任务不会动态增加 CPU，重跑仍需新输出目录和新建 manifest。

head 元数据阶段仍采用现有的可信 head 锚定、调用间构建一致性、raw-edge 和
head-free 长度限制；本入口没有放宽这些定义。旧审核池之外的 read 可以重新进入，
但缺乏这些元数据证据或 head-free 区间超长的 read 仍会被明确排除。

完成后直接传整个新目录给配套训练入口：

```bash
GPU_ID=0 PROFILE_BATCHES=20 \
EXPERIMENT_ID="strict-poly-tcn-context-unfreeze1-$(date +%Y%m%d-%H%M%S)" \
bash scripts/run_joint_training.sh --dataset-dir "$STRICT_DATASET" configs/local.yaml --dry-run
```

脚本自动定位 `strict/`、检查整体及子阶段状态和最终 manifest 哈希。将 `GPU_ID`
换为选定的物理 GPU，去掉 `--dry-run` 才开始训练；5 epochs、TCN、最后一层解冻、
length/status/basecalling 三项 loss 默认保持与现有 context 实验一致。

## 数据与标签

- 严格筛选保留输入池的 split、构建标称长度、弱状态、head 排除边界。
  从原始数据开始时，输入池和 split 都是本次新生成的。
- 正向只考虑中心位于后半段的平台，反向只考虑前半段。原始采集坐标不翻转。
- 在整条信号上归一化、检测平台并计算邻域证据；**先排除不完整包含于本次新建
  head-free chunk 的平台，再判断多个候选和 0 nt 冲突**。不切割跨边界平台。
  全信号平台仍保存在审计中，速度估计仍排除全信号的所有平台及其邻域。
- 严格筛选按标称长度 × 方向 × **输入池 split** 固定种子 reservoir 抽样，
  train/validation/test 的池配额约为 80/10/10。从原始数据开始默认每组最多 300 条；
  可选的已有表筛选入口默认每组最多 30 条。
- 已有表筛选入口默认两张源表各检查最多 50000 行、BAM 最多 10000 条记录。
  原始数据入口则每遍最多 100000 条 BAM 记录，并检查本次新建的完整小型源表。
  SHA256 会读取完整的两张元数据表，dry-run 会报告相应计划。
- 只解码入池 UUID 的 POD5，每条最多 120000 samples。原始数据入口默认最多
  2100 次信号读取尝试，已有表筛选入口默认最多 210 次。sample count 在解码前检查；
  不会复制信号、扫描 POD5 目录或加载模型。
- 0 nt 是构建级弱负标签。head-free chunk 内存在形态支持的平台时转入复核；整条信号动态范围退化、
  读取失败或关联不可靠时不能作为负样本。
  区间外平台只用于审计，不与目标区间竞争；弱负标签不证明整条 read 没有 Poly。

正样本仍使用 `training_length_nt = nominal_length_nt`，
`label_source=construct_nominal_weak_supervision`。候选时长不被转换为新训练长度。
保持 `supervision_mode=length_only`，`poly_start_sample`、`poly_end_sample`、
`poly_is_complete` 和 `observed_sequence_length_nt` 均未知，**不监督 location**。

为了复用现有消费者，`manifest_version` 保留 v4 **接口 schema**；制作方法用新增的
`dataset_method=strict-poly-candidate-v1`、`selection_reason`、`strict_*` 审计字段和
`strict_dataset.json.workflow` 区分。旧入口、旧表、旧 checkpoint 均不修改。
修复版额外记录
`selection_scope=parent_head_free_chunk_before_candidate_selection`，便于与旧运行区分。

## 严格验收规则

1. 稳健归一化；不平滑、不删除尖峰。多尺度固定种子电平扩展沿用 Notebook 定义：
   核心 `s,d <= 0.15`；扩展 `s,d <= 0.25`，与种子电平差 `<= 0.25`；
   最多约 1 ms 连续失败；持续至少 8 ms；中位数 Q90−Q10 `<= 0.35`。
2. 普通平台最多 10% 失败窗口；优先候选最多 5%。至少两个**不同实际采样窗长**
   的合格候选 IoU > 0.7；先统计跨尺度支持再去重，不拼接平台。
3. Dorado 主比对 MAPQ 至少 20，255 视为未知；方向、参考名称、POD5 文件和信号长度
   必须与源表相符；重复主比对、split/duplex、非法或缺失 moves 不自动接受。
4. 与普通序列相接的一侧必须有边界证据：正向检查左侧、反向检查右侧。邻域为
   20 ms，至少三个同尺度窗口，整窗不得跨入候选。邻域/内部窗口 s 的中位数比值
   `>= 2`，或归一化中位电平差 `> 0.25` 才提供支持。最近邻域仍连续满足相同电平的
   平台条件时，视为平台延续，不能把内部片段当作清楚边界。
5. 速度来自所有检出平台及其 5 ms 邻域之外、不重叠的 20-base move 窗口，至少两个，
   使用 samples/nt 中位数。按原始采集顺序计算，不能再次反转反向 read 的 moves。
   候选时长/标称预期时长在 `[0.5, 2]` 内才自动接受；缺失或冲突只进入复核。
6. 只在 head-free chunk 内判断歧义。该区间内多个形态支持候选不由标称长度强行决胜，
   不选最长平台、不补凑边界。区间外平台不能提前触发歧义或弱零标签冲突。
7. 两侧信号转换有支持、单侧/端点未知、边界未定分别记录。靠近原始端点 5 ms
   不直接否决，但不能确认完整性；这些仍是信号形态边界，不是分子真值。

唯一通过的正候选加左右各 15 ms 上下文，裁剪到**源 head-free chunk** 内；
0 nt 保留源 head-free chunk。固定模型输入默认 12000 samples，真实短区间由现有
dataset 右侧 padding。源 chunk 外的候选在选择前排除，上下文后超长时进入复核，
绝不截断候选；导出时仍防御性复查候选包含关系。
统一训练入口从制作摘要读取模型输入大小，并检查所有长度表记录使用同一值。

## 可选：重新筛选已有配套 manifest

所有命令在项目根目录运行，POD5/pysam 仅使用 Bonito 独立环境。示例路径相对于
配置的 work_root 或 data_root；请填入实际的**配套源表**。先查看计划：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STRICT_DATASET="metadata/strict-poly-candidate-v1-trial-$(date +%Y%m%d-%H%M%S)"
STRICT_ARGS=(
  --config configs/local.yaml
  --source-manifest metadata/existing-length/manifest.tsv
  --source-sequence-manifest metadata/existing-sequence/sequence_manifest.tsv
  --dorado-bam dorado_calls/condition/dorado_basecalls.bam
  --output-dir "$STRICT_DATASET"
  --poly-types polyA
  --nominal-lengths 0 20 40 60 80 100 150
  --pool-per-group 300
  --max-signal-reads 2100
  --max-source-rows 50000
  --max-bam-records 10000
)
src/bonito/.venv/bin/python -m holypoly.datasets.build_strict_poly_dataset \
  "${STRICT_ARGS[@]}" --dry-run
```

确认资源计划后，加 `--run` 执行同一参数组。若选中的 UUID 不在前 10000 条 BAM
记录内，会保留明确缺失原因，**不会**扩大扫描。若需要完整的一次 BAM 元数据扫描，
将参数改成 `--max-bam-records 0 --allow-full-bam-scan`，重新 dry-run 后再执行。
`--pool-per-group` 限制审核池，不是保证通过数量；增大它时同步调整信号预算。

```bash
src/bonito/.venv/bin/python -m holypoly.datasets.build_strict_poly_dataset \
  "${STRICT_ARGS[@]}" --run
```

输出只写新外部目录：

| 文件 | 用途 |
| --- | --- |
| `strict_dataset.json` | 资源、规则、来源哈希、版本、实时 progress、完成/失败状态 |
| `source_pool.tsv` | 入池源 read，保留原 split 与原弱标签 |
| `read_audit.tsv` | 每条 read 的结论、失败原因、全信号/区间内候选数和最终是否选入 |
| `candidate_audit.jsonl` | 全信号候选、head-free 选择范围与计数、边界证据和速度参照 |
| `eligible_before_balance.tsv` | 平衡前通过的弱标签样本，仅用于检查 |
| `length/manifest.tsv` | 训练用长度/弱状态表，仅完整成功时可用 |
| `sequence/sequence_manifest.tsv` | 同一批 read 的 reference-head 序列表 |

在原 split 内，按标称长度 × 方向（含 0 nt 两个方向）取共同可用数量进行均衡；
不移动 read 到其他 split。只要请求的任一分层缺失，就返回非零退出码和
`incomplete_no_training_manifest`，保留审计、不输出训练表、不放宽门槛。
如果导出后验证失败，状态为 `failed`；训练脚本拒绝非 `complete` 数据集。

序列标签只保留原配套表中同一 read 的 reference-head 标签，不生成 Poly 序列。
原配对哈希先验证，再将新序列表绑定到**新长度 manifest 的 SHA256**，记录父表哈希，
并调用现有两套训练 manifest 消费者做最终验证。

## 与当前 TCN 训练衔接

```bash
GPU_ID=0 PROFILE_BATCHES=20 \
EXPERIMENT_ID="strict-poly-tcn-context-unfreeze1-$(date +%Y%m%d-%H%M%S)" \
bash scripts/run_joint_training.sh --dataset-dir "$STRICT_DATASET" configs/local.yaml --dry-run
```

把 `GPU_ID` 换为明确选择的单 GPU。预检核验数据集完成状态及两个 manifest 哈希。
默认配置为 TCN、`sum_mean_duration`、sum scale 0.001、解冻最后一层、5 epochs、
batch 32、sequence batch 1、head/encoder LR 0.001/0.00001、三项 loss 1/1/0.1 和
Huber 10 nt。去掉 `--dry-run` 才会启动训练。
普通序列速度用于离线筛选。新数据从原 Bonito 开始，使用新实验 ID。同一实验可使用
原数据和配置 `--resume`。原 `run_strict_poly_candidate_tcn.sh` 兼容入口用于新训。

结果仍在 `results/training_evaluation/<实验ID>`，模型和日志沿用现有外部目录结构。
脚本会运行 best-validation checkpoint 的测试和绘图；其结果仍应描述为**筛选后弱标签
子集上的实验结果**，不能报告独立长度准确率或把未入选 read 从失败率统计中隐去。

本方法没有验证真实数据通过率。严格筛选可能产生很小或不完整的数据集；优先检查审计
原因和扩大预先声明的审核池，不依据 validation/test 结果反复调阈值。0 nt 筛选会改变
负样本分布，正样本筛选也依赖标称长度；后续还需要独立、未经筛选的 read 集合评价。
