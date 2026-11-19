# 新方法：完整 head-free 输入＋平台质控

方法名为 `head-free-platform-qc-v1`，原始数据入口为
`holypoly.datasets.build_head_free_platform_qc_from_raw`，联合训练使用
`scripts/run_joint_training.sh --dataset-dir`，独立初始规则为
`configs/head_free_platform_qc_v1.json`。

这是一条独立的新方法，保留原 head-poly v4 制作方式和
`strict-poly-candidate-v1` 平台裁剪方式。它从原始 BAM、POD5 和参考 FASTA 重新建池，
不要求或复用旧 manifest。共享内部实现通过不可变 method policy 区分行为，
没有修改现有 CLI 的默认规则。

```text
正向信号： [---- HEAD ----][------------- Poly 侧信号 -------------]
平台证据：                            [=======)
训练输入：                 [--------------------------------------)

反向信号： [------------- Poly 侧信号 -------------][---- HEAD ----]
平台证据：          [=======)
训练输入： [--------------------------------------)
```

## 入选与训练区间

- 正长度 read 继续使用当前严格平台验收：稳健归一化、多尺度固定电平扩展、
  跨尺度支持、方向位置、序列侧边界证据、唯一受支持候选，以及候选时长/标称预期时长
  `[0.5, 2]` 的相容性检查。新配置的初始数值与严格裁剪方法相同，可独立记录和调整。
- 检测在整条原始电流上进行，候选必须完整包含于元数据 head-free 区间；
  所有检出平台及其邻域仍用于排除普通序列速度估计窗口。
- 平台通过验收后，**训练的 chunk_start/end_sample 完全保留本次元数据池的值**：
  正向 `[head_boundary_sample, raw_signal_length)`；反向 `[0, head_boundary_sample)`。
  不在平台两端裁剪，不借用 head 填充，不根据平台截取局部输入。
- 平台坐标、边界支持和时长比仅写入 `platform_qc_*` 审计字段；
  `training_interval_policy=full_head_free_raw_edge`。
- 仍继承默认 12000 samples 的 head-free 输入上限，过长整段明确排除，绝不截短来适配模型。
  短输入由现有消费者右侧 padding，padding mask 保留真实长度。

完整 head-free 输入不等于经验证的完整 Poly。原始端点仍可能含采集终止信号，
head/move 边界仍是暂定边界；本方法没有确认分子完整性，也没有新增平台必须贴近 head
或覆盖固定比例的规则。平台条件仍是待标定初始值。

## 0 nt：沿用旧方法的元数据弱负标签

0 nt read 只需通过原 head-poly 元数据准入：构建/比对关联、head、moves、POD5 UUID
和样本数、head-free 区间存在且长度合法等原有检查。它不经过新增的严格平台、位置、
时长、信号动态范围或严格 moves 二次验收，也不因为缺少 reference-head 序列标签被删除。

0 nt 电流不会在本方法的信号 QC 阶段解码；不调用平台检测，也不会因强平台被转入复核。
审计状态为 `weak_zero_metadata_only`，`signal_evaluated=false`，候选数量和采样率等字段留空，
详细审计明确标记 `not_evaluated_metadata_only_zero`，不会伪装成“已检测且未检出平台”。
`reference_head_sequence_available` 单独记录序列标签可用性。

这是构建级弱负标签，不能证明每条 read 都没有 Poly。0 nt 也不表示检测失败或无标签。
平台检测使用的信号读取预算只计算正长度 read；元数据阶段仍会访问 0 nt 的 BAM/POD5
元数据，这不属于电流解码。

## 标签、序列和既有训练接口

所有长度目标仍是 `training_length_nt = true_length_nt = nominal_length_nt`，
`label_source=construct_nominal_weak_supervision`。没有改为用电流时长估计的 nt 标签，
也没有独立逐 read 长度真值。

保持 v4 训练 schema、`supervision_mode=length_only`；`poly_start_sample`、
`poly_end_sample`、`poly_is_complete` 均留空，不监督 location。
`dataset_method=head-free-platform-qc-v1` 区分制作方法。

正长度入选 read 必须有配套 reference-head 序列；0 nt 有序列标签就保留，没有就只参与
长度/状态训练。序列 manifest 可以是最终长度 manifest 的子集，这与旧训练接口一致。
序列标签仅来自参考 head，不生成标称长度的重复 A/T；序列表重新绑定最终长度表 SHA256。

联合训练默认使用三项 loss、TCN 和最后一层解冻。普通序列速度只用于离线 QC。
统一入口核验方法名、完整状态及两张成品表的路径和哈希。新数据从原 Bonito 开始，
同一实验可使用原数据和配置 `--resume`。原 `run_head_free_platform_qc_tcn.sh`
兼容入口用于新训。

## 从原始数据制作

在项目根目录运行，输入相对配置的 data_root，输出相对 work_root。
下面为最终 5000 条/长度/方向的示例；7 个长度、仅 polyA 时共 35000 条，
每组 train/validation/test 为 4000/500/500。请填写实际 condition 路径。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
QC_DATASET="metadata/head-free-platform-qc-final5000-$(date +%Y%m%d-%H%M%S)"
QC_ARGS=(
  --config configs/local.yaml
  --bonito-bam basecalls/condition/basecalls.bam
  --dorado-bam dorado_calls/condition/dorado_basecalls.bam
  --pod5-dir pod5/condition
  --reference fixed_len_ref.fa
  --output-dir "$QC_DATASET"
  --poly-types polyA
  --per-group 50000
  --target-per-group 5000
  --max-records 0
  --allow-full-bam-scan
  --max-signal-reads 350000
  --workers 8
  --pod5-reader-cache-size 8
  --seed 20261117
  --stop-when-full
)
src/bonito/.venv/bin/python -u -m holypoly.datasets.build_head_free_platform_qc_from_raw \
  "${QC_ARGS[@]}" --dry-run
```

查看资源计划后执行：

```bash
src/bonito/.venv/bin/python -u -m holypoly.datasets.build_head_free_platform_qc_from_raw \
  "${QC_ARGS[@]}" --run
```

全扫描为 Bonito 一遍、Dorado 两遍；moves 在本次序列扫描时同步收集。
主进程顺序读 POD5，CPU 子进程仅计算，按审核顺序汇总，满组跳过且全部分层足额后早停。
固定种子在 QC 前建立 split，之后不迁移 read；0 nt 按请求方向分别计数。
`--max-signal-reads` 不包含元数据直接接受的 0 nt；内存为元数据映射加 O(workers) 条检测数组。

审核池耗尽、信号预算耗尽或某分层不足时，状态为 `incomplete_no_training_manifest`，
保留审计，不输出不足目标数量的训练表。输入数据量、磁盘竞争、正长度通过率和审计大小
都会影响耗时；示例是小时量级任务，先使用有限 `--max-records` 和较小配额检查链路。
不自动启动全量扫描或训练，所有结果写入新的外部 work_root 目录。

```text
<QC_DATASET>/
  head_free_qc_raw_dataset.json
  metadata_pool/
    length/manifest.tsv
    sequence/sequence_manifest.tsv
  head_free_qc/
    head_free_qc_dataset.json
    source_pool.tsv
    read_audit.tsv
    candidate_audit.jsonl
    eligible_before_balance.tsv
    length/manifest.tsv
    sequence/sequence_manifest.tsv
```

只有内外两个状态均为 `complete` 才用于训练。`sequence_reads` 可能小于 `selected_reads`，
差值必须全部是缺少序列标签的 0 nt read，不影响长度分层完整性。

## 开始训练

```bash
GPU_ID=0 PROFILE_BATCHES=20 \
EXPERIMENT_ID="head-free-platform-qc-tcn-$(date +%Y%m%d-%H%M%S)" \
bash scripts/run_joint_training.sh --dataset-dir "$QC_DATASET" configs/local.yaml --dry-run
```

选择明确的物理 GPU 编号；去掉 `--dry-run` 才开始训练。输出沿用既有外部
`results/training_evaluation/<实验ID>` 布局。结果仍应作为筛选后的构建弱标签实验报告，
不能当作正式逐 read 准确率或跨批次泛化证据。
