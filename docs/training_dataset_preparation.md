# Holypoly 训练数据集制作说明

更新日期：2026-10-08

## 1. 文档目的

本文档说明 Holypoly 当前可用的训练数据、正式训练集的制作流程、标签定义、
质量控制要求，以及 Holypoly 与 Bonito 原生训练数据的区别。

项目目标是从 Oxford Nanopore cDNA 原始信号中定位 poly(A) 或其反向互补形式
poly(T)，并估计 Poly 区域长度。当前只支持 Bonito v5 Transformer，不涉及旧版
Bonito RNN、多 GPU 或 distributed training。

## 2. 当前数据状态

当前外部 work root 中存在一份 Milestone 3 smoke manifest，可通过
`HOLYPOLY_M3_MANIFEST` 指定。仓库不保存该 manifest、真实 POD5、BAM、模型权重
或 checkpoint。

当前 smoke 数据统计如下：

| 项目 | 数量或状态 |
| --- | --- |
| 总 chunk 数 | 16 |
| train | 12 |
| validation | 4 |
| POD5 文件 | 1 |
| poly(A) | 16 |
| poly(T) | 0 |
| `poly_present` | 16 |
| `zero_nt` | 0 |
| `failure` | 0 |
| `training_length_nt` | 全部为 20 nt |
| signal flank | provisional |

这份数据可用于验证以下工程链路：

```text
POD5 -> fixed-size chunk -> Bonito v5 encoder -> Poly heads
     -> auxiliary loss -> backward -> checkpoint
```

它不能用于报告正式准确率，也不足以有效训练：

- 多长度回归，因为只有一个正长度值；
- poly(A)/poly(T) 分类，因为没有 poly(T)；
- `zero_nt/failure` 状态分类，因为没有对应样本；
- 跨 POD5、run、样本或批次的泛化；
- 高可信 signal-level 边界定位，因为当前 flank 仍是 provisional；
- 正式 MAE，因为当前长度为构建级弱监督，不是独立验证的逐 read 真值。

因此，当前数据应称为 **pipeline smoke dataset**，而不是正式训练集或测试集。

## 3. Holypoly 与 Bonito 训练数据的区别

### 3.1 Bonito 原生训练

Bonito 原生 basecalling 训练通常读取：

```text
chunks.npy
references.npy
reference_lengths.npy
```

对应训练 batch 为：

```text
signal:             [batch, 1, raw_samples]
reference:          [batch, max_reference_bases]
reference_lengths:  [batch]
```

其中 `reference_lengths` 是每条 basecalling 目标序列的有效碱基数，不是 Poly
区域长度。

### 3.2 Holypoly 当前训练

Holypoly 加载预训练 Bonito v5，默认冻结卷积层、Transformer、upsample 和原 CRF
head，只训练 Poly location 和 length 辅助 head。

模型直接输入为：

```text
signal:            [batch, 1, raw_samples]  float32
raw_padding_mask:  [batch, raw_samples]     bool
```

监督标签为：

```text
location_targets:    [batch, encoder_time]  int64
training_length_nt:  [batch]                float32
status_targets:      [batch]                int64
```

当前训练不使用 Bonito 的 `reference`、`reference_lengths` 或 basecalling loss。
因此，Holypoly manifest 不能直接作为 Bonito 原生训练数据。

只有在未来明确决定联合优化 basecalling loss 时，才需要额外制作与每个 signal
chunk 严格对齐的碱基 reference 和 reference length。

## 4. 标签和坐标定义

### 4.1 坐标系统

所有 signal 和 Poly 边界统一采用原始 POD5 signal sample 坐标：

```text
0-based、左闭右开：[start_sample, end_sample)
```

manifest 中的绝对坐标包括：

```text
chunk_start_sample
chunk_end_sample
poly_start_sample
poly_end_sample
```

当前 Bonito v5 实测 raw-sample 到 encoder timestep 的 stride 为 6。该值只能用于
已经验证过的当前 v5 模型，不能自动应用于其他模型。

### 4.2 Poly 类型

location head 使用三个类别：

```text
0 = background
1 = poly(A)
2 = poly(T)
```

`poly_type` 必须表示该 read 的原始信号方向中实际需要识别的类型。不能只根据构建
参考名称推断，应结合：

- BAM 比对方向；
- basecalled sequence 中的 A/T 区域；
- read 与参考序列的方向关系；
- cDNA 接头和预期 Poly 端位置；
- 必要时的 move table 映射。

`orientation` 和 `poly_type` 必须分别保存。方向不明确时不能强行生成 A/T 标签，
应标记为无标签或失败，并记录原因。

### 4.3 Read 状态

当前状态定义为：

```text
poly_present
zero_nt
failure
```

| 状态 | Poly 边界 | 长度目标 | location 目标 |
| --- | --- | --- | --- |
| `poly_present` | 必须存在 | 正有限值 | poly(A) 或 poly(T) |
| `zero_nt` | 不存在 | 0 | 全 background |
| `failure` | 不存在 | 空值或 NaN | ignore index |

`0 nt` 是合法负标签，不等同于检测失败。

### 4.4 长度字段

以下字段不得互相替代：

```text
nominal_length_nt
observed_sequence_length_nt
training_length_nt
```

- `nominal_length_nt`：构建设计或参考名称给出的标称长度；
- `observed_sequence_length_nt`：Dorado 或其他 basecaller 在序列中观察到的长度；
- `training_length_nt`：本次训练明确选择的回归目标。

只有经过独立方法验证的单条 read 长度，才能称为 read-level ground truth。若使用
构建标称长度或 basecaller 观察长度，必须将其描述为弱监督或 provisional 标签，
并填写 `label_source`。

## 5. 正式训练集制作流程

### 5.1 数据清点和只读体检

输入数据应位于只读 `data_root`，生成结果只能写入 `work_root`。首先运行：

```bash
uv run holypoly doctor --config configs/local.yaml
```

该步骤只检查目录、权限、文件数量和少量文件头，不读取完整数据集。

在开始批量处理前，应记录：

- POD5 文件数量和总大小；
- BAM/FASTQ 文件数量和总大小；
- read 数量估计；
- 预计输出大小；
- 预计运行时间；
- CPU、内存、磁盘和 GPU 需求。

### 5.2 建立 read-level 清单

从 BAM、POD5、参考序列和构建元数据建立 read-level 清单，至少包含：

```text
read_id
pod5_path
bam_path
pod5_file_id
alignment_orientation
nominal_length_nt
```

现有 `src/holypoly/datasets/length_groups.py` 可以从 BAM 比对参考名称解析标称长度，并按
`nominal_length_nt` 整理 read。该结果只表示构建级分类，还不能直接训练 location
head。

### 5.3 确定 Poly 类型和序列区间

对每条候选 read：

1. 获取 basecalled sequence 和比对方向；
2. 确定该 read 应表现为 poly(A) 还是 poly(T)；
3. 在序列中定位候选 Poly run；
4. 保存观察长度和识别规则；
5. 对方向冲突、多个候选区间或边界不明确的 read 记录失败原因。

Poly 区域长期定义允许少量非目标碱基，最大连续中断长度为 2。但在将该规则写入
生产标签生成器前，需要明确并验证：

- run 的起止判定；
- 多个候选 run 的选择规则；
- 中断是否计入长度；
- poly(A) 与 poly(T) 是否完全对称处理；
- read 端点截断时的状态定义。

### 5.4 将序列区间映射到原始信号

推荐使用 Dorado move table 或其他经过验证的序列—信号映射，将 Poly 序列区间转换
为 raw-sample 区间：

```text
basecalled Poly interval
        -> move table mapping
        -> provisional signal interval
        -> flank/boundary refinement
        -> manual or independent audit
```

必须记录：

```text
poly_start_sample
poly_end_sample
flank_source
flank_status
label_source
```

provisional flank 可以用于 pipeline 验证或弱监督训练，但不能直接称为真实边界。

### 5.5 选择训练长度

建议的证据优先级为：

1. 独立验证的 read-level 长度；
2. 高可信 sequence/move-table 推导长度；
3. basecaller 观察长度；
4. 构建标称长度弱监督。

不能由程序静默选择 truth。使用哪一个字段作为 `training_length_nt`，必须由数据集
版本或配置显式确定，并写入 `label_source`。

例如，使用构建标称长度时应明确记录：

```text
training_length_nt = nominal_length_nt
label_source = construct_nominal_weak_supervision
```

### 5.6 制作完整的状态样本

正式数据集应覆盖：

- 多个正 Poly 长度；
- poly(A)；
- poly(T)；
- `zero_nt`；
- `failure`；
- 不同 read 质量区间；
- 不同 POD5、run、样本和测序批次；
- 不同方向和可能的截断情况。

不能为了类别平衡将不确定 read 强行标成 `zero_nt` 或 `poly_present`。

### 5.7 截取固定长度 signal chunk

当前默认输入长度为：

```text
chunk_size_samples = 12000
```

补齐后的模型输入长度必须满足：

```text
chunk_size_samples % 12 == 0
```

无 head 的 v3 manifest 将实际保留区间写在 `chunk_start_sample/chunk_end_sample`，
将补齐后的固定输入大小写在 `model_input_samples`。Dataset 仅截取该区间，
归一化后右侧补零；补零位置通过 raw/encoder padding mask 排除定位 loss 和
长度池化。真实区间可以短于 12000，也不必是 12 的倍数。超过输入上限者拒绝。
`head_excluded=true` 和 `head_boundary_sample` 用于检查正反向输入均不跨入 head。
排除保证依据 Dorado 比对与 move table，仍需要人工审计实际边界。

对 `poly_present` 记录，当前训练要求整个 Poly 区间位于 chunk 内：

```text
chunk_start_sample <= poly_start_sample
poly_end_sample <= chunk_end_sample
```

如果 Poly 跨 chunk、位于 read 边界或无法完整覆盖，不能静默裁剪。应明确标记为
截断、无标签或失败，直到跨 chunk 策略得到独立验证。

### 5.8 划分 train、validation 和 test

最低要求是不同 split 之间不得出现重复 `read_id`。

正式独立验证建议进一步按以下采集单位隔离 test：

- POD5 文件；
- sequencing run；
- 生物样本；
- 构建批次；
- 测序批次；
- 物种。

train 和 validation 可以来自相同 POD5 文件，但必须避免同一 read 跨 split 泄漏。
同一 read 的多个 chunk 必须全部位于同一 split。

## 6. Manifest schema

每行表示一个真实信号区间及其固定模型输入大小。当前权威 schema 位于
`src/holypoly/training/data.py`。

必需字段包括：

```text
manifest_version
read_id
pod5_path
bam_path
pod5_file_id
split
orientation
raw_signal_length
chunk_start_sample
chunk_end_sample
poly_start_sample
poly_end_sample
poly_type
training_length_nt
nominal_length_nt
observed_sequence_length_nt
label_status
poly_is_complete
failure_reason
label_source
flank_source
flank_status
selection_reason
```

路径可以是绝对路径，也可以相对于 manifest 所在目录解释。真实 manifest 必须位于
外部 work root，不得提交到 Git。

## 7. 数据质量控制

生成 manifest 后至少检查：

### 7.1 结构检查

- 文件非空；
- 必需字段完整；
- 枚举值合法；
- 数值字段可以解析；
- `(read_id, chunk_start_sample, chunk_end_sample)` 不重复；
- 不同 split 不包含相同 `read_id`；
- 所有引用的 POD5 文件存在且位于 data root；
- 输出路径位于 work root，而不是 data root。

### 7.2 坐标检查

```text
0 <= chunk_start_sample < chunk_end_sample <= raw_signal_length
```

对 `poly_present`：

```text
chunk_start_sample <= poly_start_sample
poly_start_sample < poly_end_sample
poly_end_sample <= chunk_end_sample
poly_is_complete = true
```

### 7.3 标签一致性检查

- `poly_present` 必须具有 `polyA` 或 `polyT`；
- `poly_present` 的训练长度必须为正有限值；
- `zero_nt` 必须使用 `poly_type=none` 和 `training_length_nt=0`；
- `failure` 必须使用 `poly_type=none`，长度留空，并填写 `failure_reason`；
- `nominal_length_nt`、`observed_sequence_length_nt` 和
  `training_length_nt` 不得被静默覆盖；
- 所有弱监督标签必须记录来源和限制。

### 7.4 分布检查

至少汇总：

- 每个 split 的 read/chunk 数；
- Poly 长度分布；
- poly(A)/poly(T) 数量；
- `poly_present/zero_nt/failure` 数量；
- 方向分布；
- POD5、run、样本和批次分布；
- label source 和 flank status 分布；
- chunk 内 Poly 相对位置分布。

## 8. 训练前验证

配置外部路径：

```bash
export HOLYPOLY_DATA_ROOT=/path/to/read-only/data
export HOLYPOLY_WORK_ROOT=/path/to/writable/work
export HOLYPOLY_M3_MANIFEST=/path/to/writable/work/metadata/train.tsv
export HOLYPOLY_BONITO_MODEL=/path/to/bonito-v5-model
export HOLYPOLY_M3_CHECKPOINT=/path/to/writable/work/models/holypoly-heads.pt
```

首先执行：

```bash
uv run holypoly train --dry-run
```

dry-run 应只读取 manifest 元数据并检查 Bonito `config.toml`，不读取 POD5 signal、
不加载模型权重，也不创建 checkpoint。

通过后，再在显式单 GPU 环境运行小规模训练：

```bash
CUDA_VISIBLE_DEVICES=<physical-gpu-id> \
uv run holypoly train --epochs 1 --batch-size 1
```

在扩大数据规模前，应先确认：

- forward 和 loss 有限；
- backward 成功；
- 冻结参数没有梯度；
- 两个 Poly head 具有梯度；
- checkpoint 可以保存和恢复；
- 固定随机种子下结果可重复；
- basecalling 输出接口未改变。

## 9. 当前工具与限制

当前仓库已经包含：

- 外部数据只读体检；
- 按参考名称整理标称长度 read；
- manifest 读取和严格校验；
- POD5 signal chunk Dataset；
- location、length 和 status 标签构建；
- 冻结 Bonito encoder 的单 GPU训练；
- checkpoint、恢复和基础指标；
- synthetic test 和真实数据 smoke test。

当前已有面向固定长度构建的精简 builder：
`src/holypoly/datasets/prepare.py`，用法见 [prepare_dataset.md](prepare_dataset.md)。
它完成 UUID 关联、0A provisional flank 推定、Dorado move 映射、均衡抽样、
无 read 泄漏划分及现有训练接口的 manifest 校验。
边界仍未经独立验证，不支持截断 Poly 训练或通用构建结构。完整流程包括：

```text
BAM + move table + POD5 + reference/construct metadata
                        -> read/POD5 association
                        -> orientation determination
                        -> poly(A/T) sequence detection
                        -> sequence-to-signal mapping
                        -> boundary and label audit
                        -> balanced sampling
                        -> leakage-safe split
                        -> versioned training manifest
```

上述精简 builder 已明确采用标称长度弱监督、最多连续 2 个非目标碱基中断、
双侧锚点完整性检查及已验证 0A 负样本。

经确认的启动训练妥协为：正长度构建缺少可靠 Poly 边界时，允许
`supervision_mode=length_only`。该模式保留标称长度，未知边界、完整性和观察长度
留空，输入覆盖 head 后的可见 signal（可以含 flank），定位与状态 loss 均忽略。
它不声称 read 的 Poly 完整；v3 短候选右侧补零，无法放入固定输入的候选仍被拒绝。
完整操作说明见 [无 head 数据集准备](prepare_dataset.md)。旧 manifest 保持兼容；
仅重新制作的数据带有 head 排除保证，应使用新的 checkpoint 从头训练。
0A 仍须通过双侧锚点检查。缺少 `supervision_mode` 的旧 manifest 按完整监督处理。

扩展为其他数据流程前仍须确认：

1. `training_length_nt` 的首选证据来源；
2. Poly 中断规则的精确定义；
3. 截断 Poly 的训练状态和 loss 行为；
4. poly(A)/poly(T) 与 read orientation 的转换规则；
5. signal flank 的生成和独立审核方法；
6. `zero_nt` 和 failure 样本的来源及判定标准。

## 10. 推荐下一步

1. 为正式 dataset builder 定义输入、输出和标签策略；
2. 使用少量 read 人工审计方向、move table 映射和 signal 边界；
3. 制作同时包含多个长度、poly(A)、poly(T)、`zero_nt` 和 failure 的小型开发集；
4. 增加 manifest builder 的单元测试和 synthetic fixtures；
5. 完成单 GPU 小规模训练和 checkpoint 恢复验证；
6. 在独立 POD5/run 上建立只用于最终评价的 test split；
7. 只有获得可靠 read-level 标签后，才报告正式 MAE 和 tolerance accuracy。
