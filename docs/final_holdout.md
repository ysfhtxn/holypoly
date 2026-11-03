# 独立定长比较集（完整 POD5）

用途：冻结一份未用于训练或调参的 read 清单，让 Holypoly 和其他工具从相同完整
原始信号出发比较。仅 polyA，按构建标称长度分组，0 nt 为单独负组。
标称长度仍是构建级弱标签，不是逐 read 真实长度。

## 制作步骤

1. 在创建新候选 manifest **之前**，汇总全部历史长度/序列训练 manifest，
   至少排除 train/validation；推荐同时排除历史 test。
   `holypoly.datasets.export_holdout.history_exclusions` 支持合并、SHA256 溯源和
   单个 manifest 内的跨 split 重复检查。UUID 清单只存入外部 work root。
2. 采用现有 v4 `head-poly / raw-edge` 规则重新筛选，使用
   `prepare --exclude-read-ids` 在配额计算、边界检查和 POD5 元数据访问之前
   去除旧 read。不读取全量 POD5 信号。
3. 导出器对剩余候选按长度等量抽样；每组数量为请求上限和最小可用组数量的
   最小值。缺失任意长度组时失败，不静默省略该组。

示例参数（输入文件相对 data root，输出及排除清单相对 work root）：

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --structure head-poly --signal-end-policy raw-edge \
  --bonito-bam basecalls/<sample>/basecalls.bam \
  --dorado-bam dorado_calls/<sample>/dorado_basecalls.bam \
  --pod5-dir pod5/<sample> --reference fixed_len_ref.fa \
  --output-dir metadata/<new-candidates> \
  --exclude-read-ids metadata/<exclusion-inventory>/read_ids.txt \
  --max-records 0 --per-group 500 --poly-types polyA \
  --stop-when-full --seed 20261102

PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.export_holdout \
  --config configs/local.yaml \
  --manifest metadata/<new-candidates>/manifest.tsv \
  --exclude-manifests metadata/<historical-1>/manifest.tsv \
                      metadata/<historical-2>/sequence_manifest.tsv \
  --exclude-all-history-splits \
  --output-dir processed/<new-final-holdout> \
  --lengths 0 20 40 60 80 100 150 --per-group 500 --seed 20261102
```

`--exclude-manifests` 必须列全历史来源；不要把新候选 manifest 作为历史来源。
导出器的 `--dry-run` 只检查 metadata，不创建文件、不读 POD5 信号。
命令耗时取决于 BAM 大小，执行全量扫描前应报告规模并确认授权。

## 输出与验证

* `balanced.pod5`：完整原始 read，不裁剪、不归一化。
* `labels.tsv`：构建标称弱标签、来源、`final_test` 用途、原/新 POD5 路径、信号哈希。
* `read_ids.txt`：固定比较清单。
* `summary.json`：历史 manifest 哈希、排除和均衡统计、随机种子、代码版本、
  输出哈希及完成标志。

导出后逐条重读，核对 UUID、原始 int16 样本哈希、长度、校准和关键采集信息，
并检查原文件大小/mtime 未变。只导出选定 read，不运行模型或按预测成功筛选。

新候选 manifest 中的 train/validation/test 是准备工具自动分配的来源字段，
不代表可用于训练；导出的整个队列保留为 `final_test`，不得用于调参。
`labels.tsv` 是独立比较集 sidecar，不是训练 manifest。

## 后续运行现有 POD5 预测入口

导出目录位于 work root，而现有预测入口要求 POD5 在 data root 下。
仅对此次命令将 data root 显式设为导出目录即可，无需移动 POD5：

```bash
FINAL_POD5_DIR="${HOLYPOLY_WORK_ROOT}/processed/<new-final-holdout>"

HOLYPOLY_DATA_ROOT="$FINAL_POD5_DIR" \
DATASET_NAME=fixed-len-final RESULT_CATEGORY=training_evaluation \
GPU_ID=0 MAX_READS=0 BATCH_SIZE=8 POLY_TYPES=polyA \
CHECKPOINT="models/<frozen-run>/heads.epoch_0005.pt" \
bash scripts/holypoly_predict_cdna.sh "$FINAL_POD5_DIR"
```

先设置 `HOLYPOLY_WORK_ROOT` 为既有外部工作目录，checkpoint 保持冻结。
这一步需要单 GPU，不属于数据导出步骤，不会由导出工具自动执行。
预测输出与 `labels.tsv` 按 read_id 联结后才能统计各标称长度的偏差及失败率。

## 解释限制

* 原始信号完整复制不等于生物学 Poly 完整，边界与逐 read 真值仍未独立验证。
* 来源是同一采集批次；read 不重叠不能证明跨批次泛化。
* v4 候选规则、可对齐性及 BAM 顺序早停会产生选择偏差。
* 比较时保留所有 read 的失败状态，同时报告覆盖率和成功子集的标称长度偏差。
* 此排除检查依赖提供的历史 manifest 清单；旧测试曾被用于调参时也应排除。
