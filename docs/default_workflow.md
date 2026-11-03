# 当前默认数据与训练流程

2026-10-28 用户确认：以后新数据集采用 v4 长度/状态 manifest + 配套
reference-head sequence manifest，新常规训练采用 length + status + basecalling。
历史数据、模型和结果保留；单独长度训练、状态校准和消融仅在明确需要时使用。

新训练验证/测试和绘图统一写入 `results/training_evaluation/<run>/`；
实际 POD5 预测写入 `results/cdna_prediction/<experiment>/`。
命名及历史续训兼容规则见 [结果目录约定](result_layout.md)。

## 制作数据集的两个步骤

1. `holypoly.datasets.prepare`：排除 head，保留 raw-edge Poly 候选，长度目标
   为构建标称弱标签。A/T 联合集按正长度×类型、0 nt 两方向严格均衡。
2. `holypoly.datasets.prepare_sequence`：仅为上述 manifest 的 read 挑选参考
   head 内部的可靠序列窗口，继承同一 split，记录源 manifest 哈希。
   不改长度标签，不伪造 Poly 序列，也不把候选边缘当成已确认的 Poly 终点。

每个新数据集最终有一个 `manifest.tsv` 和一个匹配的 `sequence_manifest.tsv`。
head 筛选失败会记录审计，原长度样本不会被删除；head 配套数据数量可能较少，
不保证其类型/长度分布也均衡。先核对 accepted 数量、split、方向和失败原因。
不能复用源自另一 manifest 的旧 sequence manifest。

长度制作命令见 [v4 制作方法](prepare_dataset_v4.md)，配套 head 的完整筛选规则见
[参考 head 序列监督](reference_joint_training.md)。新目录必须尚不存在。
`--per-group` 是目标上限：当前 poly2p5per 在现有筛选规则下，40 nt polyT
只有 577 条，严格均衡时请求 5000 条也会将各组统一缩减到 577 条。

## A/T 8078 条联合集的配套制作示例

现有长度 manifest 已制作完成，无需重做。先生成其配套 head 数据：

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare_sequence \
  --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-poly-v4-polyAT-balanced-5000pergroup/manifest.tsv \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-poly-v4-polyAT-balanced-5000pergroup-reference-head \
  --max-records 0 --dry-run
```

`--dry-run` 仅读取 manifest、参考及文件元数据，不扫描 BAM、不写文件。
检查后去掉它，由用户启动实际制作：会顺序读取完整 Dorado BAM 一次，当前约
3.12 GiB，CPU 运行，可能数分钟至数十分钟；不读 POD5 信号、不用 GPU，
8 千条候选的 metadata 预计数 MB 至十余 MB。先查看新 `summary.json`，
确认 train/validation 有 accepted head 窗口，再进入训练。

## 默认三项联合训练

当前新 polyA 数据集已完成两步制作：35000 条长度样本，34429 条 accepted
reference-head 样本。长度 split 为 train/validation/test = 28000/3500/3500，
head split 为 27538/3441/3450。先 dry-run，再去掉开关开始训练：

```bash
GPU_ID=0 EPOCHS=5 \
LAMBDA_LENGTH=1 LAMBDA_STATUS=1 LAMBDA_BASECALL=0.1 \
bash scripts/run_reference_joint.sh \
  metadata/poly2p5per-head-poly-v4-polyA-5000pergroup-20261028-164859/manifest.tsv \
  metadata/poly2p5per-head-poly-v4-polyA-5000pergroup-20261028-164859-reference-head/sequence_manifest.tsv \
  --dry-run
```

脚本和 `joint_command` 均默认启用 status，无需额外设置 `STATUS_SUPERVISION=1`。
去掉 `--dry-run` 才开始训练、最佳 validation 轮次测试和绘图。物理设备由
`GPU_ID` 选择，内部使用 `cuda:0`。默认 batch=32、head batch=1；训练读取
所选 POD5 信号，单 GPU，可能为小时级；5 轮 checkpoint 通常需要数 GiB，
实际空间和训练资源由启动检查报告。先 dry-run，资源不足时减小 batch。

```text
total_loss = 1.0 × Huber(length, nominal_length; delta=10 nt)
           + 1.0 × CE(shared_status_logits, weak_status)
           + 0.1 × Bonito_native_CTC_CRF(head_signal, reference_head)
```

使用 sum pooling；location 不监督。仅解冻 encoder 最后一层 Transformer 和
Poly heads，原 basecalling head 保持冻结。三项系数可通过环境变量或 CLI 覆盖。
status 当前为普通 CE，没有自动类别加权；6:1 的状态数量差仍需独立验证。
数据增强尚未实现或默认启用。

默认从原 Bonito 权重和新 Poly heads 开始。只有确认 checkpoint 未见过当前
validation/test reads 后才能设置 `INITIALIZE_FROM`；重新分 split 不会消除
旧 checkpoint 已见过这些 reads 的事实。

脚本自动按最佳 validation 轮次测试和绘图。长度图使用实际 nt 数值横轴，
仅 polyA 的预测表只显示 polyA 与 0 nt；不用为不存在的类型预留位置。

旧两项 loss 实验恢复时显式使用 `--no-supervise-status` 并保持原配置。
`STATUS_SUPERVISION=0` 可用于明确的脚本消融，常规新训练保持默认三项 loss。
历史 `holypoly train` 和旧消融脚本保留原实现，不是当前默认实验流程。

评价仍针对弱标签一致性；分别报告 A/T、各长度、0 nt 两方向，重点检查
validation 的 0 nt 召回、20 nt 误判及 100/150 nt 偏差。不将这些指标描述为
独立逐 read 真值下的正式准确率。
