# 参考 head 序列监督与联合训练

2026-10-28 起，这是新数据与新常规训练的默认流程，入口见
[当前默认流程](default_workflow.md)。保留 v4 原 manifest、长度标签、原网络结构和旧实验。
head 序列 manifest 与长度 manifest 配套制作；历史冻结入口用于旧实验和对照。

## 标签与训练方式

序列监督来自已有参考 FASTA 中的 **head 内部序列**。FASTA 仍按 v4 的约定解释为
head；不把标称长度复制成 `AAAA…`，也不把 Dorado 的整条 basecall 当真值。

1. 仅使用指定 v4 manifest 的 read ID，严格继承 train/validation/test split。
2. 用 Dorado BAM 比对和 move table 找到 head 内部的一段信号。默认取 96 bp，
   距 head 两端至少 32 bp，优先选靠 head 末端的有效窗口。
3. 要求连续完全匹配参考、无 indel、MAPQ ≥20、每个碱基 Q≥15；窗口不得含超过
   4 bp 的同聚物，并须在 head 和该条 basecall 中唯一。排除共享 move 边界的切口。
   这里的同聚物过滤只用于挑选序列监督窗口，不改变 Poly 定义。
4. 信号窗口最多 4096 samples，不与原 Poly 候选重叠；检查方向和 raw 长度。
   反向 BAM 的目标取反向互补，使其与原始信号采集方向一致。
5. 未见到、重复 primary、筛选失败等情况均保留在 `audit.tsv`，不伪造序列标签。

序列标签记录 `reference_head_exact_match`，信号边界记录
`dorado_moves_provisional`，并保留源 manifest、参考的 SHA256。
这些是参考支持的筛选标签；不是独立确认的逐 read 真值。

每次更新分别 forward 原 Poly 候选和独立 head 片段，二者使用同一个 encoder：

```text
joint_loss = lambda_length × Huber(predicted_length, 原 v4 长度标签)
             + lambda_status × CE(shared_status_logits, 弱状态标签)
             + lambda_basecall × Bonito 原生 CTC-CRF(head_signal, reference_head)
```

原始 v4 manifest 是 `length_only`。联合入口默认启用 `--supervise-status`：
沿用原 manifest 的 `label_status`，记录来源为 `construct_nominal_weak_supervision`；
0 nt 构建监督 `zero_nt`，正标称长度构建监督 `poly_present`。
这不是逐 read 生物学真值，不按 basecaller 长度改标签，也不引入 `<5 nt` 新定义。
原 manifest、长度目标和 `length_only` 字段保持原样，location 始终 ignore。

Status loss 为原两个 head 的 status logits 均值上的三分类交叉熵，
只计算一次；两个现有 status 分支均可获得梯度，无需改网络结构。
保留 `failure` 类，但 v4 没有 failure 标签，不能由此声称学会检测失败。
联合入口目前使用普通 CE，不做类别重采样或加权；训练类别数量写入 `joint_config`，
后续根据 validation 的负样本召回和正样本误判判断是否需要平衡。

保留 Bonito 原 CRF head、
解码及参数名称；冻结卷积、upsample、CRF 和较早 Transformer，仅解冻最后 N 层。
不使用 `torch.nn.CTCLoss` 替代原生 CTC-CRF。长度头固定为 sum pooling。

脚本默认：5 轮、长度 batch=32（直接 CLI 默认 4）、head batch=1、最后 1 层、head LR=1e-3、
encoder LR=1e-5、Huber delta=10 nt、basecall 权重=0.1、seed=20260923。
它们是小实验起点，尚未证实为最佳参数。
Status 在脚本和 CLI 中均默认开启；可显式设置
`LAMBDA_LENGTH=1 LAMBDA_STATUS=1 LAMBDA_BASECALL=0.1`。
旧两项 loss 续训需使用 `--no-supervise-status`，脚本中的显式两项消融用
`STATUS_SUPERVISION=0`，旧 checkpoint 和结果不被改写。
length/status 权重未显式指定时取配置文件，启用状态后 location 权重记为 0。

## 制作序列 manifest

在项目根目录运行。以下路径是示例：`metadata/experiment/manifest.tsv` 换成已有
小型 v4 manifest；BAM、参考相对于 data root，其他路径相对于 work root。
所有输出目录必须是新目录。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
src/bonito/.venv/bin/python -m holypoly.datasets.prepare_sequence \
  --config configs/local.yaml \
  --manifest metadata/experiment/manifest.tsv \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --reference fixed_len_ref.fa \
  --output-dir metadata/experiment-reference-head \
  --max-records 100000 --dry-run
```

去掉 `--dry-run` 才会扫描和生成文件。dry-run 仅读取 manifest/参考及文件元数据，
不扫描 BAM、不加载权重、不读 POD5 信号。

默认最多扫描前 100,000 条 BAM 记录。若需覆盖小 manifest 在完整 BAM 中分散的
read ID，显式指定 `--max-records 0` 全量扫描一次，仍只为已有 ID 制作序列标签。
当前 poly2p5per BAM 约 3.12 GiB；全扫可能需数分钟至数十分钟，使用 CPU，
不读 POD5 信号、不用 GPU。1400 个候选的 metadata 通常为几 MB。
有界扫描只检查已扫描前缀内的重复 primary，且可能受 BAM 顺序影响。

输出 `sequence_manifest.tsv`、`audit.tsv`、`summary.json`。先看 accepted 数量、
split_counts 和失败原因；至少需有 train 和 validation 的 accepted read 才能训练。
head 筛选不会删除或改动原来的长度样本。

## 训练、测试与绘图

```bash
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv --dry-run
```

去掉 `--dry-run` 开始训练；默认物理 GPU 0，内部为 `cuda:0`。可用环境变量覆盖：

```bash
GPU_ID=0 EPOCHS=5 LAMBDA_BASECALL=0.1 ENCODER_LEARNING_RATE=0.00001 \
INITIALIZE_FROM=models/frozen-sum-huber/heads.epoch_0008.pt \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv
```

三项 loss 已默认启用；显式记录权重的命令示例为：

```bash
STATUS_SUPERVISION=1 LAMBDA_LENGTH=1 LAMBDA_STATUS=1 LAMBDA_BASECALL=0.1 \
GPU_ID=0 EPOCHS=5 \
INITIALIZE_FROM=models/frozen-sum-huber/heads.epoch_0008.pt \
bash scripts/run_reference_joint.sh \
  metadata/experiment/manifest.tsv \
  metadata/experiment-reference-head/sequence_manifest.tsv
```

脚本环境变量分别映射到 `--supervise-status`、`--lambda-length`、
`--lambda-status`、`--lambda-basecall`；`HUBER_DELTA_NT` 对应 `--huber-delta-nt`。
所有设置进入日志、run summary 和 checkpoint。旧两项 loss 的 run 不会被改写。

`INITIALIZE_FROM` 可选；须为匹配 sum/Huber、hidden size、delta、Bonito 配置的冻结
checkpoint，重新创建 optimizer。必须确认它只使用当前 train split 训练过，不能用
当前 validation/test read 训练过的模型；旧 checkpoint 不含足以自动证明这一点的
数据来源。未指定时从原 Bonito 权重和新随机长度头开始。
初始化允许改变 loss 权重；长度 loss 定义及 delta 必须匹配，源 checkpoint
的 loss 配置也记录到初始化信息。它是新实验，epoch/global step 重新计数。

每个长度 batch 配一个独立打乱的 head batch，短 loader 循环重读。
因此每轮可能未用完所有 accepted head read；日志记录调用数和唯一 read 数。
validation/test 遍历各自全部 accepted head，序列逐条 forward，避免批量 padding
进入 CRF loss。完整实验会反复读取所选 POD5 read，预计耗时可能为小时级；
部分 encoder backward 比冻结训练占用更多显存，应先用小 manifest。

脚本会创建新的 `models/<run>`、`logs/<run>`、`results/training_evaluation/<run>` 和 pipeline 日志：

- 训练前保存 head validation 的原始 CRF loss/解码编辑率作为 baseline。
- 每轮保留完整 checkpoint；按 validation 长度偏差最小选择最佳轮次。
- 对最佳轮次执行原长度 test；有 head test 样本时同时导出其 CRF/编辑率。
- 绘制训练曲线、CRF/参考编辑率曲线和长度 test 图。

启用 status 后，训练日志中的 `status_loss`、`status_accuracy` 应有有效数值；
location loss 仍为 0。测试 `summary.json` 增加 `status_metrics` 和
`status_metrics_by_nominal_length`：混淆矩阵、0 nt 假阳性比例、正样本误判为零比例、
正样本召回及被判 failure 的比例。分母为成功预测且有对应弱标签的 read，
输入/预测失败另计。检查 20 nt 组是否被过度判零，100/150 nt 回归是否恶化。

`predictions.tsv` 新增 shared present/zero/failure 三项概率，方便后续只用 validation
选择阈值。当前状态决策使用 shared logits 的 argmax，不自动选阈值或强制判零。
原始 `regression_length_nt` 继续驱动所有长度指标；旧 `decoded_length_nt` 使用单独
length status 分支，不等同于共享状态解码，不应混作联合状态的最终长度结果。

完整 checkpoint 预计每份约 0.4 GiB，5 轮含 latest 约 2.4 GiB，另留临时文件空间。
实际大小以运行结果为准。新目录已存在时拒绝覆盖。

续训使用独立入口，`--epochs` 是目标总轮数；保持所有其余设置相同：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src src/bonito/.venv/bin/python \
  -m holypoly.training.joint_command \
  --config configs/local.yaml \
  --manifest metadata/experiment/manifest.tsv \
  --sequence-manifest metadata/experiment-reference-head/sequence_manifest.tsv \
  --run-id <existing-joint-run> --epochs 10 --resume
```

不要用旧冻结训练命令续训 joint checkpoint。恢复会检查 joint/loss 配置、数据哈希、
完整日志和 baseline；不覆盖旧 epoch 快照。断电发生在日志/快照发布中间时，可能
需要人工核对二者再恢复，程序不会自动删除冲突记录。原 epoch 的 head test 文件保留。
三项 loss 实验续训保持 status 开启和原有权重参数；不能把旧两项 loss
实验直接按三项 loss resume。旧两项续训显式加 `--no-supervise-status`。

## 如何判断收益

与使用相同 split 的冻结 sum/Huber baseline 对照：检查 0/100/150 nt 的预测偏差、
整体弱标签偏差及 ±10 nt 比例，同时比较训练前后的 head validation CRF/编辑率。
若长度改善而 head 编辑率明显变差，优先降低 encoder LR 或减少解冻层数，
必要时提高 basecall 约束，再用 validation 选择。
权重影响方向需结合实际梯度和验证曲线判断，不能仅比较两个 loss 数值。

head 编辑率只表示被筛选的 head 片段与参考的一致程度，不代表整条 read 的正式
basecalling 准确率；head 监督也不保证能解决同聚物长度估计或长 Poly 低估。
不同构建的 head 可能不同，仍需后续独立批次验证。

已验证合成输入上的标签坐标、split、梯度、checkpoint、原生 CRF 和解码链路；
真实小 manifest 通过 metadata dry-run；默认三项 loss 的开关及旧两项显式关闭
有合成验证。合成测试不证明其能改善 0 nt 假阳性或长度估计。

关键实现：`datasets/prepare_sequence.py`、`training/sequence_data.py`、
`models/joint_model.py`、`training/joint_engine.py`、`training/joint_command.py`。

## 本次变更范围

- 新增上述 5 个模块，以及 `scripts/run_reference_joint.sh`。
- `training/engine.py`：可选 joint checkpoint 配置及恢复检查。
- `training/evaluate.py`：评估汇总携带 joint 配置。
- `scripts/plot_training.py`：新增 CRF/编辑率曲线，按实际 loss 命名。
- `pyproject.toml`：为新的 Torch Dataset 添加现有模式的 mypy 例外，不新增依赖。
- 测试：新增 `tests/isolated/datasets/test_prepare_sequence.py` 和
  `tests/isolated/training/test_reference_joint.py`；扩展
  `tests/unit/training/test_plot_inputs.py`。
- 文档：本文件、README、`docs/common_commands.md` 和 `docs/checkpoints.md`。

Bonito 子模块、原始数据和历史实验结果未修改。
