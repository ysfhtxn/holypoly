# 可选双分支 TCN 预测头

通过 `head_type=legacy/tcn` 切换，默认继续使用 legacy。TCN 接收 Bonito 原 CRF
投影前的 `[B,T,C]` 特征；Bonito、原 CRF、原始数据处理和损失定义均不改变。

两个分支有独立的投影层和四个残差块。各自先做
`LayerNorm(C) → Linear(C,128) → GELU`，再依次使用 dilation 为 1、2、4、8 的块。
每个块是 `LN → Conv → GELU → Dropout(0.1) → LN → Conv → Dropout(0.1) + residual`。
卷积都是非因果的，kernel=3、stride=1、padding=dilation；LayerNorm 沿通道执行。
padding 在输入投影、LayerNorm 后卷积前、每次卷积后和残差输出处清零。

- 长度分支采用 masked sum；长度回归路径为
  `[sum(H) × tcn_length_sum_scale, mean(H), log1p(N_valid)]`
  `→ 257 → 64 → 1 → Softplus`，隐藏激活为 GELU。
  长度状态路径保持未缩放的 `sum(H) → 128 → 64 → 3`，直接返回原始 logits。
- 位置分支采用 masked mean，再接 `128 → 64 → 3` MLP，只输出全局状态。
- 三状态保持 `poly_present / zero_nt / failure`；现有两路 logits 平均、CE 状态
  loss 和长度 loss 原样保留，0 nt 仍参与长度回归。
- 特征 mask 沿用 Bonito 下采样后有效长度，每条 read 至少一帧有效特征。

兼容说明：图中的 `[B,1]` 长度在公共输出处 squeeze 为既有的 `[B]`，不改变 loss
和预测代码。`PolyLocationOutput.location_logits` 保留 `[B,T,3]` 全零占位，供
既有输出结构消费；TCN 没有逐帧位置分类器，不提供定位或起止坐标。它用于当前
`length_only` 联合训练，不应用于有逐帧 location 标签的训练。

TCN 通道数固定为 128，TCN 块 dropout 固定为 0.1；顶层 `dropout=0` 表示投影处
不额外加 dropout。当前只支持 sum 长度模式，不接入可选的时间参照残差分支；不兼容
的组合会明确报错。

`tcn_length_sum_scale` 是固定的有限正数，新 TCN 模型默认 0.001，直接乘在长度
sum 项上，不除以单条 read 的有效帧数。重复有效帧仍按数量累积 sum 项；
0.001 是控制长序列累积尺度的初始设置，不保证训练始终不会塌缩。该值写入
`head_config`，不新增可训练参数，也不改变 Softplus、损失、状态分支或 Bonito。

Python 构造可传 `tcn_length_sum_scale=0.001`；联合训练 CLI 使用
`--tcn-length-sum-scale 0.001`，脚本环境变量为 `TCN_LENGTH_SUM_SCALE=0.001`。
旧 TCN checkpoint 没有该字段时按历史的 1.0 恢复；预测和评估自动读取。
续训会拒绝缩放不匹配；继续旧未缩放实验需显式设置 `TCN_LENGTH_SUM_SCALE=1`。
使用修复后的尺度应新建实验 ID，并重新初始化预测头，避免继续已塌缩的实验。

新 TCN 默认 `tcn_length_features=sum_mean_duration`。长度 MLP 接收缩放 sum 的
128 维、masked mean 的 128 维和 `log1p(N_valid)` 的 1 维；只有首层输入宽度从
128 改为 257。`N_valid` 来自现有 Bonito 特征 padding mask，每条 read 至少为 1，
不包含 padding。这是候选区间的有效 encoder 帧数，不是真实 Poly 时长或新标签。
状态分支、TCN 块、固定 sum 缩放、Softplus、数据处理和损失公式保持原样。

该模式写入 `head_config`；旧 checkpoint 缺少字段时自动按 `sum`（128 维）加载。
预测和评估沿用 checkpoint 自动重建，不迁移旧 MLP 权重。新结构需要新建实验，
不能直接 resume 旧结构。继续旧 TCN 实验时设置 `TCN_LENGTH_FEATURES=sum`，并使用
原缩放值。Python 参数是 `tcn_length_features`，CLI 参数为
`--tcn-length-features sum_mean_duration`；脚本环境变量为
`TCN_LENGTH_FEATURES=sum_mean_duration`。设置 `sum` 可做原结构对照。

## 使用

沿用已有且互相配套的两份 manifest，设置一个全新的实验 ID：

```bash
HEAD_TYPE=tcn TCN_LENGTH_SUM_SCALE=0.001 USE_TIMING_REFERENCE=0 TIMING_REFERENCE_MANIFEST="" \
TCN_LENGTH_FEATURES=sum_mean_duration \
EXPERIMENT_ID=poly-tcn-trial INITIALIZE_FROM="" RESUME=0 \
bash scripts/run_joint_training.sh \
  metadata/experiment/length/manifest.tsv \
  metadata/experiment/sequence/sequence_manifest.tsv \
  configs/local.yaml --dry-run
```

去掉 `--dry-run` 才会训练；真实训练前仍需按项目规则确认读取量、GPU 和输出空间。
只训练 TCN 预测头时设置 `UNFREEZE_LAST_LAYERS=0`，encoder 和 CRF 全部冻结并保持
eval 状态；TCN 仍处于 train 状态。新实验默认只计算一次 head 序列验证基线，
每轮复用并标记 `sequence_metrics_source=frozen_baseline`；训练阶段跳过 head 序列
读取和计算，记录 `sequence_examples=0`、`sequence_metrics_source=disabled_frozen`，
不生成未测量的 CRF loss。实际参数更新来自长度和状态 loss。
`ENCODER_LEARNING_RATE` 保留正数即可，冻结模式
没有 encoder 参数进入更新。更改冻结策略需要新建实验，不能恢复旧的部分解冻实验。
恢复同结构的 TCN 实验时设置 `HEAD_TYPE=tcn`、原实验 ID 和 `RESUME=1`。切换为
`HEAD_TYPE=legacy` 使用旧头。也可直接通过联合入口的 `--head-type tcn` 选择。

Python 模型构造示例：

```python
model = HolypolyModel(basecaller, head_type="tcn", length_head_mode="sum")
```

TCN checkpoint 在 `head_config` 和联合训练配置中记录 `head_type=tcn`。
评估/预测沿用已有 checkpoint 加载路径自动恢复结构。旧 checkpoint 没有该字段时
继续按 legacy 加载；默认 legacy 的 state_dict 名称、初始化顺序和配置保持原样。
跨结构 resume 明确拒绝，不自动丢弃预测头参数或优化器状态。本次没有增加跨结构
encoder 权重迁移；现有 `--initialize-from` 仍要求兼容的冻结模型 checkpoint。
已有模型、数据和实验结果均不被改写，新实验沿用现有的目录防覆盖检查。

## 训练速度和进度

联合训练的长度/状态批次使用 `forward_poly()`，跳过未使用的原 CRF 投影；默认
`model(...)` 仍返回完整 Bonito 输出。两条路径共享 encoder 和 Poly heads，
不改变 checkpoint 参数名、标签或损失公式。部分解冻模式继续逐 step 计算原生
CRF loss 并反向传播。

`FROZEN_SEQUENCE_MONITOR=auto` 为默认值：新建冻结实验使用上述基线策略；续训
沿用 checkpoint 的策略，没有策略记录的旧 checkpoint 使用原来的 `per-step`。
可显式设置 `baseline` 或 `per-step`；仅影响 encoder 完全冻结时的序列监测。
策略写入 checkpoint 的 `input_pipeline`。切换策略会改变序列 DataLoader 的随机数
消耗，不保证与原策略逐步数值一致；需要严格复现旧实验时保持 `per-step`。
冻结模式仍在最终选出的 checkpoint 上运行一次独立 test split 的序列评估。

初始序列基线、训练、长度/状态验证、序列验证和最终序列 test 都显示进度；各阶段
和 checkpoint 保存显示开始、结束及耗时。进度条显示已处理数量、速度和 ETA，
ETA 根据当前阶段已完成的工作估计。每轮日志包含 `stage_seconds`，保存耗时及完整
阶段汇总写入运行日志和完成后的 `run_summary.json`。复用基线时明确显示复用，
对应序列阶段耗时记录为 0。训练跳过的 CRF 项不加入该行 total；验证 total 仍含
已测量的 CRF 监测项，因此这两行 total 不应直接比较。

需要定位瓶颈时，在原训练命令前增加：

```bash
PROFILE_BATCHES=50
```

该变量须与原命令一起传入（例如 `PROFILE_BATCHES=50 HEAD_TYPE=tcn ... bash ...`）。
只在本次启动的第一个训练 epoch，跳过 5 个 warmup batch 后测量最多 50 个 batch；
若该 epoch 更短则报告实际数量。结果立即打印并写入训练日志的 `batch_profile`，
分别报告数据等待、传输、encoder、Poly heads、辅助 loss/backward、序列任务、
优化器和指标统计。测量时会同步 CUDA，包含计时开销；不是正常训练吞吐量。
默认 `PROFILE_BATCHES=0`，不执行逐段 CUDA 同步。直接 Python 入口的对应选项是
`--profile-batches` 和 `--frozen-sequence-monitor`。

指标在设备上累积，epoch 结束再回传，减少逐 batch 的 CPU/GPU 同步。本次不引入
特征缓存、混合精度或变长裁剪，也不增加数据读取并发；这些需根据实测结果选择。

## 验证

新增测试覆盖多种 batch/时间长度、单帧输入、两分支参数独立、卷积配置、带偏置的
padding 隔离、masked sum/mean、非有限 padding 哨兵、梯度、原 CRF 输出回归、
checkpoint/optimizer 恢复、跨结构恢复拒绝，以及 CLI 和脚本参数传递。
测试使用合成特征、微型 Bonito 模型和临时 manifest，不使用真实数据或已有权重。
