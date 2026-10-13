# 构建级弱监督训练集

脚本：`src/holypoly/datasets/prepare.py`。在 Bonito 独立环境运行，无新增依赖，
不加载模型、不使用 GPU、不修改原始文件。

## 运行

使用已有的 `configs/local.yaml`，或者用 `--data-root`、`--work-root` /
`HOLYPOLY_DATA_ROOT`、`HOLYPOLY_WORK_ROOT` 指定外部目录。CLI 优先于环境变量，
环境变量优先于配置。输入相对路径相对于 data root，输出相对于 work root。

```bash
PYTHONPATH=src src/bonito/.venv/bin/python -m holypoly.datasets.prepare \
  --config configs/local.yaml \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per \
  --reference fixed_len_ref.fa \
  --output-dir metadata/poly2p5per-head-free-v3 \
  --max-records 10000 \
  --per-group 3
```

默认每轮读取至多 10000 条 BAM 记录。两个 BAM 顺序不同，受限扫描可能没有足够
的共同 UUID；此时明确报错，不能据此判断完整数据缺少样本。
flank 支持不足时仍可生成只监督长度的正样本，不再因 flank 推定失败中止。
输出目录必须是尚不存在的 work root 子目录，并位于 data root 之外。

默认在 stderr 显示三个阶段的进度条：Bonito 构建关联、Dorado flank 候选推定、
Dorado 边界与 POD5 元数据检查，包含耗时、速率和预计剩余时间。
受限扫描的总数是 `--max-records` 上限；文件提前结束或收集够 200 个 flank
候选时，该阶段可能未到 100% 就结束。完整扫描按 BAM 的压缩字节位置估算
百分比，并显示已扫描记录数；BGZF 分块和各 read 处理成本不同，进度与剩余
时间都是估算值。不额外计数扫描，也不要求 BAM 索引。
使用 Bonito 环境已有的 `tqdm`，无新增依赖；`--no-progress` 可关闭进度条，
stdout 仍仅输出最终 JSON，便于重定向。

确认小规模审计后，将 `--max-records 0` 用于完整扫描，并更换输出目录。
完整扫描读取 Bonito BAM 一遍、Dorado BAM 至多两遍；本批数据的 BAM 读取上限约
6.90 GiB（按压缩文件大小计），另需读取 POD5 索引和 read 元数据。
CPU 内存取决于 UUID 和合格候选数，百万 read 预计需要数 GB；HDD 可能需要数十分钟
至数小时，尚未实测。manifest 通常数 MB，完整 UUID 审计表可能数百 MB。
脚本在读取前打印资源提示。本次实现没有执行完整扫描。

## 输入定义与算法

1. FASTA 整段序列是 Poly 前的 head；参考名称包含 `<number>A` 标称长度。
2. 从 Bonito BAM 的 `RNAME` 分配构建，去掉 `UUID:chunk_index:chunk_count`
   后缀；同一 UUID 的构建冲突进入审计表。
3. Dorado 必须已有对这些 head 的比对，且构建与 Bonito 一致。检查最后 24 个
   head 碱基的覆盖和匹配率，并要求 head 最后一个碱基确实有对应 query 碱基。
4. 从最多 200 条有 UUID 关联的 0A read 提取 head 后的 24-base 候选。
   按允许的替换误差聚合支持，选出有至少 5 条支持的唯一最佳观察序列，作为
   **provisional flank**。必须在至少两个正标称长度中找到总计至少 5 条完整候选，
   通过跨长度检查后，才将锚点用于定位监督和 0A 负样本。支持不足时改用
   `length_only` 正样本。这些检查不能证明 flank 的生物学身份或真实边界。
5. 在 head 后搜索 flank。正样本间隔要求至少 80% 为 A，且最多连续 2 个非 A；
   0A 要求 head 与 flank 紧邻。所有判定都在参考方向进行，因此反向 BAM 也检查 A。
   不把正构建的漏检当作 zero_nt，也不把 0A 中观察到的正长度强制改为 0。
6. 用 `mv` 中实际 stride 和 `ts` 映射至原始未裁剪 sample；反向 read 先将
   query 区间恢复为测序方向，并标记 polyT。多个 move 共用一个 block 时，边界
   分辨率有限；拒绝退化区间、split read、duplex 和信号长度不一致的记录。
7. 检查 `fn` 指定的 POD5 文件及 UUID 的 `num_samples`。所有输入仅保留 head
   后的可见信号，包含 Poly、可能的 flank 和其他下游背景。不得向 head 一侧扩展
   来填满输入。双锚点通过检查者使用 `supervision_mode=full`；正构建没有可信
   flank 时使用 `supervision_mode=length_only`，不把整个下游区间标为 Poly。
   forward read 使用末端信号，reverse read 使用测序起始端信号，保持 raw 坐标。
   可见信号范围来自 move blocks，包含 `ts`，末端界限为最后一个 move block
   之后的 sample（不超过 `ns`），不包含 move table 覆盖之外的未知信号。
   真实区间不足 12000 samples 时，归一化后在右侧补零至 12000；raw padding mask
   和 encoder padding mask 排除补零位置的定位 loss 和长度池化。共享 encoder 的
   两个任务使用同一个无 head 输入。候选区间超过输入上限者进入审计表，不截断。
   head 最后一个碱基与首个下游碱基共用 move block 时，记录
   `head_downstream_share_move_block` 并拒绝，避免混入 head 或切掉可能的 Poly。
   `raw_signal_length` 使用 POD5 实际 sample 数；Dorado 的 `ns` 可能舍去末端
   不足一个 move block 的信号，只接受 `0 <= raw_signal_length - ns < move_stride`，
   并保存 `dorado_signal_samples`。超过该范围的差异仍拒绝，原始坐标不改变。
8. 按存在的“正标称长度 × polyA/polyT”分层，下采样至最少层的数量，同时受
   `--per-group` 限制；缺失层列入 summary，不补造样本。zero_nt 数量至多为一层
   的数量，不单独按方向平衡。每个保留层至少 3 条 read；不足 3 条的层不进入
   小型训练集，记录为 `not_selected_insufficient_group`，并列入缺失层报告。
9. 各层按固定种子分 train/validation/test，约 80/10/10，小层保证各 split
   至少一条。每个 UUID 只有一个 chunk，不跨 split。Poly 靠近输入一端，存在位置
   偏差；划分仍属于同一批次的开发数据，不能当作跨批次独立验证。

`--anchor-bases`、`--anchor-error-rate`、`--min-flank-reads`、
`--min-poly-fraction`、`--chunk-size-samples` 和 `--seed` 可调整。
锚点匹配只容忍替换，不做新的序列重比对；插入/缺失可能造成拒绝。

## 输出及训练

- `manifest.tsv`：使用 `training/data.py` 的实际 schema，写完调用其 loader 校验。
  `training_length_nt` 是训练读取字段，`true_length_nt` 同值仅为兼容审计；
  两者均是标称弱监督，`label_source=construct_nominal_weak_supervision`。
  `supervision_mode` 为可选字段，旧 manifest 默认 `full`。
  `length_only` 必须将 `poly_start_sample`、`poly_end_sample`、`poly_is_complete`
  及 `observed_sequence_length_nt` 留空，保持未知，而不把 flank 算作观察 Poly 长度。
  `candidate_start_sample/candidate_end_sample` 记录输入候选区间，仅用于审计。
  新版本为 `construct-nominal-head-excluded-v3`，包含 `head_excluded=true`、
  `head_boundary_sample` 和 `model_input_samples`。chunk 坐标只描述真实信号，
  `model_input_samples` 描述补零后的 tensor 大小。loader 校验 forward 的 chunk
  起点不早于 head 边界，reverse 的 chunk 终点不晚于该边界。
- `audit.tsv`：每个 Bonito primary UUID 的构建、标称长度及保留/拒绝原因。
  flank 推定或跨长度检查失败时，优先生成 `length_only` 正样本，并保留审计。
  边界失败记录不作为 failure 训练标签，避免将标签制作失败等同于检测失败。
- `summary.json`：flank 候选序列、跨长度支持、分层数量、缺失层、参数、输入路径、
  随机种子、时间、候选/选中分层数量、监督模式数量、flank QC 原因和 Git/Bonito commit。
  只在完整准备成功后生成。

信号不复制为 `.npy`：现有 `PolyTrainingDataset` 按 manifest 从原 POD5 提取
chunk，并按训练模型配置归一化、补零。这样直接对齐当前训练接口，保留下游背景，
Bonito 不作修改。完整监督的 `observed_sequence_length_nt` 是两侧锚点间的
query 碱基数，含允许的非 A/T 中断，并非仅目标碱基的计数。

训练数据消费代码将 `length_only` 的所有 location targets 和 status target 设为
`-100`。这类样本只进入长度回归，不进入 location/status loss；全部为
`length_only` 的 batch 也能产生有限 loss 和 backward。混合 batch 的定位和状态
统计只计算有对应标签的样本。length loss 权重和冻结 encoder 策略保持不变。
若数据全为 `length_only`，定位和状态 head 尚未训练；此时只使用原始回归输出
`length_nt` 检查学习趋势，不能将按状态解码的长度作为可靠检测结果。

```bash
uv run holypoly train --config configs/local.yaml \
  --manifest metadata/poly2p5per-head-free-v3/manifest.tsv --dry-run
```

模型目录须另外由配置、环境变量或 `--model-dir` 提供。
chunk size 若更改，训练时也必须传入同样的 `--chunk-size-samples`。

构建标称长度不代表逐 read 真实长度，move/锚点边界也未经独立验证。不同长度
使用不同 head；v3 按接受的 Dorado 比对与 move 边界排除 head，但不能消除这些
边界本身的误差。正式性能评价需要可靠逐 read 标签和独立样本。
length_only 样本可能包含截断 Poly，其标称目标存在额外噪声；`poly_is_complete`
保持未知。下一步先检查两种监督模式数量及各层保留率，再开展小规模单 GPU 训练。

旧 v2 manifest、模型与运行结果不自动转换或覆盖。需要重新制作 v3 manifest，
使用新 checkpoint 和日志路径从头训练，不传入旧 checkpoint 的 `--resume`。
小规模审计后可提高 `--per-group` 并以 `--max-records 0` 完整扫描；为每次准备使用
新输出目录。先检查 summary 的 `head_excluded_reads`、`padded_reads`、
`supervision_counts`、`missing_positive_strata` 与 audit 拒绝原因。

开发时只读抽查两个 BAM 各前 1000 条记录：72 条关联 read 通过双侧锚点与 move
检查，其中 65 条为 0A、7 条为正长度候选；尚未检查 POD5 或 chunk 长度。
该样本受文件顺序影响，不能作为全数据保留率或模型性能。缺少下游锚点是主要限制。

## 真实小型流程测试

准备数据后，通过 `holypoly train --dry-run` 检查配置，显式选择一个 GPU 训练
一轮，再用 `--epochs 2 --resume` 检查 checkpoint 恢复。模型、manifest 和
checkpoint 路径均由配置、CLI 或环境变量提供，不将真实文件放入仓库。

`tests/isolated/training/test_prepared_pipeline.py` 是独立的 opt-in integration test。
显式设置 `HOLYPOLY_DATA_ROOT`、`HOLYPOLY_WORK_ROOT`、`HOLYPOLY_M3_MANIFEST`、
`HOLYPOLY_BONITO_MODEL` 和 `HOLYPOLY_M3_CHECKPOINT`，再运行：

```bash
CUDA_VISIBLE_DEVICES=<physical-gpu-id> PYTHONPATH=src \
  src/bonito/.venv/bin/python -m pytest -q \
  tests/isolated/training/test_prepared_pipeline.py -m integration
```

该测试读取各 split 的真实 POD5，检查混合监督 forward/backward、冻结 encoder
与 CRF 的权重和梯度、原 basecall 输出接口、优化器恢复及重复推理一致性，并在
`<work_root>/results/<checkpoint-directory-name>/` 写入 `pipeline_checks.json` 和
`test_predictions.tsv`。test split 只导出预测，不进行 optimizer 更新。
缺少外部路径时明确跳过；必须只暴露一个 GPU。Torch、Triton、CUDA 的运行缓存
也应通过对应环境变量指向 work root 的 cache 子目录。
已有 `test_real_2p5per_pipeline_smoke` 专门测试旧固定 manifest；测试新的 v3 数据时
应调用上述文件，避免将新 manifest 套入旧 fixture 的断言。

此前 v2 小型运行使用前 10000 条 Bonito/Dorado 记录，每个可用层抽 3 条，共 33 条；
train/validation/test 各 11 条，完整监督 10 条、length_only 23 条。
覆盖 0/20/40/60/80/100/150 nt，40 nt polyT 因候选不足、60 nt polyT 因没有候选
未纳入。单 GPU 训练、恢复训练及真实数据 integration 均通过。该结果仅为
链路 smoke test，不代表正式准确率。
v3 目前只完成 synthetic 验证；上述真实运行结果不证明 v3 在真实数据上的行为。
