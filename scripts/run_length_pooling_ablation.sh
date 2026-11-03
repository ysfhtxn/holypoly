#!/usr/bin/env bash
# Compare masked mean/sum pooling with identical layers and a shared objective.
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/run_length_pooling_ablation.sh <manifest.tsv> [config.yaml] [--dry-run]
Manifest paths are relative to work_root, or absolute inside it.
Train mean and sum sequentially from fresh identical initialization on GPU_ID (0).
Only pooling differs: same layers, parameter count, manifest, loss and batch order.
Neither head receives the explicit raw-duration input of mean_duration.
Select each best checkpoint by validation nt MAE, then test, plot and compare.
Defaults: EPOCHS=20 BATCH_SIZE=8 SEED=20260923 LEARNING_RATE=0.001
          LENGTH_LOSS=huber HUBER_DELTA_NT=10 SUBSET_PER_GROUP=0 GPU_ID=0
Optional environment: the defaults above and EXPERIMENT_ID.
SUBSET_PER_GROUP=200 optionally selects one shared, split-preserving small manifest.
--dry-run validates inputs, settings, output paths and resource estimates; no writes.
All outputs are new; previous results and the input manifest are never overwritten.
HELP
  exit 0
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MANIFEST="$1"
shift
CONFIG="configs/local.yaml"
if [[ $# -gt 0 && "$1" != "--dry-run" ]]; then
  CONFIG="$1"
  shift
fi
DRY_RUN=false
if [[ $# -gt 0 && "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unexpected argument: $1. Use --help for usage." >&2
  exit 2
fi

EXPERIMENT_ID="${EXPERIMENT_ID:-v4-length-pooling-ablation-$(date +%Y%m%d-%H%M%S)}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-20260923}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
LENGTH_LOSS="${LENGTH_LOSS:-huber}"
GPU_ID="${GPU_ID:-0}"
HUBER_DELTA_NT="${HUBER_DELTA_NT:-10}"
SUBSET_PER_GROUP="${SUBSET_PER_GROUP:-0}"
if [[ ! "$EXPERIMENT_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "EXPERIMENT_ID must be a single directory name." >&2
  exit 2
fi
for number in "$EPOCHS" "$BATCH_SIZE"; do
  if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPOCHS and BATCH_SIZE must be positive integers." >&2
    exit 2
  fi
done
for number in "$SEED" "$SUBSET_PER_GROUP" "$GPU_ID"; do
  if [[ ! "$number" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "SEED, SUBSET_PER_GROUP and GPU_ID must be non-negative integers." >&2
    exit 2
  fi
done
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
PARENT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
BONITO_PYTHON="$PROJECT_ROOT/src/bonito/.venv/bin/python"
for executable in "$PARENT_PYTHON" "$BONITO_PYTHON" "$PROJECT_ROOT/.venv/bin/holypoly"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing installed environment executable: $executable" >&2
    exit 2
  fi
done

# Pure metadata preflight, including optional subset feasibility, before any writes.
WORK_ROOT="$("$PARENT_PYTHON" - "$CONFIG" "$MANIFEST" "$EXPERIMENT_ID" \
  "$EPOCHS" "$BATCH_SIZE" "$SEED" "$LEARNING_RATE" "$LENGTH_LOSS" \
  "$HUBER_DELTA_NT" "$SUBSET_PER_GROUP" "$GPU_ID" <<'PY'
import csv
import math
import sys
from collections import Counter
from pathlib import Path

from holypoly.core.config import load_config
from holypoly.datasets.subset_manifest import select_subset
from holypoly.training.loss_config import normalize_loss_config

config_path, manifest_value, experiment = sys.argv[1:4]
epochs, batch_size, seed = map(int, sys.argv[4:7])
learning_rate = float(sys.argv[7])
loss = sys.argv[8]
delta = float(sys.argv[9])
per_group = int(sys.argv[10])
gpu_id = int(sys.argv[11])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise ValueError("LEARNING_RATE must be a finite positive number")
c = load_config(Path(config_path))
loss_config = normalize_loss_config({**vars(c.auxiliary_loss),
                                     "length_loss": loss, "huber_delta_nt": delta})
if c.paths.data_root is None or c.paths.work_root is None:
    raise ValueError("Configure data_root/work_root first")
data, work = (p.expanduser().resolve() for p in (c.paths.data_root, c.paths.work_root))
project = Path.cwd().resolve()
if not data.is_dir() or not work.is_dir():
    raise ValueError("Configured data_root/work_root must exist")
if work.is_relative_to(data) or work.is_relative_to(project):
    raise ValueError("work_root must be external and outside raw data")

def safe(path):
    resolved = path.resolve()
    if (not resolved.is_relative_to(work) or resolved == work
            or resolved.is_relative_to(data) or resolved.is_relative_to(project)):
        raise ValueError(f"Unsafe artifact path: {path}")
    return resolved

manifest = safe(work / Path(manifest_value).expanduser())
if not manifest.is_file():
    raise FileNotFoundError(manifest)
with manifest.open(newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t" if manifest.suffix == ".tsv" else ",")
    required = {"read_id", "split", "nominal_length_nt", "poly_type",
                "manifest_version", "supervision_mode", "raw_signal_length"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("Manifest is missing required comparison fields")
    rows = list(reader)
if not rows:
    raise ValueError("Manifest is empty")
seen = set()
for row in rows:
    if None in row or any(v is None for v in row.values()):
        raise ValueError("Manifest contains malformed rows")
    if (row["manifest_version"] != "construct-nominal-head-poly-v4"
            or row["supervision_mode"] != "length_only"):
        raise ValueError("This comparison requires a v4 length_only manifest")
    if not row["read_id"] or row["read_id"] in seen:
        raise ValueError("Manifest contains empty or duplicate read IDs")
    seen.add(row["read_id"])
    nominal = float(row["nominal_length_nt"])
    if not math.isfinite(nominal) or nominal < 0 or int(row["raw_signal_length"]) <= 0:
        raise ValueError("Invalid nominal length or raw signal length")
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError("Manifest contains unsupported splits")
if per_group:
    rows = select_subset(rows, per_group=per_group, seed=seed)
splits = Counter(row["split"] for row in rows)
if any(not splits[split] for split in ("train", "validation", "test")):
    raise ValueError("Comparison needs nonempty train, validation and test splits")
for category in ("metadata", "logs", "models", "results/training_evaluation", "cache"):
    for suffix in ("", "-subset", "-mean", "-sum"):
        path = work / category / (experiment + suffix)
        safe(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Choose a new EXPERIMENT_ID: {path}")
epoch_samples = sum(int(r["raw_signal_length"]) for r in rows if r["split"] != "test")
test_samples = sum(int(r["raw_signal_length"]) for r in rows if r["split"] == "test")
signal_gib = (epoch_samples * epochs * 2 + test_samples * 2) * 4 / 1024**3
print(f"Experiment: {experiment} | pooling=mean,sum | loss={loss} | GPU {gpu_id} | {epochs} epochs/head", file=sys.stderr)
print(f"Shared rows: {len(rows)} | splits={dict(splits)} | batch={batch_size} | seed={seed}", file=sys.stderr)
print(f"lr={learning_rate} | Huber delta={delta} nt | weights="
      f"{[loss_config[k] for k in ('lambda_location', 'lambda_length', 'lambda_status')]}", file=sys.stderr)
print(f"Estimated logical float32 signal reads: {signal_gib:.2f} GiB across all runs; "
      f"checkpoints roughly {2 * (epochs + 1) * 0.30:.1f} GiB (model dependent).", file=sys.stderr)
print("Previous 1,400-read subset: estimate 2-3 hours for two 20-epoch runs on RTX 3080; "
      "runtime scales with data/GPU. No BAM scan or signal copies.", file=sys.stderr)
print(work)
PY
)"
if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run complete: no signal/model weights loaded and no outputs written."
  exit 0
fi

LOG_DIR="$WORK_ROOT/logs/$EXPERIMENT_ID"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
trap 'code=$?; echo "Pooling comparison failed (exit $code) near line $LINENO." >&2; exit "$code"' ERR
echo "Experiment: $EXPERIMENT_ID | pooling=mean,sum | loss=$LENGTH_LOSS | GPU $GPU_ID | $EPOCHS epochs/head"
echo "Each head starts fresh with the same layers/seed; selection uses validation nt MAE."
echo "Batch=$BATCH_SIZE | seed=$SEED | lr=$LEARNING_RATE | Huber delta=$HUBER_DELTA_NT nt"
export XDG_CACHE_HOME="$WORK_ROOT/cache/$EXPERIMENT_ID"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"

if [[ "$SUBSET_PER_GROUP" != 0 ]]; then
  "$PARENT_PYTHON" -m holypoly.datasets.subset_manifest \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --output-dir "metadata/$EXPERIMENT_ID-subset" \
    --per-group "$SUBSET_PER_GROUP" --seed "$SEED"
  MANIFEST="metadata/$EXPERIMENT_ID-subset/manifest.tsv"
fi

for mode in mean sum; do
  RUN_ID="$EXPERIMENT_ID-$mode"
  echo "Training $mode pooling with $LENGTH_LOSS"
  "$PROJECT_ROOT/.venv/bin/holypoly" train \
    --config "$CONFIG" --manifest "$MANIFEST" --length-head-mode "$mode" \
    --length-loss "$LENGTH_LOSS" --huber-delta-nt "$HUBER_DELTA_NT" \
    --checkpoint "models/$RUN_ID/heads.pt" --loss-log "logs/$RUN_ID/train.jsonl" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
    --random-seed "$SEED" --data-order-seed "$SEED" \
    --chunk-size-samples 12000 --hidden-features 128 --dropout 0.0 \
    --weight-decay 0.0 --gradient-clip-norm 2.0 --num-workers 0 --device cuda:0

  BEST_CHECKPOINT="$("$PARENT_PYTHON" - "$WORK_ROOT" "$RUN_ID" "$EPOCHS" <<'PY'
import json
import math
import sys
from pathlib import Path

work, run, epochs = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in (work / "logs" / run / "train.jsonl").read_text().splitlines()]
if [(r["epoch"], r["split"]) for r in rows] != [
    (epoch, split) for epoch in range(1, epochs + 1) for split in ("train", "validation")
]:
    raise ValueError("Training log is incomplete or contains duplicate epochs")
validation = [r for r in rows if r["split"] == "validation" and math.isfinite(r["length_mae_nt"])]
if not validation:
    raise ValueError("No finite validation length MAE available")
best = min(validation, key=lambda r: (r["length_mae_nt"], r["epoch"]))
checkpoint = work / "models" / run / f"heads.epoch_{best['epoch']:04d}.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
output = work / "results" / "training_evaluation" / run
output.mkdir(parents=True, exist_ok=False)
(output / "best_checkpoint.json").write_text(json.dumps({
    "selection_rule": "minimum validation length MAE, earliest epoch on ties",
    "epoch": best["epoch"], "validation_mae_nt": best["length_mae_nt"],
    "checkpoint": str(checkpoint), "loss_config": best["loss_config"],
}, indent=2) + "\n")
print(checkpoint)
PY
)"
  "$BONITO_PYTHON" -m holypoly.training.evaluate \
    --config "$CONFIG" --manifest "$MANIFEST" --checkpoint "$BEST_CHECKPOINT" \
    --output-dir "results/training_evaluation/$RUN_ID/test-best-validation" \
    --split test --batch-size "$BATCH_SIZE" --max-reads 0
  "$PARENT_PYTHON" scripts/plot_training.py \
    --config "$CONFIG" --log "logs/$RUN_ID/train.jsonl" \
    --output "results/training_evaluation/$RUN_ID/training-plots"
  "$PARENT_PYTHON" scripts/plot_training.py \
    --config "$CONFIG" --predictions "results/training_evaluation/$RUN_ID/test-best-validation/predictions.tsv" \
    --supervision-mode length_only --seed "$SEED" \
    --output "results/training_evaluation/$RUN_ID/test-best-validation/plots"
done

"$PARENT_PYTHON" - "$WORK_ROOT" "$EXPERIMENT_ID" "$LENGTH_LOSS" "$HUBER_DELTA_NT" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys

work, experiment = Path(sys.argv[1]), sys.argv[2]
expected_loss, expected_delta = sys.argv[3], float(sys.argv[4])
overall, by_length = [], []
reference_targets = None
reference_context = None
for mode in ("mean", "sum"):
    directory = work / "results" / "training_evaluation" / f"{experiment}-{mode}"
    result = directory / "test-best-validation"
    summary = json.loads((result / "summary.json").read_text())
    best = json.loads((directory / "best_checkpoint.json").read_text())
    if summary["partial_evaluation"] or summary["selected_reads"] != summary["split_total_reads"]:
        raise ValueError("Comparison requires evaluation of the complete shared test split")
    if (summary["loss_config"]["length_loss"] != expected_loss
            or summary["head_config"]["length_head_mode"] != mode
            or summary["epoch"] != best["epoch"]
            or summary["checkpoint"] != best["checkpoint"]
            or summary["loss_config"] != best["loss_config"]):
        raise ValueError("Evaluated checkpoint does not match selected head/loss/epoch")
    if expected_loss == "huber" and summary["loss_config"]["huber_delta_nt"] != expected_delta:
        raise ValueError("Evaluated Huber delta differs from requested setting")
    context = {key: summary[key] for key in (
        "manifest", "model_dir", "random_seed", "data_order_seed", "batch_size",
        "model_config", "loss_config", "git_commit", "bonito_commit",
    )}
    context["hidden_features"] = summary["head_config"]["hidden_features"]
    context["dropout"] = summary["head_config"]["dropout"]
    if reference_context is not None and context != reference_context:
        raise ValueError("Training/evaluation settings differ between pooling modes")
    reference_context = context
    with (result / "predictions.tsv").open() as stream:
        predictions = list(csv.DictReader(stream, delimiter="\t"))
    targets = {r["read_id"]: float(r["training_length_nt"]) for r in predictions}
    if len(targets) != len(predictions) or len(predictions) != summary["selected_reads"]:
        raise ValueError("Missing or duplicate test predictions")
    if reference_targets is not None and targets != reference_targets:
        raise ValueError("Test read IDs/targets differ between pooling modes")
    if any(not math.isfinite(float(r["regression_length_nt"])) for r in predictions):
        raise ValueError("Nonfinite regression predictions; inspect failures before comparing")
    reference_targets = targets
    with (result / "metrics_by_group.tsv").open() as stream:
        groups = list(csv.DictReader(stream, delimiter="\t"))
    positives = [r for r in groups if float(r["nominal_length_nt"]) > 0]
    count = sum(int(r["evaluable_reads"]) for r in positives)
    positive_mae = (sum(float(r["mae_nt"]) * int(r["evaluable_reads"])
                        for r in positives) / count if count else None)
    metadata = {"length_loss": expected_loss, "length_head_mode": mode,
                "best_epoch": best["epoch"], "validation_mae_nt": best["validation_mae_nt"],
                "huber_delta_nt": summary["loss_config"]["huber_delta_nt"]}
    overall.append({**metadata, **summary["metrics"], "positive_mae_nt": positive_mae})
    by_length.extend({**metadata, **r} for r in groups)
output = work / "results" / "training_evaluation" / experiment
output.mkdir(parents=True, exist_ok=False)
for filename, rows in (("comparison.tsv", overall), ("comparison_by_length.tsv", by_length)):
    with (output / filename).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
print("Comparison: construct weak-label agreement, not independent per-read accuracy.")
for row in overall:
    print(row["length_head_mode"], "best epoch:", row["best_epoch"], "test MAE:", row["mae_nt"],
          "positive MAE:", row["positive_mae_nt"], "within 10nt:", row["within_10nt_fraction"])
print("Tables:", output)
PY
echo "Finished: $EXPERIMENT_ID"
echo "Per-head results: $WORK_ROOT/results/training_evaluation/$EXPERIMENT_ID-<mean|sum>/test-best-validation/"
