#!/usr/bin/env bash
# Reuse one v4 manifest; never rebuild data or overwrite previous experiments.
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/run_length_head_ablation.sh <manifest.tsv> [config.yaml]
Manifest paths are relative to the configured work_root, or absolute inside it.
Runs mean, duration_only and mean_duration sequentially on physical GPU 0.
Each model uses the same manifest, seed, batch size, epochs and loss weights.
Defaults: EPOCHS=5 BATCH_SIZE=32 SEED=20260923 LEARNING_RATE=0.001
SUBSET_PER_GROUP=200 selects a small 80/10/10 subset; default 0 uses all rows.
Optional environment: EXPERIMENT_ID, SUBSET_PER_GROUP, EPOCHS, BATCH_SIZE, SEED, LEARNING_RATE.
New timestamped output directories only; existing outputs cause an error.
HELP
  exit 0
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MANIFEST="$1"
CONFIG="${2:-configs/local.yaml}"
EXPERIMENT_ID="${EXPERIMENT_ID:-poly2p5per-v4-length-head-ablation-$(date +%Y%m%d-%H%M%S)}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-20260923}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SUBSET_PER_GROUP="${SUBSET_PER_GROUP:-0}"
if [[ ! "$SUBSET_PER_GROUP" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SUBSET_PER_GROUP must be 0 or a positive integer." >&2
  exit 2
fi
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
export CUDA_VISIBLE_DEVICES=0
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

# Validate paths before writing any log, checkpoint, plot or cache.
WORK_ROOT="$("$PARENT_PYTHON" - "$CONFIG" "$MANIFEST" "$EXPERIMENT_ID" <<'PY'
import csv
import sys
from pathlib import Path
from holypoly.core.config import load_config

c = load_config(Path(sys.argv[1]))
if c.paths.data_root is None or c.paths.work_root is None:
    raise ValueError("Configure data_root/work_root first")
data, work = (p.expanduser().resolve() for p in (c.paths.data_root, c.paths.work_root))
project = Path.cwd().resolve()
if not data.is_dir() or not work.is_dir():
    raise ValueError("Configured data_root/work_root must exist")
if work.is_relative_to(data) or work.is_relative_to(project):
    raise ValueError("work_root must be external and outside raw data")
manifest = (work / Path(sys.argv[2]).expanduser()).resolve()
if (not manifest.is_file() or not manifest.is_relative_to(work)
        or manifest.is_relative_to(data) or manifest.is_relative_to(project)):
    raise ValueError("manifest must exist inside external work_root")
with manifest.open() as stream:
    count = 0
    for row in csv.DictReader(stream, delimiter="\t" if manifest.suffix == ".tsv" else ","):
        count += 1
        if (row.get("manifest_version") != "construct-nominal-head-poly-v4"
                or row.get("supervision_mode") != "length_only"):
            raise ValueError("This comparison requires a v4 length_only manifest")
    if not count:
        raise ValueError("Manifest is empty")
for category in ("metadata", "logs", "models", "results/training_evaluation"):
    for suffix in ("", "-subset", "-mean", "-duration_only", "-mean_duration"):
        path = (work / category / (sys.argv[3] + suffix)).resolve()
        if (not path.is_relative_to(work) or path == work
                or path.is_relative_to(data) or path.is_relative_to(project)):
            raise ValueError(f"Unsafe output path: {path}")
        if path.exists():
            raise FileExistsError(f"Choose a new EXPERIMENT_ID: {path}")
print(work)
PY
)"
LOG_DIR="$WORK_ROOT/logs/$EXPERIMENT_ID"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
trap 'code=$?; echo "Comparison failed (exit $code) near line $LINENO." >&2; exit "$code"' ERR
echo "Experiment: $EXPERIMENT_ID | GPU 0 | existing manifest: $MANIFEST"
echo "Three sequential runs, $EPOCHS epochs each. No BAM/POD5 metadata scan."
echo "The duration-only run still computes the frozen encoder to retain the output contract."
echo "Expect about 5.5 GiB of checkpoints at five epochs across all three modes."
echo "POD5 reads are limited to this manifest; no raw signal copies."

if [[ "$SUBSET_PER_GROUP" != 0 ]]; then
  echo "Selecting $SUBSET_PER_GROUP rows per length/type, retaining the source splits."
  "$PARENT_PYTHON" -m holypoly.datasets.subset_manifest \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --output-dir "metadata/$EXPERIMENT_ID-subset" \
    --per-group "$SUBSET_PER_GROUP" --seed "$SEED"
  MANIFEST="metadata/$EXPERIMENT_ID-subset/manifest.tsv"
fi

for mode in mean duration_only mean_duration; do
  RUN_ID="$EXPERIMENT_ID-$mode"
  echo "Training $mode"
  "$PROJECT_ROOT/.venv/bin/holypoly" train \
    --config "$CONFIG" --manifest "$MANIFEST" --length-head-mode "$mode" \
    --checkpoint "models/$RUN_ID/heads.pt" --loss-log "logs/$RUN_ID/train.jsonl" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" --random-seed "$SEED" \
    --data-order-seed "$SEED" \
    --chunk-size-samples 12000 --hidden-features 128 --dropout 0.0 \
    --weight-decay 0.0 --gradient-clip-norm 2.0 --num-workers 0 --device cuda:0
  printf -v CHECKPOINT 'models/%s/heads.epoch_%04d.pt' "$RUN_ID" "$EPOCHS"
  "$BONITO_PYTHON" -m holypoly.training.evaluate \
    --config "$CONFIG" --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
    --output-dir "results/training_evaluation/$RUN_ID/test-final" \
    --split test --batch-size "$BATCH_SIZE" --max-reads 0
  "$PARENT_PYTHON" scripts/plot_training.py \
    --config "$CONFIG" --log "logs/$RUN_ID/train.jsonl" \
    --output "results/training_evaluation/$RUN_ID/training-plots"
  "$PARENT_PYTHON" scripts/plot_training.py \
    --config "$CONFIG" --predictions "results/training_evaluation/$RUN_ID/test-final/predictions.tsv" \
    --supervision-mode length_only --seed "$SEED" \
    --output "results/training_evaluation/$RUN_ID/test-final/plots"
done

"$PARENT_PYTHON" - "$WORK_ROOT" "$EXPERIMENT_ID" <<'PY'
import csv
import json
import sys
from pathlib import Path

work, experiment = Path(sys.argv[1]), sys.argv[2]
overall, by_length = [], []
for mode in ("mean", "duration_only", "mean_duration"):
    result = work / "results" / "training_evaluation" / f"{experiment}-{mode}" / "test-final"
    summary = json.loads((result / "summary.json").read_text())
    with (result / "metrics_by_group.tsv").open() as stream:
        groups = list(csv.DictReader(stream, delimiter="\t"))
    positives = [r for r in groups if float(r["nominal_length_nt"]) > 0]
    count = sum(int(r["evaluable_reads"]) for r in positives)
    positive_mae = (sum(float(r["mae_nt"]) * int(r["evaluable_reads"])
                        for r in positives) / count if count else None)
    overall.append({"mode": mode, **summary["metrics"], "positive_mae_nt": positive_mae})
    by_length.extend({"mode": mode, **r} for r in groups)
output = work / "results" / "training_evaluation" / experiment
output.mkdir(parents=True)
for filename, rows in (("comparison.tsv", overall), ("comparison_by_length.tsv", by_length)):
    with (output / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
print("Comparison (construct weak labels, not independent per-read accuracy):")
for row in overall:
    print(row["mode"], "MAE:", row["mae_nt"], "positive MAE:", row["positive_mae_nt"],
          "within 10nt:", row["within_10nt_fraction"])
print("Tables:", output)
PY
echo "Finished. Previous experiments and the input manifest were not changed."
