#!/usr/bin/env bash
# v4 weak-supervision baseline; uses the existing mean-pooled length head.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/run_poly2p5per_v4.sh [config.yaml]
Only poly2p5per, polyA, GPU 0. Prepare -> train -> test -> plot.
Defaults: PER_GROUP=1000 EPOCHS=5 BATCH_SIZE=32 SEED=20260923
Optional environment: RUN_ID, PER_GROUP, EPOCHS, BATCH_SIZE, SEED, LEARNING_RATE.
Each invocation uses new timestamped output directories; no automatic resume.
Dependencies must already be installed in .venv and src/bonito/.venv.
HELP
  exit 0
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
CONFIG="${1:-configs/local.yaml}"
PER_GROUP="${PER_GROUP:-6000}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-20260923}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
RUN_ID="${RUN_ID:-poly2p5per-head-poly-v4-polyA-${PER_GROUP}pergroup-mean-$(date +%Y%m%d-%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "RUN_ID must be a single directory name." >&2
  exit 2
fi
for number in "$PER_GROUP" "$EPOCHS" "$BATCH_SIZE"; do
  if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then
    echo "PER_GROUP, EPOCHS and BATCH_SIZE must be positive integers." >&2
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
    echo "Required environment executable is missing: $executable" >&2
    exit 2
  fi
done

# Validate every output destination before creating the pipeline log directory.
WORK_ROOT="$("$PARENT_PYTHON" - "$CONFIG" "$RUN_ID" <<'PY'
import sys
from pathlib import Path
from holypoly.core.config import load_config

config = load_config(Path(sys.argv[1]))
if config.paths.data_root is None or config.paths.work_root is None:
    raise ValueError("Configure data_root and work_root first")
data = config.paths.data_root.expanduser().resolve()
work = config.paths.work_root.expanduser().resolve()
project = Path.cwd().resolve()
if not data.is_dir() or not work.is_dir():
    raise ValueError("Configured data_root and work_root must exist")
if work.is_relative_to(data) or work.is_relative_to(project):
    raise ValueError("work_root must be outside raw data and the repository")
for category in ("metadata", "models", "logs", "results/training_evaluation"):
    path = (work / category / sys.argv[2]).resolve()
    if (not path.is_relative_to(work) or path == work
            or path.is_relative_to(data) or path.is_relative_to(project)):
        raise ValueError(f"Unsafe output destination: {path}")
    if path.exists():
        raise FileExistsError(f"Choose a new RUN_ID; output exists: {path}")
print(work)
PY
)"

# Fail before the BAM scan if the plotting environment or GPU is unavailable.
"$PARENT_PYTHON" -c 'import matplotlib'
"$BONITO_PYTHON" -c \
  'import torch; from holypoly.training.engine import require_single_cuda_device; require_single_cuda_device(torch.device("cuda:0"))'

MANIFEST="metadata/$RUN_ID/manifest.tsv"
CHECKPOINT="models/$RUN_ID/heads.pt"
LOSS_LOG="logs/$RUN_ID/train.jsonl"
TEST_OUTPUT="results/training_evaluation/$RUN_ID/test-final"
LOG_DIR="$WORK_ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
trap 'code=$?; echo "Pipeline failed (exit $code) near line $LINENO; see pipeline.log." >&2; exit "$code"' ERR

echo "Run: $RUN_ID | CUDA_VISIBLE_DEVICES=0, model device=cuda:0"
echo "Prepare: full BAM UUID scan, POD5 metadata; minutes to hours on HDD."
echo "Train/test: selected POD5 signals only, one GPU; runtime depends on hardware."
echo "Audit files may reach hundreds of MB; no signal copies."
echo "Current mean-pooling head; construct nominal weak labels; length-only supervision."

echo "[1/5] Prepare dataset"
"$BONITO_PYTHON" -m holypoly.datasets.prepare \
  --config "$CONFIG" \
  --structure head-poly --signal-end-policy raw-edge \
  --bonito-bam basecalls/poly2p5per/basecalls.bam \
  --dorado-bam dorado_calls/poly2p5per/dorado_basecalls.bam \
  --pod5-dir pod5/poly2p5per --reference fixed_len_ref.fa \
  --output-dir "metadata/$RUN_ID" \
  --max-records 0 --per-group "$PER_GROUP" --poly-types polyA \
  --stop-when-full --chunk-size-samples 12000 --seed "$SEED"

# Do not silently proceed with missing groups or unmet quotas.
"$PARENT_PYTHON" - "$WORK_ROOT/metadata/$RUN_ID/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if (summary["missing_positive_strata"]
        or any(summary["quota_deficits"].values())):
    raise ValueError("Dataset quotas are incomplete; inspect summary.json")
if (summary["manifest_version"] != "construct-nominal-head-poly-v4"
        or set(summary["supervision_counts"]) != {"length_only"}):
    raise ValueError("Expected a v4 length-only dataset")
print("Dataset quotas met:", summary["supervision_counts"])
PY

echo "[2/5] Train frozen-encoder heads ($EPOCHS epochs)"
"$PROJECT_ROOT/.venv/bin/holypoly" train \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" --loss-log "$LOSS_LOG" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" --random-seed "$SEED" \
  --chunk-size-samples 12000 --hidden-features 128 --dropout 0.0 \
  --weight-decay 0.0 --gradient-clip-norm 2.0 --num-workers 0 --device cuda:0

printf -v FINAL_CHECKPOINT 'models/%s/heads.epoch_%04d.pt' "$RUN_ID" "$EPOCHS"
echo "[3/5] Evaluate final epoch on the complete test split"
"$BONITO_PYTHON" -m holypoly.training.evaluate \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --checkpoint "$FINAL_CHECKPOINT" --output-dir "$TEST_OUTPUT" \
  --split test --batch-size "$BATCH_SIZE" --max-reads 0

echo "[4/5] Plot training history"
"$PARENT_PYTHON" scripts/plot_training.py \
  --config "$CONFIG" --log "$LOSS_LOG" \
  --output "results/training_evaluation/$RUN_ID/training-plots"

echo "[5/5] Plot test predictions (raw regression output)"
"$PARENT_PYTHON" scripts/plot_training.py \
  --config "$CONFIG" --predictions "$TEST_OUTPUT/predictions.tsv" \
  --supervision-mode length_only --seed "$SEED" \
  --output "$TEST_OUTPUT/plots"

echo "Finished. Weak-label agreement only, not independently verified accuracy."
echo "Test summary: $WORK_ROOT/$TEST_OUTPUT/summary.json"
echo "Training figures: $WORK_ROOT/results/training_evaluation/$RUN_ID/training-plots"
echo "Test figures: $WORK_ROOT/$TEST_OUTPUT/plots"
echo "Pipeline log: $LOG_DIR/pipeline.log"
