#!/usr/bin/env bash
# Existing status classifiers only; never prepares data or reads the test split.
set -euo pipefail
if [[ $# -lt 2 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/run_status_calibration.sh <v4-manifest> <initial-checkpoint> [config.yaml] [--dry-run]
Paths are relative to work_root. Uses only train/validation; no BAM scan.
Defaults: GPU_ID=0 EPOCHS=20 BATCH_SIZE=32 LEARNING_RATE=0.001 SEED=20260923
Optional: EXPERIMENT_ID (fresh directories required).
Only the two existing status_classifier layers train; projections/regression/Bonito stay frozen.
Class weights come from train only; selection uses validation 0-nt/20-nt harmonic recall.
Stores full latest/best checkpoints plus small per-epoch status snapshots; plots validation curves.
HELP
  exit 0
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MANIFEST="$1"
INITIALIZE_FROM="$2"
shift 2
CONFIG="configs/local.yaml"
if [[ $# -gt 0 && "$1" != "--dry-run" ]]; then CONFIG="$1"; shift; fi
DRY_RUN=false
if [[ $# -gt 0 && "$1" == "--dry-run" ]]; then DRY_RUN=true; shift; fi
if [[ $# -gt 0 ]]; then echo "Unexpected argument: $1" >&2; exit 2; fi
EXPERIMENT_ID="${EXPERIMENT_ID:-v4-status-calibration-$(date +%Y%m%d-%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
if [[ ! "$GPU_ID" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "GPU_ID must select one GPU" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
BONITO_PYTHON="$PROJECT_ROOT/src/bonito/.venv/bin/python"
PARENT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
ARGS=(--config "$CONFIG" --manifest "$MANIFEST" --initialize-from "$INITIALIZE_FROM"
  --run-id "$EXPERIMENT_ID" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-32}"
  --learning-rate "${LEARNING_RATE:-0.001}" --random-seed "${SEED:-20260923}")
"$BONITO_PYTHON" -m holypoly.training.status_command "${ARGS[@]}" --dry-run
if [[ "$DRY_RUN" == true ]]; then exit 0; fi
WORK_ROOT="$("$PARENT_PYTHON" - "$CONFIG" <<'PY'
from pathlib import Path
import sys
from holypoly.core.config import load_config
print(load_config(Path(sys.argv[1])).paths.work_root.resolve())
PY
)"
PIPELINE_LOG="$WORK_ROOT/logs/$EXPERIMENT_ID-pipeline"
if [[ -e "$PIPELINE_LOG" || -L "$PIPELINE_LOG" ]]; then echo "Pipeline output exists" >&2; exit 2; fi
mkdir -p "$PIPELINE_LOG"
exec > >(tee "$PIPELINE_LOG/pipeline.log") 2>&1
trap 'code=$?; echo "Status calibration failed (exit $code) near line $LINENO" >&2; exit "$code"' ERR
export XDG_CACHE_HOME="$WORK_ROOT/cache/$EXPERIMENT_ID"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
"$BONITO_PYTHON" -m holypoly.training.status_command "${ARGS[@]}"
"$PARENT_PYTHON" scripts/plot_training.py --config "$CONFIG" \
  --log "logs/$EXPERIMENT_ID/train.jsonl" --output "results/training_evaluation/$EXPERIMENT_ID/training-plots"
echo "Finished: $EXPERIMENT_ID | physical GPU $GPU_ID | validation only"
echo "Results: $WORK_ROOT/results/training_evaluation/$EXPERIMENT_ID"
