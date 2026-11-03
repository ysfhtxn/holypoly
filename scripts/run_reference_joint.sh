#!/usr/bin/env bash
# Standard three-loss training; dataset preparation remains a separate audit step.
set -euo pipefail
if [[ $# -lt 2 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/run_reference_joint.sh <length-manifest> <sequence-manifest> [config.yaml] [--dry-run]
Paths are relative to work_root. Requires an audited prepare_sequence output.
Defaults: sum + Huber(delta=10), EPOCHS=5 BATCH_SIZE=32 SEQUENCE_BATCH_SIZE=1 GPU_ID=0
          UNFREEZE_LAST_LAYERS=1 HEAD_LEARNING_RATE=0.001 ENCODER_LEARNING_RATE=0.00001
          STATUS_SUPERVISION=1 LAMBDA_BASECALL=0.1 SEED=20260923
Length + weak status + native basecalling losses are enabled by default; location stays off.
Optional: STATUS_SUPERVISION=0 is for explicit two-loss ablations.
          LAMBDA_LENGTH=1 LAMBDA_STATUS=1 HUBER_DELTA_NT=10 (config weights if unset).
Optional: INITIALIZE_FROM=<compatible frozen sum/Huber checkpoint>, EXPERIMENT_ID.
Always creates new models/logs/results directories. Does not scan BAM or rebuild data.
Selects best validation length MAE, then tests length, plots length and reference-head curves.
HELP
  exit 0
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MANIFEST="$1"
SEQUENCE_MANIFEST="$2"
shift 2
CONFIG="configs/local.yaml"
if [[ $# -gt 0 && "$1" != "--dry-run" ]]; then CONFIG="$1"; shift; fi
DRY_RUN=false
if [[ $# -gt 0 && "$1" == "--dry-run" ]]; then DRY_RUN=true; shift; fi
if [[ $# -gt 0 ]]; then echo "Unexpected argument: $1" >&2; exit 2; fi
if [[ "${STATUS_SUPERVISION:-1}" == "1" ]]; then
  EXPERIMENT_ID="${EXPERIMENT_ID:-v4-reference-status-joint-$(date +%Y%m%d-%H%M%S)}"
else
  EXPERIMENT_ID="${EXPERIMENT_ID:-v4-reference-joint-$(date +%Y%m%d-%H%M%S)}"
fi
GPU_ID="${GPU_ID:-0}"
if [[ ! "$GPU_ID" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "GPU_ID must select one GPU" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
PARENT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
BONITO_PYTHON="$PROJECT_ROOT/src/bonito/.venv/bin/python"
ARGS=(--config "$CONFIG" --manifest "$MANIFEST" --sequence-manifest "$SEQUENCE_MANIFEST"
  --run-id "$EXPERIMENT_ID" --epochs "${EPOCHS:-5}" --batch-size "${BATCH_SIZE:-32}"
  --sequence-batch-size "${SEQUENCE_BATCH_SIZE:-1}" --unfreeze-last-layers "${UNFREEZE_LAST_LAYERS:-1}"
  --head-learning-rate "${HEAD_LEARNING_RATE:-0.001}" --encoder-learning-rate "${ENCODER_LEARNING_RATE:-0.00001}"
  --lambda-basecall "${LAMBDA_BASECALL:-0.1}" --random-seed "${SEED:-20260923}")
case "${STATUS_SUPERVISION:-1}" in
  0) ARGS+=(--no-supervise-status) ;;
  1) ARGS+=(--supervise-status) ;;
  *) echo "STATUS_SUPERVISION must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${LAMBDA_LENGTH:-}" ]]; then ARGS+=(--lambda-length "$LAMBDA_LENGTH"); fi
if [[ -n "${LAMBDA_STATUS:-}" ]]; then ARGS+=(--lambda-status "$LAMBDA_STATUS"); fi
ARGS+=(--huber-delta-nt "${HUBER_DELTA_NT:-10}")
if [[ -n "${INITIALIZE_FROM:-}" ]]; then ARGS+=(--initialize-from "$INITIALIZE_FROM"); fi
"$BONITO_PYTHON" -m holypoly.training.joint_command "${ARGS[@]}" --dry-run
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
trap 'code=$?; echo "Reference joint pipeline failed (exit $code) near line $LINENO" >&2; exit "$code"' ERR
export XDG_CACHE_HOME="$WORK_ROOT/cache/$EXPERIMENT_ID"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
"$BONITO_PYTHON" -m holypoly.training.joint_command "${ARGS[@]}"
BEST="$("$PARENT_PYTHON" - "$WORK_ROOT" "$EXPERIMENT_ID" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads((Path(sys.argv[1]) / 'results' / 'training_evaluation' / sys.argv[2] / 'best_checkpoint.json').read_text())['checkpoint'])
PY
)"
"$BONITO_PYTHON" -m holypoly.training.evaluate --config "$CONFIG" --manifest "$MANIFEST" \
  --checkpoint "$BEST" --output-dir "results/training_evaluation/$EXPERIMENT_ID/test-best-validation" \
  --split test --batch-size "${BATCH_SIZE:-4}" --max-reads 0
"$PARENT_PYTHON" scripts/plot_training.py --config "$CONFIG" \
  --log "logs/$EXPERIMENT_ID/train.jsonl" --output "results/training_evaluation/$EXPERIMENT_ID/training-plots"
"$PARENT_PYTHON" scripts/plot_training.py --config "$CONFIG" \
  --predictions "results/training_evaluation/$EXPERIMENT_ID/test-best-validation/predictions.tsv" \
  --supervision-mode length_only --seed "${SEED:-20260923}" \
  --output "results/training_evaluation/$EXPERIMENT_ID/test-best-validation/plots"
echo "Finished: $EXPERIMENT_ID | physical GPU $GPU_ID"
echo "Results: $WORK_ROOT/results/training_evaluation/$EXPERIMENT_ID"
