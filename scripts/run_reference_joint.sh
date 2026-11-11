#!/usr/bin/env bash
# Standard three-loss training; dataset preparation remains a separate audit step.
set -euo pipefail
if [[ $# -lt 2 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/run_reference_joint.sh <length-manifest> <sequence-manifest> [config.yaml] [--resume] [--dry-run]
Paths are relative to work_root. Requires an audited prepare_sequence output.
Defaults: sum + Huber(delta=10), EPOCHS=5 BATCH_SIZE=32 SEQUENCE_BATCH_SIZE=1 GPU_ID=0
          EVAL_BATCH_SIZE=<BATCH_SIZE if set, otherwise 4> (test inference only)
          UNFREEZE_LAST_LAYERS=1 HEAD_LEARNING_RATE=0.001 ENCODER_LEARNING_RATE=0.00001
          STATUS_SUPERVISION=1 LAMBDA_BASECALL=0.1 SEED=20260923
          PREFETCH_BATCHES=2 POD5_READER_CACHE_SIZE=8 (0 disables each optimization)
Length + weak status + native basecalling losses are enabled by default; location stays off.
Optional: STATUS_SUPERVISION=0 is for explicit two-loss ablations.
          LAMBDA_LENGTH=1 LAMBDA_STATUS=1 HUBER_DELTA_NT=10 (config weights if unset).
Optional: INITIALIZE_FROM=<compatible frozen sum/Huber checkpoint>, EXPERIMENT_ID.
          USE_TIMING_REFERENCE=0 (default); set 1 with TIMING_REFERENCE_MANIFEST=<table.tsv>.
          Timing tables must be prepared separately; no automatic BAM scan.
Resume: RESUME=1 or --resume; requires EXPERIMENT_ID=<existing-run> and no INITIALIZE_FROM.
        EPOCHS is the target TOTAL epoch count, greater than the saved epoch.
        Keep original model/loss settings; length batch size may change.
New training creates new models/logs/results directories. Resume restores model/optimizer
and appends training logs; new evaluation/plots go under <run>/resume-<time>-<PID>/.
Does not scan BAM or rebuild data.
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
if [[ $# -gt 0 && "$1" != --* ]]; then CONFIG="$1"; shift; fi
DRY_RUN=false
RESUME="${RESUME:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --resume) RESUME=1 ;;
    *) echo "Unexpected argument: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then echo "RESUME must be 0 or 1" >&2; exit 2; fi
if [[ "$RESUME" == 1 ]]; then
  if [[ -z "${EXPERIMENT_ID:-}" ]]; then echo "Resume requires EXPERIMENT_ID=<existing-run>" >&2; exit 2; fi
  if [[ -n "${INITIALIZE_FROM:-}" ]]; then echo "Resume cannot use INITIALIZE_FROM" >&2; exit 2; fi
fi
if [[ "${STATUS_SUPERVISION:-1}" == "1" ]]; then
  EXPERIMENT_ID="${EXPERIMENT_ID:-v4-reference-status-joint-$(date +%Y%m%d-%H%M%S)}"
else
  EXPERIMENT_ID="${EXPERIMENT_ID:-v4-reference-joint-$(date +%Y%m%d-%H%M%S)}"
fi
GPU_ID="${GPU_ID:-0}"
if [[ ! "$GPU_ID" =~ ^(0|[1-9][0-9]*)$ ]]; then echo "GPU_ID must select one GPU" >&2; exit 2; fi
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE:-4}}"
if [[ ! "$EVAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then echo "EVAL_BATCH_SIZE must be positive" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
PARENT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
BONITO_PYTHON="$PROJECT_ROOT/src/bonito/.venv/bin/python"
ARGS=(--config "$CONFIG" --manifest "$MANIFEST" --sequence-manifest "$SEQUENCE_MANIFEST"
  --run-id "$EXPERIMENT_ID" --epochs "${EPOCHS:-5}" --batch-size "${BATCH_SIZE:-32}"
  --prefetch-batches "${PREFETCH_BATCHES:-2}" --pod5-reader-cache-size "${POD5_READER_CACHE_SIZE:-8}"
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
TIMING_EVAL_ARGS=()
case "${USE_TIMING_REFERENCE:-0}" in
  0) if [[ -n "${TIMING_REFERENCE_MANIFEST:-}" ]]; then echo "TIMING_REFERENCE_MANIFEST requires USE_TIMING_REFERENCE=1" >&2; exit 2; fi ;;
  1)
    : "${TIMING_REFERENCE_MANIFEST:?Set TIMING_REFERENCE_MANIFEST to the matching prepared table}"
    ARGS+=(--use-timing-reference --timing-reference-manifest "$TIMING_REFERENCE_MANIFEST")
    TIMING_EVAL_ARGS+=(--timing-reference-manifest "$TIMING_REFERENCE_MANIFEST") ;;
  *) echo "USE_TIMING_REFERENCE must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$RESUME" == 1 ]]; then ARGS+=(--resume); fi
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
EVALUATION_ROOT="results/training_evaluation/$EXPERIMENT_ID"
if [[ "$RESUME" == 1 ]]; then
  RESUME_TAG="resume-$(date +%Y%m%d-%H%M%S)-$$"
  PIPELINE_LOG="${PIPELINE_LOG}-${RESUME_TAG}"
  EVALUATION_ROOT="${EVALUATION_ROOT}/${RESUME_TAG}"
fi
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
  --checkpoint "$BEST" --output-dir "$EVALUATION_ROOT/test-best-validation" \
  --split test --batch-size "$EVAL_BATCH_SIZE" --max-reads 0 "${TIMING_EVAL_ARGS[@]}"
"$PARENT_PYTHON" scripts/plot_training.py --config "$CONFIG" \
  --log "logs/$EXPERIMENT_ID/train.jsonl" --output "$EVALUATION_ROOT/training-plots"
"$PARENT_PYTHON" scripts/plot_training.py --config "$CONFIG" \
  --predictions "$EVALUATION_ROOT/test-best-validation/predictions.tsv" \
  --supervision-mode length_only --seed "${SEED:-20260923}" \
  --output "$EVALUATION_ROOT/test-best-validation/plots"
echo "Finished: $EXPERIMENT_ID | physical GPU $GPU_ID"
echo "Results: $WORK_ROOT/$EVALUATION_ROOT"
echo "Pipeline log: $PIPELINE_LOG/pipeline.log"
