#!/usr/bin/env bash
# Joint length, status, and reference-head training, followed by test and plots.
set -euo pipefail
if [[ $# -eq 0 || "$1" == "--help" ]]; then
  cat <<'HELP'
Usage: GPU_ID=<gpu> bash scripts/run_joint_training.sh --dataset-dir <dataset> [config.yaml] [--resume] [--dry-run]
       bash scripts/run_joint_training.sh <length-manifest> <sequence-manifest> [config.yaml] [--resume] [--dry-run]
Dataset directories and manifest paths are relative to work_root.
Directory mode checks completed head-poly, strict-poly-candidate, and head-free QC builds.
It uses the recorded chunk size and requires GPU_ID. Two-manifest mode defaults to GPU_ID=0.
Defaults: Huber(delta=10), EPOCHS=5 BATCH_SIZE=32 SEQUENCE_BATCH_SIZE=1
          EVAL_BATCH_SIZE=<BATCH_SIZE if set, otherwise 4> (test inference only)
          UNFREEZE_LAST_LAYERS=1 HEAD_LEARNING_RATE=0.001 ENCODER_LEARNING_RATE=0.00001
          UNFREEZE_LAST_LAYERS=0 freezes the full encoder; new runs monitor head sequences once.
          FROZEN_SEQUENCE_MONITOR=auto|baseline|per-step (auto preserves old resume policy).
          PROFILE_BATCHES=0 (set e.g. 50 for bounded timing after 5 warmup batches).
          STATUS_SUPERVISION=1 LAMBDA_BASECALL=0.1 SEED=20260923
          PREFETCH_BATCHES=2 POD5_READER_CACHE_SIZE=8 (0 disables each optimization)
Length + weak status + native basecalling losses are enabled by default; location stays off.
Optional: STATUS_SUPERVISION=0 is for explicit two-loss ablations.
          HEAD_TYPE defaults to legacy (sum pooling); QC directories default to tcn.
          TCN has 128 channels and no timing fusion.
          TCN_LENGTH_SUM_SCALE=0.001 (fixed positive multiplier before TCN length MLP).
          TCN_LENGTH_FEATURES=sum_mean_duration (default) or sum (old TCN checkpoints).
          LAMBDA_LENGTH=1 LAMBDA_STATUS=1 HUBER_DELTA_NT=10 (config weights if unset).
Optional: INITIALIZE_FROM=<compatible frozen sum/Huber checkpoint>, EXPERIMENT_ID.
          QC directory mode disallows INITIALIZE_FROM and RESUME_FROM.
          USE_TIMING_REFERENCE=0 (default); set 1 with TIMING_REFERENCE_MANIFEST=<table.tsv>.
          Timing tables must be prepared separately; no automatic BAM scan.
Resume: RESUME=1 or --resume; requires EXPERIMENT_ID=<existing-run> and no INITIALIZE_FROM.
        EPOCHS is the target TOTAL epoch count, greater than the saved epoch.
        Keep original model/loss settings; length batch size and learning rates may change.
        HEAD_LEARNING_RATE / ENCODER_LEARNING_RATE apply after optimizer restore.
        ALLOW_LEARNING_RATE_CHANGE=0 requires matching rates (default: 1).
Branch: RESUME_FROM=models/<parent>/heads.epoch_0005.pt with a NEW EXPERIMENT_ID.
        Leave RESUME=0; restores optimizer and trains epochs 6..EPOCHS in new directories.
        Copies snapshots/logs only through the selected epoch; dry-run never copies.
New training creates new models/logs/results directories. Resume restores model/optimizer
and appends training logs; new evaluation/plots go under <run>/resume-<time>-<PID>/.
Does not scan BAM or rebuild data.
Selects best validation length MAE, then tests length, plots length and reference-head curves.
HELP
  exit 0
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
DATASET_DIR=""
if [[ "$1" == --dataset-dir ]]; then
  if [[ -z "${2:-}" || "$2" == --* ]]; then echo "--dataset-dir requires a directory" >&2; exit 2; fi
  DATASET_DIR="$2"
else
  if [[ $# -lt 2 || "$1" == --* || "$2" == --* ]]; then echo "Supply two manifests or --dataset-dir; see --help" >&2; exit 2; fi
  MANIFEST="$1"
  SEQUENCE_MANIFEST="$2"
fi
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
if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ "$RESUME" == 1 || -n "${INITIALIZE_FROM:-}" ]]; then echo "RESUME_FROM cannot use RESUME or INITIALIZE_FROM" >&2; exit 2; fi
  if [[ -z "${EXPERIMENT_ID:-}" ]]; then echo "RESUME_FROM requires a new EXPERIMENT_ID" >&2; exit 2; fi
fi
EXPECTED_WORKFLOW="${HOLYPOLY_EXPECT_DATASET_WORKFLOW:-}"
if [[ -n "$EXPECTED_WORKFLOW" ]]; then
  if [[ "$RESUME" != 0 || -n "${INITIALIZE_FROM:-}" || -n "${RESUME_FROM:-}" ]]; then
    echo "Use a fresh experiment without an old checkpoint with this QC compatibility script" >&2
    exit 2
  fi
  HEAD_TYPE=tcn
  STATUS_SUPERVISION=1
  USE_TIMING_REFERENCE=0
  TIMING_REFERENCE_MANIFEST=""
fi
if [[ -n "$DATASET_DIR" && -z "${GPU_ID:-}" ]]; then echo "Set GPU_ID to one explicit physical GPU" >&2; exit 2; fi
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

RUN_PREFIX=v4-reference-status-joint
if [[ "${STATUS_SUPERVISION:-1}" == 0 ]]; then RUN_PREFIX=v4-reference-joint; fi
CHUNK_SIZE_SAMPLES=""
if [[ -n "$DATASET_DIR" ]]; then
  DATASET_INPUTS="$("$PARENT_PYTHON" -m holypoly.training.dataset_bundle \
    "$CONFIG" "$DATASET_DIR" --expect-workflow "$EXPECTED_WORKFLOW")"
  IFS=$'\t' read -r DATASET_WORKFLOW TRAINING_SUBDIR CHUNK_SIZE_SAMPLES <<< "$DATASET_INPUTS"
  MANIFEST="$DATASET_DIR/$TRAINING_SUBDIR/length/manifest.tsv"
  SEQUENCE_MANIFEST="$DATASET_DIR/$TRAINING_SUBDIR/sequence/sequence_manifest.tsv"
  if [[ "$DATASET_WORKFLOW" != head-poly-dataset ]]; then
    if [[ -n "${INITIALIZE_FROM:-}" || -n "${RESUME_FROM:-}" ]]; then
      echo "QC dataset directory mode requires a fresh run or --resume with its matching experiment" >&2
      exit 2
    fi
    HEAD_TYPE="${HEAD_TYPE:-tcn}"
    LAMBDA_LENGTH="${LAMBDA_LENGTH:-1}"
    LAMBDA_STATUS="${LAMBDA_STATUS:-1}"
    case "$DATASET_WORKFLOW" in
      strict-poly-candidate-v1) RUN_PREFIX="strict-poly-$HEAD_TYPE-context-unfreeze${UNFREEZE_LAST_LAYERS:-1}" ;;
      head-free-platform-qc-v1) RUN_PREFIX="head-free-platform-qc-$HEAD_TYPE-unfreeze${UNFREEZE_LAST_LAYERS:-1}" ;;
    esac
  fi
  printf 'Verified %s dataset at %s (head %s, %s samples, physical GPU %s)\n' \
    "$DATASET_WORKFLOW" "$DATASET_DIR" "${HEAD_TYPE:-legacy}" "$CHUNK_SIZE_SAMPLES" "$GPU_ID" >&2
fi
EXPERIMENT_ID="${EXPERIMENT_ID:-$RUN_PREFIX-$(date +%Y%m%d-%H%M%S)}"

ARGS=(--config "$CONFIG" --manifest "$MANIFEST" --sequence-manifest "$SEQUENCE_MANIFEST"
  --head-type "${HEAD_TYPE:-legacy}"
  --tcn-length-sum-scale "${TCN_LENGTH_SUM_SCALE:-0.001}"
  --tcn-length-features "${TCN_LENGTH_FEATURES:-sum_mean_duration}"
  --frozen-sequence-monitor "${FROZEN_SEQUENCE_MONITOR:-auto}" --profile-batches "${PROFILE_BATCHES:-0}"
  --run-id "$EXPERIMENT_ID" --epochs "${EPOCHS:-5}" --batch-size "${BATCH_SIZE:-32}"
  --prefetch-batches "${PREFETCH_BATCHES:-2}" --pod5-reader-cache-size "${POD5_READER_CACHE_SIZE:-8}"
  --sequence-batch-size "${SEQUENCE_BATCH_SIZE:-1}" --unfreeze-last-layers "${UNFREEZE_LAST_LAYERS:-1}"
  --head-learning-rate "${HEAD_LEARNING_RATE:-0.001}" --encoder-learning-rate "${ENCODER_LEARNING_RATE:-0.00001}"
  --lambda-basecall "${LAMBDA_BASECALL:-0.1}" --random-seed "${SEED:-20260923}")
if [[ -n "$CHUNK_SIZE_SAMPLES" ]]; then ARGS+=(--chunk-size-samples "$CHUNK_SIZE_SAMPLES"); fi
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
if [[ -n "${RESUME_FROM:-}" ]]; then ARGS+=(--resume-from "$RESUME_FROM"); fi
case "${ALLOW_LEARNING_RATE_CHANGE:-1}" in
  1) ARGS+=(--allow-learning-rate-change) ;;
  0) ARGS+=(--no-allow-learning-rate-change) ;;
  *) echo "ALLOW_LEARNING_RATE_CHANGE must be 0 or 1" >&2; exit 2 ;;
esac
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
trap 'code=$?; echo "Joint training pipeline failed (exit $code) near line $LINENO" >&2; exit "$code"' ERR
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
