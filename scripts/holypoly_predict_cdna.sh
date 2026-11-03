#!/usr/bin/env bash
# Sequence-assisted exploratory inference; never changes network or training results.
set -euo pipefail
if [[ $# -lt 1 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: CHECKPOINT=models/<run>/heads.epoch_0005.pt bash scripts/holypoly_predict_cdna.sh <pod5-dir> [config.yaml]
POD5 path is relative to data_root; outputs use work_root/results/cdna_prediction/<experiment>.
Defaults: GPU_ID=0 MAX_READS=1000 BATCH_SIZE=8 POLY_TYPES=polyA
Optional: MODEL_DIR=<existing Bonito v5 model>, EXPERIMENT_ID, BASECALL_BAM=<existing BAM>
          DATASET_NAME=<dataset>, e.g. fixed-len or hek293t
          READ_IDS=<work-root-relative text file>, one read UUID per line, no header
          RESULT_CATEGORY=cdna_prediction or training_evaluation
          MIN_TARGET_BASES=10 END_MARGIN_BASES=100 CHUNK_SIZE_SAMPLES=12000
DATASET_NAME adds a dataset name: holypoly-predict-<dataset>-cdna-<time>-<PID>.
EXPERIMENT_ID overrides the complete output name; unset DATASET_NAME keeps the old name.
MAX_READS=0 explicitly processes all reads. BASECALL_BAM must contain mv/ts/ns tags.
READ_IDS selects reads before basecalling; cannot combine with BASECALL_BAM.
For a complete held-out list use MAX_READS=0 RESULT_CATEGORY=training_evaluation.
Set POLY_TYPES='polyA polyT' only when both directions are intended for evaluation.
No known primer/reference or Poly truth labels are required. This is not validated
biological tail detection. No candidate is never converted to zero nt.
HELP
  exit 0
fi
if [[ $# -gt 2 ]]; then echo 'Too many arguments' >&2; exit 2; fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
CONFIG="${2:-configs/local.yaml}"
: "${CHECKPOINT:?Set CHECKPOINT to the existing trained checkpoint}"
GPU_ID="${GPU_ID:-0}"
MAX_READS="${MAX_READS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
if [[ ! "$GPU_ID" =~ ^(0|[1-9][0-9]*)$ || ! "$MAX_READS" =~ ^(0|[1-9][0-9]*)$ || ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo 'GPU_ID, MAX_READS and BATCH_SIZE must be valid single-device integer settings' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# The submodule root must precede src so Python loads bonito/bonito/__init__.py.
export PYTHONPATH="$PROJECT_ROOT/src/bonito:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PARENT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
BONITO_PYTHON="$PROJECT_ROOT/src/bonito/.venv/bin/python"
[[ -x "$PARENT_PYTHON" && -x "$BONITO_PYTHON" ]] || { echo 'Required environments are missing' >&2; exit 2; }
DATASET_NAME="${DATASET_NAME:-}"
if [[ -n "$DATASET_NAME" && ! "$DATASET_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo 'DATASET_NAME must start with a letter or digit and contain only letters, digits, dots, underscores or hyphens' >&2
  exit 2
fi
EXPERIMENT_ID="${EXPERIMENT_ID:-holypoly${DATASET_NAME:+-$DATASET_NAME}-cdna-$(date +%Y%m%d-%H%M%S)-$$}"
# Plan validation uses paths only: no signals, BAM records or checkpoint loading.
mapfile -t PLAN < <("$PARENT_PYTHON" - "$CONFIG" "$1" "$CHECKPOINT" "${MODEL_DIR:-}" "$EXPERIMENT_ID" "${BASECALL_BAM:-}" "$PROJECT_ROOT" "${READ_IDS:-}" "${RESULT_CATEGORY:-cdna_prediction}" <<'PY'
from pathlib import Path
import sys
from uuid import UUID
from holypoly.core.config import load_config
config = load_config(Path(sys.argv[1]))
if config.paths.data_root is None or config.paths.work_root is None:
    raise ValueError('data_root and work_root are required')
data, work = config.paths.data_root.resolve(), config.paths.work_root.resolve()
pod5 = (data / sys.argv[2]).resolve()
checkpoint = (work / sys.argv[3]).resolve()
model_value = sys.argv[4] or config.training.bonito_model
if model_value is None:
    raise ValueError('Set MODEL_DIR or training.bonito_model')
model = (work / model_value).resolve()
experiment = sys.argv[5]
if Path(experiment).name != experiment or experiment in {'.', '..'}:
    raise ValueError('EXPERIMENT_ID must be a single directory name')
category = sys.argv[9]
if category not in {'cdna_prediction', 'training_evaluation'}:
    raise ValueError('RESULT_CATEGORY must be cdna_prediction or training_evaluation')
read_ids = (work / sys.argv[8]).resolve() if sys.argv[8] else None
if read_ids is not None:
    if sys.argv[6]:
        raise ValueError('READ_IDS cannot combine with BASECALL_BAM; re-basecall the selected reads')
    if not read_ids.is_file() or not read_ids.is_relative_to(work) or read_ids.is_relative_to(data):
        raise ValueError('READ_IDS must exist inside work_root and outside raw data')
    identifiers = read_ids.read_text().splitlines()
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('READ_IDS must contain nonempty unique read UUIDs')
    for identifier in identifiers:
        if str(UUID(identifier)) != identifier:
            raise ValueError('READ_IDS requires canonical UUIDs, one per line, no header')
output = (work / 'results' / category / experiment).resolve()
cache = (work / 'cache' / experiment).resolve()
bam = (work / sys.argv[6]).resolve() if sys.argv[6] else output / 'basecalls.bam'
if (not work.is_dir() or work.is_relative_to(data) or not pod5.is_dir()
    or not pod5.is_relative_to(data) or not checkpoint.is_file()
    or not (model / 'config.toml').is_file() or not output.is_relative_to(work)
    or output.is_relative_to(data) or output.is_relative_to(Path(sys.argv[7]))
    or output.exists() or output.is_symlink()
    or any(path.is_relative_to(output) for path in (pod5, checkpoint, model))):
    raise ValueError('Invalid inputs or unsafe/existing output directory')
if not cache.is_relative_to(work) or cache.is_relative_to(data) or cache.is_relative_to(Path(sys.argv[7])):
    raise ValueError('Cache must remain outside raw data and project')
if sys.argv[6] and not bam.is_file():
    raise FileNotFoundError(bam)
for path in (work, pod5, checkpoint, model, output, bam, cache):
    print(path)
print(read_ids if read_ids is not None else '')
PY
)
if [[ ${#PLAN[@]} -ne 8 ]]; then echo 'Input validation failed' >&2; exit 2; fi
WORK_ROOT="${PLAN[0]}"
POD5_INPUT="${PLAN[1]}"
CHECKPOINT_PATH="${PLAN[2]}"
MODEL_PATH="${PLAN[3]}"
OUTPUT="${PLAN[4]}"
BAM_INPUT="${PLAN[5]}"
READ_IDS_PATH="${PLAN[7]}"
export XDG_CACHE_HOME="${PLAN[6]}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
# Resolve the real Bonito CLI before creating outputs or reading any signals.
"$BONITO_PYTHON" -m bonito basecaller --help > /dev/null
mkdir -p "$(dirname -- "$OUTPUT")"
mkdir "$OUTPUT"
exec > >(tee "$OUTPUT/pipeline.log") 2>&1
echo "Dataset: ${DATASET_NAME:-unspecified}; experiment: $EXPERIMENT_ID"
echo "Selected read limit: $MAX_READS (0=all); physical GPU: $GPU_ID"
echo 'POD5 UUID metadata is read; no raw data is changed. Output: predictions TSV and distribution plots.'
if [[ -z "${BASECALL_BAM:-}" ]]; then
  echo 'Bonito basecalls the selected reads; signal volume/runtime depend on read lengths.'
  BASECALL_ARGS=()
  if [[ -n "$READ_IDS_PATH" ]]; then
    cp -- "$READ_IDS_PATH" "$OUTPUT/read_ids.txt"
    BASECALL_ARGS+=(--read-ids "$OUTPUT/read_ids.txt")
    echo 'Read selection: only UUIDs from read_ids.txt; MAX_READS applies after filtering.'
  fi
  "$BONITO_PYTHON" -m bonito basecaller "$MODEL_PATH" "$POD5_INPUT" \
    --device cuda:0 --recursive --no-trim --batchsize "$BATCH_SIZE" \
    --max-reads "$MAX_READS" "${BASECALL_ARGS[@]}" > "$BAM_INPUT" 2> "$OUTPUT/basecalling.log"
fi
read -r -a TYPES <<< "${POLY_TYPES:-polyA}"
"$BONITO_PYTHON" -m holypoly.inference.predict --config "$CONFIG" \
  --pod5-dir "$POD5_INPUT" --bam "$BAM_INPUT" --checkpoint "$CHECKPOINT_PATH" \
  --model-dir "$MODEL_PATH" --output-dir "$OUTPUT/prediction" \
  --max-reads "$MAX_READS" --poly-types "${TYPES[@]}" \
  --chunk-size-samples "${CHUNK_SIZE_SAMPLES:-12000}" \
  --min-target-bases "${MIN_TARGET_BASES:-10}" --end-margin-bases "${END_MARGIN_BASES:-100}"
"$PARENT_PYTHON" scripts/plot_cdna_predictions.py --config "$CONFIG" \
  --predictions "$OUTPUT/prediction/predictions.tsv" --output "$OUTPUT/plots"
echo "Finished: $OUTPUT"
