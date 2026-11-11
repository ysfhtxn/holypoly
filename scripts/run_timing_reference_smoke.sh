#!/usr/bin/env bash
# Small timing-enabled joint experiment; source data and old runs remain untouched.
set -euo pipefail
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash scripts/run_timing_reference_smoke.sh <length-manifest> <sequence-manifest> <moves.bam> [config.yaml] [--dry-run]
Length and sequence manifests must be an existing, matching reference-head pair.
Manifests are relative to work_root; BAM is relative to data_root (absolute paths allowed).
Defaults: PER_GROUP=100 EPOCHS=3 BATCH_SIZE=8 EVAL_BATCH_SIZE=8 GPU_ID=0 SEED=20260923
          RUN_BASELINE=0 (set 1 to also train a timing-disabled control).
Optional: EXPERIMENT_ID, HEAD_LEARNING_RATE, ENCODER_LEARNING_RATE,
          PREFETCH_BATCHES, POD5_READER_CACHE_SIZE.
For 7 source groups: 700 reads, 560 train / 70 validation / 70 test.
Keeps original splits/labels, derives a validated paired sequence subset, scans BAM
once for timing, then trains length/status/basecalling, evaluates and plots.
Missing timing references are retained, never relabeled as zero nt.
Starts fresh from configured Bonito, never resumes or initializes from old Poly weights.
--dry-run reads metadata and file sizes only: no BAM scan, signals, weights, GPU or writes.
Each real run creates new metadata/models/logs/results under external work_root.
HELP
  exit 0
fi
if [[ $# -lt 3 ]]; then echo 'Expected length manifest, sequence manifest and BAM; use --help' >&2; exit 2; fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
SOURCE_MANIFEST="$1"
SOURCE_SEQUENCE="$2"
BAM="$3"
shift 3
CONFIG="configs/local.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then CONFIG="$1"; shift; fi
MODE=run
if [[ $# -gt 0 && "$1" == --dry-run ]]; then MODE=dry-run; shift; fi
if [[ $# -gt 0 ]]; then echo "Unexpected argument: $1" >&2; exit 2; fi
EXPERIMENT_ID="${EXPERIMENT_ID:-v4-timing-smoke-$(date +%Y%m%d-%H%M%S)-$$}"
PER_GROUP="${PER_GROUP:-100}"
export EPOCHS="${EPOCHS:-3}" BATCH_SIZE="${BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}" GPU_ID="${GPU_ID:-0}" SEED="${SEED:-20260923}"
RUN_BASELINE="${RUN_BASELINE:-0}"
if [[ ! "$EXPERIMENT_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo 'EXPERIMENT_ID must be a single directory name' >&2; exit 2
fi
for value in "$PER_GROUP" "$EPOCHS" "$BATCH_SIZE" "$EVAL_BATCH_SIZE"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo 'Counts and batch sizes must be positive integers' >&2; exit 2; fi
done
for value in "$GPU_ID" "$SEED"; do
  if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then echo 'GPU_ID and SEED must be nonnegative integers' >&2; exit 2; fi
done
if [[ "$RUN_BASELINE" != 0 && "$RUN_BASELINE" != 1 ]]; then echo 'RUN_BASELINE must be 0 or 1' >&2; exit 2; fi
if [[ "${RESUME:-0}" != 0 || -n "${INITIALIZE_FROM:-}" ]]; then
  echo 'This smoke script starts fresh; unset RESUME and INITIALIZE_FROM' >&2; exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
BONITO_PYTHON="$PROJECT_ROOT/src/bonito/.venv/bin/python"
PARENT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$BONITO_PYTHON" || ! -x "$PARENT_PYTHON" ]]; then echo 'Required Python environments are missing' >&2; exit 2; fi
INPUT_DIR="metadata/$EXPERIMENT_ID-inputs"
# Read metadata, validate original sequence provenance, then derive both manifests.
# Only source_manifest_sha256 changes in selected sequence rows; labels stay fixed.
WORK_ROOT="$("$BONITO_PYTHON" - "$CONFIG" "$SOURCE_MANIFEST" "$SOURCE_SEQUENCE" "$BAM" \
  "$INPUT_DIR" "$EXPERIMENT_ID" "$PER_GROUP" "$SEED" "$EPOCHS" "$RUN_BASELINE" "$MODE" <<'PY'
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

from holypoly.core.config import load_config
from holypoly.datasets.prepare_sequence import sha256
from holypoly.datasets.subset_manifest import select_subset
from holypoly.training.data import load_training_manifest
from holypoly.training.sequence_data import load_sequence_manifest

config = load_config(Path(sys.argv[1]))
if config.paths.data_root is None or config.paths.work_root is None:
    raise ValueError('Configure data_root and work_root')
data, work = (path.expanduser().resolve() for path in (config.paths.data_root, config.paths.work_root))
project = Path.cwd().resolve()
if not data.is_dir() or not work.is_dir() or work.is_relative_to(data) or work.is_relative_to(project):
    raise ValueError('Existing external data/work roots must be separate')

def safe(value):
    path = (work / Path(value).expanduser()).resolve()
    if path == work or not path.is_relative_to(work) or path.is_relative_to(data) or path.is_relative_to(project):
        raise ValueError(f'Unsafe work path: {value}')
    return path

source, sequence, output = safe(sys.argv[2]), safe(sys.argv[3]), safe(sys.argv[5])
bam = (data / Path(sys.argv[4]).expanduser()).resolve(strict=True)
if not bam.is_file():
    raise ValueError('BAM must be an existing file')
model_value = config.training.bonito_model
if model_value is None or not (work / Path(model_value).expanduser() / 'config.toml').is_file():
    raise ValueError('Set training.bonito_model in the selected config')
run_id, per_group, seed, epochs = sys.argv[6], int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
paths = [output, safe(f'logs/{run_id}-preparation')]
for suffix in ('-timing', '-baseline') if sys.argv[10] == '1' else ('-timing',):
    paths.extend(safe(f'{category}/{run_id}{suffix}') for category in ('models', 'logs', 'results/training_evaluation'))
    paths.append(safe(f'logs/{run_id}{suffix}-pipeline'))
for path in paths:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Choose a new EXPERIMENT_ID: {path}')
    if any(item.is_relative_to(path) for item in (source, sequence, bam)):
        raise ValueError('Output cannot contain an input')

def read_table(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t' if path.suffix == '.tsv' else ',')
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed source table')
    return fields, rows

fields, rows = read_table(source)
records = load_training_manifest(source)
original_sha = sha256(source)
load_sequence_manifest(sequence, records, source_manifest_sha256=original_sha, data_root=data)
seq_fields, seq_rows = read_table(sequence)
selected = select_subset(rows, per_group=per_group, seed=seed)
selected_ids = {row['read_id'] for row in selected}
paired = [dict(row) for row in seq_rows if row['read_id'] in selected_ids]
if any(not any(row['split'] == split for row in paired) for split in ('train', 'validation', 'test')):
    raise ValueError('Selected subset needs accepted reference-head sequences in all splits; increase PER_GROUP or change SEED')
summary = {
    'experiment_id': run_id, 'source_manifest': str(source), 'source_manifest_sha256': original_sha,
    'source_sequence_manifest': str(sequence), 'source_sequence_manifest_sha256': sha256(sequence),
    'per_group': per_group, 'seed': seed, 'selected_reads': len(selected),
    'split_counts': dict(Counter(row['split'] for row in selected)),
    'paired_sequence_counts': dict(Counter(row['split'] for row in paired)),
    'selected_raw_samples': sum(int(row['raw_signal_length']) for row in selected),
    'bam': str(bam), 'bam_bytes': bam.stat().st_size, 'bam_passes': 1,
    'epochs_per_run': epochs, 'training_runs': 2 if sys.argv[10] == '1' else 1,
    'physical_gpu': int(os.environ['GPU_ID']), 'batch_size': int(os.environ['BATCH_SIZE']),
    'resources': 'One CPU BAM pass; one explicitly selected GPU for training. BAM: minutes for multi-GiB; training: minutes to tens of minutes, hardware dependent. Snapshots may use 2-4 GiB each, plus latest checkpoint.',
    'selection_policy': 'random within existing length/type/split, independent of timing availability',
    'sequence_derivation': 'validated original pair; retain selected IDs/splits/targets/intervals, bind to subset SHA256',
    'limitations': 'Small development smoke test; nominal weak labels, no read-level accuracy claim.',
    'dry_run': sys.argv[11] == 'dry-run',
}
print(json.dumps(summary, indent=2), file=sys.stderr, flush=True)
if not summary['dry_run']:
    (output / 'length').mkdir(parents=True)
    (output / 'sequence').mkdir()
    def write_table(path, fields, rows):
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
    manifest = output / 'length/manifest.tsv'
    write_table(manifest, fields, selected)
    digest = sha256(manifest)
    for row in paired:
        row['source_manifest_sha256'] = digest
    derived = output / 'sequence/sequence_manifest.tsv'
    write_table(derived, seq_fields, paired)
    load_sequence_manifest(derived, load_training_manifest(manifest), source_manifest_sha256=digest, data_root=data)
    summary.update(subset_manifest_sha256=digest, subset_sequence_manifest_sha256=sha256(derived))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(work)
PY
)"
if [[ "$MODE" == dry-run ]]; then echo 'Dry-run complete: no BAM scan, GPU work or outputs written.'; exit 0; fi
LOG_DIR="$WORK_ROOT/logs/$EXPERIMENT_ID-preparation"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/pipeline.log") 2>&1
trap 'code=$?; echo "Timing smoke failed (exit $code) near line $LINENO" >&2; exit "$code"' ERR
MANIFEST="$INPUT_DIR/length/manifest.tsv"
SEQUENCE_MANIFEST="$INPUT_DIR/sequence/sequence_manifest.tsv"
echo "Small timing experiment: $EXPERIMENT_ID | GPU $GPU_ID | $EPOCHS epochs | batch $BATCH_SIZE"
"$BONITO_PYTHON" -m holypoly.datasets.prepare_timing_reference \
  --config "$CONFIG" --manifest "$MANIFEST" --bam "$BAM" \
  --output-dir "$INPUT_DIR/timing" --scan-bam
# Stop rather than train a timing branch with no usable train/validation evidence.
"$PARENT_PYTHON" - "$WORK_ROOT/$INPUT_DIR/timing/summary.json" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
print('Timing reference coverage:', json.dumps(summary['coverage_by_split']))
if any(summary['coverage_by_split'][split]['available'] == 0 for split in ('train', 'validation')):
    raise SystemExit('No usable timing in train/validation; inspect window_audit.tsv before training.')
PY
export STATUS_SUPERVISION=1 LAMBDA_LENGTH=1 LAMBDA_STATUS=1 LAMBDA_BASECALL=0.1 HUBER_DELTA_NT=10
export UNFREEZE_LAST_LAYERS=1 RESUME=0 INITIALIZE_FROM=""
export HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-0.001}" ENCODER_LEARNING_RATE="${ENCODER_LEARNING_RATE:-0.00001}"
if [[ "$RUN_BASELINE" == 1 ]]; then
  EXPERIMENT_ID="$EXPERIMENT_ID-baseline" USE_TIMING_REFERENCE=0 TIMING_REFERENCE_MANIFEST="" \
    bash scripts/run_reference_joint.sh "$MANIFEST" "$SEQUENCE_MANIFEST" "$CONFIG"
fi
EXPERIMENT_ID="$EXPERIMENT_ID-timing" USE_TIMING_REFERENCE=1 \
  TIMING_REFERENCE_MANIFEST="$INPUT_DIR/timing/timing_reference.tsv" \
  bash scripts/run_reference_joint.sh "$MANIFEST" "$SEQUENCE_MANIFEST" "$CONFIG"
echo "Inputs and timing audits: $WORK_ROOT/$INPUT_DIR"
echo "Training, test and plots: $WORK_ROOT/results/training_evaluation/$EXPERIMENT_ID-timing"
if [[ "$RUN_BASELINE" == 1 ]]; then echo "Control: $WORK_ROOT/results/training_evaluation/$EXPERIMENT_ID-baseline"; fi
