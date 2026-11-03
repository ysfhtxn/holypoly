#!/usr/bin/env bash
# Fork completed epoch-5 experiments, keeping all prior artifacts unchanged.
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/resume_length_head_ablation.sh <source-experiment-id> [config.yaml]
Copy each epoch-5 checkpoint and log into new directories, then resume to epoch 20.
Use the same small manifest, model settings, data-order seed and physical GPU 0.
Select the lowest validation MAE across epochs 1-20, then test and plot it.
Optional environment: NEW_EXPERIMENT_ID. Existing outputs are never overwritten.
HELP
  exit 0
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
SOURCE_EXPERIMENT="$1"
CONFIG="${2:-configs/local.yaml}"
NEW_EXPERIMENT_ID="${NEW_EXPERIMENT_ID:-$SOURCE_EXPERIMENT-to20-$(date +%Y%m%d-%H%M%S)}"
for name in "$SOURCE_EXPERIMENT" "$NEW_EXPERIMENT_ID"; do
  if [[ ! "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Experiment IDs must be single directory names." >&2
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
    echo "Missing environment executable: $executable" >&2
    exit 2
  fi
done

# Resolve paths and validate all three sources/destinations before any writes.
WORK_ROOT="$("$PARENT_PYTHON" - "$CONFIG" "$SOURCE_EXPERIMENT" "$NEW_EXPERIMENT_ID" <<'PY'
import json
import sys
from pathlib import Path
from holypoly.core.config import load_config
from holypoly.core.result_paths import training_result_dir
from holypoly.training.loss_config import normalize_loss_config

config = load_config(Path(sys.argv[1]))
if config.paths.data_root is None or config.paths.work_root is None:
    raise ValueError("Configure data_root and work_root first")
data, work = (p.expanduser().resolve() for p in (config.paths.data_root, config.paths.work_root))
project = Path.cwd().resolve()
if not data.is_dir() or not work.is_dir():
    raise ValueError("Configured roots must exist")
if work.is_relative_to(data) or work.is_relative_to(project):
    raise ValueError("work_root must be external and outside raw data")
def safe(path):
    resolved = path.resolve()
    if (not resolved.is_relative_to(work) or resolved == work
            or resolved.is_relative_to(data) or resolved.is_relative_to(project)):
        raise ValueError(f"Unsafe artifact path: {resolved}")
    return resolved
for suffix in ("", "-mean", "-duration_only", "-mean_duration"):
    for category in ("models", "logs", "results/training_evaluation"):
        destination = safe(work / category / (sys.argv[3] + suffix))
        if destination.exists():
            raise FileExistsError(f"Choose a new NEW_EXPERIMENT_ID: {destination}")
manifests = set()
for mode in ("mean", "duration_only", "mean_duration"):
    run = sys.argv[2] + "-" + mode
    source_result = training_result_dir(work, run, resume=True)
    summary = json.loads(safe(source_result / "test-final/summary.json").read_text())
    expected = {"length_head_mode": mode, "hidden_features": 128, "dropout": 0.0}
    if (summary["epoch"] != 5 or summary["head_config"] != expected
            or summary["random_seed"] != 20260923 or summary["data_order_seed"] != 20260923):
        raise ValueError("This script expects the completed five-epoch comparison settings")
    if normalize_loss_config(summary["loss_config"]) != vars(config.auxiliary_loss):
        raise ValueError("Current loss settings differ from the source experiment")
    manifest = safe(Path(summary["manifest"]))
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifests.add(manifest)
    checkpoint = safe(work / "models" / run / "heads.epoch_0005.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    records = [json.loads(line) for line in safe(work / "logs" / run / "train.jsonl").read_text().splitlines()]
    if [(r["epoch"], r["split"]) for r in records] != [
        (epoch, split) for epoch in range(1, 6) for split in ("train", "validation")
    ]:
        raise ValueError("Expected complete source train/validation logs for epochs 1-5")
if len(manifests) != 1:
    raise ValueError("Source experiments do not use the same manifest")
print(work)
PY
)"

echo "Resume epochs 6-20 sequentially on GPU 0; previous results remain unchanged."
echo "Estimate from the small run: about 2.5-3 hours, approximately 14 GiB new checkpoints."
mkdir -p "$WORK_ROOT/logs/$NEW_EXPERIMENT_ID"
exec > >(tee -a "$WORK_ROOT/logs/$NEW_EXPERIMENT_ID/pipeline.log") 2>&1
trap 'code=$?; echo "Resume pipeline failed (exit $code) near line $LINENO." >&2; exit "$code"' ERR

for mode in mean duration_only mean_duration; do
  SOURCE_RUN="$SOURCE_EXPERIMENT-$mode"
  RUN_ID="$NEW_EXPERIMENT_ID-$mode"
  # Copy only the epoch-5 starting point; older snapshots remain in the source run.
  mkdir -p "$WORK_ROOT/models/$RUN_ID" "$WORK_ROOT/logs/$RUN_ID"
  cp -n "$WORK_ROOT/models/$SOURCE_RUN/heads.epoch_0005.pt" "$WORK_ROOT/models/$RUN_ID/heads.pt"
  cp -n "$WORK_ROOT/logs/$SOURCE_RUN/train.jsonl" "$WORK_ROOT/logs/$RUN_ID/train.jsonl"
  MANIFEST="$("$PARENT_PYTHON" - "$WORK_ROOT" "$SOURCE_RUN" <<'PY'
import json
import sys
from pathlib import Path
from holypoly.core.result_paths import training_result_dir
source = training_result_dir(Path(sys.argv[1]), sys.argv[2], resume=True)
print(json.loads((source / "test-final/summary.json").read_text())["manifest"])
PY
)"
  "$PROJECT_ROOT/.venv/bin/holypoly" train \
    --config "$CONFIG" --manifest "$MANIFEST" --length-head-mode "$mode" \
    --checkpoint "models/$RUN_ID/heads.pt" --loss-log "logs/$RUN_ID/train.jsonl" \
    --epochs 20 --resume --batch-size 32 --learning-rate 0.001 \
    --random-seed 20260923 --data-order-seed 20260923 \
    --chunk-size-samples 12000 --hidden-features 128 --dropout 0.0 \
    --weight-decay 0.0 --gradient-clip-norm 2.0 --num-workers 0 --device cuda:0

  BEST_CHECKPOINT="$("$PARENT_PYTHON" - "$WORK_ROOT" "$RUN_ID" "$SOURCE_RUN" <<'PY'
import json
import math
import sys
from pathlib import Path

work, run, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(line) for line in (work / "logs" / run / "train.jsonl").read_text().splitlines()]
validation = [r for r in rows if r.get("split") == "validation" and math.isfinite(r["length_mae_nt"])]
best = min(validation, key=lambda r: (r["length_mae_nt"], r["epoch"]))
epoch = best["epoch"]
checkpoint = work / "models" / (source if epoch <= 5 else run) / f"heads.epoch_{epoch:04d}.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
output = work / "results" / "training_evaluation" / run
output.mkdir(parents=True)
(output / "best_checkpoint.json").write_text(json.dumps({
    "selection_rule": "minimum validation length MAE, earliest epoch on ties",
    "epoch": epoch, "validation_mae_nt": best["length_mae_nt"],
    "checkpoint": str(checkpoint),
}, indent=2) + "\n")
print(checkpoint)
PY
)"
  "$BONITO_PYTHON" -m holypoly.training.evaluate \
    --config "$CONFIG" --manifest "$MANIFEST" --checkpoint "$BEST_CHECKPOINT" \
    --output-dir "results/training_evaluation/$RUN_ID/test-best-validation" \
    --split test --batch-size 32 --max-reads 0
  "$PARENT_PYTHON" scripts/plot_training.py \
    --config "$CONFIG" --log "logs/$RUN_ID/train.jsonl" \
    --output "results/training_evaluation/$RUN_ID/training-plots"
  "$PARENT_PYTHON" scripts/plot_training.py \
    --config "$CONFIG" --predictions "results/training_evaluation/$RUN_ID/test-best-validation/predictions.tsv" \
    --supervision-mode length_only --seed 20260923 \
    --output "results/training_evaluation/$RUN_ID/test-best-validation/plots"
done
echo "Finished: $NEW_EXPERIMENT_ID"
echo "Per-mode results: $WORK_ROOT/results/training_evaluation/$NEW_EXPERIMENT_ID-<mode>/"
