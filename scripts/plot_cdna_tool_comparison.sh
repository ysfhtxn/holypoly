#!/usr/bin/env bash
# Plot completed poly(A) results only; never rerun inference or basecalling.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/plot_cdna_tool_comparison.sh <holypoly-run-dir> <dorado-run-dir> [sample-name] [config]
Run-directory arguments are relative to work_root or absolute work-root paths.
Requires completed Holypoly prediction/predictions.tsv and prediction/summary.json.
Creates Holypoly-only and Holypoly/Dorado overlaid distributions in a NEW directory.
Reads the Dorado BAM once, on CPU; no models, POD5 signals, GPU or installation.
Optional: BIN_WIDTH_NT=5
HELP
  if [[ "${1:-}" == "--help" ]]; then exit 0; fi
  exit 2
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
HOLYPOLY_RUN="${1%/}"
DORADO_RUN="${2%/}"
SAMPLE_NAME="${3:-cdna}"
CONFIG="${4:-configs/local.yaml}"
if [[ ! "$SAMPLE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'sample-name must use letters, digits, dots, underscores or hyphens' >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
WORK_ROOT="$("$PROJECT_ROOT/.venv/bin/python" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from holypoly.core.config import load_config
root = load_config(Path(sys.argv[1])).paths.work_root
if root is None:
    raise ValueError('configure work_root')
print(root.resolve())
PY
)"
OUTPUT="results/tool_comparison/${SAMPLE_NAME}-holypoly-dorado-$(date +%Y%m%d-%H%M%S)-$$"
export MPLCONFIGDIR="$WORK_ROOT/cache/cdna-tool-comparison/matplotlib"
echo 'CPU-only extraction: one full Dorado BAM pass; expect minutes, 1-2 GiB RAM.'
echo 'Extracted tables and figures may use roughly 50-200 MiB for 700k reads.'
echo "New output: $WORK_ROOT/$OUTPUT"
"$PROJECT_ROOT/.venv/bin/python" scripts/compare_cdna_tools.py \
  --config "$CONFIG" --holypoly "$HOLYPOLY_RUN/prediction/predictions.tsv" \
  --dorado-bam "$DORADO_RUN/basecalls.bam" \
  --sample-name "$SAMPLE_NAME" --bin-width-nt "${BIN_WIDTH_NT:-5}" --output "$OUTPUT"
