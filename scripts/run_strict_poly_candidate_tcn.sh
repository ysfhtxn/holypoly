#!/usr/bin/env bash
# Compatibility entry point for strict-poly-candidate-v1 training.
set -euo pipefail
LAUNCHER="$(dirname -- "${BASH_SOURCE[0]}")/run_joint_training.sh"
if [[ $# -eq 0 || "$1" == --help ]]; then exec bash "$LAUNCHER" --help; fi
HOLYPOLY_EXPECT_DATASET_WORKFLOW=strict-poly-candidate-v1 \
  exec bash "$LAUNCHER" --dataset-dir "$@"
