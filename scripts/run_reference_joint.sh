#!/usr/bin/env bash
# Compatibility entry point for existing joint training commands.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_joint_training.sh" "$@"
