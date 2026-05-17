#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPFM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SPFM_DIR}" || exit 1

# Choose experiment config with first arg, defaulting to model.
EXP_CONFIG="${1:-experiments/dit.yaml}"
echo "[train.sh] Using config: ${EXP_CONFIG}"

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" run_experiment.py --config "${EXP_CONFIG}"
