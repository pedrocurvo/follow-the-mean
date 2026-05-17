#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 experiments/<name>.yaml [--dry-run]"
  echo "Available experiments:"
  find "${ROOT_DIR}/experiments" -maxdepth 1 -name '*.yaml' -printf '  experiments/%f\n' | sort
  exit 1
fi

python "${ROOT_DIR}/scripts/run_experiment.py" "$@"
