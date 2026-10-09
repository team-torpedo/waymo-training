#!/usr/bin/env bash
# Run the full pipeline inside the venv. Extra args go to main.py.
#   bash scripts/run_pipeline.sh --skip download
set -euo pipefail
cd "$(dirname "$0")/.."
exec venv/bin/python main.py "$@"
