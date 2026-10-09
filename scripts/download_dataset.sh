#!/usr/bin/env bash
# Convenience wrapper around the download stage. Extra args go to the Python CLI.
#   bash scripts/download_dataset.sh --set dataset.download.max_segments.train=50
set -euo pipefail
cd "$(dirname "$0")/.."
exec venv/bin/python stages/download.py "$@"
