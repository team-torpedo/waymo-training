#!/usr/bin/env bash
# Run the full perception pipeline inside the detected environment.
# Usage:
#   bash scripts/run_pipeline.sh [OPTIONS]
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./run_project.sh "$@"
