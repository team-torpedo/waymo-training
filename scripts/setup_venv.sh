#!/usr/bin/env bash
# Create the "venv" virtual environment and install everything.
#
#   bash scripts/setup_venv.sh            # CPU (or macOS)
#   bash scripts/setup_venv.sh --gpu      # Linux + NVIDIA GPU (tensorflow[and-cuda])
#   bash scripts/setup_venv.sh --dev      # also install pytest + ruff
#   PYTHON=python3.11 bash scripts/setup_venv.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV_DIR="venv"
GPU=0
DEV=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    --dev) DEV=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON not found. Install Python 3.10-3.12 or set PYTHON=/path/to/python." >&2
  exit 1
fi

PYV="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PYV" in
  3.10|3.11|3.12) ;;
  *) echo "WARNING: Python $PYV detected. TensorFlow/PyTorch wheels are best supported on 3.10-3.12." >&2 ;;
esac

if [ ! -d "$VENV_DIR" ]; then
  echo ">> Creating virtual environment in ./$VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo ">> Reusing existing ./$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [ "$GPU" -eq 1 ]; then
  echo ">> Installing GPU build of TensorFlow"
  # Same as requirements.txt but with the CUDA extra for TensorFlow
  sed 's/^tensorflow>=/tensorflow[and-cuda]>=/' requirements.txt > .requirements.gpu.tmp
  python -m pip install -r .requirements.gpu.tmp
  rm -f .requirements.gpu.tmp
else
  python -m pip install -r requirements.txt
fi

if [ "$DEV" -eq 1 ]; then
  python -m pip install -r requirements-dev.txt
fi

python -m pip install -e . --no-deps

echo
echo ">> Done. Activate with:  source $VENV_DIR/bin/activate"
python - <<'PY'
try:
    import tensorflow as tf
    print("TensorFlow", tf.__version__, "GPUs:", len(tf.config.list_physical_devices("GPU")))
except Exception as exc:  # noqa: BLE001
    print("TensorFlow import failed:", exc)
try:
    import torch
    print("PyTorch", torch.__version__, "CUDA:", torch.cuda.is_available())
except Exception as exc:  # noqa: BLE001
    print("PyTorch import failed:", exc)
PY
