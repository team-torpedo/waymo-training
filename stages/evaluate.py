"""Evaluate the trained detection and lane models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waymo_training.cli import stage_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(stage_main("evaluate", "Evaluate the trained detection and lane models."))
