"""Small shared helpers."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

CAMERA_IDS = {"FRONT": 1, "FRONT_LEFT": 2, "FRONT_RIGHT": 3, "SIDE_LEFT": 4, "SIDE_RIGHT": 5}
CAMERA_NAMES = {v: k for k, v in CAMERA_IDS.items()}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def write_json(path: str | Path, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def camera_ids(names: list[str]) -> list[int]:
    try:
        return [CAMERA_IDS[n.upper()] for n in names]
    except KeyError as exc:
        raise ValueError(f"Unknown camera {exc}. Valid: {sorted(CAMERA_IDS)}") from None


def enabled_tasks(cfg) -> dict[str, bool]:
    return {
        "detection": bool(cfg.get("pipeline.tasks.detection", True)),
        "lane": bool(cfg.get("pipeline.tasks.lane", True)),
    }
