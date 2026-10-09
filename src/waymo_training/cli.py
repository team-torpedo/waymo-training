"""Shared command line handling for main.py and the per-stage scripts."""

from __future__ import annotations

import argparse
import sys

from .config import Config, load_config
from .pipeline import run_stages
from .utils import setup_logging


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config-dir", default=None, help="directory of YAML configs (default: ./config)")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set lane.train.epochs=5 (repeatable)",
    )
    return p


def load_from_args(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config_dir, args.overrides)
    setup_logging(cfg.project.log_level)
    return cfg


def stage_main(stage: str, description: str) -> int:
    args = build_parser(description).parse_args()
    cfg = load_from_args(args)
    run_stages(cfg, [stage])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit("Use main.py or the scripts in stages/.")
