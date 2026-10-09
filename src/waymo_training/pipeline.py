"""Stage runners and the full pipeline. Heavy imports (TensorFlow, Ultralytics) are lazy."""

from __future__ import annotations

import time
from typing import Callable

from .config import Config
from .utils import enabled_tasks, ensure_dir, get_logger, set_seed, write_json

log = get_logger(__name__)


def stage_download(cfg: Config) -> None:
    from .data.download import run_download

    run_download(cfg)


def stage_preprocess(cfg: Config) -> None:
    from .data.preprocess import run_preprocess

    run_preprocess(cfg)


def stage_train_detection(cfg: Config) -> None:
    if not enabled_tasks(cfg)["detection"]:
        log.info("Detection task disabled; skipping train_detection")
        return
    from .detection import DETECTORS

    DETECTORS.build(cfg.detection.backend, cfg).train()


def stage_train_lane(cfg: Config) -> None:
    if not enabled_tasks(cfg)["lane"]:
        log.info("Lane task disabled; skipping train_lane")
        return
    from .lane.train import train_lane

    train_lane(cfg)


def stage_evaluate(cfg: Config) -> dict:
    tasks = enabled_tasks(cfg)
    results: dict = {}
    if tasks["detection"]:
        from .detection import DETECTORS

        det = DETECTORS.build(cfg.detection.backend, cfg)
        weights = cfg.get("evaluation.detection.weights")
        results["detection"] = det.evaluate(weights)
        det.visualize(weights)
    if tasks["lane"]:
        from .lane.evaluate import evaluate_lane

        results["lane"] = evaluate_lane(cfg)
    write_json(ensure_dir(cfg.path("paths.eval_dir")) / "summary.json", results)
    return results


STAGES: dict[str, Callable[[Config], object]] = {
    "download": stage_download,
    "preprocess": stage_preprocess,
    "train_detection": stage_train_detection,
    "train_lane": stage_train_lane,
    "evaluate": stage_evaluate,
}


def run_stages(cfg: Config, stages: list[str]) -> None:
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}. Available: {list(STAGES)}")
    set_seed(int(cfg.project.seed))
    for name in stages:
        log.info("=== stage: %s ===", name)
        t0 = time.time()
        STAGES[name](cfg)
        log.info("=== stage %s done in %.1fs ===", name, time.time() - t0)


def run_pipeline(cfg: Config) -> None:
    run_stages(cfg, [str(s) for s in cfg.pipeline.stages])
