"""Slow end-to-end YOLO check. Opt in with:  WAYMO_TEST_YOLO=1 pytest tests/test_yolo_smoke.py"""

import os

import pytest

pytest.importorskip("ultralytics")
if not os.environ.get("WAYMO_TEST_YOLO"):
    pytest.skip("set WAYMO_TEST_YOLO=1 to run the slow YOLO smoke test", allow_module_level=True)

from waymo_training.config import load_config  # noqa: E402
from waymo_training.data.preprocess import preprocess_detection  # noqa: E402
from waymo_training.detection import DETECTORS  # noqa: E402


def test_train_evaluate(synth_cfg):
    cfg = load_config(overrides=[
        f"paths.data_root={synth_cfg.path('paths.data_root')}",
        f"paths.outputs_dir={synth_cfg.path('paths.outputs_dir')}",
        "dataset.cameras=[FRONT]",
        "detection.model.weights=yolo11n.yaml",   # train from scratch: no weight download
        "detection.train.epochs=1",
        "detection.train.imgsz=64",
        "detection.train.batch=2",
        "detection.train.device=cpu",
        "detection.train.workers=0",
        "detection.train.amp=false",
        "detection.train.extra={plots: false}",
        "evaluation.detection.num_visualizations=2",
    ])
    preprocess_detection(cfg)
    det = DETECTORS.build(cfg.detection.backend, cfg)
    weights = det.train()
    assert weights.exists()
    metrics = det.evaluate()
    assert "mAP50-95" in metrics
    assert det.visualize() is not None
