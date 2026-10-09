"""Evaluate a trained lane model: IoU / Dice / precision / recall / accuracy + overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ..config import Config
from ..utils import ensure_dir, get_logger, write_json
from .dataset import build_dataset
from .train import latest_model_path

log = get_logger(__name__)


def _overlay(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = 0.5 * out[m] + 0.5 * np.array(color, np.float32)
    return out.astype(np.uint8)


def evaluate_lane(cfg: Config, model_path: str | Path | None = None) -> dict:
    ev = cfg.evaluation.lane
    model_path = Path(model_path or ev.get("model_path") or latest_model_path(cfg))
    split = str(cfg.get("evaluation.split", "val"))
    thr = float(ev.threshold)
    out_dir = ensure_dir(cfg.path("paths.eval_dir") / "lane")
    vis_dir = ensure_dir(out_dir / "visualizations")
    n_vis = int(ev.get("num_visualizations", 0))

    model = tf.keras.models.load_model(str(model_path), compile=False)
    ds, n_total = build_dataset(cfg, split, training=False)
    stride = max(1, n_total // n_vis) if n_vis > 0 else 0

    tp = fp = fn = tn = 0
    seen = saved = 0
    for x, y in ds:
        pred = model(x, training=False).numpy()[..., 0] > thr
        gt = y.numpy()[..., 0] > 0.5
        tp += int(np.logical_and(pred, gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred, gt).sum())
        tn += int(np.logical_and(~pred, ~gt).sum())
        if stride:
            for i in range(x.shape[0]):
                if (seen + i) % stride == 0 and saved < n_vis:
                    img = x[i].numpy().astype(np.uint8)
                    panel = np.concatenate(
                        [img, _overlay(img, gt[i], (0, 255, 0)), _overlay(img, pred[i], (255, 0, 0))],
                        axis=1,
                    )
                    cv2.imwrite(str(vis_dir / f"{saved:03d}.jpg"), panel[..., ::-1])
                    saved += 1
        seen += x.shape[0]

    eps = 1e-9
    metrics = {
        "model": str(model_path),
        "split": split,
        "images": n_total,
        "threshold": thr,
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }
    write_json(out_dir / "metrics.json", metrics)
    log.info("Lane metrics: %s", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    return metrics
