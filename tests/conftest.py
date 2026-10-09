"""Synthetic Waymo-v2-shaped parquet fixtures so the pipeline can be tested without the dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waymo_training.config import load_config  # noqa: E402

H, W = 128, 192
LANE_ID = 21


def _frame(i: int):
    img = np.full((H, W, 3), 80, np.uint8)
    img[:, :, 1] = (np.arange(W)[None, :] * 255 // W).astype(np.uint8)
    x = 40 + 10 * i
    img[:, x : x + 8] = 255  # a bright "lane" stripe
    sem = np.full((H, W), 20, np.uint16)  # road
    sem[:, x : x + 8] = LANE_ID
    return img, sem


def make_segment(base: Path, split_folder: str, segment: str, n_frames: int = 3) -> None:
    ts_col, cam_col, seg_col = [], [], []
    images, panoptic = [], []
    for i in range(n_frames):
        img, sem = _frame(i)
        ok, jpg = cv2.imencode(".jpg", img)
        ok2, png = cv2.imencode(".png", (sem.astype(np.uint16) * 1000))
        assert ok and ok2
        ts_col.append(1_000_000 + i)
        cam_col.append(1)  # FRONT
        seg_col.append(segment)
        images.append(jpg.tobytes())
        panoptic.append(png.tobytes())

    key = {
        "key.segment_context_name": seg_col,
        "key.frame_timestamp_micros": ts_col,
        "key.camera_name": cam_col,
    }
    for comp in ("camera_image", "camera_box", "camera_segmentation"):
        (base / split_folder / comp).mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.table({**key, "[CameraImageComponent].image": images}),
        base / split_folder / "camera_image" / f"{segment}.parquet",
    )
    # camera_box: one row per box; frame 0 gets two boxes, frame 1 one, frame 2 none (unlabeled)
    rows = [(0, 1, 50.0, 60.0, 20.0, 30.0), (0, 2, 100.0, 70.0, 10.0, 25.0), (1, 4, 30.0, 40.0, 16.0, 16.0)]
    pq.write_table(
        pa.table(
            {
                "key.segment_context_name": [segment] * len(rows),
                "key.frame_timestamp_micros": [1_000_000 + r[0] for r in rows],
                "key.camera_name": [1] * len(rows),
                "key.camera_object_id": [f"o{k}" for k in range(len(rows))],
                "[CameraBoxComponent].box.center.x": [r[2] for r in rows],
                "[CameraBoxComponent].box.center.y": [r[3] for r in rows],
                "[CameraBoxComponent].box.size.x": [r[4] for r in rows],
                "[CameraBoxComponent].box.size.y": [r[5] for r in rows],
                "[CameraBoxComponent].type": [r[1] for r in rows],
            }
        ),
        base / split_folder / "camera_box" / f"{segment}.parquet",
    )
    pq.write_table(
        pa.table(
            {
                **key,
                "[CameraSegmentationLabelComponent].panoptic_label_divisor": [1000] * n_frames,
                "[CameraSegmentationLabelComponent].panoptic_label": panoptic,
            }
        ),
        base / split_folder / "camera_segmentation" / f"{segment}.parquet",
    )


@pytest.fixture()
def synth_cfg(tmp_path):
    raw = tmp_path / "data" / "raw"
    for seg in ("segA", "segB"):
        make_segment(raw, "training", seg, n_frames=4)
    make_segment(raw, "validation", "segC", n_frames=4)
    overrides = [
        f"paths.data_root={tmp_path / 'data'}",
        f"paths.outputs_dir={tmp_path / 'outputs'}",
        "dataset.cameras=[FRONT]",
        "lane.data.image_size=[64, 96]",
        "lane.data.batch_size=2",
        "lane.train.epochs=1",
        "lane.model.params.pretrained=null",
        "evaluation.lane.num_visualizations=2",
    ]
    return load_config(overrides=overrides)
