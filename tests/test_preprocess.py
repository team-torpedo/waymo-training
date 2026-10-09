import cv2
import yaml

from waymo_training.data.preprocess import preprocess_detection, preprocess_lane


def test_detection_conversion(synth_cfg):
    stats = preprocess_detection(synth_cfg)
    # frames 0 and 1 have labels; frame 2/3 are unlabeled and skipped (require_labels: true)
    assert stats["train"]["images"] == 2 * 2
    assert stats["val"]["images"] == 2
    yolo = synth_cfg.path("paths.yolo_dir")
    data = yaml.safe_load((yolo / "data.yaml").read_text())
    assert data["names"] == {0: "vehicle", 1: "pedestrian", 2: "sign", 3: "cyclist"}
    labels = sorted((yolo / "labels" / "train").glob("*.txt"))
    first = [ln.split() for ln in labels[0].read_text().strip().splitlines()]
    assert len(first) == 2
    cls, cx, cy, w, h = first[0]
    assert cls == "0"
    assert abs(float(cx) - 50 / 192) < 1e-4 and abs(float(h) - 30 / 128) < 1e-4


def test_detection_class_filtering(synth_cfg):
    from waymo_training.config import load_config

    cfg = load_config(overrides=[
        f"paths.data_root={synth_cfg.path('paths.data_root')}",
        f"paths.outputs_dir={synth_cfg.path('paths.outputs_dir')}",
        "dataset.cameras=[FRONT]",
        "detection.classes={1: vehicle, 4: cyclist}",
    ])
    preprocess_detection(cfg)
    data = yaml.safe_load((cfg.path("paths.yolo_dir") / "data.yaml").read_text())
    assert data["names"] == {0: "vehicle", 1: "cyclist"}


def test_lane_masks(synth_cfg):
    stats = preprocess_lane(synth_cfg)
    assert stats["train"]["pairs"] == 8 and stats["val"]["pairs"] == 4
    lane = synth_cfg.path("paths.lane_dir")
    mask_path = sorted((lane / "train" / "masks").glob("*.png"))[0]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    assert set(mask.flatten().tolist()) == {0, 255}
    assert (mask == 255).sum() == 8 * mask.shape[0]  # 8 px wide stripe
