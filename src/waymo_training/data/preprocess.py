"""Preprocessing: Convert Waymo v2 parquet into YOLO detection and Lane segmentation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

from ..config import Config
from ..utils import camera_ids, enabled_tasks, ensure_dir, get_logger, write_json

log = get_logger(__name__)


def _find_col(columns: list[str], suffix: str) -> str:
    for c in columns:
        if c == suffix or c.endswith("." + suffix) or c.endswith("]" + suffix):
            return c
    for c in columns:
        if c.endswith(suffix):
            return c
    raise KeyError(f"Column ending with '{suffix}' not found in {columns}")


def _load_parquet(path: Path) -> dict[str, list]:
    table = pq.read_table(path)
    return {col: table[col].to_pylist() for col in table.column_names}


def preprocess_detection(cfg: Config) -> dict[str, dict[str, int]]:
    raw_root = cfg.path("paths.raw_dir")
    yolo_dir = ensure_dir(cfg.path("paths.yolo_dir"))
    splits = cfg.dataset.splits
    allowed_cams = set(camera_ids(list(cfg.dataset.cameras)))

    # YOLO class mapping
    classes_dict = cfg.detection.classes.to_dict() if hasattr(cfg.detection.classes, "to_dict") else dict(cfg.detection.classes)
    class_map = {int(k): idx for idx, k in enumerate(classes_dict.keys())}
    class_names = {idx: str(name) for idx, name in enumerate(classes_dict.values())}

    req_labels = bool(cfg.get("dataset.preprocess.detection.require_labels", True))
    min_box_px = float(cfg.get("dataset.preprocess.detection.min_box_px", 2))

    stats: dict[str, dict[str, int]] = {}

    for split_key, split_folder in splits.items():
        img_out = ensure_dir(yolo_dir / "images" / split_key)
        lbl_out = ensure_dir(yolo_dir / "labels" / split_key)

        img_dir = raw_root / split_folder / "camera_image"
        box_dir = raw_root / split_folder / "camera_box"

        total_images = 0
        total_boxes = 0

        if not img_dir.exists():
            log.warning("Image directory %s does not exist; skipping detection split %s", img_dir, split_key)
            stats[split_key] = {"images": 0, "boxes": 0}
            continue

        # Process segment by segment
        img_files = sorted(img_dir.glob("*.parquet"))
        for img_p in img_files:
            seg_name = img_p.stem
            box_p = box_dir / f"{seg_name}.parquet"
            box_data = _load_parquet(box_p) if box_p.exists() else {}

            # Index boxes by (segment, timestamp, camera)
            boxes_by_frame: dict[tuple[str, int, int], list[tuple[int, float, float, float, float]]] = {}
            if box_data:
                b_cols = list(box_data.keys())
                seg_c = _find_col(b_cols, "segment_context_name")
                ts_c = _find_col(b_cols, "frame_timestamp_micros")
                cam_c = _find_col(b_cols, "camera_name")
                cx_c = _find_col(b_cols, "box.center.x")
                cy_c = _find_col(b_cols, "box.center.y")
                sx_c = _find_col(b_cols, "box.size.x")
                sy_c = _find_col(b_cols, "box.size.y")
                type_c = _find_col(b_cols, "type")

                n_boxes = len(box_data[seg_c])
                for i in range(n_boxes):
                    cam = int(box_data[cam_c][i])
                    if cam not in allowed_cams:
                        continue
                    b_type = int(box_data[type_c][i])
                    if b_type not in class_map:
                        continue
                    sx = float(box_data[sx_c][i])
                    sy = float(box_data[sy_c][i])
                    if sx < min_box_px or sy < min_box_px:
                        continue
                    cx = float(box_data[cx_c][i])
                    cy = float(box_data[cy_c][i])
                    k = (str(box_data[seg_c][i]), int(box_data[ts_c][i]), cam)
                    boxes_by_frame.setdefault(k, []).append((class_map[b_type], cx, cy, sx, sy))

            # Load images
            img_data = _load_parquet(img_p)
            i_cols = list(img_data.keys())
            i_seg_c = _find_col(i_cols, "segment_context_name")
            i_ts_c = _find_col(i_cols, "frame_timestamp_micros")
            i_cam_c = _find_col(i_cols, "camera_name")
            i_img_c = _find_col(i_cols, "image")

            n_imgs = len(img_data[i_seg_c])
            for i in range(n_imgs):
                cam = int(img_data[i_cam_c][i])
                if cam not in allowed_cams:
                    continue
                seg = str(img_data[i_seg_c][i])
                ts = int(img_data[i_ts_c][i])
                k = (seg, ts, cam)

                frame_boxes = boxes_by_frame.get(k, [])
                if req_labels and not frame_boxes:
                    continue

                file_stem = f"{seg}_{ts}_{cam}"
                out_img_path = img_out / f"{file_stem}.jpg"
                out_lbl_path = lbl_out / f"{file_stem}.txt"
                if out_img_path.exists() and out_lbl_path.exists():
                    total_images += 1
                    continue

                raw_bytes = img_data[i_img_c][i]
                # Decode image to get dimensions
                np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]

                # Save image
                with open(out_img_path, "wb") as fh:
                    fh.write(raw_bytes)

                # Save YOLO label txt
                lbl_lines = []
                for cls_idx, cx, cy, sx, sy in frame_boxes:
                    norm_cx = max(0.0, min(1.0, cx / w))
                    norm_cy = max(0.0, min(1.0, cy / h))
                    norm_w = max(0.0, min(1.0, sx / w))
                    norm_h = max(0.0, min(1.0, sy / h))
                    lbl_lines.append(f"{cls_idx} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

                out_lbl_path.write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""))

                total_images += 1
                total_boxes += len(lbl_lines)

        stats[split_key] = {"images": total_images, "boxes": total_boxes}
        log.info("Detection %s: %d images, %d boxes", split_key, total_images, total_boxes)

    # Write data.yaml for YOLO
    yolo_yaml = {
        "path": str(yolo_dir.resolve()),
        "train": f"images/{list(splits.keys())[0]}",
        "val": f"images/{list(splits.keys())[1]}",
        "names": class_names,
    }
    (yolo_dir / "data.yaml").write_text(yaml.safe_dump(yolo_yaml, sort_keys=False))
    return stats


def preprocess_lane(cfg: Config) -> dict[str, dict[str, int]]:
    raw_root = cfg.path("paths.raw_dir")
    lane_dir = ensure_dir(cfg.path("paths.lane_dir"))
    splits = cfg.dataset.splits
    allowed_cams = set(camera_ids(list(cfg.dataset.cameras)))
    semantic_classes = set(int(c) for c in cfg.lane.semantic_classes)
    min_lane_fraction = float(cfg.get("dataset.preprocess.lane.min_lane_fraction", 0.0))

    stats: dict[str, dict[str, int]] = {}

    for split_key, split_folder in splits.items():
        img_out = ensure_dir(lane_dir / split_key / "images")
        mask_out = ensure_dir(lane_dir / split_key / "masks")

        img_dir = raw_root / split_folder / "camera_image"
        seg_dir = raw_root / split_folder / "camera_segmentation"

        total_pairs = 0

        if not seg_dir.exists() or not img_dir.exists():
            log.warning("Missing images/segmentation for split %s; skipping lane", split_key)
            stats[split_key] = {"pairs": 0}
            continue

        seg_files = sorted(seg_dir.glob("*.parquet"))
        for seg_p in seg_files:
            seg_name = seg_p.stem
            img_p = img_dir / f"{seg_name}.parquet"
            if not img_p.exists():
                continue

            seg_data = _load_parquet(seg_p)
            s_cols = list(seg_data.keys())
            s_seg_c = _find_col(s_cols, "segment_context_name")
            s_ts_c = _find_col(s_cols, "frame_timestamp_micros")
            s_cam_c = _find_col(s_cols, "camera_name")
            s_div_c = _find_col(s_cols, "panoptic_label_divisor")
            s_label_c = _find_col(s_cols, "panoptic_label")

            # Index segmentation by (seg, ts, cam)
            seg_by_frame: dict[tuple[str, int, int], tuple[int, bytes]] = {}
            for i in range(len(seg_data[s_seg_c])):
                cam = int(seg_data[s_cam_c][i])
                if cam not in allowed_cams:
                    continue
                k = (str(seg_data[s_seg_c][i]), int(seg_data[s_ts_c][i]), cam)
                seg_by_frame[k] = (int(seg_data[s_div_c][i]), seg_data[s_label_c][i])

            # Load images
            img_data = _load_parquet(img_p)
            i_cols = list(img_data.keys())
            i_seg_c = _find_col(i_cols, "segment_context_name")
            i_ts_c = _find_col(i_cols, "frame_timestamp_micros")
            i_cam_c = _find_col(i_cols, "camera_name")
            i_img_c = _find_col(i_cols, "image")

            for i in range(len(img_data[i_seg_c])):
                cam = int(img_data[i_cam_c][i])
                if cam not in allowed_cams:
                    continue
                seg = str(img_data[i_seg_c][i])
                ts = int(img_data[i_ts_c][i])
                k = (seg, ts, cam)
                if k not in seg_by_frame:
                    continue

                file_stem = f"{seg}_{ts}_{cam}"
                out_img_path = img_out / f"{file_stem}.jpg"
                out_mask_path = mask_out / f"{file_stem}.png"
                if out_img_path.exists() and out_mask_path.exists():
                    total_pairs += 1
                    continue

                divisor, panoptic_bytes = seg_by_frame[k]
                panoptic_png = cv2.imdecode(np.frombuffer(panoptic_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if panoptic_png is None:
                    continue

                semantic_id = panoptic_png // divisor
                mask = np.isin(semantic_id, list(semantic_classes)).astype(np.uint8) * 255

                if min_lane_fraction > 0.0:
                    lane_fraction = (mask == 255).mean()
                    if lane_fraction < min_lane_fraction:
                        continue

                raw_bytes = img_data[i_img_c][i]

                # Write image and mask
                with open(out_img_path, "wb") as fh:
                    fh.write(raw_bytes)
                cv2.imwrite(str(out_mask_path), mask)

                total_pairs += 1

        stats[split_key] = {"pairs": total_pairs}
        log.info("Lane %s: %d pairs", split_key, total_pairs)

    return stats


def run_preprocess(cfg: Config) -> dict[str, Any]:
    tasks = enabled_tasks(cfg)
    summary: dict[str, Any] = {}
    if tasks["detection"]:
        summary["detection"] = preprocess_detection(cfg)
    if tasks["lane"]:
        summary["lane"] = preprocess_lane(cfg)
    write_json(ensure_dir(cfg.path("paths.processed_dir")) / "manifest.json", summary)
    return summary
