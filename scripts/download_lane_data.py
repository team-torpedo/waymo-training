"""Download and preprocess only the lane segmentation data (camera_image + camera_segmentation)."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waymo_training.config import load_config
from waymo_training.data.download import _list_gcs_files
from waymo_training.data.preprocess import preprocess_lane
from waymo_training.utils import ensure_dir, get_logger

log = get_logger("download_lane_data")


def get_available_segments(bucket: str, split_folder: str, components: list[str], tool: str = "gsutil") -> list[str]:
    sets = []
    for comp in components:
        prefix = f"{bucket}/{split_folder}/{comp}"
        files = _list_gcs_files(prefix, tool=tool)
        sets.append({Path(f).stem for f in files})
    return sorted(list(set.intersection(*sets)))


def get_already_processed_segments(data_dir: Path, split_key: str) -> set[str]:
    img_dir = data_dir / "processed" / "lane" / split_key / "images"
    if not img_dir.exists():
        return set()
    return {f.name.split("_")[0] for f in img_dir.glob("*.jpg")}


def download_segment_batch(bucket: str, split_folder: str, components: list[str], segments: list[str], raw_root: Path, tool: str = "gsutil"):
    for comp in components:
        comp_dir = ensure_dir(raw_root / split_folder / comp)
        for seg in segments:
            fname = f"{seg}.parquet"
            local_path = comp_dir / fname
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            gcs_uri = f"{bucket}/{split_folder}/{comp}/{fname}"
            log.info("Downloading %s/%s/%s...", split_folder, comp, fname)
            cmd = [tool, "cp", gcs_uri, str(local_path)]
            subprocess.run(cmd, check=True)


def check_free_gb() -> float:
    stat = os.statvfs(".")
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


def get_dir_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess lane dataset only.")
    parser.add_argument("--train-segments", type=int, default=25, help="Number of training segments (default: 25)")
    parser.add_argument("--val-segments", type=int, default=7, help="Number of validation segments (default: 7)")
    parser.add_argument("--batch-size", type=int, default=5, help="Download batch size (default: 5)")
    args = parser.parse_args()

    cfg = load_config()
    bucket = str(cfg.dataset.gcs_bucket).rstrip("/")
    tool = str(cfg.get("dataset.download.tool", "gsutil"))
    # Only camera_image and camera_segmentation are needed for lane detection
    components = ["camera_image", "camera_segmentation"]
    raw_root = cfg.path("paths.raw_dir")
    data_root = cfg.path("paths.data_root")

    targets = {"train": args.train_segments, "val": args.val_segments}

    log.info("Lane Data Downloader: Targeting %d train and %d val segments...", targets["train"], targets["val"])

    for split_key, split_folder in cfg.dataset.splits.items():
        target_count = targets[split_key]
        processed = get_already_processed_segments(data_root, split_key)
        log.info("Split '%s': %d segments currently in data/processed/lane (target: %d)",
                 split_key, len(processed), target_count)

        if len(processed) >= target_count:
            log.info("Split '%s' already has %d segments. Skipping.", split_key, len(processed))
            continue

        available = get_available_segments(bucket, split_folder, components, tool=tool)
        needed_segs = [s for s in available[:target_count] if s not in processed]
        log.info("Split '%s': Need to download and process %d segments.", split_key, len(needed_segs))

        for i in range(0, len(needed_segs), args.batch_size):
            chunk = needed_segs[i : i + args.batch_size]
            free_space = check_free_gb()
            log.info("--- [%s] Batch %d/%d (%d segments). Free disk space: %.1f GB ---",
                     split_key, i // args.batch_size + 1, (len(needed_segs) + args.batch_size - 1) // args.batch_size, len(chunk), free_space)

            if free_space < 3.0:
                log.error("Low disk space (< 3.0 GB). Halting safely.")
                return

            # 1. Download chunk (only camera_image + camera_segmentation)
            download_segment_batch(bucket, split_folder, components, chunk, raw_root, tool=tool)

            # 2. Preprocess lane chunk
            log.info("Preprocessing lane image/mask pairs...")
            preprocess_lane(cfg)

            # 3. Delete raw parquets immediately to preserve disk space
            log.info("Cleaning raw parquet chunk to reclaim disk space...")
            shutil.rmtree(raw_root / split_folder, ignore_errors=True)

            size_mb = get_dir_size_mb(data_root / "processed" / "lane")
            log.info("Batch done. Lane processed data size: %.1f MB. Free disk space: %.1f GB",
                     size_mb, check_free_gb())

    final_size_mb = get_dir_size_mb(data_root / "processed" / "lane")
    log.info("=== Lane data download complete! Total lane dataset size: %.1f MB ===", final_size_mb)
    log.info("To start lane training: python stages/train_lane.py --set lane.train.weights=outputs/lane/lane_seg-20261106-023223/best.keras")


if __name__ == "__main__":
    main()
