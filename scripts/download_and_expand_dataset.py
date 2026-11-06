"""Safely expand the dataset to ~10 GB in small batches to preserve disk space."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waymo_training.config import load_config
from waymo_training.data.download import _list_gcs_files, get_required_components
from waymo_training.data.preprocess import run_preprocess
from waymo_training.utils import ensure_dir, get_logger

log = get_logger("expand_dataset")


def get_available_segments(bucket: str, split_folder: str, components: list[str], tool: str = "gsutil") -> list[str]:
    sets = []
    for comp in components:
        prefix = f"{bucket}/{split_folder}/{comp}"
        files = _list_gcs_files(prefix, tool=tool)
        sets.append({Path(f).stem for f in files})
    return sorted(list(set.intersection(*sets)))


def get_already_processed_segments(data_dir: Path, split_key: str) -> set[str]:
    img_dir = data_dir / "processed" / "yolo" / "images" / split_key
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


def get_dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def main():
    cfg = load_config()
    bucket = str(cfg.dataset.gcs_bucket).rstrip("/")
    tool = str(cfg.get("dataset.download.tool", "gsutil"))
    components = get_required_components(cfg)
    raw_root = cfg.path("paths.raw_dir")
    data_root = cfg.path("paths.data_root")

    # Target: 25 train segments, 7 val segments (total 32 segments -> ~10 GB)
    targets = {"train": 25, "val": 7}

    log.info("Checking existing processed dataset...")
    for split_key, split_folder in cfg.dataset.splits.items():
        target_count = targets[split_key]
        processed = get_already_processed_segments(data_root, split_key)
        log.info("Split '%s': %d segments currently processed (target: %d)", split_key, len(processed), target_count)

        if len(processed) >= target_count:
            log.info("Split '%s' already has %d segments. Skipping.", split_key, len(processed))
            continue

        available = get_available_segments(bucket, split_folder, components, tool=tool)
        needed_segs = [s for s in available[:target_count] if s not in processed]
        log.info("Split '%s': Need to download and process %d new segments.", split_key, len(needed_segs))

        # Process in batches of 5 segments to keep disk space safe
        batch_size = 5
        for i in range(0, len(needed_segs), batch_size):
            chunk = needed_segs[i : i + batch_size]
            free_space = check_free_gb()
            log.info("--- Starting batch %d/%d (%d segments) for %s. Free space: %.1f GB ---",
                     i // batch_size + 1, (len(needed_segs) + batch_size - 1) // batch_size, len(chunk), split_key, free_space)

            if free_space < 4.0:
                log.error("Low disk space (< 4.0 GB). Halting expansion safely.")
                return

            # 1. Download chunk
            download_segment_batch(bucket, split_folder, components, chunk, raw_root, tool=tool)

            # 2. Preprocess chunk
            log.info("Preprocessing batch into %s...", data_root / "processed")
            run_preprocess(cfg)

            # 3. Clean raw chunk to reclaim disk space immediately
            log.info("Cleaning raw parquet files for %s to reclaim disk space...", split_folder)
            shutil.rmtree(raw_root / split_folder, ignore_errors=True)

            current_size = get_dir_size_gb(data_root / "processed")
            log.info("Batch done. Current processed data size: %.2f GB. Free disk space: %.1f GB",
                     current_size, check_free_gb())

    final_size = get_dir_size_gb(data_root / "processed")
    log.info("=== Dataset expansion complete! Total processed data size: %.2f GB ===", final_size)


if __name__ == "__main__":
    main()
