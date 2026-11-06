"""Download Waymo Open Dataset v2 Parquet files from Google Cloud Storage."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..config import Config
from ..utils import enabled_tasks, ensure_dir, get_logger

log = get_logger(__name__)


def get_required_components(cfg: Config) -> list[str]:
    comp = cfg.get("dataset.download.components")
    if comp:
        return list(comp)

    tasks = enabled_tasks(cfg)
    needed: list[str] = []
    if tasks["detection"] or tasks["lane"]:
        needed.append("camera_image")
    if tasks["detection"]:
        needed.append("camera_box")
    if tasks["lane"]:
        needed.append("camera_segmentation")

    # Preserve order and uniqueness
    seen = set()
    return [c for c in needed if not (c in seen or seen.add(c))]


def _list_gcs_files(prefix: str, tool: str = "gsutil") -> list[str]:
    cmd = [tool, "ls", f"{prefix}/*.parquet"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip().endswith(".parquet")]
    except subprocess.CalledProcessError as exc:
        log.error("Failed to list GCS files in %s: %s", prefix, exc.stderr)
        raise RuntimeError(f"GCS listing failed for {prefix}: {exc.stderr}") from exc


def run_download(cfg: Config) -> dict[str, list[str]]:
    bucket = str(cfg.dataset.gcs_bucket).rstrip("/")
    raw_root = ensure_dir(cfg.path("paths.raw_dir"))
    splits = cfg.dataset.splits
    tool = str(cfg.get("dataset.download.tool", "gsutil"))
    parallel = bool(cfg.get("dataset.download.parallel", True))
    components = get_required_components(cfg)

    downloaded: dict[str, list[str]] = {}

    log.info("Downloading components %s using %s", components, tool)

    for split_key, split_folder in splits.items():
        max_seg = cfg.get(f"dataset.download.max_segments.{split_key}")
        max_seg = int(max_seg) if max_seg is not None else None

        # Find available segments per component
        segments_per_comp: list[set[str]] = []
        for comp in components:
            prefix = f"{bucket}/{split_folder}/{comp}"
            files = _list_gcs_files(prefix, tool=tool)
            segs = {Path(f).stem for f in files}
            segments_per_comp.append(segs)

        # Intersect to ensure all required components exist for selected segments
        if not segments_per_comp:
            log.warning("No components found for split %s", split_key)
            continue

        common_segments = sorted(list(set.intersection(*segments_per_comp)))
        if max_seg is not None:
            chosen_segments = common_segments[:max_seg]
        else:
            chosen_segments = common_segments

        log.info("Split '%s': %d available with all components, downloading %d segments",
                 split_key, len(common_segments), len(chosen_segments))

        downloaded[split_key] = chosen_segments

        for comp in components:
            comp_dir = ensure_dir(raw_root / split_folder / comp)
            to_download: list[tuple[str, Path]] = []

            for seg in chosen_segments:
                fname = f"{seg}.parquet"
                local_path = comp_dir / fname
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue
                gcs_uri = f"{bucket}/{split_folder}/{comp}/{fname}"
                to_download.append((gcs_uri, local_path))

            if not to_download:
                log.info("All %s files for %s split already downloaded.", comp, split_key)
                continue

            log.info("Downloading %d files for %s/%s...", len(to_download), split_folder, comp)
            for gcs_uri, local_path in to_download:
                cp_cmd = [tool]
                if tool == "gsutil" and parallel:
                    cp_cmd.append("-m")
                cp_cmd.extend(["cp", gcs_uri, str(local_path)])
                subprocess.run(cp_cmd, check=True)

    log.info("Download completed.")
    return downloaded
