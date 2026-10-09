# Changelog

All notable changes are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-10-09

### Added

- Initial pipeline: download → preprocess → train (YOLO detection, Keras lane segmentation) → evaluate.
- YAML configuration with merge, `${...}` interpolation and `--set` overrides.
- Registries for dataset readers, detector backends, lane models, losses, optimizers and schedulers.
- Waymo Open Dataset v2 (Parquet) reader for `camera_image`, `camera_box`, `camera_segmentation`.
- Lane models `unet`, `mobilenet_unet`, `fcn`; losses `bce`, `dice`, `bce_dice`, `focal`.
- venv/installation scripts (Linux/macOS/Windows), gcloud installer, Makefile, CI workflow.
- Documentation in `docs/`; synthetic-data tests.
