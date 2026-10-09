# Troubleshooting

**`Listing gs://... failed` / 403 / AccessDenied** — Your Google account has not been granted access, or
you are not logged in. Accept the license at <https://waymo.com/open/>, then
`gcloud auth login && gcloud auth application-default login`. Access can take up to ~48 h.

**`gsutil not found`** — `bash scripts/install_gcloud.sh`, restart the shell, or set
`dataset.download.tool: gcloud`.

**`No lane images in ...`** — Preprocess found no frames with segmentation labels. Segmentation exists only for
a subset of segments; check `data/raw/training/camera_segmentation/` is non-empty and raise `max_segments`.

**`no segments with camera_image + camera_box found`** — The download didn't finish or `paths.raw_dir` differs
from where you downloaded. Check the folder layout in [dataset.md](dataset.md).

**`image_size ... must be divisible by 32`** — Pick sizes like `[320, 512]`, `[384, 640]`, `[480, 768]`.

**Lane IoU stays near 0** — Usually "all background" collapse or masks that are nearly empty. Use
`bce_dice`/`focal`, verify masks (`data/processed/lane/train/masks/*.png` should show thin white markings),
try `lane.semantic_classes: [20, 21, 22]` for a denser target, and train longer.

**CUDA out of memory** — YOLO: lower `detection.train.batch` / `imgsz`. Lane: lower `lane.data.batch_size` or
`image_size`, enable `mixed_precision`.

**TensorFlow doesn't see the GPU** — Use `scripts/setup_venv.sh --gpu` on Linux/WSL2. Native Windows
TensorFlow has no GPU support. Check with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.

**TensorFlow and PyTorch fight over CUDA / numpy errors** — Make sure you use a fresh `venv`, Python 3.10–3.12,
and that `pip check` is clean. As a last resort run the two tasks in separate venvs (`pipeline.tasks.*`
lets you enable one at a time) — they only share files on disk.

**Slow preprocessing** — It decodes JPEG headers and PNG masks per frame. Restrict `dataset.cameras` or
`max_segments`, or run on an SSD.

**YOLO tries to download weights offline** — Pre-download `yolo11n.pt` and point
`detection.model.weights` at the local file.
