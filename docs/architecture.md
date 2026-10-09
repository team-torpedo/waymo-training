# Architecture

```
            ┌────────────┐   parquet    ┌────────────┐  YOLO images/labels  ┌────────────────┐
 GCS bucket │  download  │ ───────────▶ │ preprocess │ ───────────────────▶ │ train_detection│─┐
            └────────────┘  data/raw    │            │  lane image+mask     ├────────────────┤ │  ┌──────────┐
                                        │            │ ───────────────────▶ │ train_lane     │─┼▶│ evaluate │
                                        └────────────┘  data/processed      └────────────────┘ │  └──────────┘
                                                                              outputs/…/latest.json
```

Each stage is a function in `waymo_training/pipeline.py`, registered in `STAGES`. `main.py`
runs `pipeline.stages` from the config (or `--stages` / `--skip`); every script in `stages/`
runs exactly one. Stages communicate only through files on disk, so you can rerun any of them.

## What makes parts swappable

`waymo_training/registry.py` provides a tiny `Registry` (name → factory). The YAML names pick
the implementation:

| Config key            | Registry                   | Built-ins                           |
| --------------------- | -------------------------- | ----------------------------------- |
| `dataset.reader`      | `data.readers.READERS`     | `waymo_v2`                          |
| `detection.backend`   | `detection.base.DETECTORS` | `ultralytics`                       |
| `lane.model.name`     | `lane.models.MODELS`       | `unet`, `mobilenet_unet`, `fcn`     |
| `lane.loss.name`      | `lane.losses.LOSSES`       | `bce`, `dice`, `bce_dice`, `focal`  |
| `lane.optimizer.name` | `lane.losses.OPTIMIZERS`   | `adam`, `adamw`, `sgd`, `rmsprop`   |
| `lane.scheduler.name` | `lane.losses.SCHEDULERS`   | `constant`, `cosine`, `exponential` |

Unknown names fail with a message listing what is available. To add one, see
[extending.md](extending.md).

YOLO variants (`yolo11n`, `yolo11s`, `yolov8m`, …) are swapped by changing
`detection.model.weights`; Ultralytics resolves the architecture.

## Configuration flow

`config.load_config()` loads every `*.yaml` in `config/` alphabetically (`local.yaml` last), deep-merges
them, applies `--set` overrides, then resolves `${section.key}` / `${env:NAME}` references. The resulting
`Config` object allows `cfg.lane.model.name` or `cfg.get("lane.model.name")`, and `cfg.path("paths.raw_dir")`
returns an absolute `Path` (relative paths resolve against the repo root).

## Data model

- `data/readers.py` yields `Sample` objects (image bytes, size, boxes, semantic map) per
  (segment, frame, camera). It reads parquet in batches, so memory stays bounded.
- `data/preprocess.py` turns samples into on-disk datasets. Full-resolution JPEGs are written
  unchanged; masks are PNGs (0/255). Resizing happens at load time (`tf.data`) or inside YOLO.

## Model outputs and "latest"

Training writes `outputs/<task>/latest.json` pointing at the newest weights. `evaluate` uses it
unless `evaluation.detection.weights` / `evaluation.lane.model_path` is set.

## Frameworks

Lane uses TensorFlow/Keras 3; detection uses Ultralytics YOLO, which is PyTorch-based (there is
no maintained TensorFlow YOLO trainer). Both live in one venv and never share tensors — the
only interface between them is the dataset on disk. TensorFlow and Ultralytics are imported lazily,
so preprocessing and configuration work even if one is not installed.
