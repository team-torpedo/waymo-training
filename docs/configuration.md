# Configuration reference

All `*.yaml` files in `config/` are merged. Keep them as they are, or add your own files
(e.g. `config/experiment_a.yaml`), or use `config/local.yaml` for personal overrides (gitignored,
loaded last). To switch to an entirely different set: `--config-dir path/to/dir`.

Precedence (low → high): files alphabetically → `local.yaml` → `--set key=value`.

```bash
python main.py --set lane.train.epochs=5 --set "detection.classes={1: vehicle, 2: pedestrian}"
```

Values after `=` are parsed as YAML (`true`, `3`, `[a, b]`, `{k: v}`, `null`).
References: `${paths.data_root}` (keeps native type when the whole value is a reference) and
`${env:NAME}` / `${env:NAME,default}`.

## `config.yaml` — project, paths, pipeline

| Key                                  | Meaning                                                                     |
| ------------------------------------ | --------------------------------------------------------------------------- |
| `project.seed`                       | Random seed (Python, NumPy, Keras, YOLO)                                    |
| `project.log_level`                  | `DEBUG` / `INFO` / `WARNING`                                                |
| `paths.*`                            | Where data and outputs live. Relative = relative to repo root               |
| `pipeline.stages`                    | Order of stages run by `main.py`                                            |
| `pipeline.tasks.detection` / `.lane` | Turn a whole task off (download, preprocess, train, evaluate all honour it) |

## `dataset.yaml` — data

| Key                                           | Meaning                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| `dataset.reader`                              | Dataset reader registry key (`waymo_v2`)                                 |
| `dataset.gcs_bucket`                          | Source bucket                                                            |
| `dataset.splits`                              | Maps our `train`/`val` to bucket folders `training`/`validation`         |
| `dataset.cameras`                             | Subset of `FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT`        |
| `dataset.download.tool`                       | `gsutil` or `gcloud`                                                     |
| `dataset.download.max_segments.{train,val}`   | Segments (~20 s drives, ~200 frames × 5 cameras) per split; `null` = all |
| `dataset.download.components`                 | `null` = derive from enabled tasks, or an explicit list                  |
| `dataset.preprocess.detection.require_labels` | Skip frames without `camera_box` rows (don't treat unlabeled as empty)   |
| `dataset.preprocess.detection.min_box_px`     | Drop boxes smaller than this                                             |
| `dataset.preprocess.lane.min_lane_fraction`   | Drop frames whose lane mask is nearly empty                              |

## `detection.yaml` — YOLO

| Key                       | Meaning                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| `detection.backend`       | Detector registry key (`ultralytics`)                                                         |
| `detection.classes`       | Waymo box type id → name. Order = YOLO class index. Delete lines to ignore classes            |
| `detection.model.weights` | `yolo11n.pt`, `yolo11s.pt`, `yolov8m.pt` … (pretrained) or `*.yaml` (from scratch)            |
| `detection.train.*`       | `epochs, imgsz, batch, device, workers, patience, optimizer, lr0, cos_lr, amp, cache, resume` |
| `detection.train.extra`   | Dict forwarded to Ultralytics `train()` — any of its hyper-parameters/augmentations           |
| `detection.val.*`         | `imgsz, conf, iou` for evaluation                                                             |

`device: null` lets Ultralytics choose; use `0`, `"0,1"` or `"cpu"` to force.

## `lane.yaml` — lane segmentation

| Key                                                                             | Meaning                                                                                          |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `lane.semantic_classes`                                                         | Waymo segmentation class ids merged into the lane mask (21 lane marker, 22 road marker, 20 road) |
| `lane.data.image_size`                                                          | `[H, W]`, divisible by 32 for `mobilenet_unet` (16 for default `unet`)                           |
| `lane.data.batch_size, shuffle_buffer, cache`                                   | Input pipeline                                                                                   |
| `lane.data.augment.{hflip,brightness,contrast}`                                 | Augmentations (training only)                                                                    |
| `lane.model.name` / `.params`                                                   | Architecture + builder kwargs (see [extending](extending.md))                                    |
| `lane.loss.name` / `.params`                                                    | `bce`, `dice`, `bce_dice`, `focal`                                                               |
| `lane.optimizer.{name,lr,weight_decay,params}`                                  | `adam`, `adamw`, `sgd`, `rmsprop`                                                                |
| `lane.scheduler.{name,params}`                                                  | `constant`, `cosine`, `exponential`                                                              |
| `lane.train.{epochs,mixed_precision,early_stopping_patience,monitor,threshold}` | Training loop. `monitor` is a Keras metric (`val_iou`, `val_loss`, …)                            |

## `evaluation.yaml`

| Key                               | Meaning                                                              |
| --------------------------------- | -------------------------------------------------------------------- |
| `evaluation.split`                | Which split to evaluate on (`val`; Waymo test labels are not public) |
| `evaluation.detection.weights`    | `null` = latest trained                                              |
| `evaluation.lane.model_path`      | `null` = latest trained                                              |
| `evaluation.*.num_visualizations` | How many example images to save                                      |

## Example experiments

Fast CPU sanity run:

```bash
python main.py --set dataset.download.max_segments.train=2 --set dataset.download.max_segments.val=1 \
  --set detection.train.epochs=2 --set lane.train.epochs=2 --set detection.train.device=cpu
```

Lane only, drivable area + markings, U-Net:

```bash
python main.py --set pipeline.tasks.detection=false --set "lane.semantic_classes=[20,21,22]" \
  --set lane.model.name=unet
```
