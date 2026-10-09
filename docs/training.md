# Training

## Object detection (YOLO)

```bash
python stages/train_detection.py
python stages/train_detection.py --set detection.model.weights=yolo11s.pt --set detection.train.epochs=100
```

- Data: `data/processed/yolo/data.yaml` (written by preprocess).
- Pretrained weights are downloaded by Ultralytics on first use. For a from-scratch run use an
  architecture file, e.g. `yolo11n.yaml`.
- Any Ultralytics argument can be set through `detection.train.extra`, e.g.
  `--set "detection.train.extra={mosaic: 0.5, fliplr: 0.5}"`.
- Waymo images are 1920×1280 (front) and 1920×886 (sides). `imgsz: 640` downsizes 3× — small objects
  (distant pedestrians, signs) suffer; try `imgsz: 960` or `1280` if you have the GPU memory.
- Output: `outputs/detection/<run_name>-<timestamp>/` with `weights/best.pt`; `outputs/detection/latest.json`
  points at the newest run.

## Lane segmentation (TensorFlow)

```bash
python stages/train_lane.py
python stages/train_lane.py --set lane.model.name=unet --set lane.loss.name=focal
tensorboard --logdir outputs/lane
```

- Input: image/mask pairs in `data/processed/lane/{train,val}`; images are resized to `lane.data.image_size`
  inside the `tf.data` pipeline (bilinear for images, nearest for masks).
- Models normalise inputs internally, so preprocessing is identical at training and inference.
- Lane pixels are a tiny fraction of each image, so plain BCE tends to collapse to "all background".
  The default `bce_dice` loss counters this; `focal` is an alternative. Watch `val_iou`, not accuracy.
- Metrics logged: IoU, precision, recall (threshold `lane.train.threshold`).
- Checkpointing keeps the best `monitor` value as `best.keras`; early stopping restores best weights.
- `lane.train.mixed_precision: true` enables float16 compute on GPUs with tensor cores.
- Output: `outputs/lane/<run_name>-<timestamp>/` with `best.keras`, `last.keras`, `history.csv/json`,
  `config.yaml` (the exact resolved config of the run) and `tensorboard/`.

## Reproducibility

Each lane run stores its resolved config. YOLO runs store their args in Ultralytics' `args.yaml`.
`project.seed` seeds Python, NumPy, Keras and Ultralytics (GPU nondeterminism may still cause small differences).

## Resuming

- YOLO: `--set detection.train.resume=true` with `detection.model.weights` pointing at the run's `last.pt`.
- Lane: load `last.keras` yourself or add a `resume` option — see [extending.md](extending.md).
