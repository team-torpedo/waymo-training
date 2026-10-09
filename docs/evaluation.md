# Evaluation

```bash
python stages/evaluate.py
python stages/evaluate.py --set evaluation.detection.weights=outputs/detection/<run>/weights/best.pt \
                          --set evaluation.lane.model_path=outputs/lane/<run>/best.keras
```

Both tasks are evaluated on `evaluation.split` (default `val`; Waymo's test split has no public labels).
Disable a task with `pipeline.tasks.<task>=false`.

## Detection

Uses Ultralytics `val()`. Saved to `outputs/evaluation/detection/`:

- `metrics.json` — precision, recall, mAP@0.5, mAP@0.5:0.95, per-class mAP@0.5:0.95
- `val/` — PR curves, confusion matrix
- `predictions/` — `evaluation.detection.num_visualizations` example images with boxes

Note: these are 2D image-space COCO-style metrics, not Waymo's official 2D detection metric
(which uses its own tracking-aware AP and difficulty levels). Use the official toolkit for leaderboard numbers.

## Lane

Pixel metrics accumulated over the **whole** split (not averaged per batch), saved to
`outputs/evaluation/lane/metrics.json`:

`iou`, `dice`, `precision`, `recall`, `pixel_accuracy`, at `evaluation.lane.threshold`.

`visualizations/NNN.jpg` shows `[image | ground truth (green) | prediction (red)]`.
Pixel accuracy is dominated by background and is not informative on its own — use IoU/Dice.

## Summary

`outputs/evaluation/summary.json` combines the metrics of every evaluated task.
