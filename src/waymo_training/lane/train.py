"""Train the lane segmentation model described in config/lane.yaml."""

from __future__ import annotations

import math
from pathlib import Path

import tensorflow as tf
import yaml

from ..config import Config
from ..utils import ensure_dir, get_logger, read_json, timestamp, write_json
from .dataset import build_dataset
from .losses import LOSSES, OPTIMIZERS, SCHEDULERS
from .models import MODELS

log = get_logger(__name__)


def make_metrics(threshold: float) -> list:
    return [
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=threshold, name="iou"),
        tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
        tf.keras.metrics.Recall(thresholds=threshold, name="recall"),
    ]


def build_model(cfg: Config) -> tf.keras.Model:
    h, w = (int(v) for v in cfg.lane.data.image_size)
    params = cfg.lane.model.get("params")
    params = params.to_dict() if params else {}
    return MODELS.build(cfg.lane.model.name, input_shape=(h, w, 3), **params)


def latest_model_path(cfg: Config) -> Path:
    runs = cfg.path("paths.lane_runs_dir")
    latest = runs / "latest.json"
    if latest.exists():
        return Path(read_json(latest)["model"])
    raise FileNotFoundError(
        f"No trained lane model found in {runs}. Run train_lane first or set "
        "evaluation.lane.model_path."
    )


def train_lane(cfg: Config) -> Path:
    t = cfg.lane.train
    tf.keras.utils.set_random_seed(int(cfg.project.seed))
    if t.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision enabled")

    train_ds, n_train = build_dataset(cfg, "train", training=True)
    val_ds, n_val = build_dataset(cfg, "val", training=False)
    log.info("Lane data: %d train / %d val images", n_train, n_val)

    model = build_model(cfg)
    steps = math.ceil(n_train / int(cfg.lane.data.batch_size))
    total_steps = steps * int(t.epochs)

    sched_params = cfg.lane.scheduler.get("params")
    lr = SCHEDULERS.build(
        cfg.lane.scheduler.name,
        lr=float(cfg.lane.optimizer.lr),
        total_steps=total_steps,
        **(sched_params.to_dict() if sched_params else {}),
    )
    opt_params = cfg.lane.optimizer.get("params")
    optimizer = OPTIMIZERS.build(
        cfg.lane.optimizer.name,
        learning_rate=lr,
        weight_decay=float(cfg.lane.optimizer.get("weight_decay", 0.0) or 0.0),
        **(opt_params.to_dict() if opt_params else {}),
    )
    loss_params = cfg.lane.loss.get("params")
    loss = LOSSES.build(cfg.lane.loss.name, **(loss_params.to_dict() if loss_params else {}))
    model.compile(optimizer=optimizer, loss=loss, metrics=make_metrics(float(t.threshold)))
    model.summary(print_fn=log.info)

    runs = ensure_dir(cfg.path("paths.lane_runs_dir"))
    name = f"{cfg.lane.run_name}-{timestamp()}"
    run_dir = ensure_dir(runs / name)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    monitor = str(t.monitor)
    mode = "min" if "loss" in monitor else "max"
    best_path = run_dir / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path), monitor=monitor, mode=mode, save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard")),
    ]
    if int(t.early_stopping_patience) > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=int(t.early_stopping_patience),
                restore_best_weights=True,
            )
        )

    history = model.fit(train_ds, validation_data=val_ds, epochs=int(t.epochs), callbacks=callbacks)
    last_path = run_dir / "last.keras"
    model.save(str(last_path))
    final = best_path if best_path.exists() else last_path
    write_json(run_dir / "history.json", history.history)
    write_json(runs / "latest.json", {"run": name, "model": str(final)})
    log.info("Lane model saved: %s", final)
    return final
