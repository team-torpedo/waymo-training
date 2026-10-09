"""tf.data input pipeline for lane segmentation (image/mask pairs prepared by preprocess)."""

from __future__ import annotations

import tensorflow as tf

from ..config import Config


def list_pairs(cfg: Config, split: str) -> tuple[list[str], list[str]]:
    root = cfg.path("paths.lane_dir") / split
    images = sorted((root / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"No lane images in {root / 'images'}. Run the preprocess stage first "
            "(python stages/preprocess.py) and check that camera_segmentation was downloaded."
        )
    masks = [root / "masks" / f"{p.stem}.png" for p in images]
    missing = [m for m in masks if not m.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks missing, e.g. {missing[0]}")
    return [str(p) for p in images], [str(m) for m in masks]


def _make_load(h: int, w: int):
    def load(img_path, mask_path):
        img = tf.io.decode_jpeg(tf.io.read_file(img_path), channels=3)
        img = tf.image.resize(img, (h, w), method="bilinear", antialias=True)
        img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.float32)
        mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
        mask = tf.image.resize(mask, (h, w), method="nearest")
        mask = tf.cast(mask > 127, tf.float32)
        img.set_shape((h, w, 3))
        mask.set_shape((h, w, 1))
        return img, mask

    return load


def _make_augment(aug: Config):
    hflip = bool(aug.get("hflip", True))
    brightness = float(aug.get("brightness", 0.0) or 0.0)
    contrast = aug.get("contrast")

    def augment(img, mask):
        if hflip:
            flip = tf.random.uniform(()) < 0.5
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
        if brightness > 0:
            img = tf.image.random_brightness(img, brightness * 255.0)
        if contrast:
            img = tf.image.random_contrast(img, float(contrast[0]), float(contrast[1]))
        return tf.clip_by_value(img, 0.0, 255.0), mask

    return augment


def build_dataset(cfg: Config, split: str, training: bool) -> tuple[tf.data.Dataset, int]:
    """Return (dataset, number_of_samples)."""
    d = cfg.lane.data
    h, w = (int(v) for v in d.image_size)
    imgs, masks = list_pairs(cfg, split)
    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    if training:
        ds = ds.shuffle(min(len(imgs), int(d.shuffle_buffer)), reshuffle_each_iteration=True)
    ds = ds.map(_make_load(h, w), num_parallel_calls=tf.data.AUTOTUNE)
    if d.cache:
        ds = ds.cache()
    if training and d.get("augment"):
        ds = ds.map(_make_augment(d.augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(int(d.batch_size)).prefetch(tf.data.AUTOTUNE)
    return ds, len(imgs)
