"""Losses, optimizers and LR schedules for the lane model (all swappable from YAML)."""

from __future__ import annotations

import tensorflow as tf

from ..registry import Registry

LOSSES = Registry("lane loss")
OPTIMIZERS = Registry("lane optimizer")
SCHEDULERS = Registry("lane scheduler")

_EPS = 1e-7


# ---------------------------------------------------------------- losses
def _flat(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    return y_true, tf.clip_by_value(y_pred, _EPS, 1.0 - _EPS)


@LOSSES.register("bce")
def bce(**_):
    def bce_loss(y_true, y_pred):
        y_true, y_pred = _flat(y_true, y_pred)
        return tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_true, y_pred))

    return bce_loss


@LOSSES.register("dice")
def dice(smooth: float = 1.0, **_):
    def dice_loss(y_true, y_pred):
        y_true, y_pred = _flat(y_true, y_pred)
        axes = (1, 2, 3)
        inter = tf.reduce_sum(y_true * y_pred, axes)
        denom = tf.reduce_sum(y_true, axes) + tf.reduce_sum(y_pred, axes)
        return 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))

    return dice_loss


@LOSSES.register("bce_dice")
def bce_dice(bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0, **_):
    b, d = bce(), dice(smooth=smooth)

    def bce_dice_loss(y_true, y_pred):
        return bce_weight * b(y_true, y_pred) + dice_weight * d(y_true, y_pred)

    return bce_dice_loss


@LOSSES.register("focal")
def focal(alpha: float = 0.25, gamma: float = 2.0, **_):
    def focal_loss(y_true, y_pred):
        y_true, y_pred = _flat(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(-a_t * tf.pow(1 - p_t, gamma) * tf.math.log(p_t))

    return focal_loss


# ------------------------------------------------------------- schedulers
@SCHEDULERS.register("constant")
def constant(lr: float, total_steps: int, **_):
    return lr


@SCHEDULERS.register("cosine")
def cosine(lr: float, total_steps: int, alpha: float = 0.01, **_):
    return tf.keras.optimizers.schedules.CosineDecay(lr, max(total_steps, 1), alpha=alpha)


@SCHEDULERS.register("exponential")
def exponential(lr: float, total_steps: int, decay_rate: float = 0.9, decay_steps: int = 1000, **_):
    return tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, decay_rate)


# -------------------------------------------------------------- optimizers
@OPTIMIZERS.register("adam")
def adam(learning_rate, weight_decay: float = 0.0, **kw):
    return tf.keras.optimizers.Adam(learning_rate, weight_decay=weight_decay or None, **kw)


@OPTIMIZERS.register("adamw")
def adamw(learning_rate, weight_decay: float = 0.004, **kw):
    return tf.keras.optimizers.AdamW(learning_rate, weight_decay=weight_decay, **kw)


@OPTIMIZERS.register("sgd")
def sgd(learning_rate, weight_decay: float = 0.0, momentum: float = 0.9, **kw):
    return tf.keras.optimizers.SGD(
        learning_rate, momentum=momentum, weight_decay=weight_decay or None, **kw
    )


@OPTIMIZERS.register("rmsprop")
def rmsprop(learning_rate, weight_decay: float = 0.0, **kw):
    return tf.keras.optimizers.RMSprop(learning_rate, weight_decay=weight_decay or None, **kw)
