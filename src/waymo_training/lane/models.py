"""Lane segmentation architectures. Select with `lane.model.name` in config/lane.yaml.

Every builder takes `input_shape=(H, W, 3)` plus its own `lane.model.params`, expects raw
pixel values in [0, 255] and returns a Keras model producing a sigmoid (H, W, 1) map.
Input normalisation lives *inside* the model so training and inference stay consistent.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from ..registry import Registry

MODELS = Registry("lane model")


def _conv_block(x, filters: int, dropout: float = 0.0):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def _head(x):
    # float32 output keeps mixed-precision training numerically safe
    return layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="lane_mask")(x)


def _check_size(input_shape, multiple: int, name: str) -> None:
    h, w = input_shape[:2]
    if h % multiple or w % multiple:
        raise ValueError(f"{name}: image_size {h}x{w} must be divisible by {multiple}")


@MODELS.register("unet")
def build_unet(input_shape, base_filters: int = 32, depth: int = 4, dropout: float = 0.0):
    _check_size(input_shape, 2**depth, "unet")
    inputs = layers.Input(input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    skips = []
    f = base_filters
    for _ in range(depth):
        x = _conv_block(x, f, dropout)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
        f *= 2
    x = _conv_block(x, f, dropout)
    for skip in reversed(skips):
        f //= 2
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f, dropout)
    return tf.keras.Model(inputs, _head(x), name="unet")


@MODELS.register("mobilenet_unet")
def build_mobilenet_unet(
    input_shape,
    pretrained: str | None = "imagenet",
    freeze_backbone: bool = False,
    decoder_filters=(256, 128, 64, 32),
):
    _check_size(input_shape, 32, "mobilenet_unet")
    inputs = layers.Input(input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # MobileNetV2 expects [-1, 1]
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=x, include_top=False, weights=pretrained
    )
    backbone.trainable = not freeze_backbone
    skip_names = [
        "block_1_expand_relu",    # 1/2
        "block_3_expand_relu",    # 1/4
        "block_6_expand_relu",    # 1/8
        "block_13_expand_relu",   # 1/16
    ]
    skips = [backbone.get_layer(n).output for n in skip_names]
    x = backbone.get_layer("block_16_project").output  # 1/32
    for skip, f in zip(reversed(skips), decoder_filters):
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)
    x = layers.UpSampling2D(interpolation="bilinear")(x)  # back to full resolution
    x = _conv_block(x, 16)
    return tf.keras.Model(inputs, _head(x), name="mobilenet_unet")


@MODELS.register("fcn")
def build_fcn(input_shape, base_filters: int = 32):
    """Small, fast baseline: strided conv encoder + bilinear upsampling."""
    _check_size(input_shape, 8, "fcn")
    inputs = layers.Input(input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    f = base_filters
    for _ in range(3):
        x = layers.Conv2D(f, 3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = _conv_block(x, f)
        f *= 2
    x = layers.Conv2D(f // 2, 1, activation="relu")(x)
    x = layers.UpSampling2D(size=8, interpolation="bilinear")(x)
    return tf.keras.Model(inputs, _head(x), name="fcn")
