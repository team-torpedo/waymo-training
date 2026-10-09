# Extending the pipeline

Everything swappable is a registry entry selected by a YAML name (see [architecture](architecture.md)).
Adding a component = write a function/class, decorate it, point the config at it.

## New lane architecture

In `src/waymo_training/lane/models.py` (or any module imported by `lane/__init__.py`):

```python
@MODELS.register("my_net")
def build_my_net(input_shape, width: int = 64):
    inputs = layers.Input(input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)      # inputs are raw 0-255 pixels
    ...
    return tf.keras.Model(inputs, layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x))
```

```yaml
# config/lane.yaml
lane:
  model:
    name: my_net
    params: { width: 96 } # becomes build_my_net(input_shape, width=96)
```

Contract: output shape `(H, W, 1)` with sigmoid activation, same H/W as the input.

## New loss / optimizer / scheduler

`@LOSSES.register("name")` factories return a Keras-compatible `loss(y_true, y_pred)`.
`@OPTIMIZERS.register("name")` factories receive `learning_rate`, `weight_decay` and the YAML `params`.
`@SCHEDULERS.register("name")` factories receive `lr`, `total_steps` and the YAML `params` and return a
float or a Keras `LearningRateSchedule`.

## New detection backend

Subclass `BaseDetector` (`detection/base.py`) implementing `train()`, `evaluate()` and `visualize()`,
register it, and set `detection.backend`:

```python
@DETECTORS.register("my_detector")
class MyDetector(BaseDetector):
    def train(self): ...      # write outputs/detection/latest.json and return weights path
    def evaluate(self, weights=None): ...   # return a JSON-serialisable dict
    def visualize(self, weights=None): ...
```

If you need a non-AGPL detector (see [NOTICE.md](../NOTICE.md)), this is the seam to use — the YOLO-format
dataset produced by preprocess is understood by many frameworks.

## New dataset reader

Subclass `BaseReader` in `data/readers.py`, yield `Sample` objects, register it and set `dataset.reader`.
Preprocessing, training and evaluation only see `Sample`, so nothing else changes. This is also where a
reader for the old v1 TFRecord format or another dataset would go.

## New stage

Add a function to `pipeline.STAGES`, list it in `pipeline.stages`, and (optionally) add a 10-line script in
`stages/` modelled on the existing ones.

## Tests

`tests/conftest.py` builds synthetic Waymo-shaped parquet files; reuse the `synth_cfg` fixture for new tests.
`WAYMO_TEST_YOLO=1 pytest tests/test_yolo_smoke.py` runs the slow YOLO end-to-end check.
