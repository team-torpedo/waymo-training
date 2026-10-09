# Third-party notices

The code in this repository is released under the MIT License (see `LICENSE`).
It depends on third-party software and data that are covered by **their own,
different terms**. Read these before you train, share or deploy anything.

## Waymo Open Dataset (data)

The Waymo Open Dataset is **not** part of this repository and is **not**
covered by the MIT license. It is licensed by Waymo LLC for **non-commercial
use** under the Waymo Dataset License Agreement. You must register and accept
the terms at <https://waymo.com/open/> before downloading it.

Models trained on the dataset may inherit its restrictions. Check the license
before using trained weights outside research.

## Ultralytics YOLO (object detection)

Object detection uses the `ultralytics` package. Ultralytics YOLO is released
under **AGPL-3.0** (an Enterprise License is available from Ultralytics).
Using it as a dependency is fine for research and internal use, but if you
distribute a product or offer a network service built on it, the AGPL's
obligations may apply to your whole work. Review
<https://www.ultralytics.com/license> before any such use. The detection
backend is swappable (see `docs/extending.md`) if you need a different
license.

## TensorFlow / Keras (lane segmentation)

TensorFlow and Keras are Apache-2.0 licensed. Pretrained ImageNet weights for
MobileNetV2 are downloaded by Keras at runtime and subject to their own terms.

## Other dependencies

NumPy, pandas, PyArrow, OpenCV, Pillow, PyYAML, tqdm and TensorBoard are used
under their respective permissive licenses.
