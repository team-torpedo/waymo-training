"""Object detection backends (selected with `detection.backend`)."""

from . import yolo  # noqa: F401  (registers the 'ultralytics' backend)
from .base import DETECTORS, BaseDetector  # noqa: F401
