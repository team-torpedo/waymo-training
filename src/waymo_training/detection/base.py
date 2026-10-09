"""Detector interface. Implement it to plug another detection framework into the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Config
from ..registry import Registry

DETECTORS = Registry("detection backend")


class BaseDetector(ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def train(self) -> Path:
        """Train and return the path of the best weights. Must update latest.json."""

    @abstractmethod
    def evaluate(self, weights: str | Path | None = None) -> dict:
        """Evaluate on `evaluation.split` and return a JSON-serialisable metrics dict."""

    @abstractmethod
    def visualize(self, weights: str | Path | None = None) -> Path | None:
        """Save a few prediction images; return the output directory."""
