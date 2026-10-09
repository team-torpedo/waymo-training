"""Ultralytics YOLO backend. Change model/size/hyper-parameters in config/detection.yaml."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..utils import ensure_dir, get_logger, read_json, timestamp, write_json
from .base import DETECTORS, BaseDetector

log = get_logger(__name__)


def latest_weights(cfg: Config) -> Path:
    runs = cfg.path("paths.detection_runs_dir")
    latest = runs / "latest.json"
    if latest.exists():
        return Path(read_json(latest)["weights"])
    fallback = runs / cfg.detection.run_name / "weights" / "best.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No trained detection weights found in {runs}. Run train_detection first or set "
        "evaluation.detection.weights."
    )


@DETECTORS.register("ultralytics")
class UltralyticsDetector(BaseDetector):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ultralytics is not installed: pip install ultralytics") from exc
        self.data_yaml = cfg.path("paths.yolo_dir") / "data.yaml"

    def _need_data(self) -> None:
        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"{self.data_yaml} not found. Run the preprocess stage first (python stages/preprocess.py)."
            )

    def train(self) -> Path:
        from ultralytics import YOLO

        self._need_data()
        t = self.cfg.detection.train
        runs = ensure_dir(self.cfg.path("paths.detection_runs_dir"))
        name = f"{self.cfg.detection.run_name}-{timestamp()}"
        args = {
            "data": str(self.data_yaml),
            "epochs": t.epochs,
            "imgsz": t.imgsz,
            "batch": t.batch,
            "device": t.device,
            "workers": t.workers,
            "patience": t.patience,
            "optimizer": t.optimizer,
            "lr0": t.lr0,
            "cos_lr": t.cos_lr,
            "amp": t.amp,
            "cache": t.cache,
            "resume": t.resume,
            "seed": int(self.cfg.project.seed),
            "project": str(runs),
            "name": name,
            **(t.get("extra").to_dict() if t.get("extra") else {}),
        }
        args = {k: v for k, v in args.items() if v is not None}
        log.info("Training YOLO (%s) with %s", self.cfg.detection.model.weights, args)
        model = YOLO(str(self.cfg.detection.model.weights))
        model.train(**args)

        weights = runs / name / "weights" / "best.pt"
        if not weights.exists():
            weights = runs / name / "weights" / "last.pt"
        write_json(runs / "latest.json", {"run": name, "weights": str(weights)})
        log.info("Best weights: %s", weights)
        return weights

    def evaluate(self, weights: str | Path | None = None) -> dict:
        from ultralytics import YOLO

        self._need_data()
        weights = Path(weights) if weights else latest_weights(self.cfg)
        split = str(self.cfg.get("evaluation.split", "val"))
        v = self.cfg.detection.val
        out = ensure_dir(self.cfg.path("paths.eval_dir") / "detection")
        model = YOLO(str(weights))
        res = model.val(
            data=str(self.data_yaml),
            split=split,
            imgsz=v.imgsz,
            conf=v.conf,
            iou=v.iou,
            device=self.cfg.detection.train.device,
            project=str(out),
            name="val",
            exist_ok=True,
            plots=True,
        )
        names = list(self.cfg.get("detection.classes").to_dict().values())
        per_class_map = {}
        try:
            for idx, m in zip(res.box.ap_class_index, res.box.maps[res.box.ap_class_index]):
                per_class_map[names[int(idx)]] = float(m)
        except Exception:  # per-class info is best-effort
            pass
        metrics = {
            "weights": str(weights),
            "split": split,
            "precision": float(res.box.mp),
            "recall": float(res.box.mr),
            "mAP50": float(res.box.map50),
            "mAP50-95": float(res.box.map),
            "per_class_mAP50-95": per_class_map,
        }
        write_json(out / "metrics.json", metrics)
        return metrics

    def visualize(self, weights: str | Path | None = None) -> Path | None:
        from ultralytics import YOLO

        n = int(self.cfg.get("evaluation.detection.num_visualizations", 0))
        if n <= 0:
            return None
        split = str(self.cfg.get("evaluation.split", "val"))
        images = sorted((self.cfg.path("paths.yolo_dir") / "images" / split).glob("*.jpg"))[:n]
        if not images:
            return None
        weights = Path(weights) if weights else latest_weights(self.cfg)
        out = ensure_dir(self.cfg.path("paths.eval_dir") / "detection")
        YOLO(str(weights)).predict(
            source=[str(p) for p in images],
            save=True,
            project=str(out),
            name="predictions",
            exist_ok=True,
            imgsz=self.cfg.detection.val.imgsz,
            device=self.cfg.detection.train.device,
            verbose=False,
        )
        return out / "predictions"
