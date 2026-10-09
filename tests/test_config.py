import pytest

from waymo_training.config import load_config, parse_override


def test_defaults_load_and_interpolate():
    cfg = load_config()
    assert str(cfg.paths.raw_dir).endswith("data/raw")
    assert cfg.detection.val.imgsz == cfg.detection.train.imgsz  # native type preserved
    assert cfg.path("paths.yolo_dir").is_absolute()


def test_override_and_types():
    cfg = load_config(overrides=["lane.train.epochs=3", "lane.model.name=unet", "lane.data.augment.hflip=false"])
    assert cfg.lane.train.epochs == 3
    assert cfg.lane.model.name == "unet"
    assert cfg.lane.data.augment.hflip is False


def test_override_cascades_through_interpolation():
    cfg = load_config(overrides=["paths.data_root=/tmp/xyz"])
    assert str(cfg.paths.raw_dir) == "/tmp/xyz/raw"


def test_detection_class_keys_are_ints():
    classes = load_config().get("detection.classes").to_dict()
    assert list(classes) == [1, 2, 3, 4]


def test_bad_override():
    with pytest.raises(ValueError):
        parse_override("no_equals_sign")


def test_missing_reference():
    with pytest.raises(KeyError):
        load_config(overrides=["paths.raw_dir=${nope.missing}"])
