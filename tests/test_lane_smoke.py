import pytest

tf = pytest.importorskip("tensorflow")

from waymo_training.data.preprocess import preprocess_lane  # noqa: E402
from waymo_training.lane.evaluate import evaluate_lane  # noqa: E402
from waymo_training.lane.models import MODELS  # noqa: E402
from waymo_training.lane.train import train_lane  # noqa: E402


@pytest.mark.parametrize("name,params", [("unet", {"base_filters": 8, "depth": 3}), ("fcn", {"base_filters": 8}), ("mobilenet_unet", {"pretrained": None})])
def test_model_shapes(name, params):
    model = MODELS.build(name, input_shape=(64, 96, 3), **params)
    assert model.output_shape == (None, 64, 96, 1)


@pytest.mark.parametrize("model,loss", [("unet", "bce_dice"), ("fcn", "focal")])
def test_train_and_evaluate(synth_cfg, model, loss):
    from waymo_training.config import load_config

    cfg = load_config(overrides=[
        f"paths.data_root={synth_cfg.path('paths.data_root')}",
        f"paths.outputs_dir={synth_cfg.path('paths.outputs_dir')}",
        "dataset.cameras=[FRONT]",
        "lane.data.image_size=[64, 96]",
        "lane.data.batch_size=2",
        "lane.train.epochs=1",
        f"lane.model.name={model}",
        "lane.model.params={base_filters: 8}",
        f"lane.loss.name={loss}",
        "lane.optimizer.name=adamw",
        "lane.scheduler.name=cosine",
        "evaluation.lane.num_visualizations=2",
    ])
    preprocess_lane(cfg)
    path = train_lane(cfg)
    assert path.exists()
    metrics = evaluate_lane(cfg)
    assert 0.0 <= metrics["iou"] <= 1.0
    assert (cfg.path("paths.eval_dir") / "lane" / "metrics.json").exists()
    assert len(list((cfg.path("paths.eval_dir") / "lane" / "visualizations").glob("*.jpg"))) == 2
