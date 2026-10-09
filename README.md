# waymo-training

Configurable camera training pipeline for the [Waymo Open Dataset](https://waymo.com/open/),
built by **Team Torpedo** for our autonomous vehicle project.

| Task             | Framework                           | Default model    | Output                                 |
| ---------------- | ----------------------------------- | ---------------- | -------------------------------------- |
| Object detection | YOLO (Ultralytics, PyTorch backend) | `yolo11n`        | vehicles, pedestrians, signs, cyclists |
| Lane detection   | TensorFlow / Keras                  | MobileNetV2-UNet | binary lane-marking mask               |

Everything — dataset reader, detector, lane architecture, loss, optimizer, LR schedule,
classes, image sizes, epochs — is chosen in YAML files in [`config/`](config/). No code edits needed.

## Quick start

```bash
git clone https://github.com/team-torpedo/waymo-training.git
cd waymo-training

bash scripts/setup_venv.sh            # creates ./venv  (add --gpu for NVIDIA GPUs on Linux)
source venv/bin/activate

bash scripts/install_gcloud.sh        # once: gcloud/gsutil for the dataset download
gcloud auth login && gcloud auth application-default login   # after registering at waymo.com/open

python main.py                        # download -> preprocess -> train -> evaluate
```

Windows: `powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1`.

## Run one stage at a time

| Script                                       | What it does                                                                |
| -------------------------------------------- | --------------------------------------------------------------------------- |
| `python stages/download.py`                  | Download the needed Waymo v2 parquet components from GCS                    |
| `python stages/preprocess.py`                | Build the YOLO dataset and lane image/mask pairs                            |
| `python stages/train_detection.py`           | Train YOLO                                                                  |
| `python stages/train_lane.py`                | Train the Keras lane model                                                  |
| `python stages/evaluate.py`                  | Evaluate both models on the validation split, save metrics + visualisations |
| `python main.py [--stages ...] [--skip ...]` | The whole pipeline                                                          |

All scripts accept `--config-dir DIR` and repeated `--set key.sub=value` overrides:

```bash
python main.py --skip download --set lane.model.name=unet --set detection.model.weights=yolo11s.pt
python stages/train_lane.py --set lane.train.epochs=5 --set lane.data.batch_size=8
```

`make setup | download | preprocess | train-detection | train-lane | evaluate | all | test` are shortcuts.

## Repository layout

```
config/            YAML configs (merged together; local.yaml = personal overrides, gitignored)
docs/              Full documentation
scripts/           venv + gcloud installation, download / pipeline wrappers
stages/            One runnable script per pipeline stage
main.py            Full pipeline entry point
src/waymo_training/
  config.py        YAML loading, ${interpolation}, CLI overrides
  registry.py      name -> implementation registries (what makes parts swappable)
  data/            download, dataset readers, preprocessing
  detection/       detector backends (YOLO / Ultralytics)
  lane/            Keras models, losses, optimizers, tf.data, train, evaluate
tests/             Synthetic-data tests (no dataset needed)
```

## Important to know

- **Lane labels.** Waymo's perception dataset has no lane polylines for camera images. Lane
  masks are derived from the camera _panoptic segmentation_ labels (lane-marker class), which
  exist only for a subset of frames. See [docs/dataset.md](docs/dataset.md).
- **Licenses.** Code is MIT. The Waymo dataset is **non-commercial** and Ultralytics YOLO is
  **AGPL-3.0**. See [NOTICE.md](NOTICE.md) before sharing or deploying anything.
- **Start small.** The default config downloads only 10 training / 3 validation segments.
  Raise `dataset.download.max_segments` when your pipeline works.

## Documentation

[Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) ·
[Configuration reference](docs/configuration.md) · [Dataset](docs/dataset.md) ·
[Training](docs/training.md) · [Evaluation](docs/evaluation.md) ·
[Extending](docs/extending.md) · [Troubleshooting](docs/troubleshooting.md)

## License

MIT — see [LICENSE](LICENSE). Third-party terms: [NOTICE.md](NOTICE.md).
