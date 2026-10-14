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

### Windows (PowerShell)

Install [Python 3.10–3.12](https://www.python.org/downloads/) and Git, then run:

```powershell
git clone https://github.com/team-torpedo/waymo-training.git
Set-Location waymo-training

powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

The setup script creates `venv`, installs the project dependencies, and installs the project in
editable mode. Add `-Dev` to install the test and lint dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 -Dev
```

GPU training is not supported by this setup on native Windows. Use WSL2 with an NVIDIA GPU and the
Linux setup below for GPU training.

### Linux / macOS

```bash
git clone https://github.com/team-torpedo/waymo-training.git
cd waymo-training

bash scripts/setup_venv.sh            # creates ./venv  (add --gpu for NVIDIA GPUs on Linux)
source venv/bin/activate
```

## Download the Waymo Open Perception Dataset

The pipeline downloads Waymo Open Dataset **Perception v2** Parquet files from Google Cloud
Storage. You must register at [waymo.com/open](https://waymo.com/open/) and accept the dataset
license before downloading. Waymo controls access to its bucket; approval may take a day or two.
The dataset is not included in this repository.

1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) for your operating
   system. On Linux or macOS, the repository also provides `bash scripts/install_gcloud.sh`.
2. Authenticate with the Google account registered with Waymo:

   ```text
   gcloud auth login
   gcloud auth application-default login
   ```

3. From the repository root, verify that you can list the bucket:

   **PowerShell**

   ```powershell
   gsutil ls gs://waymo_open_dataset_v_2_0_0/training/camera_image | Select-Object -First 5
   ```

   **Linux / macOS**

   ```bash
   gsutil ls gs://waymo_open_dataset_v_2_0_0/training/camera_image | head
   ```

4. With the project virtual environment activated, download the configured subset:

   **PowerShell**

   ```powershell
   python stages\download.py
   ```

   **Linux / macOS**

   ```bash
   python stages/download.py
   ```

   By default this downloads 10 training and 3 validation segments and the Parquet components
   needed by the enabled tasks. Files are saved under `data/raw/`. To download a different number
   of segments, pass config overrides, for example:

   ```text
   python stages/download.py --set dataset.download.max_segments.train=2 --set dataset.download.max_segments.val=1
   ```

   The project supports these Perception v2 Parquet components:

   | Component | Contains | Used for |
   | --------- | -------- | -------- |
   | `camera_image` | Camera frames | Detection and lane segmentation |
   | `camera_box` | 2D object bounding boxes | Object detection |
   | `camera_segmentation` | Panoptic camera segmentation labels | Lane segmentation |

   To download **all segments** for each supported component in both the training and validation
   splits, run each command below. Download components separately so each component can include
   all of its available segments; segmentation labels are available for fewer segments than images.
   Replace `python` with `.\venv\Scripts\python.exe` if you have not activated the environment.

   **PowerShell**

   ```powershell
   python stages\download.py --set dataset.download.max_segments.train=null --set dataset.download.max_segments.val=null --set 'dataset.download.components=[camera_image]'
   python stages\download.py --set dataset.download.max_segments.train=null --set dataset.download.max_segments.val=null --set 'dataset.download.components=[camera_box]'
   python stages\download.py --set dataset.download.max_segments.train=null --set dataset.download.max_segments.val=null --set 'dataset.download.components=[camera_segmentation]'
   ```

   **Linux / macOS**

   ```bash
   python stages/download.py --set dataset.download.max_segments.train=null --set dataset.download.max_segments.val=null --set 'dataset.download.components=[camera_image]'
   python stages/download.py --set dataset.download.max_segments.train=null --set dataset.download.max_segments.val=null --set 'dataset.download.components=[camera_box]'
   python stages/download.py --set dataset.download.max_segments.train=null --set dataset.download.max_segments.val=null --set 'dataset.download.components=[camera_segmentation]'
   ```

   These commands download only the components and splits this repository is configured for; they
   do not download other Waymo Open Dataset products or a test split. Downloading every segment can
   require hundreds of gigabytes or more. See [docs/dataset.md](docs/dataset.md) for storage layout,
   component details, and label limitations.

Once the dataset is downloaded, run the full pipeline with `python main.py`; it downloads any
configured data that is still missing, preprocesses it, trains both models, and evaluates them.
Use `python main.py --skip download` to skip the download stage.

On Windows, run `python main.py` from the activated PowerShell environment.

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
