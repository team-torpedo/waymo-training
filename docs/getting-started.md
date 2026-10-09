# Getting started

## 1. Requirements

- Linux, macOS or Windows (GPU training: Linux or WSL2 with an NVIDIA GPU).
- Python **3.10 – 3.12** recommended (TensorFlow/PyTorch wheels lag behind the newest Python).
- Disk: a few GB for the default small subset; the full camera data is hundreds of GB.

## 2. Create the virtual environment

```bash
bash scripts/setup_venv.sh            # CPU / macOS
bash scripts/setup_venv.sh --gpu      # Linux + NVIDIA GPU (installs tensorflow[and-cuda])
bash scripts/setup_venv.sh --dev      # + pytest and ruff
PYTHON=python3.11 bash scripts/setup_venv.sh   # pick an interpreter
source venv/bin/activate
```

The environment is always called `venv` and lives in the repo root (it is gitignored). The script
installs `requirements.txt` and then the package itself in editable mode.

PyTorch (pulled in by Ultralytics) and TensorFlow both want CUDA libraries. The `--gpu` flag handles
TensorFlow's; PyTorch's wheels from PyPI bundle their own CUDA runtime. If you hit a conflict, see
[troubleshooting](troubleshooting.md).

## 3. Get dataset access

1. Register and accept the license at <https://waymo.com/open/> with a Google account.
   Access to the Cloud Storage bucket can take a day or two.
2. `bash scripts/install_gcloud.sh` (or install the Google Cloud SDK yourself).
3. `gcloud auth login` and `gcloud auth application-default login`.
4. Verify: `gsutil ls gs://waymo_open_dataset_v_2_0_0/training/camera_image | head`.

## 4. Run it

```bash
python main.py
```

This downloads a small subset, converts it, trains both models and evaluates them. Results:

```
data/raw/                     downloaded parquet files
data/processed/yolo/          YOLO dataset (+ data.yaml)
data/processed/lane/          image + mask pairs
outputs/detection/<run>/      YOLO run (weights/best.pt, curves, ...)
outputs/lane/<run>/           best.keras, last.keras, history.csv, tensorboard/, config.yaml
outputs/evaluation/           metrics.json per task, summary.json, prediction images
```

Watch training: `tensorboard --logdir outputs/lane` (Keras) — YOLO writes plots into its run folder.

## 5. Scale up

Edit `config/dataset.yaml`:

```yaml
dataset:
  download:
    max_segments: { train: 200, val: 50 } # or null for everything
```

then rerun `python main.py --stages download preprocess train_detection train_lane evaluate`.

## Verifying the install without the dataset

```bash
python -m pytest -q          # synthetic-data tests (needs requirements-dev.txt)
```
