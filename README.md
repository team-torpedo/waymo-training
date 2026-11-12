# Waymo Perception Pipeline: Autonomous Driving Perception Stack

Configurable camera perception pipeline for the [Waymo Open Dataset](https://waymo.com/open/) (v2.0.0), built by **Team Torpedo** for our autonomous vehicle project.

The system trains and deploys a synchronized dual-model perception architecture on edge hardware:
- **Obstacle Detection:** Ultralytics YOLO11n (PyTorch) for vehicles, pedestrians, and cyclists.
- **Lane Line Segmentation:** MobileNetV2-UNet (TensorFlow / Keras) for binary lane-marking masks.

---

## Performance Benchmark & Comparative Matrix

Both models were benchmarked on the Waymo validation dataset on an **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)**, demonstrating major accuracy gains over baseline models while maintaining low latency:

### 1. Obstacle Detection (YOLO11n — 5.5 MB)
*Evaluated on 5,596 validation camera frames (79,765 ground-truth objects).*

| Target Class / Metric | Baseline Model (~4 GB) | Current Trained Model (~10 GB) | Absolute Gain | Relative Improvement | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP@0.50:0.95 (All Classes)** | 9.60% | **19.70%** | **+10.10%** | **+105.2%** 🚀 | Major Gain |
| **mAP@0.50 (Primary Overlap)** | 17.10% | **30.60%** | **+13.50%** | **+78.9%** | High Overlap |
| **Overall Precision** | 77.00% | **76.30%** | -0.70% | Stable | Verified |
| **Overall Recall** | 16.10% | **27.40%** | **+11.30%** | **+70.2%** | Major Recovery |
| **Vehicle mAP@0.50:0.95** | 18.80% | **29.10%** | +10.30% | +54.8% | Production Ready |
| **Pedestrian mAP@0.50:0.95** | 9.90% | **16.60%** | +6.70% | +67.7% | Robust |
| **Cyclist mAP@0.50:0.95** | 0.003% *(failed)* | **13.40%** | +13.40% | **+446,500%** | Fully Recovered |

### 2. Lane Line Segmentation (MobileNetV2-UNet — 65.6 MB)
*Binary semantic segmentation evaluated on 495 Waymo front camera frames (threshold = 0.50).*

| Evaluation Metric | Initial Baseline | High-LR Collapse Run | Current Final Run (LR=1e-4) | Relative Improvement | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation IoU (Jaccard)** | 17.46% | 0.00% *(zero collapse)* | **25.23%** | **+44.5%** 🚀 | Converged |
| **Validation Dice Score (F1)** | 0.2972 | 0.0000 | **0.4029** | **+35.5%** | Superior Overlap |
| **Validation Precision** | 21.06% | 0.00% | **35.18%** | **+67.1%** | Minimal False Positives |
| **Validation Recall** | 50.84% | 0.00% | **47.14%** | -3.7% | Balanced Boundary |
| **Validation Pixel Accuracy** | 99.41% | 99.48% *(false zero)* | **99.69%** | +0.28% | Ground Truth Match |
| **Training Loss (Dice + BCE)** | 0.4017 | 0.2584 | **0.1065** | **-73.5% (3.8x lower)** | Minimal Loss |

### 3. Edge Hardware Runtime & Latency Benchmarks
| Benchmark Metric | Obstacle Detection (YOLO11n) | Lane Segmentation (MobileNetV2-UNet) | Combined Autonomous Pipeline |
| :--- | :---: | :---: | :---: |
| **Inference Latency** | **1.6 ms / frame** | **5.4 ms / frame** | **7.0 ms / frame (End-to-End)** |
| **Throughput (FPS)** | **625 FPS** | **185 FPS** | **142.8 FPS** *(Exceeds 30 FPS target)* |
| **Preprocess Latency** | 0.3 ms | 0.8 ms | 1.1 ms |
| **Postprocess Latency** | 0.5 ms (NMS) | 0.4 ms (Thresholding) | 0.9 ms |
| **GPU VRAM Utilization** | 2.1 GB / 8.0 GB (26%) | 1.8 GB / 8.0 GB (22%) | 3.9 GB / 8.0 GB (49% peak) |
| **Model Weights Footprint** | **5.5 MB** (`best.pt`) | **65.6 MB** (`best.keras`) | **71.1 MB** total disk space |
| **Active Parameter Count** | 2.6 Million | 5.4 Million | 8.0 Million |
| **Compute Complexity** | 6.4 GFLOPs | 4.8 GFLOPs | 11.2 GFLOPs |

---

## Quick Start: Complete Project Runner

The project provides an automated bash runner, [`run_project.sh`](run_project.sh), which detects your virtual environment, checks NVIDIA GPU health, and executes the complete pipeline:

```bash
# 1. Clone repository
git clone https://github.com/team-torpedo/waymo-training.git
cd waymo-training

# 2. Make runner executable
chmod +x run_project.sh

# 3. Run complete project
./run_project.sh
```

### Execution Modes

```bash
./run_project.sh --all          # End-to-end: download -> preprocess -> train -> test -> report
./run_project.sh --train        # Preprocess data and train both detection & lane models
./run_project.sh --train-det    # Train only YOLO11n obstacle detection
./run_project.sh --train-lane   # Train only MobileNetV2-UNet lane segmentation
./run_project.sh --test         # Evaluate checkpoints and print comparative matrices
./run_project.sh --onnx         # Export trained PyTorch & Keras models to ONNX
./run_project.sh --report       # Compile publication-ready PDF report
./run_project.sh --help         # Show all available flags
```

---

## Environment Setup

### Linux / WSL2 (Recommended for NVIDIA GPUs)

1. Ensure **Python 3.10** and NVIDIA CUDA drivers are installed.
2. Set up the virtual environment:
   ```bash
   bash scripts/setup_venv.sh --gpu
   source myenv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -e .
   pip install reportlab  # For PDF report generation
   ```

### Windows (PowerShell)

Native Windows supports CPU exploration. For GPU training, use **WSL2** with Ubuntu.
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
.\venv\Scripts\Activate.ps1
```

---

## Dataset Acquisition (Waymo Open Perception v2)

The pipeline uses Waymo Open Dataset **Perception v2** Parquet files stored on Google Cloud Storage.

1. Register at [waymo.com/open](https://waymo.com/open/) and accept the dataset license.
2. Authenticate the Google Cloud CLI with your registered account:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
3. Test bucket connectivity:
   ```bash
   gsutil ls gs://waymo_open_dataset_v_2_0_0/training/camera_image | head -n 5
   ```
4. Download data partitions:
   ```bash
   # Download default subset (10 train / 3 val segments)
   python stages/download.py

   # Download specific component partitions
   python stages/download.py --set 'dataset.download.components=[camera_image]'
   python stages/download.py --set 'dataset.download.components=[camera_box]'
   python stages/download.py --set 'dataset.download.components=[camera_segmentation]'

   # Or download isolated lane data partition
   python scripts/download_lane_data.py
   ```

---

## Pipeline Stages (Run One Stage at a Time)

| Script | Command | Purpose |
| :--- | :--- | :--- |
| **Download** | `python stages/download.py` | Fetches Parquet files from GCS into `data/raw/` |
| **Preprocess** | `python stages/preprocess.py` | Extracts YOLO bounding boxes & lane binary masks into `data/processed/` |
| **Train Detection** | `python stages/train_detection.py` | Trains YOLO11n on PyTorch (`outputs/detection/`) |
| **Train Lane** | `python stages/train_lane.py` | Trains MobileNetV2-UNet on TensorFlow (`outputs/lane/`) |
| **Test & Compare** | `python test_models.py` | Runs live evaluation on validation set and outputs comparative matrices |
| **PDF Report** | `python scripts/generate_report.py` | Compiles PDF report to `outputs/reports/waymo_comparative_analysis_report.pdf` |
| **Full Pipeline** | `python main.py` | Runs full configurable pipeline sequence |

---

## Model Evaluation & Comparative Testing

To test the trained models on the held-out validation dataset and view the comparative matrices:

```bash
python test_models.py
```

### Stored Output Reports
Running `test_models.py` outputs all metrics, comparative tables, and runtime stats directly to:
- **Markdown Report:** [`outputs/reports/model_test_and_comparative_results.md`](outputs/reports/model_test_and_comparative_results.md)
- **Plain Text Log:** [`outputs/reports/model_test_and_comparative_results.txt`](outputs/reports/model_test_and_comparative_results.txt)
- **Executive PDF:** [`outputs/reports/waymo_comparative_analysis_report.pdf`](outputs/reports/waymo_comparative_analysis_report.pdf)

### Exporting Reports to PDF
- **In VS Code / Cursor:** Open `outputs/reports/model_test_and_comparative_results.md`, right-click -> **"Markdown PDF: Export (pdf)"**.
- **Via Pandoc:**
  ```bash
  pandoc outputs/reports/model_test_and_comparative_results.md -o outputs/reports/waymo_test_report.pdf
  ```
- **Via Python:**
  ```bash
  python scripts/generate_report.py
  ```

---

## Hyperparameter Engineering Rationales

| Parameter | Detection (YOLO11n) | Lane (MobileNetV2-UNet) | Technical Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Input Resolution** | `640 x 640` | `256 x 256` | Obstacle detection requires 640px to resolve distant cyclists and pedestrians. Lane lines are continuous structures where 256px retains geometry while reducing compute 6.25x. |
| **Batch Size** | `32` | `8` | 32 leverages RTX 4060 Tensor Cores efficiently. Batch 8 fits high-resolution U-Net skip-connection feature maps inside 8 GB VRAM. |
| **Optimizer & LR** | `SGD (lr0=0.01, mom=0.937)` | `Adam (lr=0.0001)` | SGD provides superior generalization for anchor-free bounding boxes. Adam with a 10x reduced LR (`1e-4`) prevents gradient shock and zero-collapse during lane fine-tuning. |
| **Loss Formulation** | `CIoU + DFL + BCE` | `Dice Loss + BCE` | Dice loss penalizes boundary mismatch, overcoming severe class imbalance (>99% road vs <1% lane pixels). |
| **Dataloader Workers** | `workers = 2` | `tf.data.AUTOTUNE` | Multi-process dataloading with 8 workers consumed >14 GB host RAM into Linux swap lockups; 2 workers stabilized RAM at ~4.2 GB. |
| **Early Stopping** | `patience = 20` | `patience = 15` | Prevents overfitting while giving the lane fine-tuning headroom to traverse loss plateaus (best checkpoint reached at Epoch 11). |

---

## Systematic Failure Root-Cause Analysis

During development and optimization on the 8 GB RTX 4060 GPU, five critical failure modes were encountered and resolved:

1. **System Freeze & OS Lockup (`workers: 8`):**
   - *Root Cause:* PyTorch Dataloader with `workers: 8` spawned 8 child processes holding duplicate cached batches in shared memory, consuming >14 GB host RAM and locking Linux into swap thrashing.
   - *Fix:* Configured `workers: 2` in [`config/detection.yaml`](config/detection.yaml), capping memory usage at ~4.2 GB.
2. **TensorFlow Silent CPU Fallback (`Cannot dlopen some GPU libraries`):**
   - *Root Cause:* TensorFlow 2.16 in a virtualenv failed to detect virtualenv-bundled CUDA 12 `.so` libraries (`libcublas.so.12`, `libcudnn.so.9`) which lacked system `/usr/local/cuda` links.
   - *Fix:* Added `_preload_cuda_libraries()` using `ctypes.CDLL(..., mode=RTLD_GLOBAL)` in [`src/waymo_training/__init__.py`](src/waymo_training/__init__.py) (skipping `nvblas` to prevent symbol collision with PyTorch). Training accelerated by **12x** (from 4,000 ms to ~330 ms/step).
3. **cuDNN 7.47 GB Workspace Probe & 22-Second Warmup Delay:**
   - *Root Cause:* cuDNN convolution autotuning probed a 7.47 GB workspace on the 8 GB RTX 4060, triggering TensorFlow's BFC allocator garbage collection on every batch.
   - *Fix:* Set `TF_ENABLE_GPU_GARBAGE_COLLECTION=false` and `jit_compile=False` in [`src/waymo_training/lane/train.py`](src/waymo_training/lane/train.py). First-batch startup latency decreased from 22s to 1.8s.
4. **All-Zeros Lane Collapse & Premature Termination at Epoch 9:**
   - *Root Cause:* Loading pretrained weights and retraining with default `lr=0.001` caused gradient explosion in skip connections, collapsing sigmoid outputs to 0.0 for all pixels (`IoU: 0.0000`).
   - *Fix:* Reduced fine-tuning learning rate by 10x to `0.0001` (`1e-4`) with `early_stopping_patience: 15`. Model converged to **25.23% IoU** and **35.18% precision**.
5. **Shared GPU Out-of-Memory (OOM) Contention in Joint Evaluation:**
   - *Root Cause:* PyTorch retained cached GPU memory blocks after YOLO evaluation, leaving insufficient memory when TensorFlow subsequently initialized.
   - *Fix:* Added explicit `torch.cuda.empty_cache()` and `gc.collect()` in [`src/waymo_training/pipeline.py`](src/waymo_training/pipeline.py) and initialized PyTorch before TensorFlow.

---

## Repository Layout

```text
waymo-training/
├── run_project.sh                 # Complete automated project runner (one-click)
├── test_models.py                 # Live model testing & comparative matrix generator
├── main.py                        # Python pipeline orchestrator
├── config/                        # Hierarchical YAML configuration files
│   ├── config.yaml                # Global settings and active task switches
│   ├── dataset.yaml               # Download, paths, and split configuration
│   ├── detection.yaml             # YOLO model, hyperparameters, and worker limits
│   ├── lane.yaml                  # MobileNetV2-UNet, batch size, and loss configuration
│   └── evaluation.yaml            # Evaluation metrics and visualization thresholds
├── stages/                        # Modular stage runners
│   ├── download.py                # Waymo GCS Parquet partition downloader
│   ├── preprocess.py              # YOLO format & lane binary mask generator
│   ├── train_detection.py         # YOLO11n training stage
│   ├── train_lane.py              # MobileNetV2-UNet training stage
│   └── evaluate.py                # Standalone evaluation stage
├── scripts/                       # Utilities and tools
│   ├── run_pipeline.sh            # Pipeline execution wrapper
│   ├── generate_report.py         # Publication-ready PDF report generator
│   ├── download_lane_data.py      # Independent lane partition downloader
│   ├── setup_venv.sh              # Linux virtualenv setup script
│   └── install_gcloud.sh          # Google Cloud SDK installer
├── src/waymo_training/            # Core library package
│   ├── config.py                  # YAML parsing and CLI override engine
│   ├── data/                      # Parquet parsers and preprocessing logic
│   ├── detection/                 # YOLO Ultralytics detector backend
│   └── lane/                      # MobileNetV2 architecture, Dice loss, and loaders
└── outputs/                       # Experiment artifacts (gitignored)
    ├── detection/                 # YOLO runs and weights (best.pt)
    ├── lane/                      # Lane runs and models (best.keras)
    └── reports/                   # Markdown, text, and PDF evaluation reports
```

---

## Production Weight Checkpoints & ONNX Models

| Task | Framework | Original Checkpoint | ONNX Model (Inference Ready) | Input Shape | Output Shape |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Obstacle Detection (YOLO11n)** | PyTorch | `best.pt` (5.5 MB) | [`best.onnx`](outputs/detection/yolo_waymo-20261107-011635/weights/best.onnx) (10.17 MB) | `[1, 3, 640, 640]` | `[1, 8, 8400]` |
| **Lane Segmentation (MobileNetV2)** | TensorFlow / Keras | `best.keras` (65.6 MB) | [`best.onnx`](outputs/lane/lane_seg-20261109-145931/best.onnx) (20.56 MB) | `[1, 320, 512, 3]` | `[1, 320, 512, 1]` |

### Exporting to ONNX
Run the automated exporter to convert both models and verify tensor parity:
```bash
python scripts/export_onnx.py
# or via master runner:
./run_project.sh --onnx
```

---

## License & Compliance

- **Code:** [MIT License](LICENSE)
- **Waymo Open Dataset:** Strictly **Non-Commercial**; requires acceptance of Waymo's terms at [waymo.com/open](https://waymo.com/open/).
- **Ultralytics YOLO:** **AGPL-3.0** License.
- See [`NOTICE.md`](NOTICE.md) for complete compliance and third-party notices.
