#!/usr/bin/env bash
# ==============================================================================
# Waymo Autonomous Perception Pipeline: Complete Project Runner
# Team Torpedo - Autonomous Vehicle Perception Stack
# ==============================================================================
# This script orchestrates the complete perception pipeline:
#   1. Virtual environment and NVIDIA GPU verification
#   2. Data download & preprocessing (YOLO boxes + Lane masks)
#   3. Obstacle Detection Training (YOLO11n via PyTorch)
#   4. Lane Line Segmentation Training (MobileNetV2-UNet via TensorFlow)
#   5. Model Testing, Comparative Matrices & Report Generation
# ==============================================================================

set -euo pipefail

# ANSI Color codes for clean terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ------------------------------------------------------------------------------
# 1. Environment & Python Resolution
# ------------------------------------------------------------------------------
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_succ()  { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${CYAN}${BOLD}>>> $*${NC}"; }

# Detect virtual environment: active venv > ./myenv > ./venv
PYTHON_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${ROOT_DIR}/myenv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/myenv/bin/python"
elif [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    log_err "No valid Python interpreter found. Please create 'myenv' or activate a virtualenv."
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Configure Runtime & GPU Environment Variables
# ------------------------------------------------------------------------------
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_GPU_GARBAGE_COLLECTION=false
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================================"
    echo "        WAYMO AUTONOMOUS PERCEPTION PIPELINE: FULL PROJECT RUNNER"
    echo "================================================================================"
    echo -e "${NC}"
    log_info "Workspace Root : ${ROOT_DIR}"
    log_info "Python Binary  : ${PYTHON_BIN} ($(${PYTHON_BIN} -V))"

    # Hardware detection
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo "Unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n 1 || echo "Unknown")
        log_succ "GPU Detected   : ${GPU_NAME} (${GPU_MEM})"
    else
        log_warn "No NVIDIA GPU detected by nvidia-smi; falling back to CPU."
    fi
    echo ""
}

show_help() {
    echo "Usage: ./run_project.sh [OPTIONS]"
    echo ""
    echo "Execution Modes:"
    echo "  --all              Run complete pipeline from scratch (download -> preprocess -> train -> test -> report)"
    echo "  --train            Preprocess data and train both detection and lane models"
    echo "  --train-det        Train only YOLO11n obstacle detection"
    echo "  --train-lane       Train only MobileNetV2-UNet lane segmentation"
    echo "  --test, --eval     Run model testing on validation dataset and generate comparative matrix"
    echo "  --report           Compile PDF analysis report (outputs/reports/waymo_comparative_analysis_report.pdf)"
    echo "  --quick-test       Run quick test on small batch"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Default (no args): Auto-detects data state; trains if checkpoints missing, tests models, and updates reports."
}

# ------------------------------------------------------------------------------
# 3. Stage Execution Handlers
# ------------------------------------------------------------------------------
run_download() {
    log_step "Stage 1/5: Downloading Waymo Open Dataset Parquet Partitions..."
    "${PYTHON_BIN}" stages/download.py "$@"
    log_succ "Dataset download stage completed."
}

run_preprocess() {
    log_step "Stage 2/5: Preprocessing Sensor Data (YOLO Bounding Boxes & Lane Masks)..."
    "${PYTHON_BIN}" stages/preprocess.py "$@"
    log_succ "Preprocessing completed. Processed data ready in data/processed/."
}

run_train_detection() {
    log_step "Stage 3/5: Training Obstacle Detection (YOLO11n on PyTorch)..."
    "${PYTHON_BIN}" stages/train_detection.py "$@"
    log_succ "Obstacle detection training completed. Checkpoints in outputs/detection/."
}

run_train_lane() {
    log_step "Stage 4/5: Training Lane Line Segmentation (MobileNetV2-UNet on TensorFlow)..."
    "${PYTHON_BIN}" stages/train_lane.py "$@"
    log_succ "Lane line segmentation training completed. Checkpoints in outputs/lane/."
}

run_test_and_compare() {
    log_step "Stage 5/5: Testing Models & Generating Comparative Matrices..."
    "${PYTHON_BIN}" test_models.py "$@"
    log_succ "Model testing and comparative matrices completed."
}

run_pdf_report() {
    log_step "Compiling Executive PDF Report..."
    if "${PYTHON_BIN}" -c "import reportlab" &>/dev/null; then
        "${PYTHON_BIN}" scripts/generate_report.py
        log_succ "PDF Report compiled at: outputs/reports/waymo_comparative_analysis_report.pdf"
    else
        log_warn "ReportLab not installed in virtual environment; skipping PDF generation."
        log_info "To install: pip install reportlab"
    fi
}

# ------------------------------------------------------------------------------
# 4. Command Line Argument Dispatcher
# ------------------------------------------------------------------------------
print_banner

MODE="default"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            MODE="all"
            shift
            ;;
        --train)
            MODE="train"
            shift
            ;;
        --train-det)
            MODE="train-det"
            shift
            ;;
        --train-lane)
            MODE="train-lane"
            shift
            ;;
        --test|--eval)
            MODE="test"
            shift
            ;;
        --report)
            MODE="report"
            shift
            ;;
        --quick-test)
            MODE="quick-test"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_warn "Unrecognized option: $1"
            show_help
            exit 1
            ;;
    esac
done

START_TIME=$(date +%s)

case "$MODE" in
    all)
        log_info "Running Complete Pipeline (End-to-End)..."
        run_download
        run_preprocess
        run_train_detection
        run_train_lane
        run_test_and_compare
        run_pdf_report
        ;;
    train)
        log_info "Running Data Preprocessing and Dual Model Training..."
        run_preprocess
        run_train_detection
        run_train_lane
        run_test_and_compare
        ;;
    train-det)
        log_info "Running Obstacle Detection (YOLO11n) Training & Evaluation..."
        run_train_detection
        "${PYTHON_BIN}" test_models.py --detection-only
        ;;
    train-lane)
        log_info "Running Lane Segmentation (MobileNetV2-UNet) Training & Evaluation..."
        run_train_lane
        "${PYTHON_BIN}" test_models.py --lane-only
        ;;
    test)
        log_info "Running Model Testing and Comparative Analysis..."
        run_test_and_compare
        ;;
    report)
        log_info "Compiling Analysis PDF Report..."
        run_pdf_report
        ;;
    quick-test)
        log_info "Running Quick Validation Verification..."
        "${PYTHON_BIN}" test_models.py
        ;;
    default)
        log_info "Executing Perception Pipeline Workflow..."
        # If processed data doesn't exist, preprocess raw data
        if [[ ! -d "data/processed/yolo" || ! -d "data/processed/lane" ]]; then
            log_info "Processed dataset missing; running preprocessing stage..."
            run_preprocess
        else
            log_succ "Processed dataset verified in data/processed/."
        fi

        # If model checkpoints exist, run evaluation & report; otherwise run training
        if [[ -f "outputs/detection/latest.json" && -f "outputs/lane/latest.json" ]]; then
            log_succ "Existing trained model checkpoints detected."
            log_info "Running validation tests, comparative benchmarks, and reporting..."
            run_test_and_compare
            run_pdf_report
        else
            log_info "Model checkpoints missing; initiating training stages..."
            run_train_detection
            run_train_lane
            run_test_and_compare
            run_pdf_report
        fi
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}${BOLD}================================================================================"
echo "          WAYMO PERCEPTION PIPELINE EXECUTION COMPLETED (${ELAPSED}s)"
echo "================================================================================"
echo -e "${NC}"
echo "Key Output Files:"
echo "  - Test Results & Tables (Markdown) : outputs/reports/model_test_and_comparative_results.md"
echo "  - Test Results & Tables (Text)     : outputs/reports/model_test_and_comparative_results.txt"
if [[ -f "outputs/reports/waymo_comparative_analysis_report.pdf" ]]; then
    echo "  - Executive Report (PDF)           : outputs/reports/waymo_comparative_analysis_report.pdf"
fi
echo ""
