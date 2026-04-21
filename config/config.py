"""
config/config.py
================
Single source of truth cho toàn bộ project.
Thay đổi hyperparams hoặc đường dẫn chỉ cần sửa file này.
"""

import pathlib
import os

# =============================================================================
# PATHS — tự động resolve dựa trên vị trí file này
# =============================================================================
# Root của project (thư mục chứa config/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Đường dẫn data — hỗ trợ cả local và Kaggle
_KAGGLE_DATA = pathlib.Path("/kaggle/input/hasc-sample-data/SampleData20111104/HascToolDataPrj/SampleData/0_sequence")
_LOCAL_DATA  = PROJECT_ROOT / "SampleData20111104" / "HascToolDataPrj" / "SampleData" / "0_sequence"

DATA_ROOT: pathlib.Path = _KAGGLE_DATA if _KAGGLE_DATA.exists() else _LOCAL_DATA

# Output
OUTPUT_DIR     = PROJECT_ROOT / "outputs"
MODEL_DIR      = OUTPUT_DIR / "model"
FIGURES_DIR    = OUTPUT_DIR / "figures"
LOG_CSV        = OUTPUT_DIR / "training_log.csv"
BEST_MODEL     = OUTPUT_DIR / "best_model.keras"
LABEL_ENCODER  = OUTPUT_DIR / "label_encoder_classes.npy"
HISTORY_FILE   = OUTPUT_DIR / "history.npy"

# =============================================================================
# DATA EXTRACTION PARAMS
# =============================================================================
SUBJECTS = [
    "person101", "person102", "person103",
    "person104", "person105", "person106", "person107",
]
TRAIN_SUBJECTS = SUBJECTS[:6]   # 6 subjects để train
TEST_SUBJECT   = SUBJECTS[6]    # 1 subject để test (person107)

WINDOW_LENGTH = 700   # Độ dài mỗi đoạn time series (samples)
EXTRACT_SIZE  = 15    # Số mẫu trích xuất mỗi transition
NULL_SIZE     = 15    # Số mẫu null (không có CP)
NUM_TRIM      = 100   # Khoảng cách tối thiểu CP với đầu/cuối đoạn

# =============================================================================
# MODEL HYPERPARAMS (Deep ResNet)
# =============================================================================
KERNEL_SIZE   = (3, 25)         # Conv2D kernel
N_FILTER      = 16              # Số CNN filters
DROPOUT_RATE  = 0.3             # Dropout
N_RESBLOCK    = 3               # Số residual blocks
DENSE_WIDTHS  = [50, 40, 30]    # Width các Dense layers
MODEL_NAME    = "HASC_AutoCPD"

# =============================================================================
# TRAINING HYPERPARAMS
# =============================================================================
LEARNING_RATE     = 1e-3
EPOCHS            = 200
BATCH_SIZE        = 32
VALIDATION_SPLIT  = 0.2
EARLY_STOP_PATIENCE = 30
LR_DECAY_STEPS    = 5000
LR_DECAY_RATE     = 1

# =============================================================================
# DETECTION (Sliding Window) PARAMS
# =============================================================================
SLIDE_STEP        = 50    # Bước trượt cửa sổ
SMOOTH_WIDTH      = 700   # Độ rộng moving average
PEAK_HEIGHT       = 0.3   # Ngưỡng xác suất tối thiểu để coi là CP
PEAK_PROMINENCE   = 0.05
DETECT_BATCH_SIZE = 64

# =============================================================================
# RANDOM SEEDS
# =============================================================================
NUMPY_SEED = 2022
TF_SEED    = 2022

# =============================================================================
# RUNTIME
# =============================================================================
# Suppress TF verbose logs (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
TF_LOG_LEVEL = "2"

def setup():
    """Tạo output directories và set up TF log level."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_LOG_LEVEL
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
