# ==============================================================
# config.py — central settings for the whole project
# --------------------------------------------------------------
# Change values here instead of hunting through every file.
# ==============================================================

# --- Dataset ---
DATASET_ROOT = "/home/takunda-mamutse/Documents/merged_dataset"   # folder with your .tif images
IMAGE_EXTENSIONS = (".tif", ".tiff")                      # file types to load

# --- Training ---
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_WORKERS = 2          # background processes that load data in parallel

# --- Model ---
BACKBONE = "resnet50"    # resnet18 or resnet50
PROJECTION_INPUT = 2048  # resnet50 outputs 2048 features (resnet18 = 512)
PROJECTION_HIDDEN = 512
PROJECTION_OUTPUT = 256

# --- Momentum (EMA) schedule ---
MOMENTUM_START = 0.996   # how slow the target network updates at the start
MOMENTUM_END = 1.0       # by the end it barely moves at all

# --- Checkpoints & logs ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = "checkpoints/exp9b_hiddenonly_checkpoint.pth"
LOG_FILE        = "logs/exp9b_hiddenonly_log.txt"
CSV_LOG_FILE    = "logs/exp9b_hiddenonly_log.csv" 

# --- Collapse detection ---
COLLAPSE_THRESHOLD = 0.1  # feature std below this = warning
