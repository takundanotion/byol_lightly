# ==============================================================
# prepare_split.py — splits your CLEAN dataset into train/test
# --------------------------------------------------------------
# Run this ONCE before Tier 2 evaluation:
#   python prepare_split.py
#
# Creates:
#   data/train_files.txt  — 80% of images for BYOL training
#   data/test_files.txt   — 20% of images for evaluation
#
# Mask files are automatically excluded.
# Split is random but fixed (seed=42) — always reproducible.
# ==============================================================

import os
import random

# --- Settings — UPDATE THIS PATH to match your machine ---
DATASET_ROOT = "/home/takunda-mamutse/Documents/merged_dataset"
TRAIN_RATIO  = 0.80
SEED         = 42
OUTPUT_DIR   = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all image files — EXCLUDING mask files
all_files = sorted([
    os.path.join(DATASET_ROOT, f)
    for f in os.listdir(DATASET_ROOT)
    if f.lower().endswith((".tif", ".tiff")) and "_mask" not in f.lower()
])

print(f"Total genuine images found (masks excluded): {len(all_files)}")

# Shuffle with fixed seed for reproducibility
random.seed(SEED)
random.shuffle(all_files)

# Split
split_idx   = int(len(all_files) * TRAIN_RATIO)
train_files = all_files[:split_idx]
test_files  = all_files[split_idx:]

print(f"Training set:  {len(train_files)} images ({TRAIN_RATIO*100:.0f}%)")
print(f"Test set:      {len(test_files)} images ({(1-TRAIN_RATIO)*100:.0f}%)")

# Save to text files — one file path per line
train_path = os.path.join(OUTPUT_DIR, "train_files.txt")
test_path  = os.path.join(OUTPUT_DIR, "test_files.txt")

with open(train_path, "w") as f:
    f.write("\n".join(train_files))

with open(test_path, "w") as f:
    f.write("\n".join(test_files))

print(f"\nSaved train list → {train_path}")
print(f"Saved test list  → {test_path}")
print("\nDone. Now update config.py and dataset.py, then run byol_lightly.py")
print("to train BYOL on the TRAINING SET ONLY.")