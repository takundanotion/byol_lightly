# ==============================================================
# linear_eval.py — Experiment 6: Linear Evaluation
# --------------------------------------------------------------
# Step 1: Extract features from frozen BYOL backbone
# Step 2: Cluster features into pseudo-labels using K-Means
# Step 3: Train a linear classifier on top of frozen features
# Step 4: Evaluate and report results
#
# Run from your byol_lightly project folder:
#   python linear_eval.py
# ==============================================================

import os
import torch
import torch.nn as nn
import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
from sklearn.preprocessing import normalize
import csv

# --- Project imports ---
from models.byol_model import BYOLModel
from config import PROJECTION_INPUT


# ==============================================================
# Settings — adjust these if needed
# ==============================================================

CHECKPOINT_PATH = "checkpoints/exp5_100epochs_checkpoint_done.pth"
DATASET_ROOT    = "/home/takunda/Documents/merged_dataset"
NUM_CLUSTERS    = 5       # number of pseudo-classes (road surface types)
BATCH_SIZE      = 32
LINEAR_EPOCHS   = 50      # epochs to train the linear classifier
LEARNING_RATE   = 1e-3
TRAIN_SPLIT     = 0.8     # 80% train, 20% test
NUM_WORKERS     = 2
RESULTS_DIR     = "logs"
RESULTS_CSV     = os.path.join(RESULTS_DIR, "exp6b_weighted_5class_eval.csv")
REPORT_FILE     = os.path.join(RESULTS_DIR, "exp6b_weighted_5class_report.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================
# Dataset — same tif loading logic as training
# ==============================================================

# Standard transform for feature extraction — no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_tif_as_rgb(path):
    """Loads a .tif file and converts it to a 3-channel RGB PIL image."""
    img_array = tifffile.imread(path)

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.float32)
        img_array -= img_array.min()
        if img_array.max() > 0:
            img_array /= img_array.max()
        img_array = (img_array * 255).astype(np.uint8)

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=2)
    elif img_array.ndim == 3:
        channels = img_array.shape[2]
        if channels == 1:
            img_array = np.squeeze(img_array, axis=2)
            img_array = np.stack([img_array] * 3, axis=2)
        elif channels == 2:
            img_array = np.stack([img_array[:,:,0], img_array[:,:,1], img_array[:,:,0]], axis=2)
        elif channels == 3:
            pass
        elif channels == 4:
            img_array = img_array[:, :, :3]
        else:
            # Evenly spaced bands — same as training
            total = channels
            r = img_array[:, :, total // 4]
            g = img_array[:, :, total // 2]
            b = img_array[:, :, (total * 3) // 4]
            img_array = np.stack([r, g, b], axis=2)

    return Image.fromarray(img_array).convert("RGB")


class TIFFeatureDataset(Dataset):
    """Dataset that returns (image_tensor, pseudo_label) pairs."""

    def __init__(self, root, labels, transform):
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".tif", ".tiff"))
        ])
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = load_tif_as_rgb(self.files[idx])
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


# ==============================================================
# Step 1 — Load the trained BYOL backbone
# ==============================================================

print("\n" + "="*60)
print("STEP 1 — Loading trained BYOL backbone")
print("="*60)

model = BYOLModel().to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Final loss: {checkpoint['loss']:.4f}")
print(f"Final Feature Std: {checkpoint['feature_std']:.4f}")

# Freeze the entire backbone — nothing changes during linear eval
for param in model.parameters():
    param.requires_grad = False

model.eval()
print("Backbone frozen — no gradients will flow through it")


# ==============================================================
# Step 2 — Extract features from all images
# ==============================================================

print("\n" + "="*60)
print("STEP 2 — Extracting features from all 8008 images")
print("="*60)

all_files = sorted([
    os.path.join(DATASET_ROOT, f)
    for f in os.listdir(DATASET_ROOT)
    if f.lower().endswith((".tif", ".tiff"))
])

all_features = []

with torch.no_grad():
    for i, path in enumerate(all_files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(all_files)} images...")

        image = load_tif_as_rgb(path)
        tensor = eval_transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # Extract backbone features only — not projection or prediction head
        features = model.backbone(tensor).flatten(start_dim=1)  # (1, 2048)
        all_features.append(features.cpu().numpy())

all_features = np.vstack(all_features)   # (8008, 2048)
print(f"\nExtracted features shape: {all_features.shape}")

# L2 normalise features — important for K-Means to work well
all_features_norm = normalize(all_features, norm="l2")
print("Features L2 normalised")


# ==============================================================
# Step 3 — K-Means clustering to generate pseudo-labels
# ==============================================================

print("\n" + "="*60)
print(f"STEP 3 — K-Means clustering into {NUM_CLUSTERS} pseudo-classes")
print("="*60)

kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10,
    max_iter=300
)

pseudo_labels = kmeans.fit_predict(all_features_norm)

# Report cluster distribution
print("\nCluster distribution:")
for c in range(NUM_CLUSTERS):
    count = (pseudo_labels == c).sum()
    pct   = count / len(pseudo_labels) * 100
    print(f"  Cluster {c}: {count:>5} images ({pct:.1f}%)")

# Silhouette score — measures cluster quality
# Range: -1 (bad) to +1 (perfect). Above 0.2 is reasonable.
sil_sample = min(2000, len(all_features_norm))
sil_idx    = np.random.choice(len(all_features_norm), sil_sample, replace=False)
sil_score  = silhouette_score(
    all_features_norm[sil_idx],
    pseudo_labels[sil_idx]
)
print(f"\nSilhouette Score: {sil_score:.4f}")
print("  (closer to 1.0 = better cluster separation)")


# ==============================================================
# Step 4 — Train linear classifier on frozen features
# ==============================================================

print("\n" + "="*60)
print("STEP 4 — Training linear classifier")
print("="*60)

# Build dataset with pseudo-labels
full_dataset = TIFFeatureDataset(DATASET_ROOT, pseudo_labels, eval_transform)

# Split into train and test
train_size = int(TRAIN_SPLIT * len(full_dataset))
test_size  = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train set: {train_size} images")
print(f"Test  set: {test_size}  images")

# Linear classifier — one layer only
# input = backbone output size (2048 for ResNet50)
# output = number of clusters
linear_classifier = nn.Linear(PROJECTION_INPUT, NUM_CLUSTERS).to(device)

optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
# Weighted loss — penalises mistakes on rare damage classes more heavily
# class_counts = np.bincount(pseudo_labels)
# class_weights = 1.0 / (class_counts + 1e-6)
# class_weights = class_weights / class_weights.sum()
# weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
# criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# CSV logging
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
print("-" * 55)

best_test_acc = 0.0

for epoch in range(1, LINEAR_EPOCHS + 1):

    # --- Training ---
    linear_classifier.train()
    train_losses, train_preds, train_targets = [], [], []

    for images, labels in train_loader:
        images = images.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        with torch.no_grad():
            features = model.backbone(images).flatten(start_dim=1)

        logits = linear_classifier(features)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_preds.extend(logits.argmax(dim=1).cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

    # --- Evaluation ---
    linear_classifier.eval()
    test_losses, test_preds, test_targets = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            features = model.backbone(images).flatten(start_dim=1)
            logits   = linear_classifier(features)
            loss     = criterion(logits, labels)

            test_losses.append(loss.item())
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    train_loss = np.mean(train_losses)
    train_acc  = accuracy_score(train_targets, train_preds)
    test_loss  = np.mean(test_losses)
    test_acc   = accuracy_score(test_targets, test_preds)

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | {test_loss:>9.4f} | {test_acc:>7.1%}")

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, round(train_loss,4), round(train_acc,4), round(test_loss,4), round(test_acc,4)])


# ==============================================================
# Step 5 — Final report
# ==============================================================

print("\n" + "="*60)
print("EXPERIMENT 6 — FINAL RESULTS")
print("="*60)
print(f"Best test accuracy:    {best_test_acc:.1%}")
print(f"Final test accuracy:   {test_acc:.1%}")
print(f"Silhouette score:      {sil_score:.4f}")
print(f"Number of clusters:    {NUM_CLUSTERS}")
print(f"Training images:       {train_size}")
print(f"Test images:           {test_size}")

report = classification_report(
    test_targets, test_preds,
    target_names=[f"Surface {i}" for i in range(NUM_CLUSTERS)]
)
print(f"\nClassification Report:\n{report}")

cm = confusion_matrix(test_targets, test_preds)
print(f"Confusion Matrix:\n{cm}")

# Save full report to file
with open(REPORT_FILE, "w") as f:
    f.write("EXPERIMENT 6 — LINEAR EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Checkpoint:          {CHECKPOINT_PATH}\n")
    f.write(f"Number of clusters:  {NUM_CLUSTERS}\n")
    f.write(f"Silhouette score:    {sil_score:.4f}\n")
    f.write(f"Best test accuracy:  {best_test_acc:.1%}\n")
    f.write(f"Final test accuracy: {test_acc:.1%}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

print(f"\nFull report saved to: {REPORT_FILE}")
print(f"CSV log saved to:     {RESULTS_CSV}")
print("="*60)