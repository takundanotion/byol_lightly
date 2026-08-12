# ==============================================================
# train_test_pipeline.py — SSL on train set, classify test set
# --------------------------------------------------------------
# Step 1: Load frozen BYOL backbone (trained ONLY on train set)
# Step 2: Extract features from TRAIN set images
# Step 3: K-Means clustering on TRAIN set features -> pseudo-labels
# Step 4: Train linear classifier on TRAIN set
# Step 5: Extract features from TEST set (never seen during training)
# Step 6: Predict TEST set labels using trained classifier
# Step 7: Evaluate and save results
#
# Run from your byol_lightly project folder:
#   python train_test_pipeline.py
# ==============================================================

import os
import torch
import torch.nn as nn
import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import DataLoader
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

from models.byol_model import BYOLModel
from config import PROJECTION_INPUT


# ==============================================================
# Settings
# ==============================================================

CHECKPOINT_PATH = "checkpoints/exp9b_hiddenonly_checkpoint.pth"
TRAIN_LIST      = "data/train_files.txt"
TEST_LIST       = "data/test_files.txt"
NUM_CLUSTERS    = 2       # confirmed 2 dominant surface types from earlier experiments
BATCH_SIZE      = 32
LINEAR_EPOCHS   = 50
LEARNING_RATE   = 1e-3
RESULTS_DIR     = "logs"
RESULTS_CSV     = "logs/exp9b_train_test_eval.csv"
REPORT_FILE     = "logs/exp9b_train_test_report.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================
# Image loading — matches training band strategy (Max Variance)
# ==============================================================

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_tif_as_rgb(path):
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
            # Max Variance band selection — matches training
            band_vars = img_array.var(axis=(0, 1))
            top3 = np.argsort(band_vars)[-3:]
            top3 = np.sort(top3)
            r = img_array[:, :, top3[0]]
            g = img_array[:, :, top3[1]]
            b = img_array[:, :, top3[2]]
            img_array = np.stack([r, g, b], axis=2)

    return Image.fromarray(img_array).convert("RGB")


def load_file_list(path):
    with open(path, "r") as f:
        files = [line.strip() for line in f if line.strip() and "_mask" not in line.lower()]
    return files


def extract_features(model, file_list_path, label):
    files = load_file_list(file_list_path)
    print(f"\nExtracting features from {label} set ({len(files)} images)...")

    features = []
    with torch.no_grad():
        for i, path in enumerate(files):
            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{len(files)}...")
            image = load_tif_as_rgb(path)
            tensor = eval_transform(image).unsqueeze(0).to(device)
            feat = model.backbone(tensor).flatten(start_dim=1)
            features.append(feat.cpu().numpy())

    features = np.vstack(features)
    print(f"{label} features shape: {features.shape}")
    return features, files


# ==============================================================
# Step 1 — Load frozen BYOL backbone (trained on TRAIN set only)
# ==============================================================

print("="*60)
print("STEP 1 — Loading BYOL backbone (trained on train set only)")
print("="*60)

model = BYOLModel().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Final training loss: {checkpoint['loss']:.4f}")
print(f"Final training Feature Std: {checkpoint['feature_std']:.4f}")

for param in model.parameters():
    param.requires_grad = False
model.eval()
print("Backbone frozen.")


# ==============================================================
# Step 2 — Extract TRAIN set features
# ==============================================================

print("\n" + "="*60)
print("STEP 2 — Extracting TRAIN set features")
print("="*60)

train_features, train_files = extract_features(model, TRAIN_LIST, "TRAIN")
train_features_norm = normalize(train_features, norm="l2")


# ==============================================================
# Step 3 — K-Means clustering on TRAIN set only
# ==============================================================

print("\n" + "="*60)
print(f"STEP 3 — K-Means clustering TRAIN set into {NUM_CLUSTERS} classes")
print("="*60)

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10, max_iter=300)
train_pseudo_labels = kmeans.fit_predict(train_features_norm)

print("\nTrain cluster distribution:")
for c in range(NUM_CLUSTERS):
    count = (train_pseudo_labels == c).sum()
    print(f"  Cluster {c}: {count:>5} images ({count/len(train_pseudo_labels)*100:.1f}%)")

sil_sample = min(2000, len(train_features_norm))
sil_idx = np.random.choice(len(train_features_norm), sil_sample, replace=False)
train_sil_score = silhouette_score(train_features_norm[sil_idx], train_pseudo_labels[sil_idx])
print(f"\nTrain Silhouette Score: {train_sil_score:.4f}")


# ==============================================================
# Step 4 — Train linear classifier on TRAIN set
# ==============================================================

print("\n" + "="*60)
print("STEP 4 — Training linear classifier on TRAIN set")
print("="*60)

linear_classifier = nn.Linear(PROJECTION_INPUT, NUM_CLUSTERS).to(device)
optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
train_labels_tensor   = torch.tensor(train_pseudo_labels, dtype=torch.long).to(device)

train_dataset = torch.utils.data.TensorDataset(train_features_tensor, train_labels_tensor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9}")
print("-" * 35)

for epoch in range(1, LINEAR_EPOCHS + 1):
    linear_classifier.train()
    losses, preds, targets = [], [], []

    for feats, labels in train_loader:
        logits = linear_classifier(feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    if epoch % 10 == 0 or epoch == 1:
        acc = accuracy_score(targets, preds)
        print(f"{epoch:>6} | {np.mean(losses):>10.4f} | {acc:>8.1%}")


# ==============================================================
# Step 5 — Extract TEST set features (never seen during training)
# ==============================================================

print("\n" + "="*60)
print("STEP 5 — Extracting TEST set features (held out, unseen)")
print("="*60)

test_features, test_files = extract_features(model, TEST_LIST, "TEST")
test_features_norm = normalize(test_features, norm="l2")


# ==============================================================
# Step 6 — Classify TEST set
# ==============================================================

print("\n" + "="*60)
print("STEP 6 — Classifying TEST set (unseen images)")
print("="*60)

# Assign test images to clusters using the SAME K-Means fitted on train set
test_pseudo_labels = kmeans.predict(test_features_norm)

print("\nTest cluster distribution (from train-fitted K-Means):")
for c in range(NUM_CLUSTERS):
    count = (test_pseudo_labels == c).sum()
    print(f"  Cluster {c}: {count:>5} images ({count/len(test_pseudo_labels)*100:.1f}%)")

linear_classifier.eval()
test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)

with torch.no_grad():
    test_logits = linear_classifier(test_features_tensor)
    test_preds  = test_logits.argmax(dim=1).cpu().numpy()

test_acc = accuracy_score(test_pseudo_labels, test_preds)

test_sil_sample = min(2000, len(test_features_norm))
test_sil_idx = np.random.choice(len(test_features_norm), test_sil_sample, replace=False)
test_sil_score = silhouette_score(test_features_norm[test_sil_idx], test_pseudo_labels[test_sil_idx])


# ==============================================================
# Step 7 — Final report
# ==============================================================

print("\n" + "="*60)
print("FINAL RESULTS — TRAIN/TEST SPLIT EVALUATION")
print("="*60)
print(f"Train images:          {len(train_files)}")
print(f"Test images:           {len(test_files)}")
print(f"Train Silhouette:      {train_sil_score:.4f}")
print(f"Test Silhouette:       {test_sil_score:.4f}")
print(f"Test Accuracy:         {test_acc:.1%}")
print("(Classifier trained on train pseudo-labels, evaluated on UNSEEN test images)")

report = classification_report(
    test_pseudo_labels, test_preds,
    target_names=[f"Surface {i}" for i in range(NUM_CLUSTERS)]
)
print(f"\nClassification Report (TEST set):\n{report}")

cm = confusion_matrix(test_pseudo_labels, test_preds)
print(f"Confusion Matrix (TEST set):\n{cm}")

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(REPORT_FILE, "w") as f:
    f.write("EXPERIMENT 8 — TRAIN/TEST SPLIT EVALUATION (clean 4004 images)\n")
    f.write("="*60 + "\n\n")
    f.write(f"Checkpoint:        {CHECKPOINT_PATH}\n")
    f.write(f"Train images:      {len(train_files)}\n")
    f.write(f"Test images:       {len(test_files)}\n")
    f.write(f"Number of clusters:{NUM_CLUSTERS}\n")
    f.write(f"Train Silhouette:  {train_sil_score:.4f}\n")
    f.write(f"Test Silhouette:   {test_sil_score:.4f}\n")
    f.write(f"Test Accuracy:     {test_acc:.1%}\n\n")
    f.write("Classification Report (TEST set):\n")
    f.write(report + "\n")
    f.write("Confusion Matrix (TEST set):\n")
    f.write(str(cm) + "\n")

with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["train_images","test_images","train_silhouette","test_silhouette","test_accuracy"])
    writer.writerow([len(train_files), len(test_files), round(train_sil_score,4), round(test_sil_score,4), round(test_acc,4)])

print(f"\nReport saved to: {REPORT_FILE}")
print(f"CSV saved to:    {RESULTS_CSV}")
print("="*60)