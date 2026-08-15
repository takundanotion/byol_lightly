import os
import numpy as np
import torch
import tifffile
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from models.byol_model import BYOLModel
from config import PROJECTION_INPUT

# --- Settings ---
CHECKPOINT_PATH = "checkpoints/exp9b_verified_checkpoint.pth"
TRAIN_LIST      = "data/train_files.txt"
TEST_LIST       = "data/test_files.txt"
OUTPUT_ROOT     = "lightly_eval_data"
NUM_CLUSTERS    = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_tif_as_rgb(path):
    """Same band-selection logic used throughout this project (Max Variance)."""
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
        return [line.strip() for line in f if line.strip() and "_mask" not in line.lower()]


def extract_features(model, files, label):
    print(f"\nExtracting features from {label} set ({len(files)} images)...")
    features = []
    images = []
    with torch.no_grad():
        for i, path in enumerate(files):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(files)}...")
            image = load_tif_as_rgb(path)
            images.append(image)
            tensor = eval_transform(image).unsqueeze(0).to(device)
            feat = model.backbone(tensor).flatten(start_dim=1)
            features.append(feat.cpu().numpy())
    return np.vstack(features), images


def save_labeled_images(images, files, labels, split_name):
    for cls in set(labels):
        os.makedirs(os.path.join(OUTPUT_ROOT, split_name, f"class_{cls}"), exist_ok=True)

    for img, path, label in zip(images, files, labels):
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUTPUT_ROOT, split_name, f"class_{label}", f"{base_name}.png")
        # Save at a reasonable resolution — resize is re-applied at train time anyway
        img.save(out_path)


# ==============================================================
# Main
# ==============================================================

print("="*60)
print("STEP 1 — Loading frozen BYOL backbone")
print("="*60)

model = BYOLModel().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
for p in model.parameters():
    p.requires_grad = False
model.eval()
print(f"Loaded: {CHECKPOINT_PATH}")

print("\n" + "="*60)
print("STEP 2 — Extracting TRAIN features and fitting K-Means")
print("="*60)

train_files = load_file_list(TRAIN_LIST)
train_features, train_images = extract_features(model, train_files, "TRAIN")
train_features_norm = normalize(train_features, norm="l2")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10, max_iter=300)
train_labels = kmeans.fit_predict(train_features_norm)   # FIT on train only

print("\nTrain cluster distribution:")
for c in range(NUM_CLUSTERS):
    count = (train_labels == c).sum()
    print(f"  class_{c}: {count} images")

print("\n" + "="*60)
print("STEP 3 — Extracting TEST features and predicting labels")
print("="*60)

test_files = load_file_list(TEST_LIST)
test_features, test_images = extract_features(model, test_files, "TEST")
test_features_norm = normalize(test_features, norm="l2")

test_labels = kmeans.predict(test_features_norm)   # PREDICT only — no fitting on test

print("\nTest cluster distribution:")
for c in range(NUM_CLUSTERS):
    count = (test_labels == c).sum()
    print(f"  class_{c}: {count} images")

print("\n" + "="*60)
print("STEP 4 — Writing labeled image folders to disk")
print("="*60)

save_labeled_images(train_images, train_files, train_labels, "train")
save_labeled_images(test_images, test_files, test_labels, "val")

print(f"\nDone. Data written to: {OUTPUT_ROOT}/train and {OUTPUT_ROOT}/val")
print("Ready for lightly's official linear_eval() function.")