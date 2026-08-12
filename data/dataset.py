# ==============================================================
# data/dataset.py — loads .tif images from a flat folder OR
#                    a file list (for train/test split)
# --------------------------------------------------------------
# Handles mixed .tif types:
#   - hyperspectral (e.g. 944 channels) -> Max Variance band selection
#   - single channel grayscale (1 channel) -> converts to RGB
#   - standard RGB (3 channels) -> passes through
#
# Mask files (_mask.tif) are automatically excluded everywhere.
# ==============================================================

import os
from PIL import Image
import tifffile
import numpy as np
from torch.utils.data import Dataset
from lightly.transforms.byol_transform import BYOLTransform


class TIFDataset(Dataset):
    """
    Reads .tif / .tiff images either from:
      - a flat folder (root=..., file_list=None)
      - a specific list of file paths (file_list=..., root=None)

    Returns TWO augmented views per image for BYOL training.
    Mask files are always excluded automatically.
    """

    def __init__(self, root=None, file_list=None, extensions=(".tif", ".tiff")):
        self.transform = BYOLTransform()

        if file_list is not None:
            # Load from a text file containing one path per line
            with open(file_list, "r") as f:
                self.files = [
                    line.strip() for line in f
                    if line.strip() and "_mask" not in line.lower()
                ]
            print(f"Loaded {len(self.files)} images from file list: {file_list}")

        elif root is not None:
            # Load everything from a folder
            self.files = [
                os.path.join(root, f)
                for f in sorted(os.listdir(root))
                if f.lower().endswith(extensions) and "_mask" not in f.lower()
            ]
            print(f"Found {len(self.files)} images in {root}")

        else:
            raise ValueError("Must provide either root= or file_list=")

        if len(self.files) == 0:
            raise RuntimeError(
                "No images found. Check your root path or file_list, "
                "and confirm mask files aren't the only files present."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        img_array = tifffile.imread(path)

        # Normalise to 0-255 uint8
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.float32)
            img_array -= img_array.min()
            if img_array.max() > 0:
                img_array /= img_array.max()
            img_array = (img_array * 255).astype(np.uint8)

        # --- Handle all possible channel configurations ---

        if img_array.ndim == 2:
            # Pure grayscale (H, W) -> stack 3 times to make fake RGB
            img_array = np.stack([img_array] * 3, axis=2)

        elif img_array.ndim == 3:
            channels = img_array.shape[2]

            if channels == 1:
                # Single channel stored as (H, W, 1)
                img_array = np.squeeze(img_array, axis=2)
                img_array = np.stack([img_array] * 3, axis=2)

            elif channels == 2:
                # Two channels — use both plus a copy of the first
                img_array = np.stack([
                    img_array[:, :, 0],
                    img_array[:, :, 1],
                    img_array[:, :, 0]
                ], axis=2)

            elif channels == 3:
                # Standard RGB — no changes needed
                pass

            elif channels == 4:
                # RGBA — drop the alpha channel
                img_array = img_array[:, :, :3]

            else:
                # Hyperspectral — Max Variance band selection
                # (confirmed best strategy on clean 4004-image dataset)
                band_vars = img_array.var(axis=(0, 1))
                top3 = np.argsort(band_vars)[-3:]
                top3 = np.sort(top3)
                r = img_array[:, :, top3[0]]
                g = img_array[:, :, top3[1]]
                b = img_array[:, :, top3[2]]
                img_array = np.stack([r, g, b], axis=2)

        # Final conversion to PIL RGB image
        image = Image.fromarray(img_array).convert("RGB")

        # BYOLTransform returns [view1, view2]
        views = self.transform(image)

        return views