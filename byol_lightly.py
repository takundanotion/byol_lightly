# ==============================================================
# byol_lightly.py — main training script
# --------------------------------------------------------------
# Run this file to start or resume training:
#   python byol_lightly.py
# ==============================================================

import os
import torch
from torch.utils.data import DataLoader
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    NUM_WORKERS, CHECKPOINT_DIR, LOG_FILE, CSV_LOG_FILE
)
from data.dataset import TIFDataset
from models.byol_model import BYOLModel
from utils.stats import compute_epoch_stats, print_epoch_stats, log_to_file, log_to_csv
from utils.checkpoint import save_checkpoint, load_checkpoint


# ==============================================================
# Setup
# ==============================================================

# Make sure output folders exist before we start writing to them
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================
# Dataset & DataLoader
# ==============================================================

dataset = TIFDataset(file_list="data/train_files.txt")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,       # avoids unstable tiny last batches
    num_workers=NUM_WORKERS
)


# ==============================================================
# Model, loss, optimiser
# ==============================================================

model = BYOLModel().to(device)

criterion = NegativeCosineSimilarity()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ==============================================================
# Resume from checkpoint if available
# ==============================================================

start_epoch = load_checkpoint(model, optimizer, device)


# ==============================================================
# Training loop
# ==============================================================

for epoch in range(start_epoch, EPOCHS):

    batch_losses = []
    batch_stds   = []

    # Momentum increases from 0.996 → 1.0 over training.
    # Higher value = target network updates more slowly = more stable.
    momentum = cosine_schedule(epoch, EPOCHS, 0.996, 1.0)

    for views in loader:

        # TIFDataset returns [view1, view2] per image
        x0 = views[0].to(device)
        x1 = views[1].to(device)

        # --- Online network predictions ---
        p0 = model(x0)
        p1 = model(x1)

        # --- Target network projections (no gradients) ---
        z0 = model.forward_momentum(x0)
        z1 = model.forward_momentum(x1)

        # --- Collapse check ---
        # std across the batch — rising means diverse representations
        feature_std = p0.std(dim=0).mean().item()

        # --- Symmetric BYOL loss ---
        # Computed both ways and averaged for stability
        loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))

        # --- Backpropagation ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Update target (mentor) network via EMA ---
        update_momentum(model.backbone, model.backbone_momentum, m=momentum)
        update_momentum(model.projection_head, model.projection_head_momentum, m=momentum)

        batch_losses.append(loss.item())
        batch_stds.append(feature_std)

    # --- End of epoch ---
    stats = compute_epoch_stats(batch_losses, batch_stds)
    print_epoch_stats(epoch + 1, EPOCHS, stats, momentum)
    log_to_file(epoch + 1, stats, momentum, LOG_FILE)
    log_to_csv(epoch + 1, stats, momentum, CSV_LOG_FILE)
    save_checkpoint(epoch + 1, model, optimizer, stats)
