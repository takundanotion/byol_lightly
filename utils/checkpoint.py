# ==============================================================
# utils/checkpoint.py — saving and loading training checkpoints
# ==============================================================

import os
import torch
from config import CHECKPOINT_FILE


def save_checkpoint(epoch, model, optimizer, stats):
    """
    Saves model + optimizer state and epoch stats to disk.
    Uses a .tmp file first so a Ctrl+C mid-save never corrupts
    your last good checkpoint.
    """
    tmp_path = CHECKPOINT_FILE + ".tmp"

    torch.save(
        {
            "epoch":              epoch,
            "model_state":        model.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "loss":               stats["avg_loss"],
            "loss_std":           stats["loss_std"],
            "feature_std":        stats["avg_feature_std"],
            "feature_std_spread": stats["feature_spread"],
        },
        tmp_path
    )

    # Atomic rename — only replaces the real file once saving is complete
    os.replace(tmp_path, CHECKPOINT_FILE)
    print("Checkpoint saved.")


def load_checkpoint(model, optimizer, device):
    """
    Loads a checkpoint if one exists.
    Returns the epoch to resume from (0 if starting fresh).
    """
    if not os.path.exists(CHECKPOINT_FILE):
        print("No checkpoint found. Starting fresh.")
        return 0

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch
