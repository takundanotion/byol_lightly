# ==============================================================
# utils/stats.py — computes, prints and saves training statistics
# ==============================================================

import os
import csv
import torch
from config import COLLAPSE_THRESHOLD


def compute_epoch_stats(batch_losses, batch_stds):
    loss_tensor = torch.tensor(batch_losses)
    std_tensor  = torch.tensor(batch_stds)

    return {
        "avg_loss":        loss_tensor.mean().item(),
        "loss_std":        loss_tensor.std().item(),
        "avg_feature_std": std_tensor.mean().item(),
        "feature_spread":  std_tensor.std().item(),
    }


def print_epoch_stats(epoch, epochs, stats, momentum):
    print(
        f"Epoch {epoch:>3}/{epochs}: "
        f"Loss {stats['avg_loss']:.4f} ± {stats['loss_std']:.4f} | "
        f"Feature Std {stats['avg_feature_std']:.4f} ± {stats['feature_spread']:.4f} | "
        f"Momentum {momentum:.4f}"
    )

    if stats["avg_feature_std"] < COLLAPSE_THRESHOLD:
        print(
            f"  ⚠  Warning: Feature Std is {stats['avg_feature_std']:.4f} "
            f"— possible representation collapse"
        )


def log_to_file(epoch, stats, momentum, log_path):
    """Appends the epoch stats to a plain text log file."""
    with open(log_path, "a") as f:
        f.write(
            f"Epoch {epoch} | "
            f"Loss {stats['avg_loss']:.4f} ± {stats['loss_std']:.4f} | "
            f"Feature Std {stats['avg_feature_std']:.4f} ± {stats['feature_spread']:.4f} | "
            f"Momentum {momentum:.4f}\n"
        )


def log_to_csv(epoch, stats, momentum, csv_path):
    """
    Saves epoch stats to a CSV file.
    Creates the file with headers on epoch 1.
    Appends a new row on every subsequent epoch.
    This makes it easy to load into pandas, Excel, or a plotting script later.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch",
            "avg_loss",
            "loss_std",
            "avg_feature_std",
            "feature_spread",
            "momentum"
        ])

        # Write header only once when the file is first created
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "epoch":           epoch,
            "avg_loss":        round(stats["avg_loss"], 6),
            "loss_std":        round(stats["loss_std"], 6),
            "avg_feature_std": round(stats["avg_feature_std"], 6),
            "feature_spread":  round(stats["feature_spread"], 6),
            "momentum":        round(momentum, 6)
        })