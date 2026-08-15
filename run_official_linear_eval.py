from pathlib import Path
from typing import Dict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.nn import Module
from torch.utils.data import DataLoader

from lightly.data import LightlyDataset
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import LinearClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero

from lightly_eval_model import LightlyEvalWrapper


def linear_eval(
    model: Module,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    strategy: str,
    num_classes: int,
) -> Dict[str, float]:
    """Runs a linear evaluation on the given model.
    Parameters follow SimCLR [0] settings.
    The most important settings are:
        - Backbone: Frozen
        - Epochs: 90
        - Optimizer: SGD
        - Base Learning Rate: 0.1
        - Momentum: 0.9
        - Weight Decay: 0.0
        - LR Schedule: Cosine without warmup

    References:
        - [0]: SimCLR, 2020, https://arxiv.org/abs/2002.05709
    """
    print_rank_zero("Running linear evaluation...")

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True,
    )

    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    val_dataset = LightlyDataset(input_dir=str(val_dir), transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=90,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
        ],
        logger=CSVLogger(save_dir=str(log_dir), name="linear_eval"),
        precision=precision,
        strategy=strategy,
        num_sanity_val_steps=0,
    )
    classifier = LinearClassifier(
        model=model,
        batch_size_per_device=batch_size_per_device,
        feature_dim=model.online_classifier.feature_dim,
        num_classes=num_classes,
        topk=(1,),
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    metrics_dict: Dict[str, float] = dict()
    for metric in ["val_top1"]:   # val_top5 removed — not meaningful for 2 classes
        print_rank_zero(
            f"max linear {metric}: {max(metric_callback.val_metrics[metric])}"
        )
        metrics_dict[metric] = max(metric_callback.val_metrics[metric])
    return metrics_dict


# ==============================================================
# Run it
# ==============================================================

if __name__ == "__main__":
    import csv
    import os
    from datetime import datetime

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_PATH = "checkpoints/exp9b_verified_checkpoint.pth"
    RESULTS_DIR     = "logs"
    RESULTS_CSV     = os.path.join(RESULTS_DIR, "official_lightly_eval_results.csv")
    REPORT_FILE     = os.path.join(RESULTS_DIR, "official_lightly_eval_report.txt")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = LightlyEvalWrapper(checkpoint_path=CHECKPOINT_PATH, device=device)

    results = linear_eval(
        model=model,
        train_dir=Path("lightly_eval_data/train"),
        val_dir=Path("lightly_eval_data/val"),
        log_dir=Path("logs/lightly_official_eval"),
        batch_size_per_device=16,     # matches your project's proven batch size
        num_workers=2,
        accelerator=accelerator,
        devices=1,
        precision="32",
        strategy="auto",
        num_classes=2,                # matches your confirmed 2-cluster structure
    )

    print("\n" + "="*60)
    print("OFFICIAL LIGHTLY LINEAR EVALUATION — FINAL RESULTS")
    print("="*60)
    print(results)

    # ----------------------------------------------------------
    # Save results permanently — human-readable report + CSV row
    # ----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(REPORT_FILE, "w") as f:
        f.write("OFFICIAL LIGHTLY LINEAR EVALUATION — REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Run timestamp:      {timestamp}\n")
        f.write(f"BYOL checkpoint:    {CHECKPOINT_PATH}\n")
        f.write(f"Train dir:          lightly_eval_data/train\n")
        f.write(f"Val dir:            lightly_eval_data/val\n")
        f.write(f"Batch size:         16\n")
        f.write(f"Max epochs:         90\n")
        f.write(f"Num classes:        2\n")
        f.write(f"topk:               (1,)  [top-5 disabled — not meaningful for 2 classes]\n\n")
        f.write("Results:\n")
        for k, v in results.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nPer-epoch metrics CSV: logs/lightly_official_eval/linear_eval/version_0/metrics.csv\n")

    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "checkpoint", "val_top1", "max_epochs", "batch_size", "num_classes"])
        writer.writerow([
            timestamp,
            CHECKPOINT_PATH,
            results.get("val_top1", "n/a"),
            90,
            16,
            2,
        ])

    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"CSV row appended to: {RESULTS_CSV}")
    print("="*60)