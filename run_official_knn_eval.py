from pathlib import Path
from typing import Dict
import os
import csv
from datetime import datetime

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from lightly.data import LightlyDataset
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero

from lightly_eval_model import LightlyEvalWrapper


def knn_eval(
    model: LightningModule,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    strategy: str,
    num_classes: int,
    knn_k: int,
    knn_t: float,
) -> Dict[str, float]:
    """Runs KNN evaluation on the given model.
    Parameters follow InstDisc [0] settings.
    The most important settings are:
        - Num nearest neighbors: 200
        - Temperature: 0.1

    References:
       - [0]: InstDict, 2018, https://arxiv.org/abs/1805.01978
    """
    print_rank_zero("Running KNN evaluation...")

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    val_dataset = LightlyDataset(input_dir=str(val_dir), transform=transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
    )

    classifier = KNNClassifier(
        model=model,
        num_classes=num_classes,
        knn_k=knn_k,
        knn_t=knn_t,
        train_dataloader_idx=0,
        val_dataloader_idx=1,
        topk=(1,),
    )

    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
        logger=CSVLogger(save_dir=str(log_dir), name="knn_eval"),
        callbacks=[
            DeviceStatsMonitor(),
            metric_callback,
        ],
        strategy=strategy,
        num_sanity_val_steps=0,
    )
    trainer.validate(
        model=classifier,
        dataloaders=[train_dataloader, val_dataloader],
        verbose=False,
    )

    metrics_dict: Dict[str, float] = dict()
    for metric in ["val_knn_top1"]:   # val_knn_top5 removed — not meaningful for 2 classes
        for name, value in metric_callback.val_metrics.items():
            if name.startswith(metric):
                print_rank_zero(f"knn {name}: {max(value)}")
                metrics_dict[name] = max(value)
    return metrics_dict


# ==============================================================
# Run it
# ==============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_PATH = "checkpoints/exp9b_verified_checkpoint.pth"
    RESULTS_DIR     = "logs"
    RESULTS_CSV     = os.path.join(RESULTS_DIR, "official_lightly_knn_results.csv")
    REPORT_FILE     = os.path.join(RESULTS_DIR, "official_lightly_knn_report.txt")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = LightlyEvalWrapper(checkpoint_path=CHECKPOINT_PATH, device=device)

    results = knn_eval(
        model=model,
        train_dir=Path("lightly_eval_data/train"),
        val_dir=Path("lightly_eval_data/val"),
        log_dir=Path("logs/lightly_official_knn"),
        batch_size_per_device=16,
        num_workers=2,
        accelerator=accelerator,
        devices=1,
        strategy="auto",
        num_classes=2,
        knn_k=200,     # InstDisc paper default
        knn_t=0.1,     # InstDisc paper default
    )

    print("\n" + "="*60)
    print("OFFICIAL LIGHTLY KNN EVALUATION — FINAL RESULTS")
    print("="*60)
    print(results)

    # ----------------------------------------------------------
    # Save results permanently
    # ----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(REPORT_FILE, "w") as f:
        f.write("OFFICIAL LIGHTLY KNN EVALUATION — REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Run timestamp:      {timestamp}\n")
        f.write(f"BYOL checkpoint:    {CHECKPOINT_PATH}\n")
        f.write(f"Train dir:          lightly_eval_data/train\n")
        f.write(f"Val dir:            lightly_eval_data/val\n")
        f.write(f"Batch size:         16\n")
        f.write(f"knn_k:              200\n")
        f.write(f"knn_t:              0.1\n")
        f.write(f"Num classes:        2\n")
        f.write(f"topk:               (1,)  [top-5 disabled — not meaningful for 2 classes]\n\n")
        f.write("Results:\n")
        for k, v in results.items():
            f.write(f"  {k}: {v}\n")

    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "checkpoint", "val_knn_top1", "knn_k", "knn_t", "num_classes"])
        writer.writerow([
            timestamp,
            CHECKPOINT_PATH,
            results.get("val_knn_top1", "n/a"),
            200,
            0.1,
            2,
        ])

    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"CSV row appended to: {RESULTS_CSV}")
    print("="*60)
