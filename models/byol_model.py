# ==============================================================
# models/byol_model.py — the BYOL neural network
# --------------------------------------------------------------
# Builds the full BYOL model using Lightly's ready-made heads
# on top of a ResNet18 backbone.
# ==============================================================

import copy
import torch
import torchvision
from torch import nn
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad

from config import BACKBONE, PROJECTION_INPUT, PROJECTION_HIDDEN, PROJECTION_OUTPUT


def build_backbone():
    """
    Loads ResNet18 and removes its final classification layer.
    We only want the feature extractor part, not the classifier.
    """
    if BACKBONE == "resnet18":
        resnet = torchvision.models.resnet18()
    elif BACKBONE == "resnet50":
        resnet = torchvision.models.resnet50()
    else:
        raise ValueError(f"Unknown backbone: {BACKBONE}. Use resnet18 or resnet50.")

    # [:-1] means "all layers except the last one"
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone


class BYOLModel(nn.Module):
    """
    Full BYOL model with:
      - online network  (backbone + projection + prediction)
      - target network  (backbone + projection, frozen, updated via EMA)
    """

    def __init__(self):
        super().__init__()

        backbone = build_backbone()

        # ---- Online network (active learner) ----
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(
            PROJECTION_INPUT,
            PROJECTION_HIDDEN,
            PROJECTION_OUTPUT
        )
        # Prediction head only exists on the online side.
        # This asymmetry is what stops BYOL from collapsing.
        self.prediction_head = BYOLPredictionHead(
            PROJECTION_OUTPUT,
            PROJECTION_HIDDEN,
            PROJECTION_OUTPUT
        )

        # ---- Target network (slow mentor) ----
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Freeze target — it must NOT learn through backpropagation.
        # It only absorbs knowledge slowly via the momentum update.
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        """Online network: backbone → flatten → project → predict."""
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        """Target network: backbone → flatten → project. No gradients."""
        with torch.no_grad():
            y = self.backbone_momentum(x).flatten(start_dim=1)
            z = self.projection_head_momentum(y)
        return z
