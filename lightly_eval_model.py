import torch
import torch.nn as nn

from models.byol_model import BYOLModel
from config import PROJECTION_INPUT


class _DummyOnlineClassifier(nn.Module):
    """
    LinearClassifier only reads model.online_classifier.feature_dim.
    It does not call this module's forward() anywhere in the source
    we verified — so this only needs to hold that one attribute.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim


class LightlyEvalWrapper(nn.Module):
    """
    Wraps our trained, frozen BYOL backbone for use with Lightly's
    official linear_eval() benchmarking function.

    model(images) returns backbone features — exactly what
    LinearClassifier.forward() expects:
        features = self.model(images).flatten(start_dim=1)
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()

        byol = BYOLModel().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        byol.load_state_dict(checkpoint["model_state"])

        # Freeze completely — this is a frozen-backbone evaluation
        for p in byol.parameters():
            p.requires_grad = False
        byol.eval()

        self.backbone = byol.backbone
        self.online_classifier = _DummyOnlineClassifier(feature_dim=PROJECTION_INPUT)

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Trained epochs: {checkpoint.get('epoch', 'unknown')}")
        print(f"Backbone frozen. Feature dim: {PROJECTION_INPUT}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matches LinearClassifier's expectation: model(images) -> features
        return self.backbone(x).flatten(start_dim=1)
