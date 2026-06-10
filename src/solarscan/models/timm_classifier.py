"""timm-backed fault classifier.

Implements the ``FaultClassifier`` protocol (``classify`` for single-crop
inference) and exposes ``forward`` for training. Checkpoints bundle the backbone
name, input size and class order so a model reloads without external config.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage

from solarscan.data.transforms import build_transforms
from solarscan.taxonomy import ALL_CLASSES, FaultClass


class TimmClassifier(torch.nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        num_classes: int = 12,
        input_size: tuple[int, int] = (64, 64),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.input_size = tuple(input_size)
        self.num_classes = num_classes
        self.model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )
        self._eval_tf = build_transforms(self.input_size, train=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # --- FaultClassifier protocol ---
    @torch.inference_mode()
    def classify(self, crop: PILImage) -> tuple[FaultClass, dict[FaultClass, float]]:
        self.eval()
        device = next(self.parameters()).device
        # Training data was grayscale IR replicated to 3 channels. Reduce any input
        # (incl. RGB-colourised thermal from another camera) to luminance first, so
        # inference matches the training domain instead of feeding unseen colour.
        rgb = crop.convert("L").convert("RGB")
        x = self._eval_tf(rgb).unsqueeze(0).to(device)
        probs = F.softmax(self.forward(x), dim=1).squeeze(0).tolist()
        prob_map = {c: float(p) for c, p in zip(ALL_CLASSES, probs, strict=True)}
        predicted = max(prob_map, key=prob_map.get)
        return predicted, prob_map

    # --- persistence ---
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "backbone": self.backbone_name,
                "num_classes": self.num_classes,
                "input_size": self.input_size,
                "classes": [c.value for c in ALL_CLASSES],
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> TimmClassifier:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            backbone=ckpt["backbone"],
            num_classes=ckpt["num_classes"],
            input_size=tuple(ckpt["input_size"]),
            pretrained=False,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
