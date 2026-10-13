"""Image embedder: global pooled timm CNN features, L2-normalised."""

from __future__ import annotations

import numpy as np
import timm
import torch
from PIL import Image


class Embedder:
    def __init__(self, name: str = "efficientnet_b0", device: str = "cuda"):
        self.name = name
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(self.device)
        self.transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.model))
        self.dim: int = self.model.num_features

    @torch.inference_mode()
    def embed(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Return an (N, dim) float32 array of unit-norm embeddings."""
        if not images_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch = torch.stack([self.transform(Image.fromarray(img[:, :, ::-1])) for img in images_bgr])
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            feats = self.model(batch.to(self.device))
        return torch.nn.functional.normalize(feats.float(), dim=1).cpu().numpy()
