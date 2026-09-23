"""LoFTR (готова реалізація Kornia, detector-free)."""
from __future__ import annotations

import kornia.feature as KF
import numpy as np
import torch

from src.config import LOFTR_PRETRAINED
from src.matching.base import ImageMatcher
from src.preprocessing.images import image_to_tensor, rgb_to_gray


class LoFTRMatcher(ImageMatcher):
    name = "LoFTR"
    # Грубий рівень LoFTR працює на сітці 1/8 роздільності.
    size_multiple = 8

    def __init__(self, device: str | None = None, pretrained: str = LOFTR_PRETRAINED):
        super().__init__(device)
        self.pretrained = pretrained
        self.matcher = KF.LoFTR(pretrained=pretrained).eval().to(self.device)

    def config(self) -> dict:
        return {**super().config(), "pretrained": self.pretrained,
                "coarse_threshold": self.matcher.config["match_coarse"]["thr"]}

    def _extract(self, image: np.ndarray):
        # Окремої детекції немає: «ознаки» — це лише тензор у градаціях сірого.
        return image_to_tensor(rgb_to_gray(image), self.device), None

    @torch.inference_mode()
    def _match(self, tensor0, tensor1):
        output = self.matcher({"image0": tensor0, "image1": tensor1})
        return (output["keypoints0"].cpu().numpy(), output["keypoints1"].cpu().numpy(),
                output["confidence"].cpu().numpy())
