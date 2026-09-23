import time

import cv2
import numpy as np
import torch
import kornia.feature as KF

from src.config import IMAGE_SIZE
from src.matching.base import ImageMatcher
from src.preprocessing.images import prepare_image


class LoFTRMatcher(ImageMatcher):
    def __init__(self, device: str | None = None, image_size: tuple[int, int] = IMAGE_SIZE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device).eval()

    def _load_gray_tensor(self, image_path: str, center_crop: bool):
        image = prepare_image(image_path, self.image_size, grayscale=True, center_crop=center_crop)
        tensor = torch.from_numpy(image)[None, None].float() / 255.0
        return tensor.to(self.device), image

    def match(self, image0_path: str, image1_path: str):
        image0_tensor, image0 = self._load_gray_tensor(image0_path, center_crop=True)
        image1_tensor, image1 = self._load_gray_tensor(image1_path, center_crop=False)
        batch = {"image0": image0_tensor, "image1": image1_tensor}

        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = self.matcher(batch)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        pts0 = output["keypoints0"].detach().cpu().numpy()
        pts1 = output["keypoints1"].detach().cpu().numpy()
        confidence = output["confidence"].detach().cpu().numpy()

        return {"pts0": pts0, "pts1": pts1, "confidence": confidence,
                "time": elapsed, "image0": image0, "image1": image1}
