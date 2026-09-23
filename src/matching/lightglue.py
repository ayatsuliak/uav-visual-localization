import time

import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from src.config import IMAGE_SIZE, MAX_KEYPOINTS
from src.matching.base import ImageMatcher
from src.preprocessing.images import prepare_image


class LightGlueMatcher(ImageMatcher):
    def __init__(self, device: str | None = None, max_num_keypoints: int = MAX_KEYPOINTS,
                 image_size: tuple[int, int] = IMAGE_SIZE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def _load_image_tensor(self, image_path: str, center_crop: bool):
        image = prepare_image(image_path, self.image_size, grayscale=False, center_crop=center_crop)
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, image

    def match(self, image0_path: str, image1_path: str):
        image0_tensor, image0 = self._load_image_tensor(image0_path, center_crop=True)
        image1_tensor, image1 = self._load_image_tensor(image1_path, center_crop=False)

        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            feats0 = self.extractor.extract(image0_tensor)
            feats1 = self.extractor.extract(image1_tensor)
            matches01 = self.matcher({"image0": feats0, "image1": feats1})
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        feats0, feats1, matches01 = rbd(feats0), rbd(feats1), rbd(matches01)
        matches = matches01["matches"].detach().cpu().numpy()
        kpts0 = feats0["keypoints"].detach().cpu().numpy()
        kpts1 = feats1["keypoints"].detach().cpu().numpy()

        if len(matches) == 0:
            pts0, pts1 = kpts0[:0], kpts1[:0]
            confidence = np.empty(0, dtype=np.float32)
        else:
            pts0 = kpts0[matches[:, 0]]
            pts1 = kpts1[matches[:, 1]]
            confidence = matches01["scores"].detach().cpu().numpy()

        return {"pts0": pts0, "pts1": pts1, "confidence": confidence,
                "time": elapsed, "image0": image0, "image1": image1}
