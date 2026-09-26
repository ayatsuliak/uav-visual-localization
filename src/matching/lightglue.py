"""LightGlue + SuperPoint (готова реалізація https://github.com/cvg/LightGlue)."""
from __future__ import annotations

import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from src.config import MAX_KEYPOINTS
from src.matching.base import Features, ImageMatcher
from src.preprocessing.images import image_to_tensor


class LightGlueMatcher(ImageMatcher):
    name = "LightGlue"
    size_multiple = 1

    def __init__(self, device: str | None = None, max_num_keypoints: int = MAX_KEYPOINTS,
                 depth_confidence: float = 0.95, width_confidence: float = 0.99,
                 filter_threshold: float = 0.1):
        super().__init__(device)
        self.max_num_keypoints = max_num_keypoints
        self.lightglue_conf = {"depth_confidence": depth_confidence,
                               "width_confidence": width_confidence,
                               "filter_threshold": filter_threshold}
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint", **self.lightglue_conf).eval().to(self.device)

    def config(self) -> dict:
        return {**super().config(), "extractor": "SuperPoint",
                "max_num_keypoints": self.max_num_keypoints, **self.lightglue_conf}

    supports_feature_cache = True
    _CACHE_KEYS = ("keypoints", "keypoint_scores", "descriptors", "image_size")

    def features_to_arrays(self, feats: Features) -> dict[str, np.ndarray]:
        """Ознаки SuperPoint без batch-виміру, float32 (без втрати точності)."""
        return {key: feats.data[key][0].detach().cpu().numpy().astype(np.float32)
                for key in self._CACHE_KEYS}

    def features_from_arrays(self, arrays: dict[str, np.ndarray],
                             device: str | None = None) -> Features:
        """Зворотне до :meth:`features_to_arrays`; ``device`` — куди завантажити (RAM/VRAM)."""
        device = device or self.device
        data = {key: torch.from_numpy(np.ascontiguousarray(arrays[key]))[None].to(device)
                for key in self._CACHE_KEYS}
        w, h = (int(v) for v in arrays["image_size"])
        return Features(data=data, image_size=(w, h),
                        num_keypoints=int(arrays["keypoints"].shape[0]), time=0.0)

    def _to_device(self, feats: dict) -> dict:
        if all(v.device.type == torch.device(self.device).type for v in feats.values()):
            return feats
        return {k: v.to(self.device, non_blocking=True) for k, v in feats.items()}

    @torch.inference_mode()
    def _extract(self, image: np.ndarray):
        tensor = image_to_tensor(image, self.device)
        # resize=None: SuperPoint.extract за замовчуванням масштабує довшу сторону
        # до 1024 px; масштаб тут контролює pipeline, тому вбудований resize вимкнено.
        feats = self.extractor.extract(tensor, resize=None)
        return feats, int(feats["keypoints"].shape[1])

    @torch.inference_mode()
    def _match(self, feats0, feats1):
        # Кешовані ознаки можуть зберігатися в RAM: переносимо на пристрій моделі.
        feats0, feats1 = self._to_device(feats0), self._to_device(feats1)
        matches01 = rbd(self.matcher({"image0": feats0, "image1": feats1}))
        matches = matches01["matches"].cpu().numpy()
        if len(matches) == 0:
            empty = np.empty((0, 2))
            return empty, empty, np.empty(0, dtype=np.float32)
        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()
        return kpts0[matches[:, 0]], kpts1[matches[:, 1]], matches01["scores"].cpu().numpy()
