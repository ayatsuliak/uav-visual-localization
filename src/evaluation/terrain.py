"""Об'єктивні ознаки місцевості під кадром (експеримент 4 — проблемні зони).

Ознаки обчислюються на ортофотоплані в межах контуру огляду кадру:

* ``vegetation_fraction`` — частка пікселів рослинності за індексом Excess
  Green у хроматичних координатах: ``ExG = 2g - r - b``, де
  ``r = R / (R + G + B)`` тощо; піксель — рослинність, якщо ``ExG > t``
  (``t = 0.05``). Відео зняте взимку, карта — влітку, тому саме рослинність
  змінюється найбільше;
* ``texture`` — середній модуль градієнта Собеля яскравості (0–255): мало
  текстури -> мало ключових точок.
"""
from __future__ import annotations

import cv2
import numpy as np

EXG_THRESHOLD = 0.05


def footprint_mask(shape_hw, polygon_xy, offset_xy=(0, 0)) -> np.ndarray:
    mask = np.zeros(shape_hw, np.uint8)
    pts = np.round(np.asarray(polygon_xy, float) - np.asarray(offset_xy, float)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def terrain_stats(map_rgb: np.ndarray, footprint_xy, exg_threshold: float = EXG_THRESHOLD
                  ) -> dict:
    """Ознаки місцевості під контуром ``footprint_xy`` (4 вершини, px карти)."""
    poly = np.asarray(footprint_xy, float).reshape(-1, 2)
    H, W = map_rgb.shape[:2]
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return {"vegetation_fraction": np.nan, "texture": np.nan, "area_px": 0}
    crop = map_rgb[y0:y1, x0:x1].astype(np.float32)
    mask = footprint_mask(crop.shape[:2], poly, (x0, y0))
    if not mask.any():
        return {"vegetation_fraction": np.nan, "texture": np.nan, "area_px": 0}
    total = crop.sum(axis=2) + 1e-6
    r, g, b = (crop[..., i] / total for i in range(3))
    exg = 2 * g - r - b
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return {"vegetation_fraction": float((exg[mask] > exg_threshold).mean()),
            "texture": float(grad[mask].mean()), "area_px": int(mask.sum())}
