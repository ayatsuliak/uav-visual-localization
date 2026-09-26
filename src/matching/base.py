"""Спільний інтерфейс методів зіставлення зображень.

Решта системи працює лише з :class:`ImageMatcher` і не залежить від
конкретного методу. Зіставлення розділено на два кроки:

* ``extract(image)`` — підготовка одного зображення (для LightGlue — ключові
  точки й дескриптори SuperPoint; для LoFTR — лише тензор, бо метод не має
  окремого етапу детекції). Результат можна кешувати, наприклад для тайлів;
* ``match_features(feats0, feats1)`` — встановлення відповідностей.

Вхідні зображення — RGB ``uint8 (H, W, 3)`` **уже в робочій роздільності**;
координати відповідностей повертаються в пікселях цих вхідних зображень.
Масштабування та перенесення координат в оригінальні пікселі виконує
``src.matching.pipeline``.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Features:
    """Результат ``extract``: дані моделі + службова інформація."""
    data: Any
    image_size: tuple[int, int]          # (W, H) вхідного зображення
    num_keypoints: int | None            # None для detector-free методів
    time: float                          # час екстракції, с


@dataclass
class MatchResult:
    """Відповідності між двома зображеннями.

    ``pts0[i]`` відповідає ``pts1[i]``; ``confidence`` — оцінка впевненості,
    яку повертає метод (шкали LightGlue та LoFTR не є взаємно порівнюваними).
    """
    pts0: np.ndarray                     # (N, 2) float64, (x, y)
    pts1: np.ndarray                     # (N, 2) float64, (x, y)
    confidence: np.ndarray               # (N,) float32
    num_keypoints0: int | None = None
    num_keypoints1: int | None = None
    time_extract0: float = 0.0
    time_extract1: float = 0.0
    time_match: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def num_matches(self) -> int:
        return int(len(self.pts0))

    @property
    def time_total(self) -> float:
        return self.time_extract0 + self.time_extract1 + self.time_match


class ImageMatcher(ABC):
    """Базовий клас методу зіставлення."""

    #: Назва методу для таблиць результатів.
    name: str = "base"
    #: Розміри вхідного зображення мають бути кратними цьому числу.
    size_multiple: int = 1

    def __init__(self, device: str | None = None):
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    # --- Кроки, які реалізують конкретні методи ---------------------------
    @abstractmethod
    def _extract(self, image: np.ndarray) -> tuple[Any, int | None]:
        """Повертає ``(data, num_keypoints)`` для одного зображення."""

    @abstractmethod
    def _match(self, data0: Any, data1: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Повертає ``(pts0, pts1, confidence)`` у координатах вхідних зображень."""

    def config(self) -> dict:
        """Параметри методу, що записуються в метадані експерименту."""
        return {"name": self.name, "device": self.device, "size_multiple": self.size_multiple}

    # --- Дисковий кеш ознак (лише для методів з окремим етапом детекції) ---
    #: Чи можна зберегти результат ``extract`` і повторно використати його.
    supports_feature_cache: bool = False

    def features_to_arrays(self, feats: Features) -> dict[str, np.ndarray]:
        raise NotImplementedError(f"{self.name} does not support feature caching")

    def features_from_arrays(self, arrays: dict[str, np.ndarray],
                             device: str | None = None) -> Features:
        raise NotImplementedError(f"{self.name} does not support feature caching")

    # --- Публічний API ----------------------------------------------------
    def extract(self, image: np.ndarray) -> Features:
        self._check_image(image)
        start = self._now()
        data, num_keypoints = self._extract(image)
        elapsed = self._now() - start
        h, w = image.shape[:2]
        return Features(data=data, image_size=(w, h), num_keypoints=num_keypoints, time=elapsed)

    def match_features(self, feats0: Features, feats1: Features) -> MatchResult:
        start = self._now()
        pts0, pts1, confidence = self._match(feats0.data, feats1.data)
        elapsed = self._now() - start
        return MatchResult(
            pts0=np.asarray(pts0, dtype=np.float64).reshape(-1, 2),
            pts1=np.asarray(pts1, dtype=np.float64).reshape(-1, 2),
            confidence=np.asarray(confidence, dtype=np.float32).reshape(-1),
            num_keypoints0=feats0.num_keypoints, num_keypoints1=feats1.num_keypoints,
            time_extract0=feats0.time, time_extract1=feats1.time, time_match=elapsed,
        )

    def match(self, image0: np.ndarray, image1: np.ndarray) -> MatchResult:
        return self.match_features(self.extract(image0), self.extract(image1))

    def warmup(self, size0_wh: tuple[int, int], size1_wh: tuple[int, int],
               iterations: int = 2, seed: int = 0) -> None:
        """«Прогрівання» (ініціалізація CUDA/cuDNN, виділення пам'яті) перед вимірюванням часу.

        Використовуються випадкові зображення тих самих розмірів, що й у
        експерименті; результати відкидаються.
        """
        rng = np.random.default_rng(seed)
        for _ in range(max(0, iterations)):
            image0 = rng.integers(0, 256, (size0_wh[1], size0_wh[0], 3), dtype=np.uint8)
            image1 = rng.integers(0, 256, (size1_wh[1], size1_wh[0], 3), dtype=np.uint8)
            self.match(image0, image1)

    # --- Службові методи --------------------------------------------------
    def _check_image(self, image: np.ndarray) -> None:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB uint8 (H, W, 3) image, got {image.dtype} {image.shape}")
        h, w = image.shape[:2]
        m = self.size_multiple
        if w % m or h % m:
            raise ValueError(f"{self.name}: image size {w}x{h} must be a multiple of {m}")

    def _now(self) -> float:
        """Час із синхронізацією CUDA, щоб враховувати асинхронні обчислення на GPU."""
        if str(self.device).startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        return time.perf_counter()
