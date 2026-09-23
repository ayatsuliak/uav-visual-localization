"""Спільні операції з зображеннями та єдина конвенція кольорів/координат.

Конвенція проєкту (обов'язкова для всіх модулів):

* зображення в пам'яті — ``numpy.ndarray`` формату ``(H, W, 3)``, ``uint8``,
  порядок каналів **RGB**. Конвертація з BGR (OpenCV) виконується лише тут;
* точка — декартова пара ``(x, y) = (column, row)``, де ``x ∈ [0, W-1]``,
  ``y ∈ [0, H-1]``; індексація масиву відповідно ``image[y, x]``;
* тензор для моделей — ``(B, C, H, W)``, ``float32``, значення в ``[0, 1]``.

Читання/запис виконуються через ``np.fromfile``/``cv2.imdecode`` та
``cv2.imencode``/``tofile``, тому шляхи з кирилицею працюють і на Windows
(звичайний ``cv2.imread`` на Windows такі шляхи не відкриває).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Низькорівневе читання / запис
# ---------------------------------------------------------------------------
def _imread_unchanged(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Only 8-bit images are supported, got {image.dtype}: {path}")
    return image


def _imwrite(path: str | Path, image_bgr_or_gray: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image_bgr_or_gray)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


# ---------------------------------------------------------------------------
# Колірні простори
# ---------------------------------------------------------------------------
def rgba_to_rgb(rgba: np.ndarray,
                background_rgb: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Альфа-блендинг RGBA на однотонне тло: ``out = a*rgb + (1-a)*bg``.

    Для повністю непрозорих пікселів (a = 255) результат побітово збігається
    з вхідними RGB-значеннями, тобто операція не вносить жодних змін.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) array, got {rgba.shape}")
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if np.all(alpha == 255):
        return np.ascontiguousarray(rgb)
    a = alpha.astype(np.float32)[..., None] / 255.0
    bg = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    blended = a * rgb.astype(np.float32) + (1.0 - a) * bg
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def decoded_to_rgb(image: np.ndarray,
                   background_rgb: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Перетворює результат ``cv2.imdecode(..., IMREAD_UNCHANGED)`` в RGB."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if channels == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return rgba_to_rgb(rgba, background_rgb)
    raise ValueError(f"Unsupported number of channels: {channels}")


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Публічний API читання / запису
# ---------------------------------------------------------------------------
def read_image_rgb(path: str | Path,
                   background_rgb: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Читає зображення будь-якого 8-бітного формату як RGB ``(H, W, 3)``."""
    return decoded_to_rgb(_imread_unchanged(path), background_rgb)


def read_image_raw(path: str | Path) -> np.ndarray:
    """Читає зображення без жодних перетворень (порядок каналів OpenCV)."""
    return _imread_unchanged(path)


def write_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    """Зберігає RGB (або grayscale) зображення. PNG зберігається без втрат."""
    if image_rgb.ndim == 2:
        _imwrite(path, image_rgb)
        return
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB (H, W, 3) array, got {image_rgb.shape}")
    _imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Геометричні операції
# ---------------------------------------------------------------------------
def center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def resize(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Масштабування до ``(W, H)``. Для зменшення — INTER_AREA (без аліасингу)."""
    w, h = size_wh
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (w, h):
        return image
    interpolation = cv2.INTER_AREA if (w < src_w and h < src_h) else cv2.INTER_LINEAR
    return cv2.resize(image, (w, h), interpolation=interpolation)


def prepare_image(path: str | Path, size: tuple[int, int], grayscale: bool = False,
                  center_crop: bool = False) -> np.ndarray:
    """Завантаження для matcher: RGB (або grayscale), опційне обрізання, resize.

    ``size`` задається як ``(W, H)`` — так само, як у ``cv2.resize``.
    """
    image = read_image_rgb(path)
    if grayscale:
        image = rgb_to_gray(image)
    if center_crop:
        image = center_crop_square(image)
    return resize(image, size)


# ---------------------------------------------------------------------------
# Тензори
# ---------------------------------------------------------------------------
def image_to_tensor(image: np.ndarray, device: str | None = None):
    """``(H, W)`` або ``(H, W, C)`` uint8 -> ``torch.Tensor (1, C, H, W)`` у ``[0, 1]``."""
    import torch  # локальний імпорт: препроцесинг не має залежати від torch

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    array = image[..., None] if image.ndim == 2 else image
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float().div_(255.0)
    return tensor.to(device) if device is not None else tensor
