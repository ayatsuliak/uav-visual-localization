"""Перенесення точок кадру в координати повної карти.

Гомографія ``H`` відображає оригінальні пікселі кадру в оригінальні пікселі
тайла (див. ``src.matching.pipeline``), тому перехід до карти — лише зсув
на початок тайла:

    p~ = H · [x, y, 1]^T,   x_local = p~[0] / p~[2],   y_local = p~[1] / p~[2],
    X_map = X_origin + x_local,   Y_map = Y_origin + y_local.

Конвенція пікселів проєкту: піксель ``(x, y)`` — це центр пікселя, тому центр
кадру ``W × H`` — точка ``((W-1)/2, (H-1)/2)``, а кути кадру — центри кутових
пікселів ``(0, 0), (W-1, 0), (W-1, H-1), (0, H-1)``.
"""
from __future__ import annotations

import numpy as np


def frame_center(frame_size_wh: tuple[int, int]) -> tuple[float, float]:
    w, h = frame_size_wh
    return (w - 1) / 2.0, (h - 1) / 2.0


def frame_corners(frame_size_wh: tuple[int, int]) -> np.ndarray:
    """Кути кадру за годинниковою стрілкою від лівого верхнього: ``(4, 2)``."""
    w, h = frame_size_wh
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def apply_homography(points, H: np.ndarray) -> np.ndarray:
    """Проєктивне перетворення з явною дегомогенізацією: ``(N, 2) -> (N, 2)``.

    Якщо третя однорідна координата (майже) нульова, точка лежить на «лінії
    нескінченності» — повертається ``nan``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ np.asarray(H, np.float64).T
    w = homogeneous[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = homogeneous[:, :2] / w
    projected[np.abs(w[:, 0]) < 1e-12] = np.nan
    return projected


def local_to_map(points_local, tile_origin_xy: tuple[float, float]) -> np.ndarray:
    return np.asarray(points_local, dtype=np.float64).reshape(-1, 2) + np.asarray(
        tile_origin_xy, dtype=np.float64).reshape(1, 2)


def frame_point_to_map(point_xy, H: np.ndarray | None,
                       tile_origin_xy: tuple[float, float]) -> tuple[float, float] | None:
    """Точка кадру (px кадру) -> точка повної карти (px карти) або ``None``."""
    if H is None:
        return None
    x, y = local_to_map(apply_homography([point_xy], H), tile_origin_xy)[0]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return float(x), float(y)


def estimate_global_position(frame_size_wh: tuple[int, int], H: np.ndarray | None,
                             tile_origin_xy: tuple[float, float]) -> tuple[float, float] | None:
    """Положення БпЛА на карті як проєкція центра кадру (надирна камера)."""
    return frame_point_to_map(frame_center(frame_size_wh), H, tile_origin_xy)


def footprint_on_map(frame_size_wh: tuple[int, int], H: np.ndarray,
                     tile_origin_xy: tuple[float, float]) -> np.ndarray:
    """Контур огляду камери на карті: 4 вершини ``(X_i, Y_i)``, px карти."""
    return local_to_map(apply_homography(frame_corners(frame_size_wh), H), tile_origin_xy)


def inside_bounds(point_xy, width: float, height: float, margin: float = 0.0) -> bool:
    """``-margin <= x <= width-1+margin`` і так само для ``y``."""
    x, y = point_xy
    return bool(np.isfinite(x) and np.isfinite(y)
                and -margin <= x <= width - 1 + margin and -margin <= y <= height - 1 + margin)
