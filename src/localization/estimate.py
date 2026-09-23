"""Перенесення точки кадру в координати повної карти.

Гомографія ``H`` відображає оригінальні пікселі кадру в оригінальні пікселі
тайла (див. ``src.matching.pipeline``), тому перехід до карти — лише зсув
на початок тайла: ``x_map = x_tile + x_local``, ``y_map = y_tile + y_local``.
"""
from __future__ import annotations

import numpy as np

from src.geometry.homography import project_points


def frame_center(frame_size_wh: tuple[int, int]) -> tuple[float, float]:
    w, h = frame_size_wh
    return (w - 1) / 2.0, (h - 1) / 2.0


def frame_point_to_map(point_xy, H: np.ndarray | None,
                       tile_origin_xy: tuple[float, float]) -> tuple[float, float] | None:
    """Точка кадру (px кадру) -> точка повної карти (px карти) або ``None``."""
    if H is None:
        return None
    x, y = project_points([point_xy], H)[0]
    return float(tile_origin_xy[0] + x), float(tile_origin_xy[1] + y)


def estimate_global_position(frame_size_wh: tuple[int, int], H: np.ndarray | None,
                             tile_origin_xy: tuple[float, float]) -> tuple[float, float] | None:
    """Положення БпЛА на карті як проєкція центра кадру (надирна камера)."""
    return frame_point_to_map(frame_center(frame_size_wh), H, tile_origin_xy)
