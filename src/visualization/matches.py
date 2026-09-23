"""Візуалізація відповідностей і проєкції кадру на карту.

Точки передаються в оригінальних пікселях зображень; для відображення
кожне зображення масштабується до спільної висоти.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.images import resize, write_image_rgb

INLIER_COLOR = (40, 200, 40)
OUTLIER_COLOR = (220, 50, 50)
FOOTPRINT_COLOR = (255, 215, 0)
TILE_COLOR = (0, 160, 255)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()


def _fit_height(image: np.ndarray, height: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = height / h
    return resize(_to_rgb(image), (max(1, round(w * scale)), height)), scale


def _put_label(canvas: np.ndarray, text: str, origin=(8, 22)) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)


def draw_matches(frame: np.ndarray, tile: np.ndarray, pts_frame: np.ndarray,
                 pts_tile: np.ndarray, inlier_mask: np.ndarray | None = None,
                 display_height: int = 540, max_draw: int = 200, show_outliers: bool = True,
                 title: str | None = None, seed: int = 0) -> np.ndarray:
    """Кадр і тайл поруч; геометрично узгоджені відповідності — зеленим, решта — червоним."""
    left, s0 = _fit_height(frame, display_height)
    right, s1 = _fit_height(tile, display_height)
    canvas = np.concatenate([left, right], axis=1)
    offset = left.shape[1]

    pts_frame = np.asarray(pts_frame, dtype=np.float64).reshape(-1, 2)
    pts_tile = np.asarray(pts_tile, dtype=np.float64).reshape(-1, 2)
    mask = (np.zeros(len(pts_frame), bool) if inlier_mask is None
            else np.asarray(inlier_mask, bool))
    rng = np.random.default_rng(seed)

    def subset(indices):
        if len(indices) > max_draw:
            indices = rng.choice(indices, max_draw, replace=False)
        return indices

    groups = []
    if show_outliers:
        groups.append((subset(np.flatnonzero(~mask)), OUTLIER_COLOR, 1))
    groups.append((subset(np.flatnonzero(mask)), INLIER_COLOR, 1))
    for indices, color, thickness in groups:
        for i in indices:
            p0 = (int(round((pts_frame[i, 0] + 0.5) * s0)), int(round((pts_frame[i, 1] + 0.5) * s0)))
            p1 = (int(round((pts_tile[i, 0] + 0.5) * s1)) + offset,
                  int(round((pts_tile[i, 1] + 0.5) * s1)))
            cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    if title:
        _put_label(canvas, title)
    return canvas


def draw_footprint(background: np.ndarray, footprint: np.ndarray,
                   tile_rect_xywh: tuple[float, float, float, float] | None = None,
                   center: tuple[float, float] | None = None,
                   true_center: tuple[float, float] | None = None,
                   max_side: int = 900, title: str | None = None) -> np.ndarray:
    """Проєкція рамки кадру (у координатах ``background``) на фрагмент карти/тайл."""
    h, w = background.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    canvas = resize(_to_rgb(background), (max(1, round(w * scale)), max(1, round(h * scale))))

    def pt(p):
        return int(round((p[0] + 0.5) * scale)), int(round((p[1] + 0.5) * scale))

    if tile_rect_xywh is not None:
        x, y, tw, th = tile_rect_xywh
        cv2.rectangle(canvas, pt((x, y)), pt((x + tw - 1, y + th - 1)), TILE_COLOR, 2)
    quad = np.array([pt(p) for p in np.asarray(footprint).reshape(-1, 2)], np.int32)
    cv2.polylines(canvas, [quad], True, FOOTPRINT_COLOR, 2, cv2.LINE_AA)
    if len(quad):
        cv2.circle(canvas, tuple(quad[0]), 5, FOOTPRINT_COLOR, -1)  # лівий верхній кут кадру
    if center is not None:
        cv2.drawMarker(canvas, pt(center), (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
    if true_center is not None:
        cv2.drawMarker(canvas, pt(true_center), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
    if title:
        _put_label(canvas, title)
    return canvas


def save_figure(path: str | Path, image_rgb: np.ndarray) -> Path:
    path = Path(path)
    write_image_rgb(path, image_rgb)
    return path
