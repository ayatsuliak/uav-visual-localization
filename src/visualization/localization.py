"""Візуалізація глобальної локалізації кадру на ортофотоплані."""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.images import resize

GRID_COLOR = (255, 255, 255)        # перевірені тайли (бліді лінії)
PASSED_COLOR = (255, 170, 0)        # кандидати, що пройшли фільтр
WINNER_COLOR = (0, 90, 255)         # тайл-переможець (синій)
FOOTPRINT_COLOR = (0, 220, 60)      # контур огляду камери (зелений)
CENTER_COLOR = (255, 0, 0)          # центр кадру (червоний)


def _text(canvas, text, org, scale=0.6):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)


def draw_global_localization(map_rgb: np.ndarray, checked_rects, winner_rect=None,
                             footprint=None, center=None, passed_rects=(), view_xyxy=None,
                             max_side: int = 1800, title: str | None = None) -> np.ndarray:
    """Малює результат пошуку на карті (усі координати — px повної карти).

    ``*_rects`` — прямокутники ``(x, y, w, h)``; ``view_xyxy`` — ділянка карти
    для показу (``None`` — вся карта); зображення зменшується до ``max_side``.
    """
    H, W = map_rgb.shape[:2]
    x0, y0, x1, y1 = view_xyxy if view_xyxy is not None else (0, 0, W, H)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
    view = map_rgb[y0:y1, x0:x1]
    scale = min(1.0, max_side / max(view.shape[:2]))
    canvas = resize(view, (max(1, round(view.shape[1] * scale)),
                           max(1, round(view.shape[0] * scale)))).copy()

    def pt(x, y):
        return int(round((x - x0 + 0.5) * scale)), int(round((y - y0 + 0.5) * scale))

    def rect(r, color, thickness, image):
        x, y, w, h = r
        cv2.rectangle(image, pt(x, y), pt(x + w - 1, y + h - 1), color, thickness, cv2.LINE_AA)

    grid = canvas.copy()
    for r in checked_rects:
        rect(r, GRID_COLOR, 1, grid)
    canvas = cv2.addWeighted(grid, 0.35, canvas, 0.65, 0)
    for r in passed_rects:
        rect(r, PASSED_COLOR, 1, canvas)
    if winner_rect is not None:
        rect(winner_rect, WINNER_COLOR, 3, canvas)

    if footprint is not None and np.all(np.isfinite(footprint)):
        quad = np.array([pt(x, y) for x, y in np.asarray(footprint).reshape(-1, 2)], np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [quad], FOOTPRINT_COLOR)
        canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
        cv2.polylines(canvas, [quad], True, FOOTPRINT_COLOR, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(int(v) for v in quad[0]), 4, FOOTPRINT_COLOR, -1)

    if center is not None:
        c = pt(*center)
        cv2.drawMarker(canvas, c, CENTER_COLOR, cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
        cv2.circle(canvas, c, 5, CENTER_COLOR, -1, cv2.LINE_AA)
        _text(canvas, f"({center[0]:.1f}, {center[1]:.1f})", (c[0] + 12, c[1] - 10))
    if title:
        _text(canvas, title, (10, 26), 0.7)
    return canvas
