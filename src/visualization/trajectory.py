"""Візуалізація траєкторії БпЛА на ортофотоплані (Етап 4)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.images import resize

LINE_COLOR = (0, 200, 255)          # лінія траєкторії
TRACKED_COLOR = (0, 210, 60)        # TRACKING (зелений)
INIT_COLOR = (0, 110, 255)          # INITIALIZED (синій)
INTERP_COLOR = (255, 215, 0)        # INTERPOLATED (жовтий)
REJECTED_COLOR = (230, 30, 30)      # відкинуті аномалії (червоний хрестик)
FOOTPRINT_COLOR = (255, 255, 255)


def _text(canvas, text, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def trajectory_view(trajectory: pd.DataFrame, map_size, margin: float = 400.0):
    """Прямокутник ``(x0, y0, x1, y1)`` навколо всіх точок траєкторії (польотна зона)."""
    xs = pd.concat([trajectory["x_map"], trajectory.get("x_raw", pd.Series(dtype=float))]).dropna()
    ys = pd.concat([trajectory["y_map"], trajectory.get("y_raw", pd.Series(dtype=float))]).dropna()
    if xs.empty:
        return 0, 0, map_size[0], map_size[1]
    return (max(0.0, xs.min() - margin), max(0.0, ys.min() - margin),
            min(float(map_size[0]), xs.max() + margin), min(float(map_size[1]), ys.max() + margin))


def draw_trajectory(map_rgb: np.ndarray, trajectory: pd.DataFrame, view_xyxy=None,
                    max_side: int = 1800, footprints: list | None = None,
                    label_every: int = 10, title: str | None = None) -> np.ndarray:
    """Траєкторія ``(x_map, y_map)`` у порядку кадрів.

    Лінія з'єднує всі точки з позицією (TRACKING, INITIALIZED, INTERPOLATED);
    стрілки показують напрямок руху; відкинуті оцінки ``(x_raw, y_raw)`` —
    червоні хрестики. ``footprints`` — контури огляду, які треба намалювати.
    """
    H, W = map_rgb.shape[:2]
    x0, y0, x1, y1 = view_xyxy if view_xyxy is not None else (0, 0, W, H)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
    view = map_rgb[y0:y1, x0:x1]
    scale = min(1.0, max_side / max(view.shape[:2]))
    canvas = resize(view, (max(1, round(view.shape[1] * scale)),
                           max(1, round(view.shape[0] * scale)))).copy()
    # Трохи притлумити карту, щоб траєкторію було краще видно.
    canvas = cv2.addWeighted(canvas, 0.75, np.zeros_like(canvas), 0.25, 0)

    def pt(x, y):
        return int(round((x - x0 + 0.5) * scale)), int(round((y - y0 + 0.5) * scale))

    for fp in footprints or []:
        quad = np.array([pt(x, y) for x, y in fp], np.int32)
        cv2.polylines(canvas, [quad], True, FOOTPRINT_COLOR, 1, cv2.LINE_AA)

    positioned = trajectory.dropna(subset=["x_map", "y_map"])
    points = [pt(x, y) for x, y in zip(positioned["x_map"], positioned["y_map"])]
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if length < 1:
            continue
        # Стрілка фіксованого розміру ≈10 px незалежно від довжини відрізка.
        cv2.arrowedLine(canvas, a, b, LINE_COLOR, 2, cv2.LINE_AA, tipLength=min(0.5, 10 / length))

    for status, p in zip(positioned["status"], points):
        if status == "INTERPOLATED":
            cv2.circle(canvas, p, 6, INTERP_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 6, (0, 0, 0), 1, cv2.LINE_AA)
        elif status == "INITIALIZED":
            cv2.circle(canvas, p, 7, INIT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(canvas, p, 4, TRACKED_COLOR, -1, cv2.LINE_AA)

    if "x_raw" in trajectory:
        for x, y in trajectory[["x_raw", "y_raw"]].dropna().itertuples(index=False):
            cv2.drawMarker(canvas, pt(x, y), REJECTED_COLOR, cv2.MARKER_TILTED_CROSS, 16, 3,
                           cv2.LINE_AA)

    if label_every and len(points):
        for i, (frame_id, p) in enumerate(zip(positioned["frame_id"], points)):
            if i % label_every == 0 or i == len(points) - 1:
                _text(canvas, str(frame_id).split("_")[-1].split(".")[0], (p[0] + 8, p[1] - 8), 0.45)

    # Легенда
    legend = [("TRACKING", TRACKED_COLOR), ("INITIALIZED", INIT_COLOR),
              ("INTERPOLATED", INTERP_COLOR), ("rejected outlier", REJECTED_COLOR)]
    y = canvas.shape[0] - 12 - 22 * (len(legend) - 1)
    for name, color in legend:
        cv2.circle(canvas, (18, y - 5), 6, color, -1, cv2.LINE_AA)
        _text(canvas, name, (32, y))
        y += 22
    if title:
        _text(canvas, title, (10, 26), 0.7)
    return canvas
