"""Еталонні координати (ground truth) центру кадру на карті.

Еталон має бути незалежним від системи, що оцінюється, тому він будується з
**ручних** відповідностей точок: для опорного кадру вказуються щонайменше 4
пари «точка кадру (px кадру) ↔ та сама точка на карті (px карти)». За ними
методом найменших квадратів оцінюється гомографія ``H_gt`` (кадр -> карта), і
центр кадру ``((W-1)/2, (H-1)/2)`` проєктується на карту.

Точки слід обирати на рівні землі (розмітка, бордюри, кути доріжок, основи
стовпів), рівномірно по кадру — дахи високих будівель через паралакс
зміщують оцінку.

Файли:

* ``data/metadata/ground_truth.csv`` — ``frame_id, timestamp, x_true, y_true,
  is_keyframe, n_points, residual_px, max_residual_px, center_uncertainty_px,
  reliable, source``;

Якість розмітки опорного кадру. З 4 пар гомографія визначається точно, тому
залишкова похибка завжди 0 і нічого не перевіряє. Кадр вважається надійним
(``reliable``), якщо: пар ``>= 5``; кожна пара узгоджується з рештою
(залишкова похибка ``<= 3`` px карти — інакше одна з точок хибна); центр
стійкий до неточності кліків (95-й процентиль зміщення за шуму σ = 2 px на
кадрі та 1 px на карті ``<= 5`` px карти — інакше точки скупчені або центр
далеко поза ними).
* ``data/metadata/ground_truth_points.json`` — самі відповідності (для
  перевірки й повторного обчислення).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.localization.coordinates import apply_homography, frame_center

GT_COLUMNS = ["frame_id", "timestamp", "x_true", "y_true", "is_keyframe", "n_points",
              "residual_px", "max_residual_px", "center_uncertainty_px", "center_in_hull",
              "reliable", "source"]
MIN_POINTS = 4              # мінімум для обчислення гомографії
MIN_POINTS_RELIABLE = 5     # мінімум для перевірки: з 4 пар H визначається точно, residual = 0
MAX_UNCERTAINTY_PX = 5.0    # допустима невизначеність центру через шум кліків, px карти
MAX_POINT_RESIDUAL_PX = 3.0 # допустима залишкова похибка окремої пари, px карти
CLICK_NOISE_FRAME_PX = 2.0  # σ точності кліку на кадрі (вікно показує кадр у масштабі 1:2)
CLICK_NOISE_MAP_PX = 1.0    # σ точності кліку на карті


@dataclass
class KeyframeAnnotation:
    frame_id: str
    timestamp: float
    frame_size: tuple[int, int]               # (W, H) кадру
    frame_points: list[list[float]]           # [[x, y], ...] px кадру
    map_points: list[list[float]]             # [[X, Y], ...] px карти
    note: str = ""


@dataclass
class KeyframeEstimate:
    x_true: float
    y_true: float
    n_points: int
    residual_px: float                        # RMS похибка відповідностей, px карти
    H: np.ndarray
    max_residual_px: float = 0.0              # найбільша залишкова похибка серед пар
    # Чутливість центру до неточності кліків: 95-й процентиль зміщення центру за
    # випадкового шуму N(0, σ²) у координатах точок (Монте-Карло), px карти.
    # Велике значення — точки скупчені або центр далеко поза ними.
    center_uncertainty_px: float = float("nan")
    # Центр кадру всередині опуклої оболонки вказаних точок кадру. False — центр
    # екстрапольовано: шум кліків це вже враховує, але систематичні похибки
    # (дисторсія об'єктива, паралакс) за межами точок не оцінюються. Лише попередження,
    # на ``reliable`` не впливає.
    center_in_hull: bool = True

    @property
    def reliable(self) -> bool:
        return (self.n_points >= MIN_POINTS_RELIABLE
                and self.max_residual_px <= MAX_POINT_RESIDUAL_PX
                and bool(self.center_uncertainty_px <= MAX_UNCERTAINTY_PX))

    def warnings(self) -> list[str]:
        out = []
        if self.n_points < MIN_POINTS_RELIABLE:
            out.append(f"{self.n_points} pairs: no check possible, add >= {MIN_POINTS_RELIABLE}")
        elif self.max_residual_px > MAX_POINT_RESIDUAL_PX:
            out.append(f"a pair disagrees by {self.max_residual_px:.1f} px: fix or undo it")
        if self.center_uncertainty_px > MAX_UNCERTAINTY_PX:
            out.append(f"centre sensitive to clicks (+-{self.center_uncertainty_px:.1f} px): "
                       f"spread points around the whole frame")
        if not self.center_in_hull:
            out.append("centre outside the clicked points (extrapolated): add points on the "
                       "other side of the centre if possible")
        return out


def estimate_keyframe(annotation: KeyframeAnnotation) -> KeyframeEstimate:
    """Центр кадру на карті за ручними відповідностями (гомографія МНК, без RANSAC).

    RANSAC не використовується: усі точки вказані людиною і мають входити в
    оцінку; погану точку видно за ``residual_px`` і її слід виправити вручну.
    """
    src = np.asarray(annotation.frame_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(annotation.map_points, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError(f"{annotation.frame_id}: {len(src)} frame points vs {len(dst)} map points")
    if len(src) < MIN_POINTS:
        raise ValueError(f"{annotation.frame_id}: at least {MIN_POINTS} point pairs are required")
    H = _fit(src, dst)
    if H is None:
        raise ValueError(f"{annotation.frame_id}: degenerate point configuration")
    residuals = np.linalg.norm(apply_homography(src, H) - dst, axis=1)
    center = frame_center(annotation.frame_size)
    x, y = apply_homography([center], H)[0]
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"{annotation.frame_id}: degenerate point configuration")
    return KeyframeEstimate(float(x), float(y), len(src), float(np.sqrt(np.mean(residuals ** 2))),
                            H, float(residuals.max()),
                            click_noise_uncertainty(src, dst, center, (x, y)),
                            inside_hull(src, center))


def inside_hull(points: np.ndarray, point) -> bool:
    """Точка всередині (або на межі) опуклої оболонки ``points``."""
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    return cv2.pointPolygonTest(hull, (float(point[0]), float(point[1])), False) >= 0


def _fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    H, _ = cv2.findHomography(src, dst, 0)
    return H if H is not None and np.all(np.isfinite(H)) else None


def click_noise_uncertainty(src, dst, center, estimate, n_samples: int = 200,
                            seed: int = 0) -> float:
    """95-й процентиль зміщення центру за шуму кліків (px карти), Монте-Карло."""
    rng = np.random.default_rng(seed)
    shifts = []
    for _ in range(n_samples):
        H = _fit(src + rng.normal(0, CLICK_NOISE_FRAME_PX, src.shape),
                 dst + rng.normal(0, CLICK_NOISE_MAP_PX, dst.shape))
        if H is None:
            shifts.append(np.inf)
            continue
        p = apply_homography([center], H)[0]
        shifts.append(np.hypot(p[0] - estimate[0], p[1] - estimate[1]))
    shifts = np.nan_to_num(np.asarray(shifts, float), nan=np.inf)
    return float(np.percentile(shifts, 95))


# ---------------------------------------------------------------------------
# Файли
# ---------------------------------------------------------------------------
def load_annotations(path: str | Path) -> dict[str, KeyframeAnnotation]:
    path = Path(path)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {a["frame_id"]: KeyframeAnnotation(**{**a, "frame_size": tuple(a["frame_size"])})
            for a in data.get("keyframes", [])}


def save_annotations(path: str | Path, annotations: dict[str, KeyframeAnnotation]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [asdict(a) for a in sorted(annotations.values(), key=lambda a: a.timestamp)]
    path.write_text(json.dumps({"description": "manual frame<->map point correspondences; "
                                               "frame px (x, y), full-map px (X, Y)",
                                "keyframes": items}, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def keyframes_table(annotations: dict[str, KeyframeAnnotation]) -> pd.DataFrame:
    rows = []
    for a in sorted(annotations.values(), key=lambda a: a.timestamp):
        est = estimate_keyframe(a)
        rows.append({"frame_id": a.frame_id, "timestamp": a.timestamp, "x_true": est.x_true,
                     "y_true": est.y_true, "is_keyframe": True, "n_points": est.n_points,
                     "residual_px": est.residual_px,
                     "max_residual_px": est.max_residual_px,
                     "center_uncertainty_px": est.center_uncertainty_px,
                     "center_in_hull": est.center_in_hull, "reliable": est.reliable,
                     "source": "manual_homography"})
    return pd.DataFrame(rows, columns=GT_COLUMNS)


def interpolate_ground_truth(keyframes: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Лінійна інтерполяція в часі між сусідніми опорними кадрами.

    ``frames`` — ``frame_id, timestamp_sec`` усіх кадрів. Кадри поза проміжком
    опорних кадрів не отримують еталону (екстраполяція не виконується).
    Інтерпольовані точки позначаються ``is_keyframe = False``: на розворотах
    між опорними кадрами вони менш надійні, тому основні метрики рахуються
    за опорними кадрами.
    """
    key = keyframes.sort_values("timestamp")
    if len(key) < 2:
        return key.reset_index(drop=True)
    t_key = key["timestamp"].to_numpy(float)
    rows = []
    for f in frames.itertuples(index=False):
        t = float(f.timestamp_sec)
        if f.frame_id in set(key["frame_id"]):
            rows.append(key.loc[key["frame_id"] == f.frame_id].iloc[0].to_dict())
        elif t_key[0] <= t <= t_key[-1]:
            rows.append({"frame_id": f.frame_id, "timestamp": t,
                         "x_true": float(np.interp(t, t_key, key["x_true"])),
                         "y_true": float(np.interp(t, t_key, key["y_true"])),
                         "is_keyframe": False, "n_points": 0, "residual_px": np.nan,
                         "max_residual_px": np.nan, "center_uncertainty_px": np.nan,
                         "center_in_hull": None,
                         "reliable": None,
                         "source": "linear_interpolation"})
    return pd.DataFrame(rows, columns=GT_COLUMNS)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    gt = pd.read_csv(path)
    missing = {"frame_id", "x_true", "y_true"} - set(gt.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "is_keyframe" not in gt:
        gt["is_keyframe"] = True
    gt["is_keyframe"] = gt["is_keyframe"].astype(str).str.lower().isin(["true", "1"])
    for flag in ("reliable", "center_in_hull"):
        if flag in gt:
            gt[flag] = gt[flag].astype(str).str.lower().isin(["true", "1"])
    return gt
