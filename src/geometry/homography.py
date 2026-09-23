"""Геометрична перевірка відповідностей: RANSAC + гомографія.

Гомографія ``H`` переводить точки кадру (оригінальні пікселі) у точки тайла
(оригінальні пікселі тайла = пікселі карти зі зсувом). Тому поріг RANSAC і
похибка репроєкції вимірюються **в пікселях карти**.

Обмеження: гомографія точно описує лише плоску сцену (або чистий поворот
камери). Для надирної зйомки з висоти, значно більшої за висоту будівель,
це прийнятне наближення; паралакс дахів проявляється як залишкова похибка.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import (MAX_ANISOTROPY, MIN_INLIERS, RANSAC_CONFIDENCE, RANSAC_MAX_ITERS,
                        RANSAC_METHOD, RANSAC_SEED, RANSAC_THRESHOLD, SCALE_TOLERANCE)

_RANSAC_METHODS = {"RANSAC": cv2.RANSAC, "USAC_MAGSAC": cv2.USAC_MAGSAC}


@dataclass(frozen=True)
class RansacConfig:
    method: str = RANSAC_METHOD
    threshold: float = RANSAC_THRESHOLD          # px карти
    confidence: float = RANSAC_CONFIDENCE
    max_iters: int = RANSAC_MAX_ITERS
    seed: int | None = RANSAC_SEED


@dataclass(frozen=True)
class PlausibilityConfig:
    """Умови правдоподібності гомографії «кадр -> карта» для надирної зйомки."""
    expected_scale: float | None = None          # s_f; None -> масштаб не перевіряється
    scale_tolerance: float = SCALE_TOLERANCE     # допустимо [s_f / tol, s_f * tol]
    max_anisotropy: float = MAX_ANISOTROPY


@dataclass
class HomographyResult:
    H: np.ndarray | None
    inlier_mask: np.ndarray                      # (N,) bool
    num_matches: int
    reprojection_errors: np.ndarray              # (N,) px карти, для всіх відповідностей
    time: float = 0.0

    @property
    def num_inliers(self) -> int:
        return int(self.inlier_mask.sum())

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / self.num_matches if self.num_matches else 0.0

    def inlier_errors(self) -> np.ndarray:
        return self.reprojection_errors[self.inlier_mask]


@dataclass
class HomographyDiagnostics:
    """Параметри гомографії, обчислені в центрі кадру (локальна афінна апроксимація)."""
    scale: float                 # sqrt(|det J|) — px карти на px кадру
    rotation_deg: float          # кут повороту кадру відносно карти
    anisotropy: float            # sigma_max / sigma_min якобіана J
    orientation_preserved: bool  # det J > 0 (немає дзеркального відображення)
    footprint_convex: bool       # проєкція рамки кадру — опуклий чотирикутник
    footprint: np.ndarray        # (4, 2) проєкції кутів кадру (px тайла)
    plausible: bool
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Базові операції
# ---------------------------------------------------------------------------
def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2))
    return cv2.perspectiveTransform(points, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def reprojection_errors(pts0: np.ndarray, pts1: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Пряма похибка репроєкції ``||H·p0 - p1||`` для кожної відповідності."""
    return np.linalg.norm(project_points(pts0, H) - np.asarray(pts1, dtype=np.float64), axis=1)


def estimate_homography(pts0: np.ndarray, pts1: np.ndarray,
                        config: RansacConfig = RansacConfig()) -> HomographyResult:
    """RANSAC-оцінка гомографії ``pts0 -> pts1``.

    Відповідність вважається геометрично узгодженою, якщо похибка репроєкції
    не перевищує ``config.threshold``. Маска узгоджених відповідностей береться
    з RANSAC (OpenCV уточнює H за всіма такими відповідностями).
    """
    import time

    pts0 = np.asarray(pts0, dtype=np.float64).reshape(-1, 2)
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    n = len(pts0)
    empty = HomographyResult(H=None, inlier_mask=np.zeros(n, dtype=bool), num_matches=n,
                             reprojection_errors=np.full(n, np.nan))
    if n < 4:
        return empty

    if config.method not in _RANSAC_METHODS:
        raise ValueError(f"Unknown RANSAC method '{config.method}'")
    if config.seed is not None:
        cv2.setRNGSeed(int(config.seed))
    start = time.perf_counter()
    H, mask = cv2.findHomography(pts0, pts1, _RANSAC_METHODS[config.method],
                                 ransacReprojThreshold=config.threshold,
                                 maxIters=config.max_iters, confidence=config.confidence)
    elapsed = time.perf_counter() - start
    if H is None or mask is None or not np.all(np.isfinite(H)):
        empty.time = elapsed
        return empty
    return HomographyResult(H=H, inlier_mask=mask.ravel().astype(bool), num_matches=n,
                            reprojection_errors=reprojection_errors(pts0, pts1, H), time=elapsed)


# ---------------------------------------------------------------------------
# Правдоподібність
# ---------------------------------------------------------------------------
def homography_jacobian(H: np.ndarray, point_xy) -> np.ndarray:
    """Аналітичний якобіан 2×2 відображення ``H`` у точці ``(x, y)``."""
    H = np.asarray(H, dtype=np.float64)
    x, y = float(point_xy[0]), float(point_xy[1])
    u = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    v = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return np.array([
        [(H[0, 0] * w - u * H[2, 0]) / w ** 2, (H[0, 1] * w - u * H[2, 1]) / w ** 2],
        [(H[1, 0] * w - v * H[2, 0]) / w ** 2, (H[1, 1] * w - v * H[2, 1]) / w ** 2],
    ])


def _is_convex_quad(quad: np.ndarray) -> bool:
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    return bool(all(s > 0 for s in signs) or all(s < 0 for s in signs))


def diagnose_homography(H: np.ndarray, frame_size_wh: tuple[int, int],
                        config: PlausibilityConfig = PlausibilityConfig()) -> HomographyDiagnostics:
    """Перевіряє, чи може ``H`` описувати надирний кадр, спроєктований на карту.

    Умови (усі мають виконуватися):

    1. знаменник проєктивного перетворення має однаковий знак у всіх кутах
       кадру (рамка не перетинає «лінію нескінченності»);
    2. ``det J > 0`` — немає дзеркального відображення;
    3. проєкція рамки кадру — опуклий чотирикутник;
    4. анізотропія ``sigma_max / sigma_min <= max_anisotropy`` — кадр не
       «сплющений» (для надирної камери масштаб по осях майже однаковий);
    5. масштаб ``sqrt(|det J|)`` у межах ``[s_f / tol, s_f · tol]``, якщо ``s_f`` задано.
    """
    w, h = frame_size_wh
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    reasons: list[str] = []

    denominators = corners @ H[2, :2] + H[2, 2]
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        reasons.append("frame crosses the line at infinity")
    footprint = project_points(corners, H)

    J = homography_jacobian(H, center)
    det = float(np.linalg.det(J))
    singular = np.linalg.svd(J, compute_uv=False)
    scale = float(np.sqrt(abs(det)))
    anisotropy = float(singular[0] / singular[1]) if singular[1] > 1e-12 else float("inf")
    rotation_deg = float(np.degrees(np.arctan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1])))
    orientation_preserved = det > 0
    convex = _is_convex_quad(footprint)

    if not orientation_preserved:
        reasons.append("reflection (det J <= 0)")
    if not convex:
        reasons.append("footprint is not convex")
    if anisotropy > config.max_anisotropy:
        reasons.append(f"anisotropy {anisotropy:.2f} > {config.max_anisotropy}")
    if config.expected_scale is not None:
        low = config.expected_scale / config.scale_tolerance
        high = config.expected_scale * config.scale_tolerance
        if not (low <= scale <= high):
            reasons.append(f"scale {scale:.3f} outside [{low:.3f}, {high:.3f}]")

    return HomographyDiagnostics(scale=scale, rotation_deg=rotation_deg, anisotropy=anisotropy,
                                 orientation_preserved=orientation_preserved,
                                 footprint_convex=convex, footprint=footprint,
                                 plausible=not reasons, reasons=reasons)


def is_successful(result: HomographyResult, diagnostics: HomographyDiagnostics | None,
                  min_inliers: int = MIN_INLIERS) -> tuple[bool, str]:
    """Критерій успішного зіставлення пари. Повертає ``(success, reason)``.

    Успіх = гомографію знайдено ∧ кількість геометрично узгоджених
    відповідностей ``>= min_inliers`` ∧ гомографія правдоподібна.
    """
    if result.H is None:
        return False, "no homography" if result.num_matches >= 4 else "fewer than 4 matches"
    if result.num_inliers < min_inliers:
        return False, f"inliers {result.num_inliers} < {min_inliers}"
    if diagnostics is not None and not diagnostics.plausible:
        return False, "; ".join(diagnostics.reasons)
    return True, ""
