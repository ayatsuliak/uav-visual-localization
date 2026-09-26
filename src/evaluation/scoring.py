"""Правила вибору найкращого тайла-кандидата (Етап 3).

Це правило ранжування, а не дослідницька метрика якості: дослідницькі
метрики визначено в ``src.evaluation.metrics``. Відношення
``inliers / reprojection_error`` як основний критерій не використовується.

Крок 1 — жорсткий фільтр (gating). Кандидат відкидається, якщо не виконано
хоча б одну умову:

1. гомографію знайдено і вона правдоподібна
   (``src.geometry.homography.diagnose_homography``);
2. ``N_inliers >= min_inliers``;
3. ``inlier_ratio = N_inliers / N_all >= min_inlier_ratio``;
4. ``E_reproj <= max_reproj_error`` (середня похибка серед геометрично
   узгоджених відповідностей). Примітка: RANSAC залишає лише відповідності з
   похибкою ``<= RANSAC_THRESHOLD`` (3 px), тому при порозі 3.5 px ця умова
   виконується завжди і є запобіжником на випадок зміни порогу RANSAC;
5. проєкція центра кадру ``(x_local, y_local)`` лежить у межах тайла з
   допуском ``center_margin``: ``-m <= x_local <= S_w - 1 + m`` (так само
   для ``y``) — тайл є основним носієм центра кадру;
6. проєкція центра кадру лежить у межах карти.

Крок 2 — бал ранжування серед кандидатів, що пройшли фільтр:

    Score = N_inliers · (N_inliers / N_all) · exp(-E_reproj / sigma_r)

Переможець — кандидат із найбільшим ``Score``. За рівності балів перевага
надається більшій кількості геометрично узгоджених відповідностей.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from src.config import (CENTER_MARGIN, MAX_REPROJ_ERROR, MIN_INLIER_RATIO, MIN_INLIERS,
                        SCORE_SIGMA_R)


@dataclass(frozen=True)
class GateConfig:
    min_inliers: int = MIN_INLIERS
    min_inlier_ratio: float = MIN_INLIER_RATIO
    max_reproj_error: float = MAX_REPROJ_ERROR       # px карти
    center_margin: float = CENTER_MARGIN             # px тайла (= px карти)
    sigma_r: float = SCORE_SIGMA_R                   # px карти

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateMetrics:
    """Величини одного кандидата, потрібні для фільтра і балу."""
    tile_id: str
    matches: int
    inliers: int
    reproj_mean: float | None
    homography_found: bool
    plausible: bool
    center_local: tuple[float, float] | None          # px тайла
    tile_size: tuple[float, float]                    # (S_w, S_h)
    center_in_map: bool = True

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


@dataclass
class CandidateEvaluation:
    tile_id: str
    passed: bool
    score: float                       # 0 для відкинутих кандидатів
    reasons: list[str] = field(default_factory=list)


def ranking_score(inliers: int, matches: int, reproj_error: float, sigma_r: float) -> float:
    """``N_in · (N_in / N_all) · exp(-E / sigma_r)``."""
    if matches <= 0 or inliers <= 0:
        return 0.0
    return float(inliers * (inliers / matches) * math.exp(-reproj_error / sigma_r))


def center_inside_tile(center_local, tile_size, margin: float) -> bool:
    if center_local is None:
        return False
    x, y = center_local
    w, h = tile_size
    return bool(math.isfinite(x) and math.isfinite(y)
                and -margin <= x <= w - 1 + margin and -margin <= y <= h - 1 + margin)


def gate_reasons(c: CandidateMetrics, config: GateConfig) -> list[str]:
    """Причини відхилення кандидата (порожній список — фільтр пройдено)."""
    if not c.homography_found:
        return ["no homography"]
    reasons = []
    if not c.plausible:
        reasons.append("implausible homography")
    if c.inliers < config.min_inliers:
        reasons.append(f"inliers {c.inliers} < {config.min_inliers}")
    if c.inlier_ratio < config.min_inlier_ratio:
        reasons.append(f"inlier ratio {c.inlier_ratio:.3f} < {config.min_inlier_ratio}")
    if c.reproj_mean is None or not c.reproj_mean <= config.max_reproj_error:
        reasons.append(f"reprojection error {c.reproj_mean} > {config.max_reproj_error}")
    if not center_inside_tile(c.center_local, c.tile_size, config.center_margin):
        reasons.append("frame centre outside the tile")
    if not c.center_in_map:
        reasons.append("frame centre outside the map")
    return reasons


def evaluate_candidate(c: CandidateMetrics, config: GateConfig = GateConfig()) -> CandidateEvaluation:
    reasons = gate_reasons(c, config)
    if reasons:
        return CandidateEvaluation(c.tile_id, passed=False, score=0.0, reasons=reasons)
    return CandidateEvaluation(c.tile_id, passed=True,
                               score=ranking_score(c.inliers, c.matches, c.reproj_mean,
                                                   config.sigma_r))


def rank_candidates(candidates: list[CandidateMetrics], config: GateConfig = GateConfig()
                    ) -> list[tuple[CandidateMetrics, CandidateEvaluation]]:
    """Усі кандидати, відсортовані: спочатку ті, що пройшли фільтр, за спаданням балу."""
    evaluated = [(c, evaluate_candidate(c, config)) for c in candidates]
    return sorted(evaluated, key=lambda ce: (ce[1].passed, ce[1].score, ce[0].inliers),
                  reverse=True)
