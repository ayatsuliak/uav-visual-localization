"""Єдиний pipeline зіставлення пари: ``frame + tile -> matcher -> RANSAC -> metrics``.

Нормалізація масштабу. Кадр і тайл мають різну просторову роздільність:
один піксель кадру відповідає ``s_f = FRAME_TO_MAP_SCALE`` пікселів карти.
Перед зіставленням обидва зображення приводяться до спільної роздільності:

* тайл масштабується на ``work_scale``;
* кадр масштабується на ``s_f · work_scale`` (кадр використовується повністю,
  без обрізання).

Після зіставлення координати відповідностей переносяться назад в
**оригінальні пікселі** кадру та тайла, тому гомографія ``H`` відображає
кадр (1920×1080) на тайл, а всі геометричні величини вимірюються в пікселях
карти незалежно від ``work_scale`` і методу зіставлення.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.config import FRAME_TO_MAP_SCALE, MIN_INLIERS, WORK_SCALE
from src.evaluation.metrics import reprojection_error_stats
from src.geometry.homography import (HomographyDiagnostics, HomographyResult, PlausibilityConfig,
                                     RansacConfig, diagnose_homography, estimate_homography,
                                     is_successful)
from src.localization.estimate import estimate_global_position
from src.matching.base import Features, ImageMatcher, MatchResult
from src.preprocessing.images import resize_by_scale, to_original_coordinates


@dataclass(frozen=True)
class PipelineConfig:
    frame_to_map_scale: float = FRAME_TO_MAP_SCALE
    work_scale: float = WORK_SCALE
    min_inliers: int = MIN_INLIERS
    ransac: RansacConfig = field(default_factory=RansacConfig)
    check_plausibility: bool = True
    plausibility: PlausibilityConfig = field(default_factory=PlausibilityConfig)

    @property
    def frame_scale(self) -> float:
        """Множник масштабування кадру перед подачею в matcher."""
        return self.frame_to_map_scale * self.work_scale

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedImage:
    """Зображення в робочій роздільності + фактичні масштаби для перенесення координат."""
    image: np.ndarray                    # RGB uint8 у робочій роздільності
    scale_xy: tuple[float, float]        # (sx, sy) = робоча / оригінальна роздільність
    original_size: tuple[int, int]       # (W, H) оригіналу


@dataclass
class PairResult:
    match: MatchResult                   # відповідності в ОРИГІНАЛЬНИХ пікселях кадру і тайла
    homography: HomographyResult         # H: кадр (px) -> тайл (px)
    diagnostics: HomographyDiagnostics | None
    success: bool
    failure_reason: str
    frame: PreparedImage
    tile: PreparedImage
    tile_origin: tuple[float, float]     # (x, y) лівого верхнього кута тайла на карті

    @property
    def frame_center_map(self) -> tuple[float, float] | None:
        """Проєкція центра кадру в координати повної карти (px) або ``None``."""
        return estimate_global_position(self.frame.original_size, self.homography.H,
                                        self.tile_origin)

    def to_row(self) -> dict:
        """Плоский запис для CSV. Визначення метрик — див. README (Stage 2)."""
        m, g, d = self.match, self.homography, self.diagnostics
        errors = reprojection_error_stats(g.inlier_errors())
        center = self.frame_center_map
        return {
            "frame_work_w": self.frame.image.shape[1], "frame_work_h": self.frame.image.shape[0],
            "tile_work_w": self.tile.image.shape[1], "tile_work_h": self.tile.image.shape[0],
            "keypoints_frame": m.num_keypoints0, "keypoints_tile": m.num_keypoints1,
            "matches": g.num_matches, "inliers": g.num_inliers, "inlier_ratio": g.inlier_ratio,
            "reproj_mean": errors["mean"], "reproj_median": errors["median"],
            "reproj_rmse": errors["rmse"], "reproj_max": errors["max"],
            "mean_confidence": float(m.confidence.mean()) if m.num_matches else None,
            "h_scale": d.scale if d else None, "h_rotation_deg": d.rotation_deg if d else None,
            "h_anisotropy": d.anisotropy if d else None,
            "h_plausible": d.plausible if d else None,
            "success": self.success, "failure_reason": self.failure_reason,
            "center_x_map": center[0] if center else None,
            "center_y_map": center[1] if center else None,
            "time_extract_frame": m.time_extract0, "time_extract_tile": m.time_extract1,
            "time_match": m.time_match, "time_matcher_total": m.time_total,
            "time_ransac": g.time,
        }


class PairMatchingPipeline:
    """Зіставлення кадру з тайлом та геометрична перевірка з єдиними правилами для всіх методів."""

    def __init__(self, matcher: ImageMatcher, config: PipelineConfig = PipelineConfig()):
        self.matcher = matcher
        self.config = config
        self._tile_cache: dict[object, tuple[PreparedImage, Features]] = {}

    # --- Підготовка -------------------------------------------------------
    def prepare(self, image: np.ndarray, scale: float) -> PreparedImage:
        h, w = image.shape[:2]
        resized, scale_xy = resize_by_scale(image, scale, self.matcher.size_multiple)
        return PreparedImage(image=resized, scale_xy=scale_xy, original_size=(w, h))

    def prepare_frame(self, frame: np.ndarray) -> PreparedImage:
        return self.prepare(frame, self.config.frame_scale)

    def prepare_tile(self, tile: np.ndarray) -> PreparedImage:
        return self.prepare(tile, self.config.work_scale)

    def tile_features(self, tile: np.ndarray, cache_key=None) -> tuple[PreparedImage, Features]:
        """Ознаки тайла; за наявності ``cache_key`` обчислюються один раз.

        Час екстракції з кешу дорівнює нулю — так відображається реальна
        вартість обробки, коли ознаки карти підготовлено заздалегідь.
        """
        if cache_key is not None and cache_key in self._tile_cache:
            prepared, feats = self._tile_cache[cache_key]
            return prepared, Features(feats.data, feats.image_size, feats.num_keypoints, 0.0)
        prepared = self.prepare_tile(tile)
        feats = self.matcher.extract(prepared.image)
        if cache_key is not None:
            self._tile_cache[cache_key] = (prepared, feats)
        return prepared, feats

    def clear_cache(self) -> None:
        self._tile_cache.clear()

    def warmup(self, frame_size_wh: tuple[int, int], tile_size_wh: tuple[int, int],
               iterations: int) -> None:
        """Прогрівання на зображеннях тих самих робочих розмірів, що й у експерименті."""
        dummy_frame = np.zeros((frame_size_wh[1], frame_size_wh[0], 3), np.uint8)
        dummy_tile = np.zeros((tile_size_wh[1], tile_size_wh[0], 3), np.uint8)
        f = self.prepare_frame(dummy_frame).image.shape
        t = self.prepare_tile(dummy_tile).image.shape
        self.matcher.warmup((f[1], f[0]), (t[1], t[0]), iterations)

    # --- Основний крок ----------------------------------------------------
    def run(self, frame: np.ndarray, tile: np.ndarray, tile_origin=(0.0, 0.0),
            tile_cache_key=None, frame_features: tuple[PreparedImage, Features] | None = None
            ) -> PairResult:
        """Зіставляє кадр із тайлом.

        ``frame_features`` дає змогу повторно використати ознаки кадру під час
        перебору кількох тайлів (Stage 3).
        """
        if frame_features is None:
            prepared_frame = self.prepare_frame(frame)
            feats_frame = self.matcher.extract(prepared_frame.image)
        else:
            prepared_frame, feats_frame = frame_features
        prepared_tile, feats_tile = self.tile_features(tile, tile_cache_key)

        work = self.matcher.match_features(feats_frame, feats_tile)
        match = MatchResult(
            pts0=to_original_coordinates(work.pts0, prepared_frame.scale_xy),
            pts1=to_original_coordinates(work.pts1, prepared_tile.scale_xy),
            confidence=work.confidence, num_keypoints0=work.num_keypoints0,
            num_keypoints1=work.num_keypoints1, time_extract0=work.time_extract0,
            time_extract1=work.time_extract1, time_match=work.time_match, extra=work.extra,
        )
        return self.verify(match, prepared_frame, prepared_tile, tile_origin)

    def verify(self, match: MatchResult, frame: PreparedImage, tile: PreparedImage,
               tile_origin=(0.0, 0.0)) -> PairResult:
        """RANSAC + перевірка правдоподібності + критерій успіху."""
        cfg = self.config
        homography = estimate_homography(match.pts0, match.pts1, cfg.ransac)
        diagnostics = None
        if homography.H is not None:
            plaus = cfg.plausibility
            if plaus.expected_scale is None:
                plaus = PlausibilityConfig(expected_scale=cfg.frame_to_map_scale,
                                           scale_tolerance=plaus.scale_tolerance,
                                           max_anisotropy=plaus.max_anisotropy)
            diagnostics = diagnose_homography(homography.H, frame.original_size, plaus)
        success, reason = is_successful(
            homography, diagnostics if cfg.check_plausibility else None, cfg.min_inliers)
        return PairResult(match=match, homography=homography, diagnostics=diagnostics,
                          success=success, failure_reason=reason, frame=frame, tile=tile,
                          tile_origin=(float(tile_origin[0]), float(tile_origin[1])))
