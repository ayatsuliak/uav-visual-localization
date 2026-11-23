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

from src.config import FRAME_TO_MAP_SCALE, MIN_INLIERS, ROTATION_BORDER_PX, WORK_SCALE
from src.evaluation.metrics import reprojection_error_stats
from src.geometry.homography import (HomographyDiagnostics, HomographyResult, PlausibilityConfig,
                                     RansacConfig, diagnose_homography, estimate_homography,
                                     is_successful)
from src.localization.coordinates import estimate_global_position
from src.matching.base import Features, ImageMatcher, MatchResult
from src.preprocessing.images import (apply_affine, rescale_frame, resize_by_scale,
                                      rotate_expand, scale_matrix, to_original_coordinates)


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
    image: np.ndarray | None             # RGB uint8 у робочій роздільності (None — ознаки з кешу)
    scale_xy: tuple[float, float]        # (sx, sy) = робоча / оригінальна роздільність
    original_size: tuple[int, int]       # (W, H) оригіналу
    work_size: tuple[int, int]           # (W, H) у робочій роздільності
    # Повне перетворення «оригінал -> робоче» (3×3); None — лише масштаб scale_xy.
    transform: np.ndarray | None = None
    # Пікселі робочого зображення, що належать кадру (після повороту); None — усі.
    valid_mask: np.ndarray | None = None
    rotation_deg: float = 0.0            # поворот, застосований до кадру перед зіставленням

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Точки робочого зображення -> оригінальні пікселі."""
        if self.transform is None:
            return to_original_coordinates(points, self.scale_xy)
        return apply_affine(points, np.linalg.inv(self.transform))

    def inside_valid(self, points: np.ndarray) -> np.ndarray:
        """Маска точок, що лежать у коректній області (не на тлі після повороту)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self.valid_mask is None:
            return np.ones(len(points), dtype=bool)
        h, w = self.valid_mask.shape
        x = np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1)
        return self.valid_mask[y, x]

    @classmethod
    def from_sizes(cls, original_size, work_size) -> "PreparedImage":
        """Опис зображення без пікселів — для ознак, завантажених із кешу."""
        (w, h), (ww, wh) = original_size, work_size
        return cls(image=None, scale_xy=(ww / w, wh / h), original_size=(w, h),
                   work_size=(ww, wh))


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
            "frame_work_w": self.frame.work_size[0], "frame_work_h": self.frame.work_size[1],
            "tile_work_w": self.tile.work_size[0], "tile_work_h": self.tile.work_size[1],
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
        return PreparedImage(image=resized, scale_xy=scale_xy, original_size=(w, h),
                             work_size=(resized.shape[1], resized.shape[0]))

    def prepare_frame(self, frame: np.ndarray, rotation_deg: float = 0.0) -> PreparedImage:
        """Кадр -> роздільність карти × work_scale (``rescale_frame``).

        ``rotation_deg`` — компенсація курсу: кадр повертається на очікуваний кут
        «кадр -> карта», щоб matcher бачив кадр в орієнтації карти (SuperPoint і
        LoFTR не інваріантні до великих поворотів). Гомографія все одно
        оцінюється в оригінальних пікселях кадру.
        """
        h, w = frame.shape[:2]
        multiple = self.matcher.size_multiple
        resized, scale_xy = rescale_frame(frame, self.config.frame_to_map_scale,
                                          self.config.work_scale,
                                          1 if rotation_deg else multiple)
        if not rotation_deg:
            return PreparedImage(image=resized, scale_xy=scale_xy, original_size=(w, h),
                                 work_size=(resized.shape[1], resized.shape[0]))
        rotated, A, mask = rotate_expand(resized, rotation_deg, multiple,
                                         border=ROTATION_BORDER_PX)
        return PreparedImage(image=rotated, scale_xy=scale_xy, original_size=(w, h),
                             work_size=(rotated.shape[1], rotated.shape[0]),
                             transform=A @ scale_matrix(scale_xy), valid_mask=mask,
                             rotation_deg=float(rotation_deg))

    def frame_features(self, frame: np.ndarray, rotation_deg: float = 0.0
                       ) -> tuple[PreparedImage, Features]:
        """Ознаки кадру — обчислюються один раз для перебору багатьох тайлів."""
        prepared = self.prepare_frame(frame, rotation_deg)
        return prepared, self.matcher.extract(prepared.image)

    def prepare_tile(self, tile: np.ndarray) -> PreparedImage:
        return self.prepare(tile, self.config.work_scale)

    def tile_features(self, tile: np.ndarray | None, cache_key=None) -> tuple[PreparedImage, Features]:
        """Ознаки тайла; за наявності ``cache_key`` обчислюються один раз.

        Час екстракції з кешу дорівнює нулю — так відображається реальна
        вартість обробки, коли ознаки карти підготовлено заздалегідь.
        """
        if tile is None and cache_key not in self._tile_cache:
            raise KeyError(f"Tile '{cache_key}' is not cached and no image was given")
        if cache_key is not None and cache_key in self._tile_cache:
            prepared, feats = self._tile_cache[cache_key]
            return prepared, Features(feats.data, feats.image_size, feats.num_keypoints, 0.0)
        prepared = self.prepare_tile(tile)
        feats = self.matcher.extract(prepared.image)
        if cache_key is not None:
            self._tile_cache[cache_key] = (prepared, feats)
        return prepared, feats

    def add_cached_tile(self, cache_key, prepared: PreparedImage, feats: Features) -> None:
        """Додає готові ознаки тайла (наприклад, з дискового кешу ``src.preprocessing.cache``)."""
        self._tile_cache[cache_key] = (prepared, feats)

    def is_cached(self, cache_key) -> bool:
        return cache_key in self._tile_cache

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
    def run(self, frame: np.ndarray | None, tile: np.ndarray | None, tile_origin=(0.0, 0.0),
            tile_cache_key=None, frame_features: tuple[PreparedImage, Features] | None = None
            ) -> PairResult:
        """Зіставляє кадр із тайлом.

        ``frame_features`` дає змогу повторно використати ознаки кадру під час
        перебору кількох тайлів (Stage 3).
        """
        if frame_features is None:
            prepared_frame, feats_frame = self.frame_features(frame)
        else:
            prepared_frame, feats_frame = frame_features
        prepared_tile, feats_tile = self.tile_features(tile, tile_cache_key)

        work = self.matcher.match_features(feats_frame, feats_tile)
        keep = prepared_frame.inside_valid(work.pts0)       # не тло після повороту
        match = MatchResult(
            pts0=prepared_frame.to_original(work.pts0[keep]),
            pts1=prepared_tile.to_original(work.pts1[keep]),
            confidence=work.confidence[keep], num_keypoints0=work.num_keypoints0,
            num_keypoints1=work.num_keypoints1, time_extract0=work.time_extract0,
            time_extract1=work.time_extract1, time_match=work.time_match, extra=work.extra,
        )
        return self.verify(match, prepared_frame, prepared_tile, tile_origin)

    def verify(self, match: MatchResult, frame: PreparedImage, tile: PreparedImage,
               tile_origin=(0.0, 0.0)) -> PairResult:
        """RANSAC + перевірка правдоподібності + критерій успіху."""
        cfg = self.config
        if match.num_matches < cfg.min_inliers:
            # Критерій успіху недосяжний: RANSAC не запускається (на наборі з самих
            # викидів cv2.RANSAC відпрацьовує всі max_iters ітерацій, ≈0.1 с).
            n = match.num_matches
            homography = HomographyResult(H=None, inlier_mask=np.zeros(n, dtype=bool),
                                          num_matches=n, reprojection_errors=np.full(n, np.nan))
        else:
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
