"""Автоматичний пошук тайла для одиночного кадру (Етап 3).

    кадр -> ознаки кадру (один раз) -> для кожного тайла пулу:
        matcher -> RANSAC -> перевірка правдоподібності -> фільтр + бал
    -> найкращий тайл -> центр кадру і контур огляду в координатах карти.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.scoring import (CandidateEvaluation, CandidateMetrics, GateConfig,
                                    rank_candidates)
from src.localization.coordinates import (apply_homography, footprint_on_map, frame_center,
                                          inside_bounds, local_to_map)
from src.matching.pipeline import PairMatchingPipeline, PairResult
from src.preprocessing.images import read_image_rgb


# ---------------------------------------------------------------------------
# Пул тайлів
# ---------------------------------------------------------------------------
def grid_indices(tiles: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Номери стовпця і рядка тайла в сітці (за відсортованими ``x_origin``/``y_origin``)."""
    xs = {x: i for i, x in enumerate(sorted(tiles["x_origin"].unique()))}
    ys = {y: i for i, y in enumerate(sorted(tiles["y_origin"].unique()))}
    return tiles["x_origin"].map(xs), tiles["y_origin"].map(ys)


def select_tiles(tiles: pd.DataFrame, roi_xyxy=None, grid_step: int = 1) -> pd.DataFrame:
    """Обмежує простір пошуку.

    * ``roi_xyxy = (x_min, y_min, x_max, y_max)`` — лишаються тайли, що
      перетинаються з прямокутником (px карти);
    * ``grid_step = n`` — лише кожен n-й стовпець і рядок сітки (прорідження
      для швидких тестів LoFTR; при перекритті 50 % і n = 2 тайли стикуються
      без перекриття).
    """
    selected = tiles
    if roi_xyxy is not None:
        x0, y0, x1, y1 = roi_xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid ROI {roi_xyxy}: expected x_min < x_max, y_min < y_max")
        selected = selected[(selected["x_origin"] < x1)
                            & (selected["x_origin"] + selected["width"] > x0)
                            & (selected["y_origin"] < y1)
                            & (selected["y_origin"] + selected["height"] > y0)]
    if grid_step > 1:
        col, row = grid_indices(tiles)
        keep = (col % grid_step == 0) & (row % grid_step == 0)
        selected = selected[keep.loc[selected.index]]
    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Результати
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    tile_id: str
    tile_origin: tuple[float, float]
    tile_size: tuple[float, float]
    pair: PairResult
    metrics: CandidateMetrics
    evaluation: CandidateEvaluation | None = None

    def to_row(self) -> dict:
        center = self.metrics.center_local
        ev = self.evaluation
        return {"tile_id": self.tile_id, "x_origin": self.tile_origin[0],
                "y_origin": self.tile_origin[1],
                "passed": ev.passed if ev else None, "score": ev.score if ev else None,
                "gate_reasons": "; ".join(ev.reasons) if ev else None,
                "center_x_local": center[0] if center else None,
                "center_y_local": center[1] if center else None,
                **self.pair.to_row()}


@dataclass
class SearchResult:
    frame_size: tuple[int, int]
    candidates: list[Candidate]                 # у порядку ранжування
    timings: dict = field(default_factory=dict)

    @property
    def winner(self) -> Candidate | None:
        best = self.candidates[0] if self.candidates else None
        return best if best is not None and best.evaluation.passed else None

    @property
    def passed(self) -> list[Candidate]:
        return [c for c in self.candidates if c.evaluation.passed]

    def center_map(self) -> tuple[float, float] | None:
        w = self.winner
        if w is None:
            return None
        x, y = local_to_map([w.metrics.center_local], w.tile_origin)[0]
        return float(x), float(y)

    def footprint_map(self) -> np.ndarray | None:
        w = self.winner
        if w is None:
            return None
        return footprint_on_map(self.frame_size, w.pair.homography.H, w.tile_origin)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_row() for c in self.candidates])


# ---------------------------------------------------------------------------
# Пошук
# ---------------------------------------------------------------------------
class TileSearch:
    """Перебір пулу тайлів для одного кадру з єдиними правилами фільтра і балу."""

    def __init__(self, pipeline: PairMatchingPipeline, tiles_dir: str | Path,
                 map_size: tuple[int, int], gate: GateConfig = GateConfig()):
        self.pipeline = pipeline
        self.tiles_dir = Path(tiles_dir)
        self.map_size = map_size
        self.gate = gate

    def candidate_metrics(self, tile_id, tile_size, pair: PairResult) -> CandidateMetrics:
        g, d = pair.homography, pair.diagnostics
        center_local = None
        if g.H is not None:
            x, y = apply_homography([frame_center(pair.frame.original_size)], g.H)[0]
            if np.isfinite(x) and np.isfinite(y):
                center_local = (float(x), float(y))
        center_in_map = False
        if center_local is not None:
            cx, cy = local_to_map([center_local], pair.tile_origin)[0]
            center_in_map = inside_bounds((cx, cy), *self.map_size)
        errors = g.inlier_errors()
        return CandidateMetrics(
            tile_id=tile_id, matches=g.num_matches, inliers=g.num_inliers,
            reproj_mean=float(errors.mean()) if len(errors) else None,
            homography_found=g.H is not None, plausible=bool(d is not None and d.plausible),
            center_local=center_local, tile_size=tile_size, center_in_map=center_in_map)

    def search(self, frame: np.ndarray, tiles: pd.DataFrame, progress_every: int = 0
               ) -> SearchResult:
        start = time.perf_counter()
        frame_features = self.pipeline.frame_features(frame)
        time_frame = frame_features[1].time

        candidates, time_io = [], 0.0
        for i, tile in enumerate(tiles.itertuples(index=False), 1):
            # Без дискового кешу ознаки тайла не зберігаються в пам'яті (для LoFTR це
            # були б тензори всіх тайлів у VRAM) — кожен тайл читається й обробляється заново.
            image, cached = None, self.pipeline.is_cached(tile.tile_id)
            if not cached:
                t0 = time.perf_counter()
                image = read_image_rgb(self.tiles_dir / tile.tile_id)
                time_io += time.perf_counter() - t0
            origin = (float(tile.x_origin), float(tile.y_origin))
            size = (float(tile.width), float(tile.height))
            pair = self.pipeline.run(None, image, origin, tile_cache_key=tile.tile_id if cached else None,
                                     frame_features=frame_features)
            candidates.append(Candidate(tile.tile_id, origin, size, pair,
                                        self.candidate_metrics(tile.tile_id, size, pair)))
            if progress_every and i % progress_every == 0:
                print(f"  {i}/{len(tiles)} tiles")

        ranked = rank_candidates([c.metrics for c in candidates], self.gate)
        by_id = {c.tile_id: c for c in candidates}
        ordered = []
        for metrics, evaluation in ranked:
            candidate = by_id[metrics.tile_id]
            candidate.evaluation = evaluation
            ordered.append(candidate)

        pairs = [c.pair for c in candidates]
        timings = {
            "frame_extract": time_frame,
            "tile_extract_sum": sum(p.match.time_extract1 for p in pairs),
            "match_sum": sum(p.match.time_match for p in pairs),
            "ransac_sum": sum(p.homography.time for p in pairs),
            "tile_io": time_io,
            "search_wall": time.perf_counter() - start,
        }
        timings["per_tile_mean"] = (timings["tile_extract_sum"] + timings["match_sum"]
                                    + timings["ransac_sum"]) / max(1, len(pairs))
        return SearchResult(frame_size=frame_features[0].original_size, candidates=ordered,
                            timings=timings)
