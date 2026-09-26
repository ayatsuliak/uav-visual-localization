"""Автотести Етапу 3: координати, фільтр і бал ранжування, пул тайлів, кеш, пошук.

Нейромережі не потрібні: використовується синтетичний matcher.

Запуск:  python -m pytest tests -q      або      python tests/test_stage3_localization.py
"""
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.scoring import (CandidateMetrics, GateConfig, center_inside_tile,
                                    evaluate_candidate, rank_candidates, ranking_score)
from src.localization.coordinates import (apply_homography, estimate_global_position,
                                          footprint_on_map, frame_center, frame_corners,
                                          inside_bounds, local_to_map)
from src.localization.search import TileSearch, select_tiles
from src.matching.base import Features, ImageMatcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.cache import build_tile_cache, load_tile_cache
from src.preprocessing.images import (to_original_coordinates, to_scaled_coordinates,
                                      write_image_rgb)
from src.preprocessing.tiles import generate_tile_grid


def similarity(scale, angle_deg, tx, ty):
    a = np.radians(angle_deg)
    return np.array([[scale * np.cos(a), -scale * np.sin(a), tx],
                     [scale * np.sin(a), scale * np.cos(a), ty], [0, 0, 1.0]])


# ---------------------------------------------------------------------------
# coordinates.py
# ---------------------------------------------------------------------------
def test_center_and_corners_follow_pixel_centre_convention():
    assert frame_center((1920, 1080)) == (959.5, 539.5)
    assert frame_corners((1920, 1080)).tolist() == [[0, 0], [1919, 0], [1919, 1079], [0, 1079]]


def test_apply_homography_dehomogenizes():
    H = np.array([[2.0, 0, 10], [0, 3.0, 20], [0.001, 0, 1.0]])
    x, y = 100.0, 50.0
    w = 0.001 * x + 1.0
    assert np.allclose(apply_homography([(x, y)], H), [[(2 * x + 10) / w, (3 * y + 20) / w]])
    # Масштабування H на константу не змінює результат (однорідні координати).
    assert np.allclose(apply_homography([(x, y)], 7.5 * H), apply_homography([(x, y)], H))


def test_point_at_infinity_is_nan():
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.01, 0, -1.0]])   # w = 0 при x = 100
    assert np.all(np.isnan(apply_homography([(100.0, 5.0)], H)))
    assert estimate_global_position((201, 11), H, (0, 0)) is None


def test_local_to_map_and_global_position():
    H = similarity(0.27, 0, 10, 20)
    cx, cy = estimate_global_position((1920, 1080), H, (768, 2048))
    assert math.isclose(cx, 768 + 10 + 0.27 * 959.5) and math.isclose(cy, 2048 + 20 + 0.27 * 539.5)
    assert local_to_map([(1, 2)], (768, 2048)).tolist() == [[769, 2050]]


def test_footprint_on_map():
    H = similarity(0.25, 90, 300, 0)                             # поворот на 90°
    fp = footprint_on_map((1920, 1080), H, (1000, 2000))
    assert fp.shape == (4, 2)
    assert np.allclose(fp[0], [1300, 2000])                      # (0,0) -> зсув
    assert np.allclose(fp[1], [1300, 2000 + 0.25 * 1919])        # вісь x кадру -> вісь y карти
    side_a = np.linalg.norm(fp[1] - fp[0])
    side_b = np.linalg.norm(fp[3] - fp[0])
    assert math.isclose(side_a / side_b, 1919 / 1079)


def test_inside_bounds():
    assert inside_bounds((0, 0), 6200, 4320) and inside_bounds((6199, 4319), 6200, 4320)
    assert not inside_bounds((6200, 10), 6200, 4320)
    assert not inside_bounds((-1, 10), 6200, 4320)
    assert inside_bounds((-1, 10), 6200, 4320, margin=1)
    assert not inside_bounds((float("nan"), 10), 6200, 4320)


# ---------------------------------------------------------------------------
# scoring.py
# ---------------------------------------------------------------------------
def candidate(**overrides):
    base = dict(tile_id="t", matches=200, inliers=80, reproj_mean=1.5, homography_found=True,
                plausible=True, center_local=(250.0, 250.0), tile_size=(512.0, 512.0))
    base.update(overrides)
    return CandidateMetrics(**base)


def test_ranking_score_formula():
    expected = 80 * (80 / 200) * math.exp(-1.5 / 2.0)
    assert math.isclose(ranking_score(80, 200, 1.5, 2.0), expected)
    assert ranking_score(0, 200, 1.0, 2.0) == 0.0 and ranking_score(5, 0, 1.0, 2.0) == 0.0
    ev = evaluate_candidate(candidate())
    assert ev.passed and math.isclose(ev.score, expected)


def test_score_monotonicity():
    # Більше узгоджених відповідностей, вища частка, менша похибка -> вищий бал.
    assert ranking_score(100, 200, 1.5, 2) > ranking_score(80, 200, 1.5, 2)
    assert ranking_score(80, 160, 1.5, 2) > ranking_score(80, 200, 1.5, 2)
    assert ranking_score(80, 200, 1.0, 2) > ranking_score(80, 200, 2.0, 2)
    # Не зводиться до inliers / error: подвоєння похибки не ділить бал навпіл.
    assert not math.isclose(ranking_score(80, 200, 2.0, 2) / ranking_score(80, 200, 1.0, 2), 0.5)


def test_gate_rejections():
    cfg = GateConfig(min_inliers=15, min_inlier_ratio=0.15, max_reproj_error=3.5,
                     center_margin=30)
    cases = {
        "no homography": candidate(homography_found=False),
        "implausible homography": candidate(plausible=False),
        "inliers 14 < 15": candidate(inliers=14, matches=20),
        "inlier ratio": candidate(inliers=20, matches=200),
        "reprojection error": candidate(reproj_mean=3.6),
        "frame centre outside the tile": candidate(center_local=(-31.0, 100.0)),
        "frame centre outside the map": candidate(center_in_map=False),
    }
    for reason, c in cases.items():
        ev = evaluate_candidate(c, cfg)
        assert not ev.passed and ev.score == 0.0, reason
        assert any(r.startswith(reason) for r in ev.reasons), (reason, ev.reasons)
    # Межові значення проходять.
    assert evaluate_candidate(candidate(inliers=15, matches=100), cfg).passed
    assert evaluate_candidate(candidate(reproj_mean=3.5), cfg).passed


def test_center_margin_boundaries():
    assert center_inside_tile((-30, 0), (512, 512), 30)
    assert center_inside_tile((541, 541), (512, 512), 30)            # 511 + 30
    assert not center_inside_tile((541.01, 0), (512, 512), 30)
    assert not center_inside_tile(None, (512, 512), 30)


def test_rank_candidates_order():
    cands = [candidate(tile_id="weak", inliers=40), candidate(tile_id="best", inliers=120),
             candidate(tile_id="rejected", inliers=500, plausible=False),
             candidate(tile_id="mid", inliers=80)]
    order = [c.tile_id for c, _ in rank_candidates(cands)]
    assert order == ["best", "mid", "weak", "rejected"]


# ---------------------------------------------------------------------------
# Пул тайлів
# ---------------------------------------------------------------------------
def _tiles_df():
    grid = generate_tile_grid(1280, 1024, 512, 0.5)
    return pd.DataFrame([vars(t) for t in grid])


def test_select_tiles_roi_and_grid_step():
    tiles = _tiles_df()                                          # 4 × 3 тайли
    assert len(tiles) == 12
    roi = select_tiles(tiles, (600, 600, 700, 700))              # точка (600..700)² у 4 тайлах
    assert sorted(zip(roi.x_origin, roi.y_origin)) == [(256, 256), (256, 512), (512, 256),
                                                       (512, 512)]
    step = select_tiles(tiles, grid_step=2)
    assert sorted(set(step.x_origin)) == [0, 512] and sorted(set(step.y_origin)) == [0, 512]


# ---------------------------------------------------------------------------
# Синтетичний matcher: «правильний» тайл має зашите значення яскравості
# ---------------------------------------------------------------------------
TRUE_VALUE = 200


class SyntheticMatcher(ImageMatcher):
    """Для тайла з яскравістю TRUE_VALUE повертає відповідності за ``H_true``, інакше — шум."""
    name = "Synthetic"
    supports_feature_cache = True

    def __init__(self, H_true, frame_scale_xy, tile_scale_xy, n=200, seed=0):
        super().__init__(device="cpu")
        self.H, self.fs, self.ts, self.n = H_true, frame_scale_xy, tile_scale_xy, n
        self.rng = np.random.default_rng(seed)

    def _extract(self, image):
        return {"size": np.array(image.shape[:2]), "value": np.array([image[0, 0, 0]])}, None

    def _match(self, d0, d1):
        (h0, w0), (h1, w1) = d0["size"], d1["size"]
        work0 = self.rng.uniform([0, 0], [w0 - 1, h0 - 1], (self.n, 2))
        if int(d1["value"][0]) == TRUE_VALUE:
            orig1 = apply_homography(to_original_coordinates(work0, self.fs), self.H)
            work1 = to_scaled_coordinates(orig1, self.ts)
        else:
            work1 = self.rng.uniform([0, 0], [w1 - 1, h1 - 1], (self.n, 2))
        return work0, work1, np.ones(self.n, np.float32)

    def features_to_arrays(self, feats):
        return dict(feats.data)

    def features_from_arrays(self, arrays, device=None):
        data = {"size": arrays["size"], "value": arrays["value"]}
        h, w = arrays["size"]
        return Features(data=data, image_size=(int(w), int(h)), num_keypoints=None, time=0.0)


def _write_tiles(tmp, tiles, true_tile):
    for t in tiles.itertuples(index=False):
        value = TRUE_VALUE if t.tile_id == true_tile else 50
        write_image_rgb(Path(tmp) / t.tile_id, np.full((t.height, t.width, 3), value, np.uint8))


def _make_search(tiles, tmp, H_true):
    config = PipelineConfig(frame_to_map_scale=0.27)
    probe = PairMatchingPipeline(SyntheticMatcher(H_true, (1, 1), (1, 1)), config)
    fs = probe.prepare_frame(np.zeros((1080, 1920, 3), np.uint8)).scale_xy
    matcher = SyntheticMatcher(H_true, fs, (1.0, 1.0))
    pipeline = PairMatchingPipeline(matcher, config)
    return TileSearch(pipeline, tmp, (1280, 1024)), pipeline


def test_search_finds_true_tile_and_global_centre():
    tiles = _tiles_df()
    true_tile = tiles.loc[(tiles.x_origin == 512) & (tiles.y_origin == 256), "tile_id"].item()
    H_true = similarity(0.27, 3, -40, 100)                       # кадр -> тайл (512, 256)
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiles(tmp, tiles, true_tile)
        search, _ = _make_search(tiles, tmp, H_true)
        result = search.search(np.zeros((1080, 1920, 3), np.uint8), tiles)
    assert result.winner is not None and result.winner.tile_id == true_tile
    assert len(result.passed) == 1
    expected = estimate_global_position((1920, 1080), H_true, (512, 256))
    assert np.allclose(result.center_map(), expected, atol=1e-6)
    assert np.allclose(result.footprint_map(), footprint_on_map((1920, 1080), H_true, (512, 256)))
    assert len(result.to_frame()) == len(tiles)


def test_tile_cache_roundtrip_gives_same_result():
    tiles = _tiles_df()
    true_tile = tiles.tile_id.iloc[5]
    row = tiles.iloc[5]
    H_true = similarity(0.27, 0, 0, 100)
    with tempfile.TemporaryDirectory() as tmp:
        tiles_dir, cache_dir = Path(tmp) / "tiles", Path(tmp) / "cache"
        tiles_dir.mkdir()
        _write_tiles(tiles_dir, tiles, true_tile)
        tiles_csv = Path(tmp) / "tiles.csv"
        tiles.to_csv(tiles_csv, index=False)

        search, pipeline = _make_search(tiles, tiles_dir, H_true)
        manifest = build_tile_cache(pipeline, tiles, tiles_dir, cache_dir, tiles_csv,
                                    progress_every=0)
        assert len(list(cache_dir.glob("tile_*.npz"))) == len(tiles)
        assert manifest["tile_ids"] == list(tiles.tile_id)

        load_tile_cache(pipeline, cache_dir, tiles.tile_id, tiles_csv)
        assert all(pipeline.is_cached(t) for t in tiles.tile_id)
        for p in tiles_dir.iterdir():                            # зображення більше не потрібні
            p.unlink()
        result = search.search(np.zeros((1080, 1920, 3), np.uint8), tiles)
        assert result.winner.tile_id == true_tile
        expected = estimate_global_position((1920, 1080), H_true, (row.x_origin, row.y_origin))
        assert np.allclose(result.center_map(), expected, atol=1e-6)

        # Кеш, побудований з іншими параметрами, не приймається.
        other = PairMatchingPipeline(pipeline.matcher, PipelineConfig(work_scale=1.5))
        try:
            load_tile_cache(other, cache_dir, tiles.tile_id, tiles_csv)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError for a mismatched cache")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
