"""Автотести Етапу 2: масштабування, перенесення координат, RANSAC, критерій успіху.

Нейромережі не потрібні: замість LightGlue/LoFTR використовується «ідеальний»
matcher, який повертає відповідності за відомою гомографією (+ викиди).

Запуск:  python -m pytest tests -q      або      python tests/test_stage2_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.evaluation.metrics import reprojection_error_stats
from src.geometry.homography import (PlausibilityConfig, RansacConfig, diagnose_homography,
                                     estimate_homography, is_successful, project_points)
from src.localization.estimate import estimate_global_position
from src.matching.base import ImageMatcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.images import (resize_by_scale, scaled_size, to_original_coordinates,
                                      to_scaled_coordinates)


def similarity(scale, angle_deg, tx, ty):
    a = np.radians(angle_deg)
    return np.array([[scale * np.cos(a), -scale * np.sin(a), tx],
                     [scale * np.sin(a), scale * np.cos(a), ty],
                     [0.0, 0.0, 1.0]])


class OracleMatcher(ImageMatcher):
    """Відповідності в РОБОЧИХ координатах, узгоджені з гомографією в ОРИГІНАЛЬНИХ.

    ``H_orig`` відображає оригінальний кадр в оригінальний тайл; matcher знає
    масштаби робочих зображень і генерує точки так, як це зробив би реальний метод.
    """
    name = "Oracle"

    def __init__(self, H_orig, frame_scale_xy, tile_scale_xy, n=300, outliers=0.3,
                 noise=0.0, size_multiple=1, seed=0):
        super().__init__(device="cpu")
        self.H, self.fs, self.ts = H_orig, frame_scale_xy, tile_scale_xy
        self.n, self.outliers, self.noise = n, outliers, noise
        self.size_multiple = size_multiple
        self.rng = np.random.default_rng(seed)

    def _extract(self, image):
        return image.shape[:2], None

    def _match(self, shape0, shape1):
        h0, w0 = shape0
        h1, w1 = shape1
        work0 = self.rng.uniform([0, 0], [w0 - 1, h0 - 1], (self.n, 2))
        orig1 = project_points(to_original_coordinates(work0, self.fs), self.H)
        work1 = to_scaled_coordinates(orig1, self.ts) + self.rng.normal(0, self.noise, (self.n, 2))
        n_out = int(self.outliers * self.n)
        work1[:n_out] = self.rng.uniform([0, 0], [w1 - 1, h1 - 1], (n_out, 2))
        return work0, work1, np.ones(self.n, np.float32)


# ---------------------------------------------------------------------------
# Масштабування та координати
# ---------------------------------------------------------------------------
def test_scaled_size_respects_multiple():
    assert scaled_size(1920, 1080, 0.27) == (518, 292)
    assert scaled_size(1920, 1080, 0.27, multiple=8) == (520, 288)
    assert scaled_size(512, 512, 1.0, multiple=8) == (512, 512)


def test_coordinate_roundtrip_and_pixel_centers():
    pts = np.array([[0.0, 0.0], [10.5, 3.25], [1919.0, 1079.0]])
    scale = (0.27, 0.2667)
    assert np.allclose(to_original_coordinates(to_scaled_coordinates(pts, scale), scale), pts)
    # Центр зображення переходить у центр (узгодження центрів пікселів, як у cv2.resize).
    img = np.zeros((1080, 1920, 3), np.uint8)
    small, sxy = resize_by_scale(img, 0.25)
    h, w = small.shape[:2]
    center_small = np.array([[(w - 1) / 2, (h - 1) / 2]])
    assert np.allclose(to_original_coordinates(center_small, sxy), [[959.5, 539.5]])


def test_resize_maps_a_feature_to_predicted_location():
    img = np.zeros((400, 600, 3), np.uint8)
    img[200:204, 300:304] = 255                      # «пляма» з центром (301.5, 201.5)
    small, sxy = resize_by_scale(img, 0.5)
    ys, xs = np.nonzero(small[..., 0])
    weights = small[ys, xs, 0].astype(float)
    found = np.array([[np.average(xs, weights=weights), np.average(ys, weights=weights)]])
    assert np.allclose(to_original_coordinates(found, sxy), [[301.5, 201.5]], atol=0.1)


# ---------------------------------------------------------------------------
# Гомографія та її перевірка
# ---------------------------------------------------------------------------
def test_ransac_recovers_homography_and_rejects_outliers():
    rng = np.random.default_rng(1)
    H_true = similarity(0.27, 10, 40, 60)
    H_true[2, :2] = [1e-6, -2e-6]
    pts0 = rng.uniform(0, [1920, 1080], (400, 2))
    pts1 = project_points(pts0, H_true) + rng.normal(0, 0.3, (400, 2))
    pts1[:100] = rng.uniform(0, 512, (100, 2))       # 25 % викидів
    result = estimate_homography(pts0, pts1, RansacConfig(threshold=3.0))
    assert result.H is not None
    assert not result.inlier_mask[:100].any() or result.inlier_mask[:100].sum() < 5
    assert result.inlier_mask[100:].mean() > 0.95
    stats = reprojection_error_stats(result.inlier_errors())
    assert stats["max"] <= 3.0 and stats["median"] < 1.0
    corners = np.array([[0, 0], [1919, 1079]], float)
    assert np.allclose(project_points(corners, result.H), project_points(corners, H_true), atol=1.0)


def test_ransac_is_reproducible_with_seed():
    rng = np.random.default_rng(2)
    pts0 = rng.uniform(0, 500, (200, 2))
    pts1 = pts0 + 5 + rng.normal(0, 1.0, (200, 2))
    pts1[:120] = rng.uniform(0, 500, (120, 2))
    a = estimate_homography(pts0, pts1, RansacConfig(seed=0))
    b = estimate_homography(pts0, pts1, RansacConfig(seed=0))
    assert np.array_equal(a.inlier_mask, b.inlier_mask)


def test_too_few_matches():
    result = estimate_homography(np.zeros((3, 2)), np.zeros((3, 2)))
    assert result.H is None and result.num_inliers == 0 and result.inlier_ratio == 0.0
    assert is_successful(result, None) == (False, "fewer than 4 matches")


def test_diagnostics_of_similarity():
    d = diagnose_homography(similarity(0.27, 30, 100, 50), (1920, 1080),
                            PlausibilityConfig(expected_scale=0.27))
    assert d.plausible and d.reasons == []
    assert abs(d.scale - 0.27) < 1e-9 and abs(d.rotation_deg - 30) < 1e-6
    assert abs(d.anisotropy - 1.0) < 1e-9


def test_diagnostics_reject_implausible_homographies():
    cfg = PlausibilityConfig(expected_scale=0.27, scale_tolerance=2.0, max_anisotropy=1.5)
    mirror = similarity(0.27, 0, 0, 0) @ np.diag([-1.0, 1.0, 1.0])
    assert not diagnose_homography(mirror, (1920, 1080), cfg).orientation_preserved
    squashed = np.diag([0.27, 0.1, 1.0])
    assert any("anisotropy" in r for r in diagnose_homography(squashed, (1920, 1080), cfg).reasons)
    too_small = similarity(0.05, 0, 0, 0)
    assert any("scale" in r for r in diagnose_homography(too_small, (1920, 1080), cfg).reasons)
    horizon = similarity(0.27, 0, 0, 0)
    horizon[2] = [-1e-3, 0, 1.0]                     # знаменник змінює знак у межах кадру
    assert not diagnose_homography(horizon, (1920, 1080), cfg).plausible


# ---------------------------------------------------------------------------
# Повний pipeline з «ідеальним» matcher
# ---------------------------------------------------------------------------
def _run_oracle(frame_to_map=0.27, work_scale=1.0, size_multiple=1, **matcher_kwargs):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    tile = np.zeros((512, 512, 3), np.uint8)
    H_true = similarity(frame_to_map, 5, 20, 110)    # кадр -> тайл, оригінальні px
    config = PipelineConfig(frame_to_map_scale=frame_to_map, work_scale=work_scale)
    probe = PairMatchingPipeline(OracleMatcher(H_true, (1, 1), (1, 1), size_multiple=size_multiple),
                                 config)
    fs = probe.prepare_frame(frame).scale_xy
    ts = probe.prepare_tile(tile).scale_xy
    matcher = OracleMatcher(H_true, fs, ts, size_multiple=size_multiple, **matcher_kwargs)
    result = PairMatchingPipeline(matcher, config).run(frame, tile, tile_origin=(768, 2048))
    return result, H_true


def test_pipeline_recovers_original_resolution_homography():
    for work_scale, multiple in [(1.0, 1), (1.5, 1), (1.0, 8), (0.75, 8)]:
        result, H_true = _run_oracle(work_scale=work_scale, size_multiple=multiple)
        assert result.success, result.failure_reason
        assert result.frame.image.shape[1] % multiple == 0
        corners = np.array([[0, 0], [1919, 0], [1919, 1079], [0, 1079]], float)
        assert np.allclose(project_points(corners, result.homography.H),
                           project_points(corners, H_true), atol=1e-3)
        assert abs(result.diagnostics.scale - 0.27) < 1e-4


def test_frame_center_in_map_coordinates():
    result, H_true = _run_oracle()
    cx, cy = project_points([(959.5, 539.5)], H_true)[0]
    assert np.allclose(result.frame_center_map, (768 + cx, 2048 + cy), atol=1e-3)
    assert np.allclose(estimate_global_position((1920, 1080), H_true, (768, 2048)),
                       (768 + cx, 2048 + cy))
    row = result.to_row()
    assert row["success"] and row["inliers"] >= 200 and row["reproj_max"] < 1e-3


def test_pipeline_fails_on_random_matches():
    result, _ = _run_oracle(outliers=1.0)
    assert not result.success and result.failure_reason


def test_matcher_rejects_wrong_input_size():
    matcher = OracleMatcher(np.eye(3), (1, 1), (1, 1), size_multiple=8)
    try:
        matcher.extract(np.zeros((100, 100, 3), np.uint8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for size not divisible by 8")


def test_tile_feature_cache():
    calls = []

    class Counting(OracleMatcher):
        def _extract(self, image):
            calls.append(image.shape)
            return super()._extract(image)

    matcher = Counting(similarity(0.27, 0, 0, 0), (0.27, 0.27), (1, 1))
    pipeline = PairMatchingPipeline(matcher, PipelineConfig())
    tile = np.zeros((512, 512, 3), np.uint8)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    first = pipeline.run(frame, tile, tile_cache_key="tile_0001.png")
    second = pipeline.run(frame, tile, tile_cache_key="tile_0001.png")
    assert calls.count((512, 512, 3)) == 1                   # тайл оброблено один раз
    assert second.match.time_extract1 == 0.0 and first.match.num_matches == 300


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
