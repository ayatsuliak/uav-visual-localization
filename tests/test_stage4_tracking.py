"""Автотести Етапу 4: сусіди 3×3, кінематична перевірка, екстраполяція, стани трекера.

Пошук тайлів замінено заглушкою, яка повертає задані позиції, тому нейромережі
не потрібні.

Запуск:  python -m pytest tests -q      або      python tests/test_stage4_tracking.py
"""
import math
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.localization.tracker import (Event, SequenceTracker, Status, TrackerConfig,
                                      TrackerState, extrapolate, max_displacement,
                                      extrapolate_yaw, nearest_tile, neighbor_tiles,
                                      passes_kinematic_gate, wrap_angle)
from src.preprocessing.tiles import generate_tile_grid

CFG = TrackerConfig(neighborhood_radius=1, max_misses=3, max_speed_px_per_s=120,
                    jump_margin_px=30, init_min_score=5.0, yaw_retry_offsets_deg=())


def tiles_df(width=6200, height=4320, size=512, overlap=0.5):
    return pd.DataFrame([vars(t) for t in generate_tile_grid(width, height, size, overlap)])


TILES = tiles_df()          # 24 × 16 = 384 тайли, як у проєкті


def tile_at(col, row):
    return TILES["tile_id"].iloc[row * 24 + col]


# ---------------------------------------------------------------------------
# Сусіди 3×3
# ---------------------------------------------------------------------------
def test_neighbors_inside_grid_is_3x3():
    pool = neighbor_tiles(TILES, tile_at(3, 8), radius=1)       # tile_0196
    assert tile_at(3, 8) == "tile_0196.png"
    assert len(pool) == 9
    assert set(pool["tile_id"]) == {tile_at(c, r) for c in (2, 3, 4) for r in (7, 8, 9)}
    assert list(pool.columns) == list(TILES.columns)


def test_neighbors_at_corner_and_edge():
    assert set(neighbor_tiles(TILES, tile_at(0, 0))["tile_id"]) == {
        tile_at(0, 0), tile_at(1, 0), tile_at(0, 1), tile_at(1, 1)}
    assert len(neighbor_tiles(TILES, tile_at(23, 5))) == 6       # правий край
    assert len(neighbor_tiles(TILES, tile_at(23, 15))) == 4      # правий нижній кут


def test_neighbors_radius_2_and_irregular_edge_tile():
    assert len(neighbor_tiles(TILES, tile_at(10, 8), radius=2)) == 25
    # Останній стовпець (x = 5688) — «крайовий» тайл з кроком 56 px, але в сітці він сусідній.
    edge = neighbor_tiles(TILES, tile_at(23, 8))
    assert sorted(edge["x_origin"].unique()) == [5632, 5688]


def test_unknown_tile_raises():
    try:
        neighbor_tiles(TILES, "tile_9999.png")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_nearest_tile():
    # Центр tile_0196: (768 + 255.5, 2048 + 255.5).
    assert nearest_tile(TILES, (1023.5, 2303.5)) == "tile_0196.png"
    assert nearest_tile(TILES, (-500, -500)) == "tile_0001.png"


# ---------------------------------------------------------------------------
# Кінематика
# ---------------------------------------------------------------------------
def test_max_displacement():
    assert max_displacement(1.0, CFG) == 150.0                  # D_max при 1 FPS
    assert max_displacement(2.0, CFG) == 270.0
    assert max_displacement(-1.0, CFG) == 30.0                  # від'ємний Δt не зменшує запас


def test_kinematic_gate_boundary():
    ok, d = passes_kinematic_gate((0, 0), 0.0, (90, 120), 1.0, CFG)     # |d| = 150
    assert ok and d == 150.0
    ok, d = passes_kinematic_gate((0, 0), 0.0, (90, 120.1), 1.0, CFG)
    assert not ok and d > 150
    assert passes_kinematic_gate((0, 0), 0.0, (0, 260), 2.0, CFG)[0]   # більший Δt


def test_extrapolation():
    assert extrapolate([], 1.0) is None
    assert extrapolate([(0.0, (10, 20))], 5.0) == (10.0, 20.0)
    fixes = [(0.0, (100, 100)), (1.0, (100, 50))]
    assert np.allclose(extrapolate(fixes, 2.0), (100, 0))                # P_{k-1} + (P_{k-1} - P_{k-2})
    assert np.allclose(extrapolate(fixes, 3.0), (100, -50))              # враховується Δt
    fast = [(0.0, (0, 0)), (1.0, (1000, 0))]
    assert np.allclose(extrapolate(fast, 2.0, max_speed=120), (1120, 0))  # |v| обмежено V_max


# ---------------------------------------------------------------------------
# Стани трекера (заглушка пошуку)
# ---------------------------------------------------------------------------
@dataclass
class _Eval:
    passed: bool
    score: float


@dataclass
class _Metrics:
    inliers: int = 80
    inlier_ratio: float = 0.4
    reproj_mean: float = 1.5


class _Candidate:
    def __init__(self, xy, score=20.0, yaw=None):
        self.center_map = xy
        self.tile_id = nearest_tile(TILES, xy)
        self.tile_origin = (0.0, 0.0)
        self.metrics = _Metrics()
        self.evaluation = _Eval(True, score)
        self.pair = SimpleNamespace(diagnostics=None if yaw is None else
                                    SimpleNamespace(rotation_deg=yaw, scale=0.27))


class _Result:
    def __init__(self, candidates):
        self.passed = candidates

    @property
    def winner(self):
        return self.passed[0] if self.passed else None


class ScriptedSearch:
    """Повертає для кожного виклику наступний список позицій (None — нічого не знайдено)."""

    def __init__(self, script):
        self.script, self.calls, self.yaws = list(script), [], []

    def search(self, frame, tiles, rotation_deg=0.0):
        self.calls.append(len(tiles))
        self.yaws.append(rotation_deg)
        item = self.script.pop(0)
        if item is None:
            return _Result([])
        items = item if isinstance(item, list) else [item]
        return _Result([_Candidate(xy, score) for xy, score in
                        ((i, 20.0) if not isinstance(i[0], tuple) else i for i in items)])


def run(script, times=None, config=CFG):
    search = ScriptedSearch(script)
    tracker = SequenceTracker(search, TILES, config=config)
    tracker._accept = _accept_without_footprint.__get__(tracker)
    times = times or [float(i) for i in range(len(script))]
    steps = [tracker.step(None, f"frame_{i + 1:04d}.png", t) for i, t in enumerate(times)]
    return steps, tracker, search


def _accept_without_footprint(self, step, candidate, t, status):
    self._fill_candidate(step, candidate)
    step.status, step.event = status, Event.OK
    step.x_map, step.y_map = candidate.center_map
    self.fixes.append((t, (step.x_map, step.y_map)))
    if step.h_rotation_deg is not None:
        self.yaws.append((t, step.h_rotation_deg))
    self.misses = 0


def test_init_then_local_tracking():
    steps, tracker, search = run([(1100, 2300), (1100, 2250), (1100, 2200)])
    assert [s.status for s in steps] == [Status.INITIALIZED, Status.TRACKING, Status.TRACKING]
    assert [s.search_mode for s in steps] == ["global", "local", "local"]
    assert search.calls == [384, 9, 9]                           # глобально, потім 3×3
    assert tracker.state == TrackerState.TRACKING
    assert math.isclose(steps[2].jump_px, 50.0)


def test_outlier_is_rejected_and_extrapolated():
    steps, _, _ = run([(1100, 2300), (1100, 2250), (1900, 2250), (1100, 2150)])
    bad = steps[2]
    assert bad.status == Status.INTERPOLATED and bad.event == Event.OUTLIER_REJECTED
    assert (bad.x_raw, bad.y_raw) == (1900, 2250) and bad.jump_px > 150
    assert np.allclose((bad.x_map, bad.y_map), (1100, 2200))     # P_{k-1} + (P_{k-1} - P_{k-2})
    # Наступний кадр порівнюється з останньою ПРИЙНЯТОЮ позицією (Δt = 2 с).
    assert steps[3].status == Status.TRACKING and math.isclose(steps[3].max_jump_px, 270.0)


def test_second_candidate_used_when_best_is_outlier():
    steps, _, _ = run([(1100, 2300), [((2000, 2300), 50.0), ((1100, 2260), 10.0)]])
    assert steps[1].status == Status.TRACKING
    assert (steps[1].x_map, steps[1].y_map) == (1100, 2260)


def test_lost_after_max_misses_then_global_reinit():
    steps, tracker, search = run([(1100, 2300), (1100, 2250), None, None, None,
                                  (1500, 1500), (1500, 1450)])
    statuses = [s.status for s in steps]
    assert statuses[2:5] == [Status.INTERPOLATED] * 3
    assert all(s.event == Event.NO_MATCH for s in steps[2:5])
    assert search.calls[5] == 384 and steps[5].search_mode == "global"
    assert statuses[5] == Status.INITIALIZED and statuses[6] == Status.TRACKING


def test_low_score_does_not_initialize():
    steps, tracker, _ = run([[((1100, 2300), 2.0)], (1100, 2300)])
    assert steps[0].status == Status.LOST and steps[0].event == Event.LOW_SCORE
    assert steps[0].x_map is None and steps[0].x_raw == 1100
    assert steps[1].status == Status.INITIALIZED


def test_trajectory_rows_have_no_gaps():
    steps, _, _ = run([None, (1100, 2300), (1100, 2250), None, (1100, 2150)])
    rows = pd.DataFrame([s.to_row() for s in steps])
    assert len(rows) == 5 and rows["frame_id"].is_unique
    assert list(rows["status"]) == ["LOST", "INITIALIZED", "TRACKING", "INTERPOLATED", "TRACKING"]
    assert rows.loc[1:, "x_map"].notna().all()


# ---------------------------------------------------------------------------
# Компенсація курсу
# ---------------------------------------------------------------------------
def test_wrap_and_extrapolate_yaw():
    assert wrap_angle(190) == -170 and wrap_angle(-180) == 180 and wrap_angle(360) == 0
    assert extrapolate_yaw([], 1.0) is None
    assert extrapolate_yaw([(0.0, 30.0)], 5.0) == 30.0
    assert math.isclose(extrapolate_yaw([(0.0, 10.0), (1.0, 30.0)], 2.0), 50.0)
    # Перехід через ±180°: 170 -> -170 — це +20°/с, а не -340°/с.
    assert math.isclose(extrapolate_yaw([(0.0, 170.0), (1.0, -170.0)], 2.0), -150.0)
    assert math.isclose(extrapolate_yaw([(0.0, 0.0), (1.0, 90.0)], 2.0, max_rate=45), 135.0)


class YawAwareSearch:
    """Знаходить кадр, лише якщо компенсація курсу близька до справжнього курсу."""

    def __init__(self, positions, true_yaws, tolerance=30.0):
        self.positions, self.true_yaws, self.tolerance = positions, true_yaws, tolerance
        self.frame, self.calls = 0, []

    def search(self, frame, tiles, rotation_deg=0.0):
        self.calls.append((self.frame, rotation_deg))
        true = self.true_yaws[self.frame]
        if abs(wrap_angle(rotation_deg - true)) > self.tolerance:
            return _Result([])
        return _Result([_Candidate(self.positions[self.frame], yaw=true)])


def run_yaw(positions, true_yaws, config):
    search = YawAwareSearch(positions, true_yaws)
    tracker = SequenceTracker(search, TILES, config=config)
    tracker._accept = _accept_without_footprint.__get__(tracker)
    steps = []
    for i in range(len(positions)):
        search.frame = i
        steps.append(tracker.step(None, f"frame_{i + 1:04d}.png", float(i)))
    return steps, search


def test_yaw_prediction_follows_turn():
    config = replace(CFG, yaw_retry_offsets_deg=())
    positions = [(1100, 2300 - 40 * i) for i in range(5)]
    steps, search = run_yaw(positions, [0, 20, 40, 60, 80], config)
    assert all(s.status in (Status.INITIALIZED, Status.TRACKING) for s in steps)
    # Прогноз за кутовою швидкістю: 0, 0 (один фікс), 40, 60, 80.
    assert [round(y) for _, y in search.calls] == [0, 0, 40, 60, 80]


def test_yaw_retry_recovers_sudden_turn():
    config = replace(CFG, yaw_retry_offsets_deg=(45.0, -45.0, 90.0, -90.0))
    positions = [(1100, 2300 - 40 * i) for i in range(3)]
    steps, search = run_yaw(positions, [0, 0, -90], config)
    assert steps[2].status == Status.TRACKING
    assert steps[2].attempts == 5 and math.isclose(steps[2].yaw_used, -90)
    assert [y for f, y in search.calls if f == 2] == [0, 45, -45, 90, -90]


def test_global_search_cycles_yaw_hypotheses():
    config = replace(CFG, yaw_retry_offsets_deg=(), global_yaw_cycle_deg=(0.0, 90.0, 180.0, -90.0))
    positions = [(1100, 2300)] * 4
    steps, search = run_yaw(positions, [180, 180, 180, 180], config)
    assert [s.status for s in steps[:3]] == [Status.LOST, Status.LOST, Status.INITIALIZED]
    assert [y for _, y in search.calls[:3]] == [0, 90, 180]
    assert steps[3].status == Status.TRACKING and search.calls[3][1] == 180


def test_no_yaw_compensation_when_disabled():
    config = replace(CFG, compensate_yaw=False, yaw_retry_offsets_deg=(45.0,))
    positions = [(1100, 2300 - 40 * i) for i in range(3)]
    steps, search = run_yaw(positions, [0, 10, 20], config)
    assert all(y == 0.0 for _, y in search.calls)
    assert all(s.attempts == 1 for s in steps)


def test_pipeline_rotation_keeps_original_homography():
    """Поворот кадру перед зіставленням не змінює H у оригінальних пікселях."""
    from src.geometry.homography import diagnose_homography, project_points
    from src.localization.coordinates import apply_homography
    from src.matching.base import ImageMatcher
    from src.matching.pipeline import PairMatchingPipeline, PipelineConfig

    theta = 70.0
    a = np.radians(theta)
    H_true = np.array([[0.27 * np.cos(a), -0.27 * np.sin(a), 300.0],
                       [0.27 * np.sin(a), 0.27 * np.cos(a), 50.0], [0, 0, 1.0]])

    class Oracle(ImageMatcher):
        name, size_multiple = "Oracle", 8

        def __init__(self):
            super().__init__(device="cpu")
            self.frame_prepared = None
            self.rng = np.random.default_rng(0)

        def _extract(self, image):
            return image.shape[:2], None

        def _match(self, shape0, shape1):
            h0, w0 = shape0
            work0 = self.rng.uniform([0, 0], [w0 - 1, h0 - 1], (400, 2))
            orig0 = self.frame_prepared.to_original(work0)
            work1 = apply_homography(orig0, H_true)          # тайл: масштаб 1
            # Точки поза кадром (тло після повороту) — «випадкові», вони мають відсіятися маскою.
            outside = ~self.frame_prepared.inside_valid(work0)
            work1[outside] = self.rng.uniform(0, 512, (int(outside.sum()), 2))
            return work0, work1, np.ones(400, np.float32)

    matcher = Oracle()
    pipeline = PairMatchingPipeline(matcher, PipelineConfig(frame_to_map_scale=0.27))
    frame = np.zeros((1080, 1920, 3), np.uint8)
    prepared, feats = pipeline.frame_features(frame, rotation_deg=theta)
    matcher.frame_prepared = prepared
    assert prepared.work_size[0] % 8 == 0 and prepared.work_size[1] % 8 == 0
    # Робоче зображення орієнтоване як карта: H_work = H_true · T^{-1} без повороту.
    H_work = H_true @ np.linalg.inv(prepared.transform)
    assert abs(diagnose_homography(H_work, prepared.work_size).rotation_deg) < 1e-6

    result = pipeline.run(None, np.zeros((512, 512, 3), np.uint8), frame_features=(prepared, feats))
    assert result.success, result.failure_reason
    assert result.homography.inlier_ratio == 1.0                # тло відфільтровано
    corners = np.array([[0, 0], [1919, 0], [1919, 1079], [0, 1079]], float)
    assert np.allclose(project_points(corners, result.homography.H),
                       project_points(corners, H_true), atol=1e-3)
    assert math.isclose(result.diagnostics.rotation_deg, theta, abs_tol=1e-4)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")


# ---------------------------------------------------------------------------
# smoothing.py — фільтр Калмана / RTS
# ---------------------------------------------------------------------------
from src.localization.smoothing import SmoothingConfig, smooth_trajectory, with_position


def synthetic_trajectory(n=60, noise=4.0, seed=0, status=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x_true, y_true = 1000 + 15 * t, 2000 + 0.1 * t ** 2
    st = np.array(status if status is not None else ["TRACKING"] * n, dtype=object)
    st[0] = "INITIALIZED"
    df = pd.DataFrame({"frame_id": [f"frame_{i + 1:04d}.png" for i in range(n)], "timestamp": t,
                       "status": st, "x_map": x_true + rng.normal(0, noise, n),
                       "y_map": y_true + rng.normal(0, noise, n)})
    df.loc[df["status"] == "LOST", ["x_map", "y_map"]] = np.nan
    return df, x_true, y_true


def test_kalman_reduces_noise():
    df, xt, yt = synthetic_trajectory()
    out = smooth_trajectory(df, SmoothingConfig(meas_std_px=4.0, accel_std_px_s2=2.0))
    err = {v: np.nanmean(np.hypot(with_position(out, v)["x_map"] - xt,
                                  with_position(out, v)["y_map"] - yt)[10:])
           for v in ("raw", "kf", "rts")}
    assert err["kf"] < err["raw"] and err["rts"] < err["kf"]


def test_kalman_gaps_and_segments():
    n = 30
    status = ["TRACKING"] * n
    status[10:13] = ["INTERPOLATED"] * 3          # екстраполяція трекера — не вимірювання
    status[20:23] = ["LOST"] * 3                  # немає положення
    status[23] = "INITIALIZED"                    # новий сегмент
    df, xt, yt = synthetic_trajectory(n, noise=0.0, status=status)
    df.loc[10:12, ["x_map", "y_map"]] = 0.0       # хибна «екстраполяція» не має впливати
    out = smooth_trajectory(df, SmoothingConfig())
    # Положення INTERPOLATED береться з прогнозу фільтра, а не зі «сирих» значень.
    assert np.all(np.abs(out.loc[10:12, "x_kf"] - xt[10:13]) < 5)
    # LOST лишається без положення, новий сегмент стартує точно з вимірювання.
    assert out.loc[20:22, ["x_kf", "x_rts"]].isna().all().all()
    assert np.isclose(out.loc[23, "x_kf"], df.loc[23, "x_map"])
    # Без шуму вимірювань (і з малою «маневреністю») фільтр не зсуває траєкторію.
    assert np.nanmax(np.abs(out["x_rts"] - xt)[~out["x_rts"].isna()]) < 1.0


# ---------------------------------------------------------------------------
# Вікно за прогнозом і уточнення після глобального пошуку
# ---------------------------------------------------------------------------
from src.localization.tracker import map_window

FOOT = np.array([[-259, -146], [259, -146], [259, 146], [-259, 146]], float)   # 518×292
EMPTY_MAP = np.zeros((4320, 6200, 3), np.uint8)


def test_map_window_bounds_and_multiple():
    x0, y0, x1, y1 = map_window((6200, 4320), FOOT + (1100, 2300), margin=128)
    assert (x0, y0) == (1100 - 259 - 128, 2300 - 146 - 128)
    assert (x1 - x0) % 8 == 0 and (y1 - y0) % 8 == 0
    assert x1 >= 1100 + 259 + 120 and y1 >= 2300 + 146 + 120      # запас (мінус кратність)
    # Біля краю карти — обрізано, а не за межами.
    x0, y0, x1, y1 = map_window((6200, 4320), FOOT + (100, 4300), margin=128)
    assert x0 == 0 and y1 <= 4320
    # Завелике або невизначене вікно -> None (пошук у тайлах).
    assert map_window((6200, 4320), FOOT + (1100, 2300), 128, max_side=600) is None
    assert map_window((6200, 4320), np.full((4, 2), np.nan), 128) is None


class WindowSearch(ScriptedSearch):
    """Як ScriptedSearch, плюс ``search_regions`` за окремим сценарієм ``(xy, inliers)``."""

    def __init__(self, script, region_script):
        super().__init__(script)
        self.region_script, self.regions = list(region_script), []

    def search_regions(self, frame, regions, rotation_deg=0.0):
        (region_id, image, origin), = regions
        self.regions.append((region_id, image.shape[:2], origin))
        item = self.region_script.pop(0)
        if item is None:
            return _Result([])
        xy, inliers = item if isinstance(item[0], tuple) else (item, 80)
        candidate = _Candidate(xy)
        candidate.tile_id, candidate.metrics = region_id, _Metrics(inliers=inliers)
        return _Result([candidate])


def window_tracker(search, config):
    tracker = SequenceTracker(search, TILES, config=config, map_rgb=EMPTY_MAP)
    # У заглушці немає гомографії: контур — прямокутник 518×292 навколо центру кандидата.
    tracker._candidate_footprint = lambda c: FOOT + c.center_map
    return tracker


def test_window_requires_map():
    try:
        SequenceTracker(ScriptedSearch([]), TILES, config=replace(CFG, predicted_window=True))
    except ValueError:
        return
    raise AssertionError("predicted_window without map_rgb must fail")


def test_predicted_window_replaces_tiles():
    search = WindowSearch([(1100, 2300)], [(1100, 2280), (1100, 2260)])
    tracker = window_tracker(search, replace(CFG, predicted_window=True))
    steps = [tracker.step(None, f"frame_{i + 1:04d}.png", float(i)) for i in range(3)]
    # Кадр 1 — глобально; кадри 2–3 — з вікна, тайли 3×3 не потрібні.
    assert [s.search_mode for s in steps] == ["global", "window", "window"]
    assert [s.status for s in steps] == [Status.INITIALIZED, Status.TRACKING, Status.TRACKING]
    assert search.calls == [384] and all(s.tiles_checked == 1 for s in steps[1:])
    # Вікно кадру 3 центроване на прогнозі (1100, 2260) і вміщує контур із запасом.
    _, (h, w), (x0, y0) = search.regions[1]
    assert x0 <= 1100 - 259 - 120 and x0 + w >= 1100 + 259 + 120
    assert y0 <= 2260 - 146 - 120 and y0 + h >= 2260 + 146 + 120


def test_window_falls_back_to_tiles():
    # Вікно: нічого (кадр 2), потім кінематичний викид (кадр 3) -> обидва рази тайли 3×3.
    search = WindowSearch([(1100, 2300), (1100, 2280), (1100, 2260)], [None, (1900, 2260)])
    tracker = window_tracker(search, replace(CFG, predicted_window=True))
    steps = [tracker.step(None, f"frame_{i + 1:04d}.png", float(i)) for i in range(3)]
    assert [s.search_mode for s in steps] == ["global", "local", "local"]
    assert [s.status for s in steps[1:]] == [Status.TRACKING, Status.TRACKING]
    assert search.calls == [384, 9, 9] and steps[2].attempts == 2


def test_refine_after_global():
    config = replace(CFG, refine_after_global=True, refine_max_shift_px=30)
    # Глобальний переможець (80 inliers) -> уточнення на 12 px зі 120 inliers: приймається.
    search = WindowSearch([(1100, 2300)], [((1105, 2311), 120)])
    tracker = window_tracker(search, config)
    step = tracker.step(None, "frame_0001.png", 0.0)
    assert step.refined and (step.x_map, step.y_map) == (1105, 2311) and step.attempts == 2
    # Уточнення, що зсуває центр далі за поріг або має менше inliers, — відкидається.
    for item in (((1200, 2300), 120), ((1101, 2301), 50)):
        search.script, search.region_script = [(1100, 2300)], [item]
        tracker.reset()
        step = tracker.step(None, "frame_0002.png", 1.0)
        assert not step.refined and (step.x_map, step.y_map) == (1100, 2300)


# ---------------------------------------------------------------------------
# Курс із компаса
# ---------------------------------------------------------------------------
from src.localization.compass import CompassConfig, CompassSensor


def test_compass_sensor_offset_bias_noise():
    headings = {f"frame_{i:04d}.png": 170.0 for i in range(1, 201)}
    clean = CompassSensor(headings, CompassConfig(offset_deg=2.3))
    assert math.isclose(clean("frame_0001.png"), 172.3)
    assert clean("missing.png") is None
    # Перехід через ±180°
    assert math.isclose(CompassSensor(headings, CompassConfig(offset_deg=15))("frame_0001.png"), -175.0)
    north = {f: 30.0 for f in headings}
    noisy = CompassSensor(north, CompassConfig(bias_deg=10, noise_deg=5, seed=1))
    values = np.array([noisy(f) for f in north])
    assert abs(np.mean(values) - 40.0) < 1.5 and 3.5 < np.std(values) < 6.5
    again = CompassSensor(north, CompassConfig(bias_deg=10, noise_deg=5, seed=1))
    assert noisy("frame_0007.png") == again("frame_0007.png")   # відтворюваність


def test_compass_sets_global_yaw():
    # Справжній курс 23°: без компаса перебір 0/90/180/-90 не дає курсу в межах 15°,
    # з компасом глобальний пошук одразу виконується з правильним курсом.
    positions = [(1100, 2300 - 20 * i) for i in range(4)]
    true = [23.0] * 4
    config = replace(CFG, yaw_retry_offsets_deg=())
    search = YawAwareSearch(positions, true, tolerance=15.0)
    tracker = SequenceTracker(search, TILES, config=config,
                              compass=lambda fid, t: 25.0)          # компас з похибкою 2°
    tracker._accept = _accept_without_footprint.__get__(tracker)
    steps = []
    for i in range(4):
        search.frame = i
        steps.append(tracker.step(None, f"frame_{i + 1:04d}.png", float(i)))
    assert steps[0].status == Status.INITIALIZED and steps[0].yaw_used == 25.0
    assert steps[0].compass_yaw == 25.0
    assert all(s.status == Status.TRACKING for s in steps[1:])

    search = YawAwareSearch(positions, true, tolerance=15.0)
    tracker = SequenceTracker(search, TILES, config=config)
    tracker._accept = _accept_without_footprint.__get__(tracker)
    search.frame = 0
    assert tracker.step(None, "frame_0001.png", 0.0).status == Status.LOST


def test_compass_hypothesis_in_tracking():
    # Прогноз курсу з гомографії застарів (різкий розворот на 90°): компас додає
    # правильну гіпотезу одразу після прогнозованої, до перебору ±45/±90.
    config = replace(CFG, yaw_retry_offsets_deg=(45.0, -45.0))
    positions = [(1100, 2300), (1100, 2280), (1100, 2260)]
    search = YawAwareSearch(positions, [0.0, 0.0, 90.0], tolerance=10.0)
    tracker = SequenceTracker(search, TILES, config=config,
                              compass=lambda fid, t: 90.0 if fid == "frame_0003.png" else 0.0)
    tracker._accept = _accept_without_footprint.__get__(tracker)
    steps = []
    for i in range(3):
        search.frame = i
        steps.append(tracker.step(None, f"frame_{i + 1:04d}.png", float(i)))
    assert steps[2].status == Status.TRACKING and steps[2].yaw_used == 90.0
    assert steps[2].attempts == 2                                  # прогноз 0°, потім компас 90°
