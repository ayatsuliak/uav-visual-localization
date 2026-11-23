"""Автотести Етапу 5: еталон за ручними точками, інтерполяція, метрики похибки,
спотворення зображень, ознаки місцевості.

Запуск:  python -m pytest tests -q      або      python tests/test_stage5_evaluation.py
"""
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.ground_truth import (KeyframeAnnotation, estimate_keyframe,
                                         interpolate_ground_truth, keyframes_table,
                                         load_annotations, load_ground_truth, save_annotations)
from src.evaluation.metrics import error_statistics, localization_error, trajectory_errors
from src.evaluation.terrain import terrain_stats
from src.localization.coordinates import apply_homography
from src.preprocessing.images import (add_gaussian_noise, adjust_brightness, adjust_contrast,
                                      motion_blur)


def similarity(scale, angle_deg, tx, ty):
    a = np.radians(angle_deg)
    return np.array([[scale * np.cos(a), -scale * np.sin(a), tx],
                     [scale * np.sin(a), scale * np.cos(a), ty], [0, 0, 1.0]])


def annotation(frame_id="frame_0001.png", t=0.0, H=None, n=6, noise=0.0, seed=0):
    H = similarity(0.27, 30, 1000, 2000) if H is None else H
    rng = np.random.default_rng(seed)
    frame_pts = rng.uniform([50, 50], [1870, 1030], (n, 2))
    map_pts = apply_homography(frame_pts, H) + rng.normal(0, noise, (n, 2))
    return KeyframeAnnotation(frame_id, t, (1920, 1080), frame_pts.tolist(), map_pts.tolist())


# ---------------------------------------------------------------------------
# Еталон
# ---------------------------------------------------------------------------
def test_keyframe_centre_from_exact_points():
    H = similarity(0.27, 30, 1000, 2000)
    est = estimate_keyframe(annotation(H=H, n=4))
    expected = apply_homography([(959.5, 539.5)], H)[0]
    assert np.allclose((est.x_true, est.y_true), expected, atol=1e-3)
    assert est.n_points == 4 and est.residual_px < 1e-3


def test_keyframe_with_click_noise_has_small_error_and_residual():
    H = similarity(0.27, -80, 1500, 1200)
    est = estimate_keyframe(annotation(H=H, n=8, noise=1.0, seed=3))
    expected = apply_homography([(959.5, 539.5)], H)[0]
    assert math.hypot(est.x_true - expected[0], est.y_true - expected[1]) < 2.0
    assert 0.3 < est.residual_px < 2.0


def test_reliability_flags():
    H = similarity(0.27, 30, 1000, 2000)
    # 4 пари: точна підгонка, перевірка неможлива.
    four = estimate_keyframe(annotation(H=H, n=4))
    assert not four.reliable and any("no check" in w for w in four.warnings())
    # 8 пар по всьому кадру з реалістичним шумом кліків — надійно.
    good = estimate_keyframe(annotation(H=H, n=8, noise=0.5, seed=2))
    assert good.reliable, good.warnings()
    assert good.center_uncertainty_px < 5.0 and good.max_residual_px < 3.0
    # Скупчені точки в куті кадру: центр екстраполюється -> велика невизначеність.
    rng = np.random.default_rng(4)
    corner = rng.uniform([50, 50], [350, 250], (6, 2))
    clustered = KeyframeAnnotation("f", 0.0, (1920, 1080), corner.tolist(),
                                   apply_homography(corner, H).tolist())
    est = estimate_keyframe(clustered)
    assert not est.reliable and est.center_uncertainty_px > 5.0
    # Одна хибна пара -> велика залишкова похибка саме цієї пари.
    a = annotation(H=H, n=7, noise=0.3, seed=5)
    a.map_points[3] = [a.map_points[3][0] + 25, a.map_points[3][1]]
    bad = estimate_keyframe(a)
    assert not bad.reliable and bad.max_residual_px > 3.0


def test_centre_outside_points_is_a_warning_only():
    H = similarity(0.27, 30, 1000, 2000)
    rng = np.random.default_rng(7)
    # Точки лише в лівій третині кадру (як на кадрі 80): центр екстрапольовано.
    left = rng.uniform([50, 50], [600, 1030], (7, 2))
    est = estimate_keyframe(KeyframeAnnotation("f", 0.0, (1920, 1080), left.tolist(),
                                               apply_homography(left, H).tolist()))
    assert not est.center_in_hull
    assert any("extrapolated" in w for w in est.warnings())
    # Точки навколо центру — попередження немає.
    good = estimate_keyframe(annotation(H=H, n=8, noise=0.5, seed=2))
    assert good.center_in_hull and not any("extrapolated" in w for w in good.warnings())


def test_keyframe_rejects_too_few_or_mismatched_points():
    a = annotation(n=3)
    for bad in (a, KeyframeAnnotation("f", 0, (1920, 1080), a.frame_points, a.map_points[:2])):
        try:
            estimate_keyframe(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_annotations_roundtrip_and_table():
    anns = {"frame_0010.png": annotation("frame_0010.png", 9.0),
            "frame_0001.png": annotation("frame_0001.png", 0.0)}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "points.json"
        save_annotations(path, anns)
        loaded = load_annotations(path)
    assert set(loaded) == set(anns) and loaded["frame_0010.png"].frame_size == (1920, 1080)
    table = keyframes_table(loaded)
    assert list(table["frame_id"]) == ["frame_0001.png", "frame_0010.png"]    # за часом
    assert table["is_keyframe"].all() and (table["source"] == "manual_homography").all()


def test_interpolation_between_keyframes_only():
    key = pd.DataFrame({"frame_id": ["frame_0002.png", "frame_0004.png"], "timestamp": [1.0, 3.0],
                        "x_true": [100.0, 200.0], "y_true": [50.0, 10.0], "is_keyframe": True,
                        "n_points": 5, "residual_px": 0.5, "source": "manual_homography"})
    frames = pd.DataFrame({"frame_id": [f"frame_{i:04d}.png" for i in range(1, 6)],
                           "timestamp_sec": [0.0, 1.0, 2.0, 3.0, 4.0]})
    gt = interpolate_ground_truth(key, frames)
    assert list(gt["frame_id"]) == ["frame_0002.png", "frame_0003.png", "frame_0004.png"]
    mid = gt.iloc[1]
    assert (mid["x_true"], mid["y_true"]) == (150.0, 30.0) and not mid["is_keyframe"]
    assert gt.iloc[0]["is_keyframe"] and gt.iloc[2]["is_keyframe"]


def test_load_ground_truth_parses_flags():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gt.csv"
        path.write_text("frame_id,timestamp,x_true,y_true,is_keyframe\n"
                        "frame_0001.png,0.0,1110.2,2330.1,True\n"
                        "frame_0002.png,1.0,1111.0,2320.0,False\n", encoding="utf-8")
        gt = load_ground_truth(path)
    assert gt["is_keyframe"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def test_localization_error():
    assert localization_error((3, 4), (0, 0)) == 5.0


def test_trajectory_errors_and_statistics():
    gt = pd.DataFrame({"frame_id": ["a", "b", "c", "d", "e"],
                       "x_true": [0.0] * 5, "y_true": [0.0] * 5})
    traj = pd.DataFrame({"frame_id": ["a", "b", "c", "d", "x"],
                         "x_map": [3.0, 6.0, 30.0, np.nan, 1.0], "y_map": [4.0, 8.0, 40.0, np.nan, 1.0],
                         "status": ["TRACKING", "TRACKING", "TRACKING", "LOST", "TRACKING"]})
    merged = trajectory_errors(traj, gt)
    assert len(merged) == 5                                 # лише кадри з еталоном
    assert merged["error"].tolist()[:3] == [5.0, 10.0, 50.0]
    assert np.isnan(merged["error"].iloc[3]) and np.isnan(merged["error"].iloc[4])
    s = error_statistics(merged["error"])
    assert s["n_frames"] == 5 and s["n_predicted"] == 3 and math.isclose(s["coverage"], 0.6)
    assert math.isclose(s["mean"], 65 / 3) and s["median"] == 10.0 and s["max"] == 50.0
    assert math.isclose(s["rmse"], math.sqrt((25 + 100 + 2500) / 3))
    # success@K: частка від УСІХ кадрів з еталоном (відсутні оцінки — неуспіх), поріг включно.
    assert s["success@10"] == 0.4 and s["success@25"] == 0.4 and s["success@50"] == 0.6


def test_error_statistics_empty():
    s = error_statistics([np.nan, np.nan])
    assert s["n_predicted"] == 0 and s["mean"] is None and s["success@10"] == 0.0


# ---------------------------------------------------------------------------
# Спотворення
# ---------------------------------------------------------------------------
def test_brightness_and_contrast():
    img = np.full((4, 4, 3), 200, np.uint8)
    img[0, 0] = 100
    assert adjust_brightness(img, 0.5)[1, 1, 0] == 100
    out = adjust_contrast(img, 0.5)
    mean = img[..., 0].astype(float).mean()
    assert abs(out[0, 0, 0] - (0.5 * (100 - mean) + mean)) <= 0.5
    assert abs(out[..., 0].astype(float).mean() - mean) < 1.0          # середнє зберігається
    assert out.dtype == np.uint8


def test_noise_is_reproducible_and_has_given_sigma():
    img = np.full((200, 200, 3), 128, np.uint8)
    a, b = add_gaussian_noise(img, 10, seed=1), add_gaussian_noise(img, 10, seed=1)
    assert np.array_equal(a, b)
    assert 9.0 < (a.astype(float) - 128).std() < 11.0


def test_motion_blur_spreads_along_direction():
    img = np.zeros((21, 21, 3), np.uint8)
    img[10, 10] = 255
    out = motion_blur(img, 9, angle_deg=0.0)[..., 0].astype(float)
    assert out[10, 6] > 0 and out[10, 14] > 0 and out[6, 10] == 0     # горизонтальне
    assert abs(out.sum() - 255) < 10                                  # енергія зберігається
    assert np.array_equal(motion_blur(img, 1), img)


# ---------------------------------------------------------------------------
# Місцевість
# ---------------------------------------------------------------------------
def test_terrain_vegetation_fraction_and_texture():
    m = np.zeros((100, 200, 3), np.uint8)
    m[:, :100] = (60, 140, 50)            # зелень
    m[:, 100:] = (120, 120, 120)          # асфальт
    m[::4, 100:] = (200, 200, 200)        # смуги -> текстура
    left = terrain_stats(m, [(10, 10), (90, 10), (90, 90), (10, 90)])
    right = terrain_stats(m, [(110, 10), (190, 10), (190, 90), (110, 90)])
    half = terrain_stats(m, [(60, 10), (140, 10), (140, 90), (60, 90)])
    assert left["vegetation_fraction"] == 1.0 and right["vegetation_fraction"] == 0.0
    assert 0.4 < half["vegetation_fraction"] < 0.6
    assert right["texture"] > left["texture"]
    assert np.isnan(terrain_stats(m, [(500, 500), (600, 500), (600, 600), (500, 600)])["texture"])


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")


# ---------------------------------------------------------------------------
# geo.py — прив'язка карти (лише для еталона з журналу польоту)
# ---------------------------------------------------------------------------
from src.evaluation.geo import WorldFile, utm_forward, utm_zone_from_epsg


def test_utm_reference_points():
    # Осьовий меридіан зони 31 (3° сх. д.): на екваторі x = 500 000, y = 0.
    x, y = utm_forward(3.0, 0.0, 31)
    assert np.isclose(x, 500000.0) and np.isclose(y, 0.0, atol=1e-6)
    # На 45° пн. ш. на осьовому меридіані y = k0 · M(45°) = 0.9996 · 4 984 944.38 м.
    x, y = utm_forward(3.0, 45.0, 31)
    assert np.isclose(x, 500000.0) and abs(y - 0.9996 * 4984944.378) < 0.05
    # Симетрія відносно осьового меридіана; південна півкуля — зсув 10 000 км.
    xe, ye = utm_forward(4.0, 50.0, 31)
    xw, yw = utm_forward(2.0, 50.0, 31)
    assert np.isclose(xe - 500000.0, 500000.0 - xw) and np.isclose(ye, yw)
    _, ys = utm_forward(3.0, -10.0, 31, north=False)
    _, yn = utm_forward(3.0, 10.0, 31)
    assert np.isclose(ys, 10_000_000.0 - yn)
    assert utm_zone_from_epsg("EPSG:32634") == (34, True)
    assert utm_zone_from_epsg(32735) == (35, False)


def test_world_file_pixel_centres():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "map.pgw"
        path.write_text("0.3\n0\n0\n-0.3\n715158.9\n5503703.8\n")
        w = WorldFile.read(path)
    assert np.isclose(w.gsd, 0.3)
    px, py = w.to_pixel(715158.9, 5503703.8)                 # центр лівого верхнього пікселя
    assert np.isclose(px, 0) and np.isclose(py, 0)
    px, py = w.to_pixel(715158.9 + 0.3 * 10, 5503703.8 - 0.3 * 20)
    assert np.isclose(px, 10) and np.isclose(py, 20)
