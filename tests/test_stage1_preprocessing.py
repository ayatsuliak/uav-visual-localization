"""Автотести Етапу 1 на синтетичних даних (не потребують реальної карти/відео).

Запуск:  python -m pytest tests -q      або      python tests/test_stage1_preprocessing.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.frames import compute_frame_step, extract_frames
from src.preprocessing.images import read_image_rgb, rgba_to_rgb, write_image_rgb
from src.preprocessing.tiles import (compute_axis_origins, compute_stride, cut_map_tiles,
                                     expected_tile_count, generate_tile_grid)
from src.preprocessing.validation import validate_frames, validate_tiles


def _write_rgba_png(path, rgba):
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def test_stride_and_origins_match_spec():
    assert compute_stride(512, 0.5) == 256
    xs = compute_axis_origins(6200, 512, 256)
    ys = compute_axis_origins(4320, 512, 256)
    assert xs[:3] == [0, 256, 512] and xs[-2:] == [5632, 5688]
    assert ys[-2:] == [3584, 3808]
    assert len(xs) == 24 and len(ys) == 16
    assert expected_tile_count(6200, 4320, 512, 256) == 384
    assert len(xs) == len(set(xs))


def test_exact_fit_has_no_extra_edge_tile():
    assert compute_axis_origins(1024, 512, 256) == [0, 256, 512]
    assert compute_axis_origins(512, 512, 256) == [0]


def test_map_smaller_than_tile_raises():
    try:
        compute_axis_origins(300, 512, 256)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_rgba_blending():
    rgba = np.zeros((1, 3, 4), np.uint8)
    rgba[0, 0] = (10, 20, 30, 255)   # непрозорий -> без змін
    rgba[0, 1] = (0, 0, 0, 0)        # прозорий -> тло
    rgba[0, 2] = (0, 0, 0, 128)      # напівпрозорий -> змішування
    out = rgba_to_rgb(rgba, (255, 255, 255))
    assert out[0, 0].tolist() == [10, 20, 30]
    assert out[0, 1].tolist() == [255, 255, 255]
    assert out[0, 2].tolist() == [127, 127, 127]


def test_color_roundtrip_is_rgb():
    with tempfile.TemporaryDirectory() as tmp:
        img = np.zeros((4, 4, 3), np.uint8)
        img[..., 0] = 200                          # червоний канал у RGB
        write_image_rgb(Path(tmp) / "red.png", img)
        bgr = cv2.imread(str(Path(tmp) / "red.png"))
        assert bgr[0, 0].tolist() == [0, 0, 200]   # OpenCV бачить червоний у BGR
        assert read_image_rgb(Path(tmp) / "red.png")[0, 0].tolist() == [200, 0, 0]


def _make_map(tmp, w=1300, h=900, alpha=True):
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    if alpha:
        rgba[:50, :50, 3] = 0            # прозорий кут
    path = Path(tmp) / "map.png"
    _write_rgba_png(path, rgba)
    return path


def test_cut_tiles_pixel_identity_and_coverage():
    with tempfile.TemporaryDirectory() as tmp:
        map_path = _make_map(tmp)
        tiles_dir, csv, info = Path(tmp) / "tiles", Path(tmp) / "tiles.csv", Path(tmp) / "info.json"
        (tiles_dir).mkdir()
        (tiles_dir / "tile_9999.png").write_bytes(b"stale")  # має бути видалений
        df = cut_map_tiles(map_path, tiles_dir, csv, 512, 0.5, run_info_path=info, verbose=False)
        assert list(df.columns) == ["tile_id", "x_origin", "y_origin", "width", "height"]
        assert df["tile_id"].iloc[0] == "tile_0001.png"
        assert len(df) == expected_tile_count(1300, 900, 512, 256)
        assert df["x_origin"].max() == 1300 - 512 and df["y_origin"].max() == 900 - 512
        errors, stats = validate_tiles(map_path, tiles_dir, csv, run_info_path=info)
        assert errors == [], errors
        assert stats["min_coverage"] >= 1
        tile = cv2.imread(str(tiles_dir / "tile_0001.png"), cv2.IMREAD_UNCHANGED)
        assert tile.shape == (512, 512, 3)
        assert read_image_rgb(tiles_dir / "tile_0001.png")[0, 0].tolist() == [255, 255, 255]


def test_validation_detects_wrong_coordinates():
    with tempfile.TemporaryDirectory() as tmp:
        map_path = _make_map(tmp, alpha=False)
        tiles_dir, csv = Path(tmp) / "tiles", Path(tmp) / "tiles.csv"
        cut_map_tiles(map_path, tiles_dir, csv, 512, 0.5, verbose=False)
        df = pd.read_csv(csv)
        df.loc[1, "x_origin"] += 1                # імітація помилки в метаданих
        df.to_csv(csv, index=False)
        errors, _ = validate_tiles(map_path, tiles_dir, csv, tile_size=512, overlap=0.5)
        assert any("pixels differ" in e for e in errors), errors


def test_roi_coordinates_are_global():
    tiles = generate_tile_grid(2000, 2000, 256, 0.5, roi=(912, 1409, 768, 512))
    assert tiles[0].x_origin == 912 and tiles[0].y_origin == 1409
    assert max(t.x_origin for t in tiles) + 256 == 912 + 768
    assert max(t.y_origin for t in tiles) + 256 == 1409 + 512


def test_frame_step():
    assert compute_frame_step(30.0, 1.0) == 30
    assert compute_frame_step(29.97, 1.0) == 30
    assert compute_frame_step(30.0, 2.0) == 15
    assert compute_frame_step(30.0, None, every_n=7) == 7


def _make_video(path, n_frames=95, fps=30.0, size=(320, 180)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        for b in range(8):                    # 8-бітний код номера кадру
            if (i >> b) & 1:
                frame[10:40, 10 + b * 38:40 + b * 38] = 255
        writer.write(frame)
    writer.release()


def _decode_index(rgb):
    return sum(1 << b for b in range(8) if rgb[25, 25 + b * 38].mean() > 127)


def test_extract_frames_indices_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "v.avi"
        _make_video(video)
        out, csv, info = Path(tmp) / "frames", Path(tmp) / "frames.csv", Path(tmp) / "fi.json"
        df = extract_frames(video, out, csv, target_fps=1.0, run_info_path=info, verbose=False)
        assert df["source_frame_idx"].tolist() == [0, 30, 60, 90]
        assert df["timestamp_sec"].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert df["frame_id"].tolist() == [f"frame_{i:04d}.png" for i in range(1, 5)]
        for r in df.itertuples():
            assert _decode_index(read_image_rgb(out / r.frame_id)) == r.source_frame_idx
        errors, _ = validate_frames(out, csv, expected_size=(320, 180), run_info_path=info)
        assert errors == [], errors

        df2 = extract_frames(video, out, csv, target_fps=1.0, limit=2, verbose=False)
        assert len(df2) == 2 and len(list(out.glob("frame_*.png"))) == 2


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
