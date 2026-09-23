"""Перевірка цілісності результатів Етапу 1 (sanity check).

Кожна функція повертає список рядків-помилок; порожній список означає успіх.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.frames import FRAMES_CSV_COLUMNS
from src.preprocessing.images import read_image_raw, read_image_rgb
from src.preprocessing.tiles import (TILES_CSV_COLUMNS, compute_stride,
                                     expected_tile_count, load_map_rgb, file_sha256)


def _check_csv_schema(df: pd.DataFrame, columns: list[str], name: str) -> list[str]:
    errors = []
    if list(df.columns) != columns:
        errors.append(f"{name}: columns {list(df.columns)} != expected {columns}")
        return errors
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        errors.append(f"{name}: NaN values in columns {bad}")
    if df.empty:
        errors.append(f"{name}: no rows")
    return errors


def validate_tiles(map_path: str | Path, tiles_dir: str | Path, tiles_csv: str | Path,
                   tile_size: int | None = None, overlap: float | None = None,
                   run_info_path: str | Path | None = None,
                   background_rgb: tuple[int, int, int] = (255, 255, 255),
                   check_pixels: bool = True, sample: int | None = None,
                   seed: int = 0) -> tuple[list[str], dict]:
    """Перевіряє реєстр тайлів, файли тайлів та піксельну ідентичність з картою."""
    errors: list[str] = []
    tiles_dir = Path(tiles_dir)
    df = pd.read_csv(tiles_csv)
    errors += _check_csv_schema(df, TILES_CSV_COLUMNS, "tiles.csv")
    if errors:
        return errors, {}

    run_info = {}
    if run_info_path is not None and Path(run_info_path).is_file():
        run_info = json.loads(Path(run_info_path).read_text(encoding="utf-8"))
        tile_size = tile_size or run_info.get("tile_size")
        overlap = overlap if overlap is not None else run_info.get("overlap")
        if run_info.get("map_sha256") and run_info["map_sha256"] != file_sha256(map_path):
            errors.append("Map file changed since tiling (sha256 mismatch) - rerun prepare_map.py")

    map_rgb, _ = load_map_rgb(map_path, background_rgb)
    map_h, map_w = map_rgb.shape[:2]

    # --- схема та цілі значення ---
    for col in ["x_origin", "y_origin", "width", "height"]:
        if not pd.api.types.is_integer_dtype(df[col]):
            errors.append(f"tiles.csv: column {col} is not integer ({df[col].dtype})")
    if df["tile_id"].duplicated().any():
        errors.append(f"tiles.csv: duplicated tile_id: "
                      f"{df.loc[df['tile_id'].duplicated(), 'tile_id'].tolist()[:5]}")
    if df[["x_origin", "y_origin"]].duplicated().any():
        errors.append("tiles.csv: duplicated (x_origin, y_origin) pairs")
    if tile_size is not None:
        bad = df[(df["width"] != tile_size) | (df["height"] != tile_size)]
        if len(bad):
            errors.append(f"tiles.csv: {len(bad)} tiles have size != {tile_size}")

    # --- межі карти ---
    oob = df[(df["x_origin"] < 0) | (df["y_origin"] < 0) |
             (df["x_origin"] + df["width"] > map_w) | (df["y_origin"] + df["height"] > map_h)]
    if len(oob):
        errors.append(f"tiles.csv: {len(oob)} tiles outside the map {map_w}x{map_h}")

    # --- повне покриття (для тайлування без ROI) ---
    roi = run_info.get("roi_xywh")
    cover_x0, cover_y0, cover_w, cover_h = roi if roi else (0, 0, map_w, map_h)
    coverage = np.zeros((cover_h, cover_w), dtype=np.uint16)
    for r in df.itertuples(index=False):
        x0, y0 = r.x_origin - cover_x0, r.y_origin - cover_y0
        coverage[max(0, y0):max(0, y0 + r.height), max(0, x0):max(0, x0 + r.width)] += 1
    uncovered = int((coverage == 0).sum())
    if uncovered:
        errors.append(f"Coverage: {uncovered} pixels of the (ROI) map are not covered by any tile")

    if tile_size is not None and overlap is not None:
        expected = expected_tile_count(cover_w, cover_h, tile_size, compute_stride(tile_size, overlap))
        if expected != len(df):
            errors.append(f"Tile count {len(df)} != expected {expected}")

    # --- файли на диску ---
    on_disk = {p.name for p in tiles_dir.glob("tile_*.png")}
    listed = set(df["tile_id"])
    missing = sorted(listed - on_disk)
    extra = sorted(on_disk - listed)
    if missing:
        errors.append(f"{len(missing)} tiles listed in CSV are missing on disk, e.g. {missing[:3]}")
    if extra:
        errors.append(f"{len(extra)} stale tile files not listed in CSV, e.g. {extra[:3]}")

    # --- піксельна ідентичність ---
    checked = 0
    if check_pixels:
        rows = df[df["tile_id"].isin(on_disk)]
        if sample is not None and sample < len(rows):
            rows = rows.sample(n=sample, random_state=seed)
        for r in rows.itertuples(index=False):
            raw = read_image_raw(tiles_dir / r.tile_id)
            if raw.ndim != 3 or raw.shape[2] != 3:
                errors.append(f"{r.tile_id}: expected 3 channels (RGB, no alpha), got {raw.shape}")
                continue
            tile_rgb = read_image_rgb(tiles_dir / r.tile_id)
            if tile_rgb.shape != (r.height, r.width, 3):
                errors.append(f"{r.tile_id}: shape {tile_rgb.shape} != ({r.height}, {r.width}, 3)")
                continue
            ref = map_rgb[r.y_origin:r.y_origin + r.height, r.x_origin:r.x_origin + r.width]
            if not np.array_equal(tile_rgb, ref):
                diff = int(np.abs(tile_rgb.astype(int) - ref.astype(int)).max())
                errors.append(f"{r.tile_id}: pixels differ from map crop "
                              f"at ({r.x_origin}, {r.y_origin}), max abs diff {diff}")
            checked += 1

    stats = {"tiles": len(df), "map_size": (map_w, map_h), "pixel_checked": checked,
             "min_coverage": int(coverage.min()), "max_coverage": int(coverage.max()),
             "cols": df["x_origin"].nunique(), "rows": df["y_origin"].nunique()}
    return errors, stats


def validate_frames(frames_dir: str | Path, frames_csv: str | Path,
                    expected_size: tuple[int, int] | None = None,
                    run_info_path: str | Path | None = None,
                    check_images: bool = True) -> tuple[list[str], dict]:
    """Перевіряє ``frames.csv`` та файли кадрів (розмір, канали, монотонність)."""
    errors: list[str] = []
    frames_dir = Path(frames_dir)
    df = pd.read_csv(frames_csv)
    errors += _check_csv_schema(df, FRAMES_CSV_COLUMNS, "frames.csv")
    if errors:
        return errors, {}

    if df["frame_id"].duplicated().any():
        errors.append("frames.csv: duplicated frame_id")
    if not df["source_frame_idx"].is_monotonic_increasing or df["source_frame_idx"].duplicated().any():
        errors.append("frames.csv: source_frame_idx is not strictly increasing")
    if not df["timestamp_sec"].is_monotonic_increasing:
        errors.append("frames.csv: timestamp_sec is not increasing")
    expected_names = [f"frame_{i:04d}.png" for i in range(1, len(df) + 1)]
    if df["frame_id"].tolist() != expected_names:
        errors.append("frames.csv: frame_id is not a contiguous sequence frame_0001.png, ...")

    run_info = {}
    if run_info_path is not None and Path(run_info_path).is_file():
        run_info = json.loads(Path(run_info_path).read_text(encoding="utf-8"))
        fps, step = run_info.get("video_fps"), run_info.get("every_n")
        if fps and step:
            idx = df["source_frame_idx"].to_numpy()
            if len(idx) > 1 and not np.all(np.diff(idx) == step):
                errors.append(f"frames.csv: source_frame_idx step is not constant {step}")
            ts_expected = idx / fps
            if not np.allclose(df["timestamp_sec"].to_numpy(), ts_expected, atol=1e-4):
                errors.append("frames.csv: timestamp_sec != source_frame_idx / fps")

    if expected_size is not None:
        w, h = expected_size
        bad = df[(df["width"] != w) | (df["height"] != h)]
        if len(bad):
            errors.append(f"frames.csv: {len(bad)} frames have size != {w}x{h}")

    on_disk = {p.name for p in frames_dir.glob("frame_*.png")}
    listed = set(df["frame_id"])
    if listed - on_disk:
        errors.append(f"{len(listed - on_disk)} frames listed in CSV are missing on disk")
    if on_disk - listed:
        errors.append(f"{len(on_disk - listed)} stale frame files not listed in CSV")

    checked = 0
    if check_images:
        for r in df[df["frame_id"].isin(on_disk)].itertuples(index=False):
            raw = read_image_raw(frames_dir / r.frame_id)
            if raw.ndim != 3 or raw.shape[2] != 3:
                errors.append(f"{r.frame_id}: expected 3 channels, got {raw.shape}")
            elif raw.shape[:2] != (r.height, r.width):
                errors.append(f"{r.frame_id}: size {raw.shape[1]}x{raw.shape[0]} "
                              f"!= CSV {r.width}x{r.height}")
            checked += 1

    duration = float(df["timestamp_sec"].iloc[-1] - df["timestamp_sec"].iloc[0]) if len(df) else math.nan
    stats = {"frames": len(df), "images_checked": checked, "span_sec": duration}
    return errors, stats
