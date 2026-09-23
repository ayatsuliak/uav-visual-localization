"""Тайлування ортофотоплану з перекриттям та формування реєстру ``tiles.csv``.

Координати тайлів завжди записуються у пікселях **вихідної повної карти**
(навіть якщо тайлування виконується в межах ROI), у конвенції ``(x, y)``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import portable_path
from src.preprocessing.images import read_image_raw, decoded_to_rgb, write_image_rgb

TILES_CSV_COLUMNS = ["tile_id", "x_origin", "y_origin", "width", "height"]


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    x_origin: int
    y_origin: int
    width: int
    height: int


def compute_stride(tile_size: int, overlap: float) -> int:
    """``stride = floor(S_tile * (1 - overlap))``, щонайменше 1 px."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    return max(1, int(math.floor(tile_size * (1.0 - overlap))))


def compute_axis_origins(length: int, tile_size: int, stride: int) -> list[int]:
    """Початки тайлів уздовж однієї осі.

    Регулярна сітка ``0, stride, 2*stride, ...`` доповнюється крайовим
    тайлом ``length - tile_size`` (зсув назад від краю), якщо регулярна сітка
    не доходить до краю. Результат відсортований і без дублікатів, тому
    кожен піксель осі покритий щонайменше одним тайлом.
    """
    if length < tile_size:
        raise ValueError(f"Axis length {length} is smaller than tile size {tile_size}")
    last = length - tile_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)            # x_origin = min(x, W - S_tile)
    return origins


def expected_tile_count(width: int, height: int, tile_size: int, stride: int) -> int:
    """Аналітична кількість тайлів: ``(ceil((W-S)/stride)+1) * (ceil((H-S)/stride)+1)``."""
    nx = math.ceil((width - tile_size) / stride) + 1
    ny = math.ceil((height - tile_size) / stride) + 1
    return nx * ny


def generate_tile_grid(map_width: int, map_height: int, tile_size: int, overlap: float,
                       roi: tuple[int, int, int, int] | None = None,
                       name_template: str = "tile_{index:04d}.png") -> list[TileSpec]:
    """Сітка тайлів у порядку row-major (спершу рядок y, потім x), нумерація з 1."""
    stride = compute_stride(tile_size, overlap)
    if roi is None:
        rx, ry, rw, rh = 0, 0, map_width, map_height
    else:
        rx, ry, rw, rh = (int(v) for v in roi)
        if rx < 0 or ry < 0 or rw <= 0 or rh <= 0 or rx + rw > map_width or ry + rh > map_height:
            raise ValueError(f"ROI {roi} is outside the map {map_width}x{map_height}")

    xs = compute_axis_origins(rw, tile_size, stride)
    ys = compute_axis_origins(rh, tile_size, stride)
    tiles: list[TileSpec] = []
    index = 1
    for y in ys:
        for x in xs:
            tiles.append(TileSpec(name_template.format(index=index),
                                  rx + x, ry + y, tile_size, tile_size))
            index += 1
    return tiles


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_map_rgb(map_path: str | Path,
                 background_rgb: tuple[int, int, int] = (255, 255, 255)) -> tuple[np.ndarray, dict]:
    """Читає карту як RGB і повертає інформацію про вихідний формат."""
    raw = read_image_raw(map_path)
    channels = 1 if raw.ndim == 2 else raw.shape[2]
    info = {"source_channels": channels, "had_alpha": channels == 4,
            "alpha_fully_opaque": None}
    if channels == 4:
        info["alpha_fully_opaque"] = bool(np.all(raw[..., 3] == 255))
    return decoded_to_rgb(raw, background_rgb), info


def _clean_directory(directory: Path, pattern: str) -> int:
    removed = 0
    for stale in directory.glob(pattern):
        if stale.is_file():
            stale.unlink()
            removed += 1
    return removed


def cut_map_tiles(map_path: str | Path, output_dir: str | Path, metadata_path: str | Path,
                  tile_size: int = 512, overlap: float = 0.5,
                  roi: tuple[int, int, int, int] | None = None,
                  background_rgb: tuple[int, int, int] = (255, 255, 255),
                  run_info_path: str | Path | None = None,
                  clean: bool = True,
                  name_template: str = "tile_{index:04d}.png",
                  verbose: bool = True) -> pd.DataFrame:
    """Розбиває карту на тайли ``tile_size × tile_size`` і зберігає реєстр.

    Повертає DataFrame з колонками ``TILES_CSV_COLUMNS``.
    """
    map_path = Path(map_path)
    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    image, source_info = load_map_rgb(map_path, background_rgb)
    map_h, map_w = image.shape[:2]
    stride = compute_stride(tile_size, overlap)
    tiles = generate_tile_grid(map_w, map_h, tile_size, overlap, roi, name_template)

    if clean:
        removed = _clean_directory(output_dir, "tile_*.png")
        if verbose and removed:
            print(f"Removed {removed} stale tile files from {output_dir}")

    for tile in tiles:
        crop = image[tile.y_origin:tile.y_origin + tile.height,
                     tile.x_origin:tile.x_origin + tile.width]
        if crop.shape != (tile.height, tile.width, 3):   # захисна перевірка
            raise RuntimeError(f"Unexpected tile shape {crop.shape} for {tile}")
        write_image_rgb(output_dir / tile.tile_id, crop)

    df = pd.DataFrame([asdict(t) for t in tiles], columns=TILES_CSV_COLUMNS)
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(metadata_path)

    if run_info_path is not None:
        xs = sorted(df["x_origin"].unique().tolist())
        ys = sorted(df["y_origin"].unique().tolist())
        info = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "map_path": portable_path(map_path),
            "map_sha256": file_sha256(map_path),
            "map_width": map_w,
            "map_height": map_h,
            **source_info,
            "alpha_background_rgb": list(background_rgb),
            "roi_xywh": list(roi) if roi is not None else None,
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": stride,
            "grid_cols": len(xs),
            "grid_rows": len(ys),
            "tile_count": len(df),
            "x_origins": xs,
            "y_origins": ys,
            "tiles_dir": portable_path(output_dir),
            "coordinate_convention": "(x, y) = (column, row), map pixels, origin top-left",
            "tile_order": "row-major, index starts at 1",
        }
        Path(run_info_path).write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    if verbose:
        print(f"Map: {map_w}x{map_h}, channels={source_info['source_channels']}, "
              f"alpha_fully_opaque={source_info['alpha_fully_opaque']}")
        print(f"Tile {tile_size}px, overlap {overlap:.2f}, stride {stride}px -> "
              f"{df['x_origin'].nunique()} cols x {df['y_origin'].nunique()} rows "
              f"= {len(df)} tiles")
        print(f"Tiles saved to:   {output_dir}")
        print(f"Metadata saved:   {metadata_path}")
    return df
