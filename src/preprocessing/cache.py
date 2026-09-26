"""Дисковий кеш ознак тайлів (SuperPoint для LightGlue).

Структура каталогу (за замовчуванням ``data/processed/cache_superpoint/``):

* ``tile_XXXX.npz`` — ознаки одного тайла (float32, без втрати точності):
  ``keypoints (N, 2)``, ``keypoint_scores (N,)``, ``descriptors (N, 256)``,
  ``image_size (2,)`` у робочій роздільності + ``original_size (2,)``;
* ``manifest.json`` — параметри, з якими кеш побудовано. Кеш вважається
  дійсним, лише якщо вони збігаються з поточними (метод, ``work_scale``,
  хеш ``tiles.csv`` і карти). Інакше завантаження завершується помилкою,
  щоб не змішати ознаки, обчислені за різних налаштувань.

Для LoFTR кеш не будується: метод бездетекторний, ознаки залежать від пари.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import portable_path
from src.matching.base import Features, ImageMatcher
from src.matching.pipeline import PairMatchingPipeline, PreparedImage
from src.preprocessing.images import read_image_rgb

MANIFEST_NAME = "manifest.json"
CACHE_FORMAT_VERSION = 1


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_signature(matcher: ImageMatcher, work_scale: float, tiles_csv: str | Path,
                    map_sha256: str | None = None) -> dict:
    """Параметри, які визначають вміст кешу (без пристрою: ознаки від нього не залежать)."""
    matcher_config = {k: v for k, v in matcher.config().items() if k != "device"}
    return {"format_version": CACHE_FORMAT_VERSION, "matcher": matcher_config,
            "work_scale": float(work_scale), "tiles_csv_sha256": _sha256(tiles_csv),
            "map_sha256": map_sha256}


@dataclass
class CachedTile:
    prepared: PreparedImage
    features: Features


def build_tile_cache(pipeline: PairMatchingPipeline, tiles: pd.DataFrame, tiles_dir: str | Path,
                     cache_dir: str | Path, tiles_csv: str | Path, map_sha256: str | None = None,
                     overwrite: bool = False, progress_every: int = 50) -> dict:
    """Обчислює й зберігає ознаки всіх тайлів із ``tiles``. Повертає маніфест."""
    matcher = pipeline.matcher
    if not matcher.supports_feature_cache:
        raise ValueError(f"{matcher.name} does not support feature caching")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(matcher, pipeline.config.work_scale, tiles_csv, map_sha256)

    manifest_path = cache_dir / MANIFEST_NAME
    if manifest_path.is_file() and not overwrite:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise RuntimeError(f"Cache in {cache_dir} was built with different parameters; "
                               f"use overwrite=True (--overwrite) to rebuild it")
    if overwrite:
        for stale in cache_dir.glob("tile_*.npz"):
            stale.unlink()

    keypoints, times = [], []
    start = time.perf_counter()
    for i, tile in enumerate(tiles.itertuples(index=False), 1):
        prepared = pipeline.prepare_tile(read_image_rgb(Path(tiles_dir) / tile.tile_id))
        feats = matcher.extract(prepared.image)
        arrays = matcher.features_to_arrays(feats)
        arrays["original_size"] = np.asarray(prepared.original_size, dtype=np.int64)
        np.savez(cache_dir / f"{Path(tile.tile_id).stem}.npz", **arrays)
        keypoints.append(feats.num_keypoints)
        times.append(feats.time)
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(tiles)} tiles")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signature": signature, "tiles_csv": portable_path(tiles_csv),
        "tile_ids": list(tiles["tile_id"]), "device": matcher.device,
        "keypoints": ({"min": int(min(keypoints)), "mean": float(np.mean(keypoints)),
                       "max": int(max(keypoints))} if None not in keypoints else None),
        "extract_time_sec": {"total": float(sum(times)), "mean": float(np.mean(times))},
        "wall_time_sec": time.perf_counter() - start,
        "size_bytes": int(sum(p.stat().st_size for p in cache_dir.glob("tile_*.npz"))),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_tile_cache(pipeline: PairMatchingPipeline, cache_dir: str | Path, tile_ids,
                    tiles_csv: str | Path, map_sha256: str | None = None,
                    device: str | None = None) -> float:
    """Завантажує ознаки тайлів ``tile_ids`` у кеш pipeline.

    ``device``: ``"cpu"`` — тримати в RAM і переносити на GPU під час
    зіставлення; ``None`` — пристрій моделі (VRAM, якщо це CUDA).
    Повертає час завантаження, с.
    """
    matcher = pipeline.matcher
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No tile cache in {cache_dir}; run scripts/build_tile_cache.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = cache_signature(matcher, pipeline.config.work_scale, tiles_csv, map_sha256)
    if manifest.get("signature") != expected:
        raise RuntimeError(f"Tile cache in {cache_dir} does not match the current parameters; "
                           f"rebuild it with scripts/build_tile_cache.py --overwrite")

    start = time.perf_counter()
    for tile_id in tile_ids:
        path = cache_dir / f"{Path(tile_id).stem}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Tile {tile_id} is missing from the cache {cache_dir}")
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        feats = matcher.features_from_arrays(arrays, device)
        prepared = PreparedImage.from_sizes(tuple(int(v) for v in arrays["original_size"]),
                                            feats.image_size)
        pipeline.add_cached_tile(tile_id, prepared, feats)
    return time.perf_counter() - start
