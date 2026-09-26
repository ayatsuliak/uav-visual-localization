"""Етап 3. Одноразове обчислення ознак SuperPoint для всіх тайлів карти.

Приклади:
    python scripts/build_tile_cache.py                 # усі тайли з tiles.csv
    python scripts/build_tile_cache.py --overwrite     # перебудувати (після зміни параметрів)

Результат: data/processed/cache_superpoint/tile_XXXX.npz + manifest.json.
Кеш прив'язаний до tiles.csv, карти, work_scale і параметрів SuperPoint;
scripts/localize_frame.py відмовиться використовувати кеш, що не збігається.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from src.config import SUPERPOINT_CACHE_DIR, TILES_DIR, TILES_METADATA_PATH, WORK_SCALE
from src.matching import create_matcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.cache import build_tile_cache
from src.run_info import tiles_run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=str(SUPERPOINT_CACHE_DIR))
    parser.add_argument("--work-scale", type=float, default=WORK_SCALE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="видалити наявний кеш і побудувати заново")
    args = parser.parse_args()

    tiles = pd.read_csv(TILES_METADATA_PATH)
    matcher = create_matcher("LightGlue", device=args.device)
    pipeline = PairMatchingPipeline(matcher, PipelineConfig(work_scale=args.work_scale))
    print(f"Extracting SuperPoint features for {len(tiles)} tiles on {matcher.device} ...")
    manifest = build_tile_cache(pipeline, tiles, TILES_DIR, args.cache_dir, TILES_METADATA_PATH,
                                map_sha256=tiles_run_info().get("map_sha256"),
                                overwrite=args.overwrite)
    kp = manifest["keypoints"]
    print(f"Keypoints per tile: min {kp['min']}, mean {kp['mean']:.0f}, max {kp['max']}")
    print(f"Extraction time: {manifest['extract_time_sec']['total']:.1f} s "
          f"(wall {manifest['wall_time_sec']:.1f} s), "
          f"cache size {manifest['size_bytes'] / 2**20:.0f} MiB")
    print(f"Saved: {args.cache_dir}")


if __name__ == "__main__":
    main()
