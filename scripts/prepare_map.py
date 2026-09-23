"""Етап 1. Тайлування ортофотоплану та формування data/metadata/tiles.csv.

Приклад:
    python scripts/prepare_map.py
    python scripts/prepare_map.py --tile-size 768 --overlap 0.25
    python scripts/prepare_map.py --roi 912 1409 768 1024
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (ALPHA_BACKGROUND_RGB, DEFAULT_MAP_PATH, TILE_OVERLAP, TILE_SIZE,
                        TILES_DIR, TILES_METADATA_PATH, TILES_RUN_INFO_PATH)
from src.preprocessing.tiles import cut_map_tiles


def parse_args():
    parser = argparse.ArgumentParser(description="Cut the orthophoto map into overlapping tiles.")
    parser.add_argument("--map", default=str(DEFAULT_MAP_PATH), help="Input map (PNG, RGB/RGBA)")
    parser.add_argument("--output", default=str(TILES_DIR), help="Directory for tile_*.png")
    parser.add_argument("--metadata", default=str(TILES_METADATA_PATH), help="Output tiles.csv")
    parser.add_argument("--run-info", default=str(TILES_RUN_INFO_PATH),
                        help="JSON with tiling parameters (use '' to skip)")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=TILE_OVERLAP)
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"),
                        help="Optional ROI in full-map pixels; CSV coordinates stay global")
    parser.add_argument("--background", nargs=3, type=int, default=list(ALPHA_BACKGROUND_RGB),
                        metavar=("R", "G", "B"), help="Background for alpha blending")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Do not delete old tile_*.png files before writing")
    return parser.parse_args()


def main():
    args = parse_args()
    cut_map_tiles(
        map_path=args.map,
        output_dir=args.output,
        metadata_path=args.metadata,
        tile_size=args.tile_size,
        overlap=args.overlap,
        roi=tuple(args.roi) if args.roi else None,
        background_rgb=tuple(args.background),
        run_info_path=args.run_info or None,
        clean=not args.keep_existing,
    )


if __name__ == "__main__":
    main()
