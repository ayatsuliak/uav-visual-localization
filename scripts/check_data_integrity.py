"""Етап 1. Sanity check: тайли, кадри та їхні метадані.

Головна перевірка: кожен тайл із tiles.csv, вирізаний із university_map.png
за (x_origin, y_origin), піксель-у-піксель збігається з файлом тайла.

Приклад:
    python scripts/check_data_integrity.py              # тайли + кадри (якщо є)
    python scripts/check_data_integrity.py --skip-frames
    python scripts/check_data_integrity.py --sample 50  # піксельна перевірка 50 тайлів
Код виходу: 0 - усе коректно, 1 - знайдено помилки.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (ALPHA_BACKGROUND_RGB, DEFAULT_MAP_PATH, FRAMES_DIR,
                        FRAMES_METADATA_PATH, FRAMES_RUN_INFO_PATH, TILES_DIR,
                        TILES_METADATA_PATH, TILES_RUN_INFO_PATH)
from src.preprocessing.validation import validate_frames, validate_tiles


def report(title, errors, stats):
    status = "OK" if not errors else f"FAILED ({len(errors)} problem(s))"
    print(f"[{title}] {status}")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    for err in errors[:30]:
        print(f"    ERROR: {err}")
    if len(errors) > 30:
        print(f"    ... and {len(errors) - 30} more")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 data integrity check.")
    parser.add_argument("--map", default=str(DEFAULT_MAP_PATH))
    parser.add_argument("--tiles-dir", default=str(TILES_DIR))
    parser.add_argument("--tiles-csv", default=str(TILES_METADATA_PATH))
    parser.add_argument("--tiles-info", default=str(TILES_RUN_INFO_PATH))
    parser.add_argument("--frames-dir", default=str(FRAMES_DIR))
    parser.add_argument("--frames-csv", default=str(FRAMES_METADATA_PATH))
    parser.add_argument("--frames-info", default=str(FRAMES_RUN_INFO_PATH))
    parser.add_argument("--frame-size", nargs=2, type=int, default=[1920, 1080], metavar=("W", "H"))
    parser.add_argument("--sample", type=int, default=None,
                        help="Check pixels of N random tiles instead of all")
    parser.add_argument("--skip-tiles", action="store_true")
    parser.add_argument("--skip-frames", action="store_true")
    args = parser.parse_args()

    failed = False
    if not args.skip_tiles:
        errors, stats = validate_tiles(args.map, args.tiles_dir, args.tiles_csv,
                                       run_info_path=args.tiles_info,
                                       background_rgb=ALPHA_BACKGROUND_RGB,
                                       sample=args.sample)
        report("TILES", errors, stats)
        failed |= bool(errors)

    if not args.skip_frames:
        if not Path(args.frames_csv).is_file():
            print(f"[FRAMES] SKIPPED: {args.frames_csv} not found (run prepare_frames.py)")
        else:
            errors, stats = validate_frames(args.frames_dir, args.frames_csv,
                                            expected_size=tuple(args.frame_size),
                                            run_info_path=args.frames_info)
            report("FRAMES", errors, stats)
            failed |= bool(errors)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
