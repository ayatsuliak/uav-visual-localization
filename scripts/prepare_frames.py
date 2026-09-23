"""Етап 1. Вилучення кадрів із відео БпЛА та формування data/metadata/frames.csv.

Приклад:
    python scripts/prepare_frames.py                    # 1 FPS, усі кадри
    python scripts/prepare_frames.py --limit 30         # перші 30 кадрів (налагодження)
    python scripts/prepare_frames.py --fps 2            # 2 кадри за секунду
    python scripts/prepare_frames.py --every-n 15       # явний крок у сирих кадрах
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (DEFAULT_VIDEO_PATH, FRAME_SAMPLING_FPS, FRAMES_DIR,
                        FRAMES_METADATA_PATH, FRAMES_RUN_INFO_PATH, VIDEO_DIR,
                        VIDEO_EXTENSIONS)
from src.preprocessing.frames import extract_frames, find_default_video


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a UAV video.")
    parser.add_argument("--video", default=None,
                        help=f"Video path (default: {DEFAULT_VIDEO_PATH.name} or the only video "
                             f"in {VIDEO_DIR})")
    parser.add_argument("--output", default=str(FRAMES_DIR))
    parser.add_argument("--metadata", default=str(FRAMES_METADATA_PATH))
    parser.add_argument("--run-info", default=str(FRAMES_RUN_INFO_PATH),
                        help="JSON with extraction parameters (use '' to skip)")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--fps", type=float, default=FRAME_SAMPLING_FPS,
                          help="Target sampling rate, frames per second (default 1.0)")
    sampling.add_argument("--every-n", type=int, default=None,
                          help="Keep every N-th raw frame (overrides --fps)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Save only the first N sampled frames (debugging)")
    parser.add_argument("--start-sec", type=float, default=0.0,
                        help="Skip the beginning of the video")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Do not delete old frame_*.png files before writing")
    return parser.parse_args()


def main():
    args = parse_args()
    video = Path(args.video) if args.video else find_default_video(
        VIDEO_DIR, DEFAULT_VIDEO_PATH, VIDEO_EXTENSIONS)
    extract_frames(
        video_path=video,
        output_dir=args.output,
        metadata_path=args.metadata,
        target_fps=None if args.every_n else args.fps,
        every_n=args.every_n,
        limit=args.limit,
        start_sec=args.start_sec,
        run_info_path=args.run_info or None,
        clean=not args.keep_existing,
    )


if __name__ == "__main__":
    main()
