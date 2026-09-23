from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def audit_video(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print("=" * 60)
    print("VIDEO AUDIT")
    print("=" * 60)

    print(f"Path:              {video_path}")
    print(f"Resolution:        {width} x {height}")
    print(f"FPS:               {fps:.3f}")
    print(f"Frame count:       {frame_count}")
    print(f"Duration:          {duration:.2f} s")
    print(f"Duration:          {duration / 60:.2f} min")

    for target_fps in (1, 2, 5, 10):
        estimated_frames = int(duration * target_fps)

        print(
            f"Frames at {target_fps:>2} FPS:   "
            f"approximately {estimated_frames}"
        )

    cap.release()


def audit_map(map_path: Path) -> None:
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")

    image = cv2.imread(str(map_path))

    if image is None:
        raise RuntimeError(f"Cannot read map: {map_path}")

    height, width = image.shape[:2]

    print()
    print("=" * 60)
    print("MAP AUDIT")
    print("=" * 60)

    print(f"Path:              {map_path}")
    print(f"Resolution:        {width} x {height}")
    print("Coordinate system: pixel coordinates")
    print("GSD:               unknown")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit UAV video and orthophoto data."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to UAV video.",
    )

    parser.add_argument(
        "--map",
        type=Path,
        required=True,
        help="Path to orthophoto map.",
    )

    args = parser.parse_args()

    audit_video(args.video)
    audit_map(args.map)


if __name__ == "__main__":
    main()