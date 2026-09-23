"""Вилучення дискретної послідовності кадрів із відео БпЛА.

Кадри читаються **послідовно** (``grab`` для пропущених, ``retrieve`` для
збережених): позиціонування через ``CAP_PROP_POS_FRAMES`` для H.264/H.265
неточне і може повертати не той кадр, тому тут не використовується.
Часова мітка обчислюється як ``source_frame_idx / fps`` відео.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from src.config import portable_path
from src.preprocessing.images import bgr_to_rgb, write_image_rgb

FRAMES_CSV_COLUMNS = ["frame_id", "source_frame_idx", "timestamp_sec", "width", "height"]


def probe_video(video_path: str | Path) -> dict:
    """Метадані контейнера. ``frame_count`` — оцінка з заголовка (може бути неточна)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
        info = {
            "fps": fps,
            "frame_count_header": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": fourcc,
        }
    finally:
        cap.release()
    info["duration_sec_header"] = (info["frame_count_header"] / fps) if fps > 0 else None
    return info


def compute_frame_step(video_fps: float, target_fps: float | None = None,
                       every_n: int | None = None) -> int:
    """Крок семплінгу у сирих кадрах: ``every_n`` або ``round(video_fps / target_fps)``."""
    if every_n is not None:
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        return int(every_n)
    if target_fps is None or target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if video_fps <= 0:
        raise ValueError("Video FPS is unknown; pass every_n explicitly")
    if target_fps > video_fps:
        raise ValueError(f"target_fps={target_fps} exceeds video fps={video_fps:.3f}")
    return max(1, int(round(video_fps / target_fps)))


def _clean_directory(directory: Path, pattern: str) -> int:
    removed = 0
    for stale in directory.glob(pattern):
        if stale.is_file():
            stale.unlink()
            removed += 1
    return removed


def extract_frames(video_path: str | Path, output_dir: str | Path,
                   metadata_path: str | Path,
                   target_fps: float | None = 1.0, every_n: int | None = None,
                   limit: int | None = None, start_sec: float = 0.0,
                   run_info_path: str | Path | None = None, clean: bool = True,
                   name_template: str = "frame_{index:04d}.png",
                   verbose: bool = True) -> pd.DataFrame:
    """Зберігає кожен ``step``-й кадр (починаючи з ``start_sec``) у PNG без втрат.

    Кадри не масштабуються: зберігається повна роздільна здатність відео.
    Повертає DataFrame з колонками ``FRAMES_CSV_COLUMNS``.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe_video(video_path)
    fps = info["fps"]
    step = compute_frame_step(fps, target_fps, every_n)
    start_idx = int(round(start_sec * fps)) if start_sec > 0 else 0

    if clean:
        removed = _clean_directory(output_dir, "frame_*.png")
        if verbose and removed:
            print(f"Removed {removed} stale frame files from {output_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows: list[dict] = []
    source_idx = 0
    try:
        while True:
            want = source_idx >= start_idx and (source_idx - start_idx) % step == 0
            if not cap.grab():
                break
            if want:
                ok, frame_bgr = cap.retrieve()
                if not ok or frame_bgr is None:
                    raise RuntimeError(f"Cannot decode frame {source_idx} of {video_path}")
                frame_rgb = bgr_to_rgb(frame_bgr)
                h, w = frame_rgb.shape[:2]
                frame_id = name_template.format(index=len(rows) + 1)
                write_image_rgb(output_dir / frame_id, frame_rgb)
                rows.append({
                    "frame_id": frame_id,
                    "source_frame_idx": source_idx,
                    "timestamp_sec": round(source_idx / fps, 6) if fps > 0 else float("nan"),
                    "width": w,
                    "height": h,
                })
                if verbose and len(rows) % 10 == 0:
                    print(f"  saved {len(rows)} frames (source idx {source_idx})")
                if limit is not None and len(rows) >= limit:
                    source_idx += 1
                    break
            source_idx += 1
    finally:
        cap.release()

    if not rows:
        raise RuntimeError(f"No frames extracted from {video_path}")

    df = pd.DataFrame(rows, columns=FRAMES_CSV_COLUMNS)
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(metadata_path)

    if run_info_path is not None:
        run_info = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "video_path": portable_path(video_path),
            **{f"video_{k}": v for k, v in info.items()},
            "frames_decoded": source_idx,
            "target_fps": target_fps if every_n is None else None,
            "every_n": step,
            "effective_fps": (fps / step) if fps > 0 else None,
            "start_sec": start_sec,
            "limit": limit,
            "frame_count": len(df),
            "frames_dir": portable_path(output_dir),
            "color_space": "RGB in memory, PNG on disk (lossless)",
            "frame_order": "chronological, index starts at 1",
        }
        Path(run_info_path).write_text(json.dumps(run_info, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    if verbose:
        print(f"Video: {info['width']}x{info['height']} @ {fps:.3f} FPS, "
              f"header frames={info['frame_count_header']}, codec={info['fourcc']}")
        print(f"Step: every {step} frame(s) (~{fps / step:.3f} FPS); "
              f"decoded {source_idx} frames; saved {len(df)}")
        print(f"Frames saved to:  {output_dir}")
        print(f"Metadata saved:   {metadata_path}")
    return df


def find_default_video(video_dir: str | Path, preferred: str | Path | None,
                       extensions: tuple[str, ...]) -> Path:
    """Повертає ``preferred``, якщо існує, інакше єдине відео в ``video_dir``."""
    if preferred is not None and Path(preferred).is_file():
        return Path(preferred)
    video_dir = Path(video_dir)
    candidates = sorted(p for p in video_dir.glob("*") if p.suffix.lower() in extensions)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    names = ", ".join(p.name for p in candidates)
    raise RuntimeError(f"Several videos in {video_dir} ({names}); pass --video explicitly")
