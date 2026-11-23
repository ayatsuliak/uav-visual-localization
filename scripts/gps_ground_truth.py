"""Еталон для набору даних із журналом польоту: GNSS -> пікселі карти для кожного кадру.

Журнал використовується ЛИШЕ для оцінки результатів після прогону — система
локалізації його не читає. Центр кадру вважається проєкцією положення апарата
(камера спрямована вертикально вниз, GIMBAL.pitch ≈ -90°; перевіряється).

Приклад:
    UAV_DATASET=flight2 python scripts/gps_ground_truth.py

Результати (каталог метаданих набору даних):
    ground_truth.csv       — frame_id, timestamp, x_true, y_true (px карти), lat, lon, ...
    ground_truth_info.json — параметри побудови
    <figures>/analysis/gt_track.png — еталонна траєкторія на карті
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd

from src.config import (DATASET, DATASET_INFO, DEFAULT_MAP_PATH, FIGURES_DIR,
                        FRAMES_METADATA_PATH, GROUND_TRUTH_PATH, PROJECT_ROOT, portable_path)
from src.evaluation.geo import WorldFile, utm_forward, utm_zone_from_epsg
from src.preprocessing.images import read_image_rgb

FT = 0.3048
COLS = {"time": "OSD.flyTime [s]", "lat": "OSD.latitude", "lon": "OSD.longitude",
        "height": "OSD.height [ft]", "pitch": "GIMBAL.pitch", "yaw": "GIMBAL.yaw [360]"}


def load_log(path: Path) -> pd.DataFrame:
    log = pd.read_csv(path, usecols=list(COLS.values()), low_memory=False).rename(
        columns={v: k for k, v in COLS.items()})
    log = log.dropna(subset=["time", "lat", "lon"])
    log = log[(log["lat"] != 0) & (log["lon"] != 0)]
    # Кілька записів з однаковим часом (10 Гц з округленням) -> середнє.
    return log.groupby("time", as_index=False).mean().sort_values("time")


def interp_angle(t, times, deg):
    """Інтерполяція кута з урахуванням переходу 360 -> 0."""
    unwrapped = np.degrees(np.unwrap(np.radians(deg)))
    return np.interp(t, times, unwrapped) % 360.0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default=str(GROUND_TRUTH_PATH))
    args = p.parse_args()
    if not DATASET_INFO.get("flight_log"):
        raise SystemExit("Set UAV_DATASET to a dataset whose dataset.json has 'flight_log'")

    offset = float(DATASET_INFO["log_time_offset_s"])
    world = WorldFile.read(PROJECT_ROOT / DATASET_INFO["map_world_file"])
    zone, north = utm_zone_from_epsg(DATASET_INFO["map_crs"])
    log = load_log(PROJECT_ROOT / DATASET_INFO["flight_log"])
    frames = pd.read_csv(FRAMES_METADATA_PATH)

    t_log = frames["timestamp_sec"].to_numpy(float) + offset
    if t_log.min() < log["time"].min() or t_log.max() > log["time"].max():
        raise SystemExit(f"Frames span log time {t_log.min():.1f}-{t_log.max():.1f} s, "
                         f"log covers {log['time'].min():.1f}-{log['time'].max():.1f} s")
    lat = np.interp(t_log, log["time"], log["lat"])
    lon = np.interp(t_log, log["time"], log["lon"])
    x_utm, y_utm = utm_forward(lon, lat, zone, north)
    px, py = world.to_pixel(x_utm, y_utm)
    pitch = np.interp(t_log, log["time"], log["pitch"])

    gt = pd.DataFrame({
        "frame_id": frames["frame_id"], "timestamp": frames["timestamp_sec"],
        "x_true": px, "y_true": py, "is_keyframe": True, "reliable": True, "source": "gnss_log",
        "lat": lat, "lon": lon, "easting": x_utm, "northing": y_utm,
        "height_m": np.interp(t_log, log["time"], log["height"]) * FT,
        "gimbal_pitch_deg": pitch,
        "camera_yaw_deg": interp_angle(t_log, log["time"].to_numpy(), log["yaw"].to_numpy()),
        "log_time_s": t_log,
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(out, index=False)

    map_rgb = read_image_rgb(DEFAULT_MAP_PATH)
    H, W = map_rgb.shape[:2]
    inside = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
    step = np.hypot(np.diff(px), np.diff(py))
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET, "flight_log": DATASET_INFO["flight_log"],
        "log_time_offset_s": offset, "map": portable_path(DEFAULT_MAP_PATH),
        "map_crs": DATASET_INFO["map_crs"], "map_gsd_m": world.gsd, "frames": int(len(gt)),
        "frames_inside_map": int(inside.sum()),
        "gimbal_pitch_deg": {"min": float(pitch.min()), "max": float(pitch.max())},
        "height_m": {"min": float(gt["height_m"].min()), "max": float(gt["height_m"].max())},
        "step_px": {"mean": float(step.mean()), "max": float(step.max())},
        "note": "Еталон лише для оцінки після прогону; система локалізації журнал не читає. "
                "Похибка самого GNSS ~2-3 м.",
    }
    (out.parent / "ground_truth_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")

    canvas = map_rgb.copy()
    pts = np.stack([px, py], axis=1).round().astype(np.int32)
    cv2.polylines(canvas, [pts], False, (255, 40, 40), 5, cv2.LINE_AA)
    for i in range(0, len(pts), 30):
        cv2.circle(canvas, tuple(int(v) for v in pts[i]), 14, (255, 220, 0), -1, cv2.LINE_AA)
        cv2.putText(canvas, frames["frame_id"].iloc[i][6:10], tuple(int(v) + 18 for v in pts[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
    fig = FIGURES_DIR / "analysis" / "gt_track.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    scale = 1600 / max(W, H)
    cv2.imwrite(str(fig), cv2.cvtColor(cv2.resize(canvas, None, fx=scale, fy=scale,
                                                  interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    print(f"{len(gt)} frames, {int(inside.sum())} inside the map; GSD {world.gsd:.4f} m/px; "
          f"gimbal pitch {pitch.min():.1f}..{pitch.max():.1f} deg; "
          f"height {gt['height_m'].min():.1f}..{gt['height_m'].max():.1f} m; "
          f"step {step.mean():.1f} px/frame (max {step.max():.1f})")
    print(f"Saved: {out}\nSaved: {fig}")


if __name__ == "__main__":
    main()
