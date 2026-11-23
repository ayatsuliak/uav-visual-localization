"""Показ компаса (курс камери) для кожного кадру — імітація магнітометра з журналу польоту.

Береться лише курс (GIMBAL.yaw), без координат GNSS. Поправка Δ (схилення, зближення
меридіанів), зміщення й шум задаються під час трекінгу (scripts/track_sequence.py
--compass ...), тому файл містить «сирий» показ.

Приклад:
    UAV_DATASET=flight2 python scripts/make_compass.py

Результат: <метадані набору даних>/compass.csv — frame_id, timestamp, heading_deg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from src.config import COMPASS_PATH, DATASET_INFO, FRAMES_METADATA_PATH, PROJECT_ROOT

YAW_COLUMN = "GIMBAL.yaw [360]"
TIME_COLUMN = "OSD.flyTime [s]"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default=str(COMPASS_PATH))
    args = p.parse_args()
    if not DATASET_INFO.get("flight_log"):
        raise SystemExit("Set UAV_DATASET to a dataset whose dataset.json has 'flight_log'")
    log = pd.read_csv(PROJECT_ROOT / DATASET_INFO["flight_log"], usecols=[TIME_COLUMN, YAW_COLUMN],
                      low_memory=False).dropna()
    log = log.groupby(TIME_COLUMN, as_index=False).mean().sort_values(TIME_COLUMN)
    frames = pd.read_csv(FRAMES_METADATA_PATH)
    t = frames["timestamp_sec"].to_numpy(float) + float(DATASET_INFO["log_time_offset_s"])
    unwrapped = np.degrees(np.unwrap(np.radians(log[YAW_COLUMN].to_numpy(float))))
    heading = np.interp(t, log[TIME_COLUMN].to_numpy(float), unwrapped) % 360.0
    out = pd.DataFrame({"frame_id": frames["frame_id"], "timestamp": frames["timestamp_sec"],
                        "heading_deg": heading})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"{len(out)} frames, heading {heading.min():.1f}..{heading.max():.1f} deg -> {args.output}")


if __name__ == "__main__":
    main()
