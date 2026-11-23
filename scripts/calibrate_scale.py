"""Калібрування масштабу кадру s_f лише за зображеннями (без GNSS).

Для вибраних кадрів виконується глобальний пошук по всій карті з гіпотезами курсу
0/90/180/-90° (як у стані LOST трекера). Перевірка правдоподібності допускає масштаб
у межах [s_f/2, 2·s_f] від апріорної оцінки, тому переможець показує фактичний масштаб
гомографії. Рекомендоване s_f — медіана масштабу переможців з високим балом.

Якщо для набору даних є еталон, для довідки виводиться відстань переможця до нього;
на оцінку s_f еталон не впливає.

Приклад:
    UAV_DATASET=flight2 python scripts/calibrate_scale.py --frames 1 30 60 90 100 120
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import pandas as pd

from src.config import (FRAME_TO_MAP_SCALE, FRAMES_DIR, GLOBAL_YAW_CYCLE_DEG, GROUND_TRUTH_PATH,
                        MAP_GSD_M, RESULTS_DIR, SUPERPOINT_CACHE_DIR, TILES_DIR,
                        TILES_METADATA_PATH, WARMUP_ITERATIONS)
from src.localization.search import TileSearch
from src.matching import MATCHER_NAMES, create_matcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.cache import load_tile_cache
from src.preprocessing.images import read_image_rgb
from src.run_info import map_size, tiles_run_info

MATCHERS = {name.lower(): name for name in MATCHER_NAMES}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--matcher", type=str.lower, choices=sorted(MATCHERS), default="lightglue")
    p.add_argument("--scale", type=float, default=FRAME_TO_MAP_SCALE,
                   help="апріорне s_f для цього запуску (типово — з конфігурації)")
    p.add_argument("--min-score", type=float, default=10.0,
                   help="переможці з меншим балом не враховуються в медіані")
    args = p.parse_args()

    tiles = pd.read_csv(TILES_METADATA_PATH)
    W, H = map_size(tiles)
    matcher = create_matcher(MATCHERS[args.matcher])
    pipeline = PairMatchingPipeline(matcher, PipelineConfig(frame_to_map_scale=args.scale))
    if matcher.supports_feature_cache and (SUPERPOINT_CACHE_DIR / "manifest.json").is_file():
        load_tile_cache(pipeline, SUPERPOINT_CACHE_DIR, tiles["tile_id"], TILES_METADATA_PATH,
                        tiles_run_info().get("map_sha256"))
    search = TileSearch(pipeline, TILES_DIR, (W, H))
    gt = pd.read_csv(GROUND_TRUTH_PATH).set_index("frame_id") if GROUND_TRUTH_PATH.is_file() else None

    rows = []
    for k in args.frames:
        frame_id = f"frame_{k:04d}.png"
        frame = read_image_rgb(FRAMES_DIR / frame_id)
        if not rows:
            pipeline.warmup((frame.shape[1], frame.shape[0]),
                            (int(tiles["width"].iloc[0]), int(tiles["height"].iloc[0])),
                            WARMUP_ITERATIONS)
        best = None
        for yaw in GLOBAL_YAW_CYCLE_DEG:
            result = search.search(frame, tiles, rotation_deg=yaw)
            w = result.winner
            if w is not None and (best is None or w.evaluation.score > best[0].evaluation.score):
                best = (w, yaw)
        row = {"frame_id": frame_id}
        if best is not None:
            w, yaw = best
            d = w.pair.diagnostics
            row.update(yaw_hypothesis=yaw, tile_id=w.tile_id, score=w.evaluation.score,
                       inliers=w.metrics.inliers, h_scale=d.scale, h_rotation_deg=d.rotation_deg,
                       x_map=w.center_map[0], y_map=w.center_map[1])
            if gt is not None and frame_id in gt.index:
                e = float(np.hypot(w.center_map[0] - gt.at[frame_id, "x_true"],
                                   w.center_map[1] - gt.at[frame_id, "y_true"]))
                row.update(error_px_for_reference=e,
                           error_m_for_reference=e * MAP_GSD_M if MAP_GSD_M else None)
        rows.append(row)
        print({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items()}, flush=True)

    df = pd.DataFrame(rows)
    good = df[df.get("score", pd.Series(dtype=float)) >= args.min_score] if "score" in df else df.iloc[0:0]
    out = RESULTS_DIR / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.matcher}_s{args.scale:g}"
    df.to_csv(out / f"scale_calibration_{tag}.csv", index=False)
    summary = {"prior_frame_to_map_scale": args.scale, "matcher": MATCHERS[args.matcher],
               "frames": args.frames, "used": int(len(good)), "min_score": args.min_score,
               "h_scale_median": float(good["h_scale"].median()) if len(good) else None,
               "h_scale_iqr": ([float(good["h_scale"].quantile(q)) for q in (0.25, 0.75)]
                               if len(good) else None)}
    (out / f"scale_calibration_{tag}.json").write_text(json.dumps(summary, indent=2),
                                                                encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
