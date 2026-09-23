"""Прототип Етапу 3: перебір тайлів для одного кадру (буде переглянуто на Етапі 3).

Використовує єдиний pipeline Етапу 2: ознаки кадру обчислюються один раз,
ознаки тайлів кешуються. Порядок кандидатів: спочатку успішні зіставлення,
потім більша кількість геометрично узгоджених відповідностей. Остаточний
критерій ранжування визначається на Етапі 3.

Приклад:
    python scripts/localize_frame.py --frame frame_0002.png --method LightGlue --max-tiles 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from src.config import FRAMES_DIR, RESULTS_DIR, TILES_DIR, TILES_METADATA_PATH
from src.matching import MATCHER_NAMES, create_matcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.images import read_image_rgb


def rank_tiles(pipeline, frame, frame_name, tiles):
    prepared = pipeline.prepare_frame(frame)
    frame_features = (prepared, pipeline.matcher.extract(prepared.image))
    rows = []
    for _, tile in tiles.iterrows():
        image = read_image_rgb(TILES_DIR / tile["tile_id"])
        result = pipeline.run(frame, image, (tile["x_origin"], tile["y_origin"]),
                              tile_cache_key=tile["tile_id"], frame_features=frame_features)
        rows.append({"method": pipeline.matcher.name, "frame": frame_name,
                     "tile": tile["tile_id"], **result.to_row()})
    return pd.DataFrame(rows).sort_values(["success", "inliers", "inlier_ratio"],
                                          ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Find the most plausible map tile for a UAV frame.")
    parser.add_argument("--frame", required=True)
    parser.add_argument("--method", choices=[*MATCHER_NAMES, "both"], default="both")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    frame_path = Path(args.frame)
    if not frame_path.is_absolute():
        frame_path = FRAMES_DIR / frame_path
    frame = read_image_rgb(frame_path)
    tiles = pd.read_csv(TILES_METADATA_PATH)
    if args.max_tiles is not None:
        tiles = tiles.head(args.max_tiles)

    methods = MATCHER_NAMES if args.method == "both" else (args.method,)
    results = [rank_tiles(PairMatchingPipeline(create_matcher(m, args.device), PipelineConfig()),
                          frame, frame_path.name, tiles) for m in methods]
    result = pd.concat(results, ignore_index=True)
    out = RESULTS_DIR / "rankings" / f"tile_ranking_{frame_path.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.groupby("method").head(5)[["method", "tile", "matches", "inliers",
                                            "inlier_ratio", "success", "center_x_map",
                                            "center_y_map"]])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
