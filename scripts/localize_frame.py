import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pandas as pd

from src.config import FRAMES_DIR, TILES_DIR, TILES_METADATA_PATH, RESULTS_DIR
from src.evaluation.scoring import tile_selection_score
from src.geometry.homography import estimate_homography_metrics
from src.matching.lightglue import LightGlueMatcher
from src.matching.loftr import LoFTRMatcher
from src.localization.estimate import estimate_global_position


def rank_tiles(method_name, matcher, frame_path, tiles_dir, tile_metadata, max_tiles=None):
    rows = []
    tiles = tile_metadata.copy()
    if max_tiles is not None:
        tiles = tiles.head(max_tiles)
    for _, tile in tiles.iterrows():
        tile_path = tiles_dir / tile["tile_id"]
        try:
            result = matcher.match(str(frame_path), str(tile_path))
            metrics = estimate_homography_metrics(result["pts0"], result["pts1"])
            score = tile_selection_score(metrics)
            position = estimate_global_position(
                result["image0"].shape, metrics["H"], float(tile["x_origin"]), float(tile["y_origin"]),
                (int(tile["height"]), int(tile["width"])),
            )
            rows.append({"method": method_name, "frame": frame_path.name,
                         "tile": tile["tile_id"], "time": result["time"],
                         "matches": metrics["matches"], "inliers": metrics["inliers"],
                         "inlier_ratio": metrics["inlier_ratio"],
                         "reprojection_error": metrics["reprojection_error"],
                         "success": metrics["success"], "tile_selection_score": score,
                         "position_x": None if position is None else position[0],
                         "position_y": None if position is None else position[1]})
        except Exception as exc:
            rows.append({"method": method_name, "frame": frame_path.name,
                         "tile": tile["tile_id"], "time": None, "matches": 0,
                         "inliers": 0, "inlier_ratio": 0, "reprojection_error": None,
                         "success": False, "tile_selection_score": 0,
                         "position_x": None, "position_y": None, "error": str(exc)})
    return pd.DataFrame(rows).sort_values(
        ["success", "tile_selection_score", "inliers", "inlier_ratio"],
        ascending=[False, False, False, False],
    )


def main():
    parser = argparse.ArgumentParser(description="Find the most plausible map tile for a UAV frame.")
    parser.add_argument("--frame", required=True)
    parser.add_argument("--method", choices=["LoFTR", "LightGlue", "both"], default="both")
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()

    frame_path = Path(args.frame)
    if not frame_path.is_absolute():
        frame_path = FRAMES_DIR / frame_path
    metadata = pd.read_csv(TILES_METADATA_PATH)
    matchers = []
    if args.method in ("LoFTR", "both"):
        matchers.append(("LoFTR", LoFTRMatcher()))
    if args.method in ("LightGlue", "both"):
        matchers.append(("LightGlue", LightGlueMatcher()))

    all_results = []
    for name, matcher in matchers:
        df = rank_tiles(name, matcher, frame_path, TILES_DIR, metadata, args.max_tiles)
        all_results.append(df)

    result = pd.concat(all_results, ignore_index=True)
    out = RESULTS_DIR / "rankings" / f"tile_ranking_{frame_path.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.groupby("method").head(5))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
