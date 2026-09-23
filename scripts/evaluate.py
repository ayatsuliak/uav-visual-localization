import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pandas as pd

from src.config import EVAL_PAIRS_PATH, FRAMES_DIR, TILES_DIR, RESULTS_DIR, FIGURES_DIR, MATCHING_RESULTS_PATH
from src.evaluation.scoring import tile_selection_score
from src.geometry.homography import estimate_homography_metrics
from src.matching.lightglue import LightGlueMatcher
from src.matching.loftr import LoFTRMatcher
from src.visualization.matches import save_match_visualization_from_images


def evaluate_one(method_name, matcher, pair_id, frame_path, tile_path, scene_type, figure_dir):
    result = matcher.match(str(frame_path), str(tile_path))
    metrics = estimate_homography_metrics(result["pts0"], result["pts1"])
    score = tile_selection_score(metrics)
    save_match_visualization_from_images(
        result["image0"], result["image1"], result["pts0"], result["pts1"],
        metrics["mask"], figure_dir / f"{method_name.lower()}_inliers.png", 120,
    )
    return {
        "pair_id": pair_id, "method": method_name, "frame": frame_path.name,
        "tile": tile_path.name, "scene_type": scene_type, "time": result["time"],
        "matches": metrics["matches"], "inliers": metrics["inliers"],
        "inlier_ratio": metrics["inlier_ratio"],
        "reprojection_error": metrics["reprojection_error"],
        "success": metrics["success"], "tile_selection_score": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(EVAL_PAIRS_PATH))
    parser.add_argument("--output", default=str(MATCHING_RESULTS_PATH))
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    matchers = [("LoFTR", LoFTRMatcher()), ("LightGlue", LightGlueMatcher())]
    rows = []
    for _, row in pairs.iterrows():
        frame_path = FRAMES_DIR / row["frame"]
        tile_path = TILES_DIR / row["tile"]
        pair_dir = FIGURES_DIR / "matching" / f"pair_{int(row['pair_id']):03d}"
        for name, matcher in matchers:
            try:
                rows.append(evaluate_one(name, matcher, int(row["pair_id"]), frame_path,
                                         tile_path, row["scene_type"], pair_dir))
            except Exception as exc:
                print(f"[ERROR] {name}, pair {row['pair_id']}: {exc}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
