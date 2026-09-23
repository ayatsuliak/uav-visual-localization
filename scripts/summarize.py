import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import pandas as pd

from src.config import MATCHING_RESULTS_PATH, RESULTS_DIR


NUMERIC_AGG = {
    "time": "mean", "matches": "mean", "inliers": "mean",
    "inlier_ratio": "mean", "reprojection_error": "mean",
    "success": "mean", "tile_selection_score": "mean",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(MATCHING_RESULTS_PATH))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "summary"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = df.groupby("method").agg(pairs=("pair_id", "count"), **{
        f"avg_{key}": (key, func) for key, func in NUMERIC_AGG.items()
    }).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    if "scene_type" in df:
        scene = df.groupby(["scene_type", "method"]).agg(pairs=("pair_id", "count"), **{
            f"avg_{key}": (key, func) for key, func in NUMERIC_AGG.items()
        }).reset_index()
        scene.to_csv(out / "summary_by_scene.csv", index=False)

    print(summary)


if __name__ == "__main__":
    main()
