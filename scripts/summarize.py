"""Агрегує результати scripts/evaluate.py за методом (і типом сцени).

Приклад:
    python scripts/summarize.py
    python scripts/summarize.py --input experiments/results/raw/ws15.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from src.config import MATCHING_RESULTS_PATH, RESULTS_DIR

# Середні значення беруться лише за успішними парами для геометричних величин
# (для неуспішних вони не визначені), а для кількісних і часових — за всіма.
ALL_PAIRS = ["matches", "inliers", "inlier_ratio", "time_extract_frame", "time_extract_tile",
             "time_match", "time_matcher_total", "time_ransac"]
SUCCESS_ONLY = ["reproj_mean", "reproj_median", "h_scale"]


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    groups = df.groupby(keys)
    table = groups.agg(pairs=("pair_id", "count"), success_rate=("success", "mean"),
                       **{f"mean_{c}": (c, "mean") for c in ALL_PAIRS if c in df},
                       **{f"median_{c}": (c, "median") for c in ("inliers", "time_matcher_total")
                          if c in df})
    ok = df[df["success"].astype(bool)]
    if len(ok):
        table = table.join(ok.groupby(keys).agg(
            **{f"mean_{c}_success": (c, "mean") for c in SUCCESS_ONLY if c in ok}))
    return table.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(MATCHING_RESULTS_PATH))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "summary"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    summary = aggregate(df, ["method"])
    summary.to_csv(out / f"{stem}_summary.csv", index=False)
    if "scene_type" in df and df["scene_type"].notna().any():
        aggregate(df, ["scene_type", "method"]).to_csv(out / f"{stem}_by_scene.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.T)
    print(f"Saved: {out / f'{stem}_summary.csv'}")


if __name__ == "__main__":
    main()
