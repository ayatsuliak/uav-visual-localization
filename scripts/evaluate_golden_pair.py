from __future__ import annotations

import os
import sys

# Додає кореневу директорію проєкту до шляхів пошуку
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.matching.lightglue import LightGlueMatcher
from src.matching.loftr import LoFTRMatcher
from src.geometry.homography import estimate_homography


def evaluate_method(
    name: str,
    matcher,
    image0: np.ndarray,
    image1: np.ndarray,
) -> dict:

    start = time.perf_counter()

    matches = matcher.match(image0, image1)

    matching_time = time.perf_counter() - start

    points0 = np.asarray(matches["points0"], dtype=np.float32)
    points1 = np.asarray(matches["points1"], dtype=np.float32)

    total_matches = len(points0)

    if total_matches < 4:
        return {
            "method": name,
            "time_sec": matching_time,
            "matches": total_matches,
            "geometric_matches": 0,
            "inlier_ratio": 0.0,
            "reprojection_error": None,
            "success": False,
        }

    geometry = estimate_homography(points0, points1)

    return {
        "method": name,
        "time_sec": matching_time,
        "matches": total_matches,
        "geometric_matches": geometry["inliers"],
        "inlier_ratio": geometry["inlier_ratio"],
        "reprojection_error": geometry["reprojection_error"],
        "success": geometry["success"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frame",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tile",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/results/summary/golden_pair.json"
        ),
    )

    args = parser.parse_args()

    frame = args.frame
    tile = args.tile

    if not frame.exists():
        raise FileNotFoundError(f"Frame not found: {frame}")

    if not tile.exists():
        raise FileNotFoundError(f"Tile not found: {tile}")

    lightglue = LightGlueMatcher()
    loftr = LoFTRMatcher()

    results = []

    results.append(
        evaluate_method(
            "LightGlue",
            lightglue,
            frame,
            tile,
        )
    )

    results.append(
        evaluate_method(
            "LoFTR",
            loftr,
            frame,
            tile,
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()