from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray:

    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    height = max(h0, h1)

    canvas = np.zeros(
        (height, w0 + w1, 3),
        dtype=np.uint8,
    )

    canvas[:h0, :w0] = image0
    canvas[:h1, w0:w0 + w1] = image1

    for i, (p0, p1) in enumerate(zip(points0, points1)):

        x0, y0 = map(int, p0)
        x1, y1 = map(int, p1)

        x1_canvas = x1 + w0

        is_inlier = (
            inlier_mask is not None
            and bool(inlier_mask[i])
        )

        if is_inlier:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(
            canvas,
            (x0, y0),
            3,
            color,
            -1,
        )

        cv2.circle(
            canvas,
            (x1_canvas, y1),
            3,
            color,
            -1,
        )

        cv2.line(
            canvas,
            (x0, y0),
            (x1_canvas, y1),
            color,
            1,
        )

    return canvas


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--frame", required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    tile = cv2.imread(args.tile)

    if frame is None:
        raise RuntimeError(
            f"Cannot read frame: {args.frame}"
        )

    if tile is None:
        raise RuntimeError(
            f"Cannot read tile: {args.tile}"
        )

    # Тут для першої версії можна завантажити
    # збережені відповідності з результатів експерименту.

    data = np.load(
        "experiments/results/raw/golden_pair_matches.npz"
    )

    points0 = data["points0"]
    points1 = data["points1"]

    inlier_mask = data.get("inlier_mask")

    result = draw_matches(
        frame,
        tile,
        points0,
        points1,
        inlier_mask,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()