import cv2
import numpy as np
from pathlib import Path

from src.preprocessing.images import write_image_rgb


def save_match_visualization_from_images(image0, image1, pts0, pts1, mask=None,
                                         output_path="experiments/figures/matches.png",
                                         max_matches=80, seed=42):
    img0 = image0.copy()
    img1 = image1.copy()
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

    pts0, pts1 = np.asarray(pts0), np.asarray(pts1)
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        pts0, pts1 = pts0[mask], pts1[mask]
    if len(pts0) > max_matches:
        idx = np.linspace(0, len(pts0) - 1, max_matches).astype(int)
        pts0, pts1 = pts0[idx], pts1[idx]

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    rng = np.random.default_rng(seed)
    for p0, p1 in zip(pts0, pts1):
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        x0, y0 = map(int, p0)
        x1, y1 = int(p1[0]) + w0, int(p1[1])
        cv2.circle(canvas, (x0, y0), 3, color, -1)
        cv2.circle(canvas, (x1, y1), 3, color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image_rgb(output_path, canvas)  # зображення в пам'яті - RGB
    print(f"Saved match visualization to: {output_path}")
