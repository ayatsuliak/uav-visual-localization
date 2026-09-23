import cv2
import numpy as np

from src.config import MIN_INLIERS, RANSAC_THRESHOLD


def estimate_homography(pts0, pts1, ransac_threshold: float = RANSAC_THRESHOLD):
    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)
    if len(pts0) < 4 or len(pts1) < 4:
        return None, None
    return cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_threshold)


def calculate_reprojection_error(pts0, pts1, H, mask):
    if H is None or mask is None or not np.any(mask):
        return None
    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)
    projected = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected[mask] - pts1[mask], axis=1)
    return float(errors.mean()) if len(errors) else None


def estimate_homography_metrics(pts0, pts1, ransac_threshold: float = RANSAC_THRESHOLD,
                                min_inliers: int = MIN_INLIERS):
    matches = len(pts0)
    H, mask = estimate_homography(pts0, pts1, ransac_threshold)
    if H is None or mask is None:
        return {"matches": matches, "inliers": 0, "inlier_ratio": 0.0,
                "reprojection_error": None, "success": False, "H": None, "mask": None}

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())
    error = calculate_reprojection_error(pts0, pts1, H, mask)
    return {"matches": matches, "inliers": inliers,
            "inlier_ratio": inliers / max(1, matches),
            "reprojection_error": error, "success": inliers >= min_inliers,
            "H": H, "mask": mask}
