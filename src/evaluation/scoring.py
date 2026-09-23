def tile_selection_score(metrics):
    """A transparent ranking score for tile retrieval, not a scientific quality metric.

    Primary ordering is success, then number of inliers, then inlier ratio,
    then lower reprojection error. This avoids the old ad-hoc inliers/error ratio.
    """
    if not metrics["success"]:
        return 0.0
    error = metrics["reprojection_error"]
    error_factor = 1.0 / (1.0 + error) if error is not None else 0.0
    return float(metrics["inliers"] * (0.5 + 0.5 * metrics["inlier_ratio"]) * error_factor)
