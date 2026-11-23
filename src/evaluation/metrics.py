"""Чітко визначені метрики зіставлення та локалізації.

Метрики пари «кадр — тайл» (усі відстані — у пікселях карти):

* ``matches`` — кількість відповідностей, які повернув метод;
* ``inliers`` — кількість геометрично узгоджених відповідностей після RANSAC
  (похибка репроєкції ``<= RANSAC_THRESHOLD``);
* ``inlier_ratio`` — частка геометрично узгоджених відповідностей ``inliers / matches``;
* ``reproj_*`` — статистика прямої похибки репроєкції ``||H·p_frame - p_tile||``
  серед геометрично узгоджених відповідностей. За побудовою вона не перевищує
  порогу RANSAC, тому характеризує точність відповідностей, а не успішність;
* ``success`` — див. ``src.geometry.homography.is_successful``.

Похибка локалізації відносно еталону (ground truth), px карти:
``E = sqrt((x_pred - x_true)^2 + (y_pred - y_true)^2)``; зведена статистика —
:func:`error_statistics`.
"""
import math

import numpy as np
import pandas as pd

SUCCESS_THRESHOLDS_PX = (10, 25, 50)


def reprojection_error_stats(errors) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "median": None, "rmse": None, "max": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "max": float(values.max())}


def localization_error(predicted_xy, true_xy) -> float:
    """``E = sqrt((x_pred - x_true)^2 + (y_pred - y_true)^2)``, px карти."""
    return math.hypot(predicted_xy[0] - true_xy[0], predicted_xy[1] - true_xy[1])


def trajectory_errors(trajectory: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Поточкова похибка ``e_i = ||P_pred,i - P_gt,i||`` для кадрів, що мають еталон.

    Результат — рядки ``ground_truth`` (лише кадри з еталоном) з колонками
    ``x_map, y_map, status`` траєкторії і ``error``; ``error = NaN``, якщо для
    кадру немає оцінки положення (статус LOST або кадр не оброблено).
    """
    cols = [c for c in ("frame_id", "x_map", "y_map", "status") if c in trajectory]
    merged = ground_truth.merge(trajectory[cols], on="frame_id", how="left")
    merged["error"] = np.hypot(merged["x_map"] - merged["x_true"],
                               merged["y_map"] - merged["y_true"])
    return merged


def error_statistics(errors, thresholds=SUCCESS_THRESHOLDS_PX) -> dict:
    """Зведена статистика похибки локалізації (px карти).

    ``errors`` — похибки всіх кадрів з еталоном; ``NaN`` означає, що система
    не видала положення. Такі кадри не входять у mean/median/RMSE/max, але
    зараховуються як неуспішні в ``success@K`` і зменшують ``coverage``:

    * ``coverage = N_pred / N``;
    * ``RMSE = sqrt(mean(e_i^2))``;
    * ``success@K = |{i : e_i <= K}| / N`` (N — усі кадри з еталоном).
    """
    values = np.asarray(errors, dtype=np.float64)
    n = len(values)
    predicted = values[np.isfinite(values)]
    stats = {"n_frames": n, "n_predicted": int(len(predicted)),
             "coverage": len(predicted) / n if n else None}
    if len(predicted):
        stats.update({"mean": float(predicted.mean()), "median": float(np.median(predicted)),
                      "rmse": float(np.sqrt(np.mean(predicted ** 2))),
                      "p90": float(np.percentile(predicted, 90)),
                      "p95": float(np.percentile(predicted, 95)),
                      "max": float(predicted.max())})
    else:
        stats.update({k: None for k in ("mean", "median", "rmse", "p90", "p95", "max")})
    for k in thresholds:
        stats[f"success@{k:g}"] = float(np.sum(predicted <= k) / n) if n else None
    return stats
