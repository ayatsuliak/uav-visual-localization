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

Похибка локалізації: ``E = sqrt((x_pred - x_true)^2 + (y_pred - y_true)^2)``.
"""
import math

import numpy as np


def reprojection_error_stats(errors) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "median": None, "rmse": None, "max": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "max": float(values.max())}


def localization_error(predicted_xy, true_xy):
    dx = predicted_xy[0] - true_xy[0]
    dy = predicted_xy[1] - true_xy[1]
    return math.hypot(dx, dy)


def summarize_localization_errors(errors):
    values = [float(x) for x in errors if x is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    values.sort()
    n = len(values)
    p90 = values[min(n - 1, math.ceil(0.9 * n) - 1)]
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    return {"count": n, "mean": sum(values) / n, "median": median, "p90": p90}
