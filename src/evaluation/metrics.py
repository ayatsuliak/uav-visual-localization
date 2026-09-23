import math


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
