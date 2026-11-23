"""Згладжування траєкторії фільтром Калмана (модель сталої швидкості).

Застосовується до готової траєкторії (``trajectory.csv`` трекера) і не впливає
на рішення трекера, тому «сиру» і згладжену траєкторії можна порівняти на тих
самих прогонах.

Стан ``s = [x, y, v_x, v_y]`` (px карти, px/с). Між кадрами (``Δt`` — з
``timestamp``):

    s_k = F s_{k-1} + w,   F = [[I, Δt·I], [0, I]],
    Q = σ_a² · [[Δt⁴/4 · I, Δt³/2 · I], [Δt³/2 · I, Δt² · I]]

(дискретний білий шум прискорення ``σ_a``). Вимірювання — положення кадрів зі
статусом ``TRACKING`` / ``INITIALIZED``: ``z_k = [x, y] + v``, ``R = σ_z² · I``.
Кадри ``INTERPOLATED`` вимірюваннями не вважаються (це екстраполяція трекера),
для них береться прогноз фільтра; кадри ``LOST`` лишаються без положення.
Кожне ``INITIALIZED`` (глобальний пошук після втрати) починає новий сегмент:
фільтр ініціалізується заново, бо попередній стан уже неактуальний.

Дві оцінки:

* ``x_kf, y_kf`` — фільтр Калмана (причинний, лише минулі кадри) — те, що
  система могла б видавати в реальному часі;
* ``x_rts, y_rts`` — згладжувач Рауха–Тунга–Штрібеля (прохід назад по сегменту,
  використовує й майбутні кадри) — лише для офлайн-аналізу польоту.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import (KALMAN_ACCEL_STD_PX_S2, KALMAN_MEAS_STD_PX, MAX_SPEED_PX_PER_S)

MEASURED_STATUSES = ("TRACKING", "INITIALIZED")


@dataclass(frozen=True)
class SmoothingConfig:
    meas_std_px: float = KALMAN_MEAS_STD_PX               # σ_z, px карти
    accel_std_px_s2: float = KALMAN_ACCEL_STD_PX_S2       # σ_a, px карти / с²
    init_speed_std_px_s: float = MAX_SPEED_PX_PER_S       # невизначеність швидкості на старті

    def to_dict(self) -> dict:
        return asdict(self)


def transition(dt: float, accel_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Матриці ``F`` і ``Q`` моделі сталої швидкості для кроку ``dt``."""
    I = np.eye(2)
    F = np.block([[I, dt * I], [np.zeros((2, 2)), I]])
    q = accel_std ** 2
    Q = q * np.block([[dt ** 4 / 4 * I, dt ** 3 / 2 * I], [dt ** 3 / 2 * I, dt ** 2 * I]])
    return F, Q


def _segment(times: np.ndarray, z: np.ndarray, measured: np.ndarray,
             config: SmoothingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Фільтр і RTS для одного сегмента; перший кадр сегмента — вимірювання."""
    n = len(times)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = config.meas_std_px ** 2 * np.eye(2)
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P = np.diag([config.meas_std_px ** 2] * 2 + [config.init_speed_std_px_s ** 2] * 2)
    xs_f, Ps_f = np.zeros((n, 4)), np.zeros((n, 4, 4))
    xs_p, Ps_p, Fs = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4, 4))
    xs_f[0], Ps_f[0], xs_p[0], Ps_p[0], Fs[0] = x, P, x, P, np.eye(4)
    for k in range(1, n):
        F, Q = transition(max(0.0, times[k] - times[k - 1]), config.accel_std_px_s2)
        x, P = F @ x, F @ P @ F.T + Q
        xs_p[k], Ps_p[k], Fs[k] = x, P, F
        if measured[k]:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z[k] - H @ x)
            P = (np.eye(4) - K @ H) @ P
        xs_f[k], Ps_f[k] = x, P
    xs_s = xs_f.copy()
    P_s = Ps_f[-1]
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ Fs[k + 1].T @ np.linalg.inv(Ps_p[k + 1])
        xs_s[k] = xs_f[k] + C @ (xs_s[k + 1] - xs_p[k + 1])
        P_s = Ps_f[k] + C @ (P_s - Ps_p[k + 1]) @ C.T
    return xs_f[:, :2], xs_s[:, :2]


def smooth_trajectory(trajectory: pd.DataFrame, config: SmoothingConfig = SmoothingConfig()
                      ) -> pd.DataFrame:
    """Копія траєкторії з колонками ``x_kf, y_kf, x_rts, y_rts`` (px карти)."""
    out = trajectory.copy()
    for col in ("x_kf", "y_kf", "x_rts", "y_rts"):
        out[col] = np.nan
    times = out["timestamp"].to_numpy(float)
    z = out[["x_map", "y_map"]].to_numpy(float)
    status = out["status"].astype(str).to_numpy()
    measured = np.isin(status, MEASURED_STATUSES) & np.all(np.isfinite(z), axis=1)

    # Сегменти: від вимірювання, що починає трек (перше або INITIALIZED), до наступного старту.
    starts = [i for i in range(len(out)) if measured[i] and
              (status[i] == "INITIALIZED" or not measured[:i].any())]
    for j, start in enumerate(starts):
        stop = starts[j + 1] if j + 1 < len(starts) else len(out)
        idx = np.arange(start, stop)
        kf, rts = _segment(times[idx], z[idx], measured[idx], config)
        out.loc[out.index[idx], ["x_kf", "y_kf"]] = kf
        out.loc[out.index[idx], ["x_rts", "y_rts"]] = rts
    # LOST — положення немає (як і в сирій траєкторії): згладжування не додає покриття.
    lost = ~np.all(np.isfinite(z), axis=1)
    out.loc[lost, ["x_kf", "y_kf", "x_rts", "y_rts"]] = np.nan
    return out


def with_position(trajectory: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Траєкторія, у якій ``x_map, y_map`` замінено на варіант ``raw`` / ``kf`` / ``rts``."""
    if variant == "raw":
        return trajectory
    out = trajectory.copy()
    out["x_map"], out["y_map"] = trajectory[f"x_{variant}"], trajectory[f"y_{variant}"]
    return out
