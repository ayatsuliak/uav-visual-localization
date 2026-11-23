"""Географічна прив'язка карти: WGS 84 -> UTM -> пікселі карти.

Використовується ЛИШЕ для побудови еталона з журналу польоту (GNSS) і для
переведення похибок у метри. Система локалізації цих даних не отримує.

Пікселі — у конвенції центрів пікселів (як у всьому проєкті): піксель з
індексом ``i`` має центр у координаті ``i``. World file (``.pgw``) за
визначенням задає координати центра лівого верхнього пікселя, тому
``px = (X - C) / A``, ``py = (Y - F) / E``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014
UTM_K0 = 0.9996


def utm_zone_from_epsg(epsg: str | int) -> tuple[int, bool]:
    """``EPSG:326zz`` -> (zz, північна півкуля), ``EPSG:327zz`` -> (zz, південна)."""
    code = int(str(epsg).upper().replace("EPSG:", ""))
    if 32601 <= code <= 32660:
        return code - 32600, True
    if 32701 <= code <= 32760:
        return code - 32700, False
    raise ValueError(f"Not a WGS 84 / UTM CRS: {epsg}")


def utm_forward(lon_deg, lat_deg, zone: int, north: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Пряма проєкція UTM (ряди Крюгера, точність ~мм у межах зони)."""
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    a, e2, k0 = WGS84_A, WGS84_E2, UTM_K0
    ep2 = e2 / (1 - e2)
    lon0 = math.radians(-183 + 6 * zone)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    t = np.tan(lat) ** 2
    c = ep2 * np.cos(lat) ** 2
    A = np.cos(lat) * (lon - lon0)
    m = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))
    x = k0 * n * (A + (1 - t + c) * A ** 3 / 6
                  + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * A ** 5 / 120) + 500000.0
    y = k0 * (m + n * np.tan(lat) * (A ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * A ** 4 / 24
                                    + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2) * A ** 6 / 720))
    if not north:
        y = y + 10_000_000.0
    return x, y


@dataclass(frozen=True)
class WorldFile:
    """Афінна прив'язка ``X = A·px + B·py + C``, ``Y = D·px + E·py + F`` (центри пікселів)."""
    A: float
    D: float
    B: float
    E: float
    C: float
    F: float

    @classmethod
    def read(cls, path: str | Path) -> "WorldFile":
        values = [float(v) for v in Path(path).read_text().split()]
        if len(values) != 6:
            raise ValueError(f"{path}: expected 6 numbers, got {len(values)}")
        return cls(*values)

    def to_pixel(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        M = np.array([[self.A, self.B], [self.D, self.E]])
        inv = np.linalg.inv(M)
        dx, dy = np.asarray(x, float) - self.C, np.asarray(y, float) - self.F
        return inv[0, 0] * dx + inv[0, 1] * dy, inv[1, 0] * dx + inv[1, 1] * dy

    @property
    def gsd(self) -> float:
        """Розмір пікселя, одиниці CRS (м для UTM)."""
        return float(math.sqrt(abs(self.A * self.E - self.B * self.D)))
