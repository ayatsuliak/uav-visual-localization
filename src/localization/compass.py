"""Курс із магнітометра (компаса) як необов'язкова підказка для компенсації курсу.

Магнітометр входить до контролера польоту, працює в реальному часі та не залежить
від GNSS. Його показ ``ψ`` (курс камери, ° за годинниковою стрілкою від півночі)
переводиться в кут «кадр -> карта», яким трекер повертає кадр перед зіставленням:

    θ = ψ + Δ + b + n,

де ``Δ = COMPASS_OFFSET_DEG`` — стала поправка, відома заздалегідь (магнітне схилення
та кут зближення меридіанів сітки карти), ``b`` — зміщення і ``n ~ N(0, σ²)`` — шум,
якими в експериментах імітується неідеальний компас.

У наборах даних показ компаса береться з журналу польоту (``scripts/make_compass.py``),
бо окремого запису магнітометра немає; положення GNSS з журналу при цьому не
використовуються.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _wrap(deg: float) -> float:
    wrapped = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


@dataclass(frozen=True)
class CompassConfig:
    offset_deg: float = 0.0          # Δ: схилення + зближення меридіанів
    bias_deg: float = 0.0            # b: імітоване систематичне зміщення
    noise_deg: float = 0.0           # σ: імітований випадковий шум
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class CompassSensor:
    """``compass(frame_id, t) -> θ`` (кут «кадр -> карта», °) або ``None``, якщо показу немає."""

    def __init__(self, headings: dict[str, float], config: CompassConfig = CompassConfig()):
        self.config = config
        ids = sorted(headings)
        rng = np.random.default_rng(config.seed)
        noise = rng.normal(0.0, config.noise_deg, len(ids)) if config.noise_deg > 0 else np.zeros(len(ids))
        self._yaw = {fid: _wrap(headings[fid] + config.offset_deg + config.bias_deg + float(e))
                     for fid, e in zip(ids, noise)}

    @classmethod
    def from_csv(cls, path: str | Path, config: CompassConfig = CompassConfig()) -> "CompassSensor":
        df = pd.read_csv(path)
        return cls(dict(zip(df["frame_id"], df["heading_deg"].astype(float))), config)

    def __call__(self, frame_id: str, t: float | None = None) -> float | None:
        return self._yaw.get(frame_id)
