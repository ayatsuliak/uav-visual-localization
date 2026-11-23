"""Локалізація послідовності кадрів: локальний трекінг і часова узгодженість (Етап 4).

Стани трекера:

* ``LOST`` — холодний старт або втрата супроводу. Виконується глобальний
  пошук (вся карта, ROI або проріджена сітка — як на Етапі 3). Якщо переможець
  має бал ``>= init_min_score``, позиція приймається (статус ``INITIALIZED``)
  і трекер переходить у ``TRACKING``.
* ``TRACKING`` — пошук лише серед тайлів вікна ``(2r+1) × (2r+1)`` (для r = 1 —
  3×3, максимум 9 тайлів) навколо базового тайла. Базовий тайл — той, чий
  центр найближчий до прогнозованої позиції (екстраполяція за сталою
  швидкістю). Якщо ``max_misses`` кадрів поспіль немає прийнятої оцінки,
  трекер переходить у ``LOST``.

Кінематична перевірка. Для оцінки ``P_k`` і останньої прийнятої позиції
``P_last`` (час ``t_last``):

    ||P_k - P_last|| <= D_max(Δt),   D_max = V_max · Δt + D_margin,   Δt = t_k - t_last.

Порівняння виконується з останньою *прийнятою* позицією, тому після кількох
екстрапольованих кадрів допустимий радіус зростає разом із часом. Серед
кандидатів, що пройшли фільтр Етапу 3, обирається найкращий за балом, який
проходить і кінематичну перевірку. Якщо такого немає, позиція екстраполюється:

    P^_k = P_last + v · (t_k - t_last),   v = (P_last - P_prev) / (t_last - t_prev),

(для сталого Δt це ``P_{k-1} + (P_{k-1} - P_{k-2})``); ``|v|`` обмежується ``V_max``.

Компенсація курсу. SuperPoint/LightGlue і LoFTR не інваріантні до великих
поворотів, а під час розвороту БпЛА кадр повертається відносно карти. Кут
«кадр -> карта» відомий з гомографії прийнятих кадрів (``h_rotation_deg``),
тому перед зіставленням кадр повертається на прогнозований курс
(екстраполяція за кутовою швидкістю, обмеженою ``max_yaw_rate_deg_per_s``).
Якщо з прогнозованим курсом тайл не знайдено, у стані ``TRACKING``
перебираються зсуви ``yaw_retry_offsets_deg``; у стані ``LOST`` кожен наступний
глобальний пошук використовує черговий зсув з ``global_yaw_cycle_deg``.

Вікно за прогнозом (``predicted_window``, типово вимкнено). У стані ``TRACKING``
відомі прогнозовані позиція і курс, а контур огляду між кадрами майже не
змінюється. Тому спершу кадр зіставляється з однією ділянкою карти, що охоплює
контур останнього прийнятого кадру, зсунутий у прогнозовану позицію, із запасом
``window_margin_px``. Кадр потрапляє у вікно повністю (а не обрізається краєм
тайла), і замість до 9 зіставлень виконується одне. Якщо у вікні оцінку не
прийнято (фільтр Етапу 3 або кінематична перевірка), виконується звичайний пошук
у 3×3 тайлах з усіма гіпотезами курсу; ``search_mode`` = ``window`` позначає
кадри, прийняті з вікна.

Курс із компаса (``compass``, типово немає). Магнітометр працює без GNSS у реальному
часі. Якщо компас задано, глобальний пошук виконується з курсом компаса замість
циклу гіпотез, а в стані ``TRACKING`` курс компаса перевіряється одразу після
прогнозованого, якщо вони розходяться більше ніж на ``COMPASS_MIN_DIFF_DEG``.

Уточнення після глобального пошуку (``refine_after_global``, типово вимкнено).
Переможця глобального пошуку повторно зіставлено з ділянкою карти навколо його
контуру огляду (запас ``refine_margin_px``). Уточнена оцінка приймається, якщо
проходить фільтр Етапу 3, має не менше узгоджених відповідностей і зсуває центр
не далі ніж на ``refine_max_shift_px`` (уточнення, а не нова гіпотеза).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from src.config import (COMPASS_MIN_DIFF_DEG, INIT_MIN_SCORE, JUMP_MARGIN_PX, MAX_MISSES, MAX_SPEED_PX_PER_S,
                        MAX_YAW_RATE_DEG_PER_S, NEIGHBORHOOD_RADIUS, PREDICTED_WINDOW,
                        REFINE_AFTER_GLOBAL, REFINE_MARGIN_PX, REFINE_MAX_SHIFT_PX,
                        WINDOW_MARGIN_PX, WINDOW_MAX_SIDE_PX, YAW_RETRY_OFFSETS_DEG,
                        GLOBAL_YAW_CYCLE_DEG)
from src.localization.coordinates import footprint_on_map
from src.localization.search import grid_indices


class TrackerState(str, Enum):
    LOST = "LOST"
    TRACKING = "TRACKING"


class Status(str, Enum):
    """Статус кадру в trajectory.csv."""
    INITIALIZED = "INITIALIZED"      # позиція з глобального пошуку
    TRACKING = "TRACKING"            # позиція з локального пошуку, пройшла кінематичну перевірку
    INTERPOLATED = "INTERPOLATED"    # позиція екстрапольована (немає прийнятої оцінки)
    LOST = "LOST"                    # позиції немає


class Event(str, Enum):
    """Що сталося на кадрі (деталізація статусу)."""
    OK = "OK"
    OUTLIER_REJECTED = "OUTLIER_REJECTED"    # оцінку відкинуто кінематичною перевіркою
    NO_MATCH = "NO_MATCH"                    # жоден тайл не пройшов фільтр Етапу 3
    LOW_SCORE = "LOW_SCORE"                  # глобальний пошук: бал < init_min_score


@dataclass(frozen=True)
class TrackerConfig:
    neighborhood_radius: int = NEIGHBORHOOD_RADIUS
    max_misses: int = MAX_MISSES
    max_speed_px_per_s: float = MAX_SPEED_PX_PER_S
    jump_margin_px: float = JUMP_MARGIN_PX
    init_min_score: float = INIT_MIN_SCORE
    compensate_yaw: bool = True
    max_yaw_rate_deg_per_s: float = MAX_YAW_RATE_DEG_PER_S
    yaw_retry_offsets_deg: tuple[float, ...] = YAW_RETRY_OFFSETS_DEG
    global_yaw_cycle_deg: tuple[float, ...] = GLOBAL_YAW_CYCLE_DEG
    predicted_window: bool = PREDICTED_WINDOW
    window_margin_px: float = WINDOW_MARGIN_PX
    window_max_side_px: int = WINDOW_MAX_SIDE_PX
    refine_after_global: bool = REFINE_AFTER_GLOBAL
    refine_margin_px: float = REFINE_MARGIN_PX
    refine_max_shift_px: float = REFINE_MAX_SHIFT_PX

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Сусідні тайли
# ---------------------------------------------------------------------------
def tile_grid(tiles: pd.DataFrame) -> pd.DataFrame:
    """Копія ``tiles`` з номерами стовпця/рядка сітки (``col``, ``row``)."""
    col, row = grid_indices(tiles)
    return tiles.assign(col=col.values, row=row.values)


def neighbor_tiles(tiles: pd.DataFrame, base_tile_id: str, radius: int = 1) -> pd.DataFrame:
    """Базовий тайл і його сусіди у вікні ``(2r+1) × (2r+1)`` сітки (біля краю — менше)."""
    grid = tile_grid(tiles)
    base = grid.loc[grid["tile_id"] == base_tile_id]
    if base.empty:
        raise KeyError(f"Unknown tile {base_tile_id}")
    c, r = int(base["col"].iloc[0]), int(base["row"].iloc[0])
    mask = ((grid["col"] - c).abs() <= radius) & ((grid["row"] - r).abs() <= radius)
    return grid.loc[mask, tiles.columns].reset_index(drop=True)


def nearest_tile(tiles: pd.DataFrame, point_xy) -> str:
    """Тайл, чий центр найближчий до точки карти (px)."""
    cx = tiles["x_origin"] + (tiles["width"] - 1) / 2.0
    cy = tiles["y_origin"] + (tiles["height"] - 1) / 2.0
    distance = np.hypot(cx - point_xy[0], cy - point_xy[1])
    return str(tiles["tile_id"].iloc[int(np.argmin(distance.values))])


# ---------------------------------------------------------------------------
# Кінематика
# ---------------------------------------------------------------------------
def max_displacement(dt: float, config: TrackerConfig) -> float:
    """``D_max = V_max · Δt + D_margin``."""
    return config.max_speed_px_per_s * max(0.0, dt) + config.jump_margin_px


def passes_kinematic_gate(prev_xy, prev_t: float, xy, t: float,
                          config: TrackerConfig) -> tuple[bool, float]:
    """Повертає ``(ok, distance)``; ``ok`` — зміщення не перевищує ``D_max``."""
    distance = float(math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1]))
    return distance <= max_displacement(t - prev_t, config), distance


def extrapolate(fixes: list[tuple[float, tuple[float, float]]], t: float,
                max_speed: float | None = None) -> tuple[float, float] | None:
    """Позиція в момент ``t`` за двома останніми прийнятими фіксами ``(t_i, (x_i, y_i))``.

    Один фікс — позиція утримується; немає фіксів — ``None``.
    """
    if not fixes:
        return None
    t1, p1 = fixes[-1]
    if len(fixes) < 2:
        return float(p1[0]), float(p1[1])
    t0, p0 = fixes[-2]
    if t1 <= t0:
        return float(p1[0]), float(p1[1])
    v = (np.asarray(p1, float) - np.asarray(p0, float)) / (t1 - t0)
    speed = float(np.hypot(*v))
    if max_speed is not None and speed > max_speed:
        v *= max_speed / speed
    p = np.asarray(p1, float) + v * (t - t1)
    return float(p[0]), float(p[1])


def map_window(map_size_wh: tuple[int, int], footprint, margin: float, multiple: int = 8,
               max_side: float | None = None) -> tuple[int, int, int, int] | None:
    """Ділянка карти ``(x0, y0, x1, y1)`` (``x1, y1`` не включно), що охоплює контур
    ``footprint`` із запасом ``margin``; обрізана межами карти, сторони кратні
    ``multiple`` (LoFTR). ``None`` — контур невизначений, вікно порожнє або більше
    за ``max_side``.
    """
    pts = np.asarray(footprint, dtype=float).reshape(-1, 2)
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return None
    W, H = map_size_wh
    x0 = max(0, int(math.floor(pts[:, 0].min() - margin)))
    y0 = max(0, int(math.floor(pts[:, 1].min() - margin)))
    x1 = min(int(W), int(math.ceil(pts[:, 0].max() + margin)) + 1)
    y1 = min(int(H), int(math.ceil(pts[:, 1].max() + margin)) + 1)
    w, h = (x1 - x0) // multiple * multiple, (y1 - y0) // multiple * multiple
    if w < multiple or h < multiple:
        return None
    if max_side is not None and max(w, h) > max_side:
        return None
    return x0, y0, x0 + w, y0 + h


def wrap_angle(deg: float) -> float:
    """Кут у діапазоні (-180, 180]."""
    wrapped = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def extrapolate_yaw(yaws: list[tuple[float, float]], t: float,
                    max_rate: float | None = None) -> float | None:
    """Курс у момент ``t`` за двома останніми ``(t_i, yaw_i)``, °; перехід через ±180° враховано."""
    if not yaws:
        return None
    t1, a1 = yaws[-1]
    if len(yaws) < 2 or yaws[-1][0] <= yaws[-2][0]:
        return wrap_angle(a1)
    t0, a0 = yaws[-2]
    rate = wrap_angle(a1 - a0) / (t1 - t0)
    if max_rate is not None:
        rate = float(np.clip(rate, -max_rate, max_rate))
    return wrap_angle(a1 + rate * (t - t1))


# ---------------------------------------------------------------------------
# Трекер
# ---------------------------------------------------------------------------
@dataclass
class TrackStep:
    frame_id: str
    timestamp: float
    status: Status
    event: Event
    search_mode: str                          # "global" | "local" | "window"
    tiles_checked: int
    tiles_passed: int
    tile_id: str | None = None
    x_map: float | None = None
    y_map: float | None = None
    x_raw: float | None = None                # відкинута оцінка (OUTLIER_REJECTED)
    y_raw: float | None = None
    jump_px: float | None = None              # зміщення від останньої прийнятої позиції
    max_jump_px: float | None = None          # D_max для цього кадру
    inliers: int | None = None
    inlier_ratio: float | None = None
    reproj_error: float | None = None
    score: float | None = None
    h_rotation_deg: float | None = None       # поворот кадру відносно карти (≈ курс БпЛА)
    h_scale: float | None = None              # px карти на px кадру (≈ висота польоту)
    yaw_used: float | None = None             # поворот кадру перед зіставленням (компенсація)
    attempts: int = 0                         # кількість пошуків (гіпотез курсу) на кадрі
    refined: bool = False                     # положення уточнено повторним зіставленням
    compass_yaw: float | None = None          # кут «кадр -> карта» з компаса (якщо є)
    footprint: list | None = None             # 4 вершини контуру огляду (прийняті кадри)
    process_time: float = 0.0
    search_time: float = 0.0
    extra: dict = field(default_factory=dict)

    CSV_COLUMNS = ["frame_id", "timestamp", "status", "tile_id", "x_map", "y_map", "inliers",
                   "inlier_ratio", "reproj_error", "process_time", "event", "search_mode",
                   "tiles_checked", "tiles_passed", "score", "jump_px", "max_jump_px",
                   "x_raw", "y_raw", "h_rotation_deg", "h_scale", "yaw_used",
                   "attempts", "search_time", "refined", "compass_yaw"]

    def to_row(self) -> dict:
        row = {k: getattr(self, k) for k in self.CSV_COLUMNS}
        row["status"], row["event"] = self.status.value, self.event.value
        return row


class SequenceTracker:
    """Послідовна локалізація кадрів.

    ``search`` — об'єкт з методом ``search(frame, tiles) -> SearchResult``
    (``src.localization.search.TileSearch``); ``global_tiles`` — пул для
    глобального пошуку (вся карта, ROI або проріджена сітка). ``map_rgb`` —
    повна карта; потрібна лише для ``predicted_window`` / ``refine_after_global``
    (тоді ``search`` має також метод ``search_regions``).
    """

    def __init__(self, search, tiles: pd.DataFrame, global_tiles: pd.DataFrame | None = None,
                 config: TrackerConfig = TrackerConfig(), map_rgb: np.ndarray | None = None,
                 compass=None):
        if (config.predicted_window or config.refine_after_global) and map_rgb is None:
            raise ValueError("predicted_window / refine_after_global need the full map (map_rgb)")
        self.map_rgb = map_rgb
        self.compass = compass                  # (frame_id, t) -> θ, ° або None
        self.search = search
        self.tiles = tiles.reset_index(drop=True)
        self.global_tiles = (global_tiles if global_tiles is not None else tiles
                             ).reset_index(drop=True)
        self.config = config
        self.state = TrackerState.LOST
        self.fixes: list[tuple[float, tuple[float, float]]] = []   # прийняті позиції
        self.yaws: list[tuple[float, float]] = []                   # курс прийнятих кадрів
        self.misses = 0
        self.global_attempts = 0                                    # глобальних спроб поспіль
        self.footprint_offset: np.ndarray | None = None             # контур мінус центр (останній)

    def reset(self) -> None:
        self.state, self.fixes, self.yaws, self.misses = TrackerState.LOST, [], [], 0
        self.global_attempts, self.footprint_offset = 0, None

    def predicted_yaw(self, t: float) -> float:
        """Очікуваний кут «кадр -> карта»; 0 — без компенсації (невідомо або вимкнено)."""
        if not self.config.compensate_yaw:
            return 0.0
        yaw = extrapolate_yaw(self.yaws, t, self.config.max_yaw_rate_deg_per_s)
        return 0.0 if yaw is None else yaw

    def _yaw_hypotheses(self, t: float, retries: bool, compass_yaw: float | None = None
                        ) -> list[float]:
        base = self.predicted_yaw(t)
        hypotheses = [base]
        if (compass_yaw is not None and self.config.compensate_yaw
                and abs(wrap_angle(compass_yaw - base)) > COMPASS_MIN_DIFF_DEG):
            hypotheses.append(wrap_angle(compass_yaw))
        if retries and self.config.compensate_yaw:
            hypotheses += [wrap_angle(base + o) for o in self.config.yaw_retry_offsets_deg]
        return hypotheses

    def _compass_yaw(self, frame_id: str, t: float) -> float | None:
        return None if self.compass is None else self.compass(frame_id, t)

    def predicted_position(self, t: float) -> tuple[float, float] | None:
        return extrapolate(self.fixes, t, self.config.max_speed_px_per_s)

    def local_pool(self, t: float) -> pd.DataFrame:
        base = nearest_tile(self.tiles, self.predicted_position(t))
        return neighbor_tiles(self.tiles, base, self.config.neighborhood_radius)

    def step(self, frame: np.ndarray, frame_id: str, t: float) -> TrackStep:
        start = time.perf_counter()
        if self.state == TrackerState.LOST:
            step = self._global_step(frame, frame_id, t)
        else:
            step = self._local_step(frame, frame_id, t)
        step.process_time = time.perf_counter() - start
        return step

    # --- LOST: глобальний пошук -------------------------------------------
    def _global_step(self, frame, frame_id, t) -> TrackStep:
        # Глобальний пошук дорогий, тому одна гіпотеза курсу на кадр: останній відомий
        # курс (або 0) плюс зсув, що циклічно змінюється від кадру до кадру
        # (0°, 90°, 180°, -90°), — так після розвороту курс рано чи пізно буде вгадано.
        compass_yaw = self._compass_yaw(frame_id, t)
        yaw = self._yaw_hypotheses(t, retries=False)[0]
        cycle = self.config.global_yaw_cycle_deg if self.config.compensate_yaw else ()
        if compass_yaw is not None and self.config.compensate_yaw:
            yaw = wrap_angle(compass_yaw)           # курс відомий: без перебору гіпотез
        elif cycle:
            yaw = wrap_angle(yaw + cycle[self.global_attempts % len(cycle)])
        self.global_attempts += 1
        s0 = time.perf_counter()
        result = self.search.search(frame, self.global_tiles, rotation_deg=yaw)
        step = TrackStep(frame_id, t, Status.LOST, Event.NO_MATCH, "global",
                         tiles_checked=len(self.global_tiles), tiles_passed=len(result.passed),
                         search_time=time.perf_counter() - s0, yaw_used=yaw, attempts=1,
                         compass_yaw=compass_yaw)
        winner = result.winner
        if winner is None:
            return step
        if winner.evaluation.score < self.config.init_min_score:
            step.event = Event.LOW_SCORE
            self._fill_candidate(step, winner)
            step.x_raw, step.y_raw = winner.center_map
            step.x_map = step.y_map = None
            return step
        if self.config.refine_after_global:
            winner = self._refine(frame, winner, yaw, step)
        self._accept(step, winner, t, Status.INITIALIZED)
        self.state, self.global_attempts = TrackerState.TRACKING, 0
        return step

    # --- TRACKING: локальний пошук ----------------------------------------
    def _local_step(self, frame, frame_id, t) -> TrackStep:
        pool = self.local_pool(t)
        last_t, last_xy = self.fixes[-1]
        step = TrackStep(frame_id, t, Status.INTERPOLATED, Event.NO_MATCH, "local",
                         tiles_checked=0, tiles_passed=0)
        step.max_jump_px = max_displacement(t - last_t, self.config)
        s0 = time.perf_counter()
        rejected = None                          # найкращий кінематичний викид серед спроб

        # Спроби по черзі: (режим, курс, пошук, кількість ділянок). Спершу — одне вікно
        # навколо прогнозованого контуру (якщо ввімкнено), далі тайли 3×3 з
        # прогнозованим курсом, потім інші гіпотези курсу (розворот БпЛА між кадрами).
        attempts = []
        window = self._predicted_window(t)
        if window:
            attempts.append(("window", self.predicted_yaw(t),
                             lambda yaw: self.search.search_regions(frame, window, yaw), 1))
        step.compass_yaw = self._compass_yaw(frame_id, t)
        for yaw in self._yaw_hypotheses(t, retries=True, compass_yaw=step.compass_yaw):
            attempts.append(("local", yaw,
                             lambda yaw: self.search.search(frame, pool, rotation_deg=yaw),
                             len(pool)))
        for mode, yaw, run_search, size in attempts:
            result = run_search(yaw)
            step.attempts += 1
            step.tiles_checked += size
            step.tiles_passed += len(result.passed)
            # Найкращий за балом кандидат, що проходить кінематичну перевірку.
            for candidate in result.passed:
                ok, distance = passes_kinematic_gate(last_xy, last_t, candidate.center_map, t,
                                                     self.config)
                if ok:
                    self._accept(step, candidate, t, Status.TRACKING)
                    step.jump_px, step.yaw_used, step.search_mode = distance, yaw, mode
                    step.search_time = time.perf_counter() - s0
                    return step
            if result.passed and rejected is None:
                rejected = (result.passed[0], yaw)
        step.search_time = time.perf_counter() - s0

        if rejected is not None:                 # усі кандидати — кінематичні викиди
            best, step.yaw_used = rejected
            step.event = Event.OUTLIER_REJECTED
            self._fill_candidate(step, best)
            step.x_raw, step.y_raw = best.center_map
            step.jump_px = passes_kinematic_gate(last_xy, last_t, best.center_map, t,
                                                 self.config)[1]
        self.misses += 1
        step.x_map, step.y_map = self.predicted_position(t)
        if self.misses >= self.config.max_misses:
            self.state = TrackerState.LOST
        return step

    # --- Вікна карти -------------------------------------------------------
    def _region(self, footprint, margin: float, max_side, name: str) -> list:
        """Ділянка карти навколо контуру як ``[(id, image, (x0, y0))]`` або ``[]``."""
        H, W = self.map_rgb.shape[:2]
        box = map_window((W, H), footprint, margin, max_side=max_side)
        if box is None:
            return []
        x0, y0, x1, y1 = box
        image = np.ascontiguousarray(self.map_rgb[y0:y1, x0:x1])
        return [(f"{name}@{x0},{y0}", image, (x0, y0))]

    def _predicted_window(self, t: float) -> list:
        """Вікно навколо контуру останнього прийнятого кадру, зсунутого в прогноз."""
        if not self.config.predicted_window or self.footprint_offset is None:
            return []
        predicted = np.asarray(self.predicted_position(t), dtype=float)
        return self._region(predicted + self.footprint_offset, self.config.window_margin_px,
                            self.config.window_max_side_px, "window")

    def _refine(self, frame, candidate, yaw: float, step: TrackStep):
        """Повторне зіставлення з ділянкою карти навколо контуру ``candidate``."""
        regions = self._region(self._candidate_footprint(candidate),
                               self.config.refine_margin_px, None, "refine")
        if not regions:
            return candidate
        result = self.search.search_regions(frame, regions, yaw)
        step.attempts += 1
        refined = result.winner
        if refined is None or refined.metrics.inliers < candidate.metrics.inliers:
            return candidate
        (x0, y0), (x1, y1) = candidate.center_map, refined.center_map
        if math.hypot(x1 - x0, y1 - y0) > self.config.refine_max_shift_px:
            return candidate
        step.refined = True
        return refined

    # --- Службові ---------------------------------------------------------
    @staticmethod
    def _candidate_footprint(candidate) -> np.ndarray:
        """Контур огляду кадру на карті за гомографією кандидата (4 вершини, px карти)."""
        pair = candidate.pair
        return footprint_on_map(pair.frame.original_size, pair.homography.H,
                                candidate.tile_origin)

    @staticmethod
    def _fill_candidate(step: TrackStep, candidate) -> None:
        m = candidate.metrics
        step.tile_id, step.inliers = candidate.tile_id, m.inliers
        step.inlier_ratio, step.reproj_error = m.inlier_ratio, m.reproj_mean
        step.score = candidate.evaluation.score
        d = candidate.pair.diagnostics
        if d is not None:
            step.h_rotation_deg, step.h_scale = d.rotation_deg, d.scale

    def _accept(self, step: TrackStep, candidate, t: float, status: Status) -> None:
        self._fill_candidate(step, candidate)
        step.status, step.event = status, Event.OK
        step.x_map, step.y_map = candidate.center_map
        step.footprint = self._candidate_footprint(candidate).tolist()
        self.footprint_offset = (np.asarray(step.footprint, dtype=float)
                                 - np.array([step.x_map, step.y_map], dtype=float))
        self.fixes.append((t, (step.x_map, step.y_map)))
        if step.h_rotation_deg is not None:
            self.yaws.append((t, step.h_rotation_deg))
        self.misses = 0
