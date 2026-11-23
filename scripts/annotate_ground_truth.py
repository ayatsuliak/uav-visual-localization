"""Етап 5. Ручна розмітка еталонних координат (ground truth) опорних кадрів.

Для кожного опорного кадру вказуються щонайменше 4 пари відповідних точок
«кадр ↔ карта» на рівні землі; за ними обчислюється гомографія і центр кадру
на карті. Результат не залежить від LightGlue/LoFTR (лише ручні точки).

Керування у вікні:
    ЛКМ по кадру, потім ЛКМ по карті  — додати пару точок
    ПКМ або U       — видалити останню точку / пару
    W A S D         — зсунути карту;  + / -  — наблизити / віддалити карту
    Enter або N     — зберегти кадр і перейти до наступного (потрібно >= 4 пари)
    Space           — пропустити кадр;  B — повернутися до попереднього
    C               — очистити точки кадру;  Q або Esc — зберегти й вийти

Приклади:
    python scripts/annotate_ground_truth.py                       # опорні кадри за замовчуванням
    python scripts/annotate_ground_truth.py --frames 1 10 20 30
    python scripts/annotate_ground_truth.py --recompute           # лише перерахувати CSV з точок
    python scripts/annotate_ground_truth.py --check               # рисунки для візуальної перевірки

Результати:
    data/metadata/ground_truth_points.json — ручні відповідності
    data/metadata/ground_truth.csv         — frame_id, timestamp, x_true, y_true, is_keyframe, ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import cv2
import numpy as np
import pandas as pd

from src.config import (DEFAULT_MAP_PATH, FIGURES_DIR, FRAMES_DIR, FRAMES_METADATA_PATH,
                        GROUND_TRUTH_PATH, GROUND_TRUTH_POINTS_PATH, GT_KEYFRAMES, TRACKING_DIR)
from src.evaluation.ground_truth import (MIN_POINTS, MIN_POINTS_RELIABLE, KeyframeAnnotation,
                                         estimate_keyframe,
                                         interpolate_ground_truth, keyframes_table,
                                         load_annotations, save_annotations)
from src.localization.coordinates import apply_homography, frame_center, frame_corners
from src.preprocessing.images import read_image_rgb, resize
from src.visualization.matches import save_figure

WINDOW = "Ground truth annotation"
PAIR_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
               (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def _text(img, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class Annotator:
    """Стан розмітки одного кадру і малювання вікна (без обробки подій — для тестів)."""

    def __init__(self, map_rgb, frame_rgb, frame_id, timestamp, view_center, frame_width=960,
                 map_panel=720, view_size=720.0, annotation: KeyframeAnnotation | None = None):
        self.map_rgb, self.frame_rgb = map_rgb, frame_rgb
        self.frame_id, self.timestamp = frame_id, timestamp
        h, w = frame_rgb.shape[:2]
        self.frame_size = (w, h)
        self.frame_scale = frame_width / w
        self.frame_panel = resize(frame_rgb, (frame_width, round(h * self.frame_scale)))
        self.map_panel = map_panel
        self.view_center = np.asarray(view_center, float)
        self.view_size = float(view_size)
        self.frame_points: list[list[float]] = list(annotation.frame_points) if annotation else []
        self.map_points: list[list[float]] = list(annotation.map_points) if annotation else []

    # --- Геометрія панелей -------------------------------------------------
    @property
    def map_scale(self) -> float:            # display px на px карти
        return self.map_panel / self.view_size

    def map_origin(self) -> np.ndarray:
        return self.view_center - self.view_size / 2.0

    def map_to_display(self, xy) -> tuple[int, int]:
        p = (np.asarray(xy, float) - self.map_origin()) * self.map_scale
        return int(round(p[0])) + self.frame_panel.shape[1], int(round(p[1]))

    def display_to_map(self, x, y) -> list[float]:
        p = np.array([x - self.frame_panel.shape[1], y], float) / self.map_scale + self.map_origin()
        return [float(p[0]), float(p[1])]

    def frame_to_display(self, xy) -> tuple[int, int]:
        return (int(round((xy[0] + 0.5) * self.frame_scale - 0.5)),
                int(round((xy[1] + 0.5) * self.frame_scale - 0.5)))

    def display_to_frame(self, x, y) -> list[float]:
        return [(x + 0.5) / self.frame_scale - 0.5, (y + 0.5) / self.frame_scale - 0.5]

    # --- Дії -------------------------------------------------------------
    def click(self, x, y) -> None:
        if x < self.frame_panel.shape[1]:
            if y < self.frame_panel.shape[0] and len(self.frame_points) == len(self.map_points):
                self.frame_points.append(self.display_to_frame(x, y))
        elif len(self.frame_points) == len(self.map_points) + 1:
            self.map_points.append(self.display_to_map(x, y))

    def undo(self) -> None:
        if len(self.frame_points) > len(self.map_points):
            self.frame_points.pop()
        elif self.map_points:
            self.map_points.pop()

    def pan(self, dx, dy) -> None:
        self.view_center += np.array([dx, dy], float) * self.view_size / 4.0

    def zoom(self, factor) -> None:
        self.view_size = float(np.clip(self.view_size * factor, 120.0, 4000.0))

    def annotation(self) -> KeyframeAnnotation:
        return KeyframeAnnotation(self.frame_id, self.timestamp, self.frame_size,
                                  self.frame_points[:len(self.map_points)], list(self.map_points))

    def estimate(self):
        if len(self.map_points) < MIN_POINTS:
            return None
        try:
            return estimate_keyframe(self.annotation())
        except ValueError:
            return None

    # --- Малювання -------------------------------------------------------
    def render(self) -> np.ndarray:
        fh, fw = self.frame_panel.shape[:2]
        height = max(fh + 170, self.map_panel)
        canvas = np.zeros((height, fw + self.map_panel, 3), np.uint8)
        canvas[:fh, :fw] = self.frame_panel

        x0, y0 = self.map_origin()
        H, W = self.map_rgb.shape[:2]
        sx0, sy0 = int(np.floor(max(0, x0))), int(np.floor(max(0, y0)))
        sx1 = int(np.ceil(min(W, x0 + self.view_size)))
        sy1 = int(np.ceil(min(H, y0 + self.view_size)))
        if sx1 > sx0 and sy1 > sy0:
            crop = self.map_rgb[sy0:sy1, sx0:sx1]
            dx0, dy0 = self.map_to_display((sx0, sy0))
            dw = max(1, round((sx1 - sx0) * self.map_scale))
            dh = max(1, round((sy1 - sy0) * self.map_scale))
            patch = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_AREA if self.map_scale < 1
                               else cv2.INTER_NEAREST)
            ys, xs = slice(max(0, dy0), min(self.map_panel, dy0 + dh)), \
                slice(max(fw, dx0), min(fw + self.map_panel, dx0 + dw))
            canvas[ys, xs] = patch[ys.start - dy0:ys.stop - dy0, xs.start - dx0:xs.stop - dx0]

        cx, cy = self.frame_to_display(frame_center(self.frame_size))
        cv2.drawMarker(canvas, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)

        estimate = self.estimate()
        for i, fp in enumerate(self.frame_points):
            color = PAIR_COLORS[i % len(PAIR_COLORS)]
            p = self.frame_to_display(fp)
            cv2.circle(canvas, p, 6, color, 2, cv2.LINE_AA)
            _text(canvas, str(i + 1), (p[0] + 7, p[1] - 7))
            if i < len(self.map_points):
                q = self.map_to_display(self.map_points[i])
                cv2.circle(canvas, q, 6, color, 2, cv2.LINE_AA)
                _text(canvas, str(i + 1), (q[0] + 7, q[1] - 7))
                if estimate is not None:       # куди потрапляє точка кадру за H
                    proj = apply_homography([fp], estimate.H)[0]
                    cv2.line(canvas, q, self.map_to_display(proj), (255, 0, 0), 1, cv2.LINE_AA)

        if estimate is not None:
            quad = apply_homography(frame_corners(self.frame_size), estimate.H)
            pts = np.array([self.map_to_display(p) for p in quad], np.int32)
            cv2.polylines(canvas, [pts], True, (0, 220, 60), 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, self.map_to_display((estimate.x_true, estimate.y_true)),
                           (255, 0, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

        cv2.line(canvas, (fw, 0), (fw, height), (80, 80, 80), 1)
        pending = "click the SAME point on the map" if len(self.frame_points) > len(
            self.map_points) else "click a ground-level point on the frame"
        lines = [f"{self.frame_id}  t={self.timestamp:.1f}s  pairs: {len(self.map_points)}  -> "
                 f"{pending}",
                 "Enter/N save  Space skip  B back  U/RMB undo  C clear  WASD pan  +/- zoom  Q quit"]
        colors = [(255, 255, 255)] * len(lines)
        if estimate is not None:
            unc = estimate.center_uncertainty_px
            lines.append(f"centre on map: ({estimate.x_true:.1f}, {estimate.y_true:.1f})  "
                         f"RMS residual {estimate.residual_px:.2f} px  "
                         f"centre +-{unc:.1f} px (click noise)")
            colors.append((255, 255, 255))
            warnings = estimate.warnings()
            if len(self.map_points) >= MIN_POINTS_RELIABLE and estimate.max_residual_px > 0:
                worst = int(np.argmax(np.linalg.norm(apply_homography(
                    self.frame_points[:len(self.map_points)], estimate.H)
                    - np.asarray(self.map_points), axis=1))) + 1
                warnings = [w.replace("a pair", f"pair #{worst}") for w in warnings]
            if estimate.reliable:
                lines.append("OK: reliable keyframe - press Enter")
                colors.append((80, 230, 80))
            lines += [f"! {w}" for w in warnings]
            colors += [(255, 200, 0)] * len(warnings)
        else:
            lines.append(f"need >= {MIN_POINTS_RELIABLE} pairs spread over the WHOLE frame "
                         f"(corners and centre), on the ground, not on roofs")
            colors.append((255, 255, 255))
        for k, (line, color) in enumerate(zip(lines, colors)):
            _text(canvas, line, (10, fh + 25 + 28 * k), 0.55, color)
        return canvas


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames", type=int, nargs="+", default=list(GT_KEYFRAMES),
                   help="номери опорних кадрів (з 1)")
    p.add_argument("--trajectory", default=str(TRACKING_DIR / "trajectory.csv"),
                   help="лише для початкового центрування карти (не впливає на еталон)")
    p.add_argument("--points", default=str(GROUND_TRUTH_POINTS_PATH))
    p.add_argument("--output", default=str(GROUND_TRUTH_PATH))
    p.add_argument("--map", default=str(DEFAULT_MAP_PATH))
    p.add_argument("--no-interpolate", action="store_true",
                   help="записати лише опорні кадри, без інтерпольованих")
    p.add_argument("--recompute", action="store_true", help="лише перерахувати CSV з точок")
    p.add_argument("--check", action="store_true", help="зберегти рисунки для перевірки розмітки")
    return p.parse_args()


def frames_table() -> pd.DataFrame:
    return pd.read_csv(FRAMES_METADATA_PATH)[["frame_id", "timestamp_sec"]]


def write_ground_truth(annotations, output, interpolate=True) -> pd.DataFrame:
    keyframes = keyframes_table(annotations)
    gt = interpolate_ground_truth(keyframes, frames_table()) if interpolate else keyframes
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(output, index=False)
    return gt


def initial_view(frame_id, annotations, trajectory, map_size):
    """Центр карти для початку розмітки: наявна розмітка -> траєкторія -> центр карти."""
    if frame_id in annotations:
        est = estimate_keyframe(annotations[frame_id])
        return est.x_true, est.y_true
    if trajectory is not None:
        row = trajectory.loc[trajectory["frame_id"] == frame_id]
        if len(row) and pd.notna(row["x_map"].iloc[0]):
            return float(row["x_map"].iloc[0]), float(row["y_map"].iloc[0])
    return map_size[0] / 2.0, map_size[1] / 2.0


def annotate(args, map_rgb, annotations):
    frames = frames_table()
    trajectory = pd.read_csv(args.trajectory) if Path(args.trajectory).is_file() else None
    ids = [f"frame_{n:04d}.png" for n in args.frames]
    ids = [f for f in ids if f in set(frames["frame_id"])]
    state = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["ann"].click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["ann"].undo()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    i = 0
    while 0 <= i < len(ids):
        frame_id = ids[i]
        t = float(frames.loc[frames["frame_id"] == frame_id, "timestamp_sec"].iloc[0])
        state["ann"] = ann = Annotator(
            map_rgb, read_image_rgb(FRAMES_DIR / frame_id), frame_id, t,
            initial_view(frame_id, annotations, trajectory, map_rgb.shape[1::-1]),
            annotation=annotations.get(frame_id))
        while True:
            cv2.imshow(WINDOW, cv2.cvtColor(ann.render(), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                key = ord("q")
            if key in (13, ord("n")):
                est = ann.estimate()
                if est is None or est.n_points < MIN_POINTS_RELIABLE:
                    continue                       # потрібно щонайменше 5 пар
                annotations[frame_id] = ann.annotation()
                save_annotations(args.points, annotations)
                print(f"{frame_id}: ({est.x_true:.1f}, {est.y_true:.1f}), {est.n_points} pairs, "
                      f"RMS {est.residual_px:.2f} px, centre +-{est.center_uncertainty_px:.1f} px"
                      + ("" if est.reliable else "  [NOT RELIABLE: " + "; ".join(est.warnings())
                         + "]"))
                i += 1
                break
            if key == ord(" "):
                i += 1
                break
            if key == ord("b"):
                i = max(0, i - 1)
                break
            if key in (ord("q"), 27):
                i = -1
                break
            actions = {ord("u"): ann.undo, ord("c"): lambda: (ann.frame_points.clear(),
                                                              ann.map_points.clear()),
                       ord("w"): lambda: ann.pan(0, -1), ord("s"): lambda: ann.pan(0, 1),
                       ord("a"): lambda: ann.pan(-1, 0), ord("d"): lambda: ann.pan(1, 0),
                       ord("+"): lambda: ann.zoom(1 / 1.5), ord("="): lambda: ann.zoom(1 / 1.5),
                       ord("-"): lambda: ann.zoom(1.5)}
            if key in actions:
                actions[key]()
    cv2.destroyAllWindows()


def check_figures(annotations, map_rgb):
    out = FIGURES_DIR / "analysis" / "gt_check"
    for a in annotations.values():
        est = estimate_keyframe(a)
        ann = Annotator(map_rgb, read_image_rgb(FRAMES_DIR / a.frame_id), a.frame_id, a.timestamp,
                        (est.x_true, est.y_true), annotation=a)
        save_figure(out / f"{Path(a.frame_id).stem}.png", ann.render())
    print(f"Saved {len(annotations)} check figures to {out}")


def main():
    args = parse_args()
    annotations = load_annotations(args.points)
    map_rgb = read_image_rgb(args.map)
    if not args.recompute and not args.check:
        annotate(args, map_rgb, annotations)
    if not annotations:
        print("No annotated keyframes yet.")
        return
    gt = write_ground_truth(annotations, args.output, interpolate=not args.no_interpolate)
    key = gt[gt["is_keyframe"]]
    print(f"Keyframes: {len(key)} ({int(key['reliable'].sum())} reliable), rows in "
          f"{args.output}: {len(gt)}")
    bad = [(a.frame_id, estimate_keyframe(a)) for a in sorted(annotations.values(),
                                                              key=lambda a: a.timestamp)]
    bad = [(f, e) for f, e in bad if not e.reliable]
    if bad:
        print("Keyframes to re-annotate (excluded from key_* metrics):")
        for f, e in bad:
            print(f"  {f}: {e.n_points} pairs, centre +-{e.center_uncertainty_px:.1f} px, "
                  f"max pair residual {e.max_residual_px:.1f} px -> " + "; ".join(e.warnings()))
        print("Re-annotate with: python scripts/annotate_ground_truth.py --frames "
              + " ".join(str(int(f[6:10])) for f, _ in bad))
    extrapolated = [a.frame_id for a in sorted(annotations.values(), key=lambda a: a.timestamp)
                    if not estimate_keyframe(a).center_in_hull]
    if extrapolated:
        print("Centre outside the clicked points (extrapolated; check if possible): "
              + ", ".join(extrapolated))
    if args.check:
        check_figures(annotations, map_rgb)


if __name__ == "__main__":
    main()
