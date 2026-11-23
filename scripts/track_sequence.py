"""Етап 4. Локалізація послідовності кадрів і відновлення траєкторії.

Перший кадр (і кадри після втрати супроводу) — глобальний пошук, як на Етапі 3;
далі — пошук лише у вікні 3×3 тайли навколо прогнозованої позиції з
кінематичною перевіркою стрибків.

Приклади:
    python scripts/build_tile_cache.py                        # один раз, для LightGlue
    python scripts/track_sequence.py --limit 20
    python scripts/track_sequence.py                          # усі кадри з frames.csv
    python scripts/track_sequence.py --matcher loftr --limit 15 --stride_search 2

Результати (--output_dir, за замовчуванням experiments/results/tracking/):
    trajectory.csv        — один рядок на кадр (без пропусків)
    trajectory_info.json  — параметри запуску і підсумкова статистика
    experiments/figures/localization/trajectory_plot.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (DEFAULT_MAP_PATH, FIGURES_DIR, FRAMES_DIR, FRAMES_METADATA_PATH,
                        RANSAC_METHOD, SCORE_SIGMA_SCALE, COMPASS_OFFSET_DEG, COMPASS_PATH,
                        SUPERPOINT_CACHE_DIR, TILES_DIR, TILES_METADATA_PATH, TRACKING_DIR,
                        WARMUP_ITERATIONS, portable_path)
from src.evaluation.scoring import GateConfig
from src.localization.compass import CompassConfig, CompassSensor
from src.geometry.homography import RansacConfig
from src.localization.search import TileSearch, select_tiles
from src.localization.tracker import SequenceTracker, TrackerConfig, TrackStep
from src.matching import MATCHER_NAMES, create_matcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.cache import load_tile_cache
from src.preprocessing.images import read_image_rgb
from src.run_info import device_info, git_commit, map_size, tiles_run_info
from src.visualization.matches import save_figure
from src.visualization.trajectory import draw_trajectory, trajectory_view

MATCHERS = {name.lower(): name for name in MATCHER_NAMES}


def optional_float(value: str) -> float | None:
    """Число або ``none`` / ``off`` (параметр вимкнено)."""
    return None if value.strip().lower() in ("none", "off") else float(value)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames_dir", "--frames-dir", dest="frames_dir", default=str(FRAMES_DIR))
    p.add_argument("--matcher", type=str.lower, choices=sorted(MATCHERS), default="lightglue")
    p.add_argument("--start", type=int, default=1, help="номер першого кадру (з 1)")
    p.add_argument("--limit", type=int, default=None, help="максимальна кількість кадрів")
    p.add_argument("--stride_search", "--stride-search", dest="stride_search", type=int,
                   default=1, help="крок сітки тайлів для глобального пошуку")
    p.add_argument("--roi", type=float, nargs=4, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
                   help="обмеження глобального пошуку прямокутником (px карти)")
    p.add_argument("--output_dir", "--output-dir", dest="output_dir", default=str(TRACKING_DIR))
    p.add_argument("--device", default=None)
    p.add_argument("--score-sigma-scale", type=optional_float, default=SCORE_SIGMA_SCALE,
                   help="σ_s штрафу за масштаб у балі; none — вимкнено (типово — з config)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--cache-dir", default=str(SUPERPOINT_CACHE_DIR))
    p.add_argument("--cache-device", choices=["model", "cpu"], default="model")
    p.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    p.add_argument("--map", default=str(DEFAULT_MAP_PATH))
    p.add_argument("--full-map", action="store_true", help="малювати всю карту, а не польотну зону")
    p.add_argument("--footprint-every", type=int, default=10,
                   help="малювати контур огляду кожного n-го прийнятого кадру (0 — не малювати)")
    p.add_argument("--ransac-method", choices=["RANSAC", "USAC_MAGSAC"], default=RANSAC_METHOD,
                   help="USAC_MAGSAC швидший на тайлах без відповідника (~0.8 с/кадр проти ~1.05 с)")
    p.add_argument("--no-yaw-compensation", action="store_true",
                   help="не повертати кадр на прогнозований курс (базовий варіант)")
    p.add_argument("--predicted-window", action=argparse.BooleanOptionalAction, default=None,
                   help="TRACKING: спершу одне вікно карти навколо прогнозованого контуру "
                        "(типово — з config.PREDICTED_WINDOW)")
    p.add_argument("--window-margin", type=float, default=None,
                   help="запас вікна навколо контуру, px карти (типово — config.WINDOW_MARGIN_PX)")
    p.add_argument("--refine-global", action=argparse.BooleanOptionalAction, default=None,
                   help="уточнювати переможця глобального пошуку (типово — "
                        "config.REFINE_AFTER_GLOBAL)")
    p.add_argument("--compass", action="store_true",
                   help=f"використовувати курс із компаса ({COMPASS_PATH.name}, scripts/make_compass.py)")
    p.add_argument("--compass-offset", type=float, default=COMPASS_OFFSET_DEG,
                   help="поправка Δ: схилення + зближення меридіанів, ° (типово — з config)")
    p.add_argument("--compass-bias", type=float, default=0.0, help="імітоване зміщення компаса, °")
    p.add_argument("--compass-noise", type=float, default=0.0, help="імітований шум компаса σ, °")
    p.add_argument("--compass-seed", type=int, default=0)
    p.add_argument("--variant", default=None,
                   help="назва варіанта системи для зведених таблиць (scripts/run_experiments.py)")
    p.add_argument("--figure", default=str(FIGURES_DIR / "localization" / "trajectory_plot.png"))
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


def list_frames(frames_dir: Path, start: int, limit: int | None) -> pd.DataFrame:
    """Кадри в хронологічному порядку з часовими мітками з frames.csv."""
    files = sorted(p.name for p in frames_dir.glob("frame_*.png"))
    frames = pd.DataFrame({"frame_id": files})
    if FRAMES_METADATA_PATH.is_file():
        meta = pd.read_csv(FRAMES_METADATA_PATH)[["frame_id", "timestamp_sec"]]
        frames = frames.merge(meta, on="frame_id", how="left")
    else:
        frames["timestamp_sec"] = np.nan
    if frames["timestamp_sec"].isna().any():
        print("[WARN] Some frames have no timestamp in frames.csv; assuming 1 FPS")
        frames["timestamp_sec"] = frames["timestamp_sec"].fillna(
            pd.Series(np.arange(len(frames), dtype=float)))
    frames = frames.sort_values("timestamp_sec").iloc[max(0, start - 1):]
    return (frames.head(limit) if limit else frames).reset_index(drop=True)


def summarize(steps: list[TrackStep]) -> dict:
    df = pd.DataFrame([s.to_row() for s in steps])
    positioned = df.dropna(subset=["x_map", "y_map"])
    xy = positioned[["x_map", "y_map"]].to_numpy()
    step_lengths = np.hypot(*np.diff(xy, axis=0).T) if len(xy) > 1 else np.array([])
    by_mode = df.groupby("search_mode")["process_time"]
    return {
        "frames": len(df),
        "status_counts": df["status"].value_counts().to_dict(),
        "event_counts": df["event"].value_counts().to_dict(),
        "positioned_frames": int(len(positioned)),
        "search_mode_counts": df["search_mode"].value_counts().to_dict(),
        "refined_frames": int(df["refined"].fillna(False).astype(bool).sum()),
        "process_time_sec": {
            "total": float(df["process_time"].sum()),
            **{f"mean_{mode}": float(t.mean()) for mode, t in by_mode},
            **{f"max_{mode}": float(t.max()) for mode, t in by_mode},
        },
        "tiles_checked_mean": {mode: float(v.mean())
                               for mode, v in df.groupby("search_mode")["tiles_checked"]},
        "path_length_px": float(step_lengths.sum()),
        "step_length_px": ({"mean": float(step_lengths.mean()),
                            "median": float(np.median(step_lengths)),
                            "max": float(step_lengths.max())} if len(step_lengths) else None),
    }


def main():
    args = parse_args()
    frames = list_frames(Path(args.frames_dir), args.start, args.limit)
    if frames.empty:
        raise SystemExit(f"No frames found in {args.frames_dir}")
    method = MATCHERS[args.matcher]
    tiles = pd.read_csv(TILES_METADATA_PATH)
    global_tiles = select_tiles(tiles, args.roi, args.stride_search)
    W, H = map_size(tiles)

    matcher = create_matcher(method, device=args.device)
    pipeline = PairMatchingPipeline(matcher, PipelineConfig(
        ransac=replace(RansacConfig(), method=args.ransac_method)))
    gate, tracker_config = GateConfig(sigma_scale=args.score_sigma_scale), TrackerConfig()
    if args.no_yaw_compensation:
        tracker_config = replace(tracker_config, compensate_yaw=False)
    if args.predicted_window is not None:
        tracker_config = replace(tracker_config, predicted_window=args.predicted_window)
    if args.window_margin is not None:
        tracker_config = replace(tracker_config, window_margin_px=args.window_margin)
    if args.refine_global is not None:
        tracker_config = replace(tracker_config, refine_after_global=args.refine_global)
    full_map = None
    if tracker_config.predicted_window or tracker_config.refine_after_global:
        if not Path(args.map).is_file():
            raise SystemExit(f"--predicted-window / --refine-global need the map: {args.map}")
        full_map = read_image_rgb(args.map)

    cache_used = False
    if matcher.supports_feature_cache and not args.no_cache:
        if (Path(args.cache_dir) / "manifest.json").is_file():
            seconds = load_tile_cache(pipeline, args.cache_dir, tiles["tile_id"],
                                      TILES_METADATA_PATH, tiles_run_info().get("map_sha256"),
                                      device="cpu" if args.cache_device == "cpu" else None)
            cache_used = True
            print(f"Loaded cached features of {len(tiles)} tiles in {seconds:.2f} s")
        else:
            print(f"[WARN] No tile cache in {args.cache_dir}; run scripts/build_tile_cache.py")

    first = read_image_rgb(Path(args.frames_dir) / frames["frame_id"].iloc[0])
    pipeline.warmup((first.shape[1], first.shape[0]),
                    (int(tiles["width"].iloc[0]), int(tiles["height"].iloc[0])), args.warmup)

    compass_config, compass = None, None
    if args.compass:
        if not COMPASS_PATH.is_file():
            raise SystemExit(f"--compass: {COMPASS_PATH} not found (run scripts/make_compass.py)")
        compass_config = CompassConfig(args.compass_offset, args.compass_bias, args.compass_noise,
                                       args.compass_seed)
        compass = CompassSensor.from_csv(COMPASS_PATH, compass_config)
    tracker = SequenceTracker(TileSearch(pipeline, TILES_DIR, (W, H), gate), tiles, global_tiles,
                              tracker_config, map_rgb=full_map, compass=compass)
    print(f"Tracking {len(frames)} frames with {method} on {matcher.device} "
          f"(global pool: {len(global_tiles)} tiles)\n")
    print(f"{'frame':>14} {'t,s':>6} {'status':>12} {'event':>16} {'mode':>6} {'tiles':>5} "
          f"{'tile':>14} {'x_map':>8} {'y_map':>8} {'inl':>4} {'jump':>6} {'yaw':>6} "
          f"{'try':>3} {'time,s':>6}")
    steps = []
    for row in frames.itertuples(index=False):
        frame = read_image_rgb(Path(args.frames_dir) / row.frame_id)
        step = tracker.step(frame, row.frame_id, float(row.timestamp_sec))
        steps.append(step)

        def fmt(v, spec):
            return format(v, spec) if v is not None and not pd.isna(v) else "-"
        print(f"{row.frame_id:>14} {row.timestamp_sec:6.1f} {step.status.value:>12} "
              f"{step.event.value:>16} {step.search_mode:>6} {step.tiles_checked:5d} "
              f"{step.tile_id or '-':>14} {fmt(step.x_map, '8.1f'):>8} {fmt(step.y_map, '8.1f'):>8} "
              f"{fmt(step.inliers, '4d'):>4} {fmt(step.jump_px, '6.1f'):>6} "
              f"{fmt(step.h_rotation_deg, '6.1f'):>6} {step.attempts:3d} "
              f"{step.process_time:6.2f}", flush=True)

    # --- Збереження ---------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectory = pd.DataFrame([s.to_row() for s in steps], columns=TrackStep.CSV_COLUMNS)
    trajectory.to_csv(out / "trajectory.csv", index=False)
    summary = summarize(steps)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "frames_dir": portable_path(args.frames_dir),
        "frames": [frames["frame_id"].iloc[0], frames["frame_id"].iloc[-1]],
        "matcher": method, "matcher_config": matcher.config(),
        "pipeline": pipeline.config.to_dict(), "gate": gate.to_dict(),
        "tracker": tracker_config.to_dict(), "variant": args.variant,
        "compass": compass_config.to_dict() if compass_config else None,
        "global_search": {"roi_xyxy": args.roi, "grid_step": args.stride_search,
                          "tiles": len(global_tiles)},
        "feature_cache_used": cache_used, "summary": summary,
        "footprints": {s.frame_id: s.footprint for s in steps if s.footprint},
        "units": "pixels of the full map; times in seconds",
        "environment": device_info(matcher.device),
    }
    (out / "trajectory_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    pt = summary["process_time_sec"]
    print(f"\nStatus: {summary['status_counts']}  events: {summary['event_counts']}")
    print("Mean time per frame: " + ", ".join(f"{k[5:]} {v:.2f} s" for k, v in pt.items()
                                              if k.startswith("mean_")))
    print(f"Path length: {summary['path_length_px']:.0f} px")
    print(f"Saved: {out / 'trajectory.csv'}\nSaved: {out / 'trajectory_info.json'}")

    if args.no_figure or not Path(args.map).is_file():
        return
    map_rgb = full_map if full_map is not None else read_image_rgb(args.map)
    accepted = [s for s in steps if s.footprint]
    footprints = ([s.footprint for i, s in enumerate(accepted) if i % args.footprint_every == 0]
                  if args.footprint_every else [])
    view = None if args.full_map else trajectory_view(trajectory, (W, H))
    title = (f"{method} | {len(steps)} frames | "
             + ", ".join(f"{k}: {v}" for k, v in summary["status_counts"].items()))
    figure = Path(args.figure)
    save_figure(figure, draw_trajectory(map_rgb, trajectory, view, footprints=footprints,
                                        title=title))
    print(f"Saved: {figure}")


if __name__ == "__main__":
    main()
