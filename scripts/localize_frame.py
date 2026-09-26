"""Етап 3. Автоматичний пошук тайла та глобальна локалізація одиночного кадру.

    кадр -> пошук серед тайлів карти -> фільтр + ранжування -> найкращий тайл
         -> (X_map, Y_map) і контур огляду камери на карті 6200×4320

Інформація про правильний тайл не використовується.

Приклади:
    python scripts/build_tile_cache.py        # один раз, для LightGlue
    python scripts/localize_frame.py --frame data/processed/frames/frame_0002.png --matcher lightglue
    python scripts/localize_frame.py --matcher loftr --roi 512 1792 1792 2816
    python scripts/localize_frame.py --matcher loftr --grid-step 2     # кожен 2-й рядок/стовпець

Результати:
    experiments/results/localization_single_frame.json          — підсумок пошуку
    experiments/results/rankings/<frame>_<matcher>_candidates.csv — усі кандидати
    experiments/figures/localization/<frame>_global_loc.png       — карта з результатом
    experiments/figures/localization/<frame>_global_loc_zoom.png  — збільшений фрагмент
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (DEFAULT_MAP_PATH, FIGURES_DIR, FRAMES_DIR, LOCALIZATION_RESULT_PATH,
                        RESULTS_DIR, SUPERPOINT_CACHE_DIR, TILES_DIR, TILES_METADATA_PATH, TOP_K,
                        WARMUP_ITERATIONS, portable_path)
from src.evaluation.scoring import GateConfig
from src.localization.search import TileSearch, select_tiles
from src.matching import MATCHER_NAMES, create_matcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.cache import load_tile_cache
from src.preprocessing.images import read_image_rgb
from src.run_info import device_info, git_commit, map_size, tiles_run_info
from src.visualization.localization import draw_global_localization
from src.visualization.matches import save_figure

MATCHERS = {name.lower(): name for name in MATCHER_NAMES}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frame", default=str(FRAMES_DIR / "frame_0002.png"),
                   help="шлях до кадру або ім'я файлу в data/processed/frames")
    p.add_argument("--matcher", type=str.lower, choices=sorted(MATCHERS), default="lightglue")
    p.add_argument("--roi", type=float, nargs=4, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
                   help="шукати лише серед тайлів, що перетинаються з прямокутником (px карти)")
    p.add_argument("--grid-step", type=int, default=1,
                   help="перевіряти кожен n-й стовпець і рядок сітки тайлів")
    p.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=TOP_K)
    p.add_argument("--device", default=None)
    p.add_argument("--no-cache", action="store_true", help="не використовувати кеш ознак тайлів")
    p.add_argument("--cache-dir", default=str(SUPERPOINT_CACHE_DIR))
    p.add_argument("--cache-device", choices=["model", "cpu"], default="model",
                   help="де тримати кеш: на пристрої моделі (VRAM) або в RAM")
    p.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    p.add_argument("--output", default=str(LOCALIZATION_RESULT_PATH))
    p.add_argument("--map", default=str(DEFAULT_MAP_PATH))
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


def resolve_frame(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if (FRAMES_DIR / path.name).is_file():
        return FRAMES_DIR / path.name
    raise FileNotFoundError(f"Frame not found: {value}")


def candidate_summary(c) -> dict:
    m, ev, g = c.metrics, c.evaluation, c.pair.homography
    return {"tile_id": c.tile_id, "tile_origin": list(c.tile_origin), "passed": ev.passed,
            "score": ev.score, "inliers": m.inliers, "matches": m.matches,
            "inlier_ratio": m.inlier_ratio, "reproj_mean": m.reproj_mean,
            "h_scale": c.pair.diagnostics.scale if c.pair.diagnostics else None,
            "center_local": list(m.center_local) if m.center_local else None,
            "center_map": ([c.tile_origin[0] + m.center_local[0],
                            c.tile_origin[1] + m.center_local[1]] if m.center_local else None),
            "gate_reasons": ev.reasons, "success_stage2": c.pair.success,
            "homography": g.H.tolist() if g.H is not None else None}


def main():
    args = parse_args()
    frame_path = resolve_frame(args.frame)
    method = MATCHERS[args.matcher]
    all_tiles = pd.read_csv(TILES_METADATA_PATH)
    tiles = select_tiles(all_tiles, args.roi, args.grid_step)
    if tiles.empty:
        raise SystemExit("No tiles selected: check --roi / --grid-step")
    W, H = map_size(all_tiles)

    matcher = create_matcher(method, device=args.device)
    config = PipelineConfig()
    pipeline = PairMatchingPipeline(matcher, config)
    gate = GateConfig()

    cache_used, cache_load_time = False, None
    if matcher.supports_feature_cache and not args.no_cache:
        if (Path(args.cache_dir) / "manifest.json").is_file():
            cache_load_time = load_tile_cache(
                pipeline, args.cache_dir, tiles["tile_id"], TILES_METADATA_PATH,
                map_sha256=tiles_run_info().get("map_sha256"),
                device="cpu" if args.cache_device == "cpu" else None)
            cache_used = True
            print(f"Loaded cached features of {len(tiles)} tiles in {cache_load_time:.2f} s")
        else:
            print(f"[WARN] No tile cache in {args.cache_dir}; features will be extracted on the "
                  f"fly. Run scripts/build_tile_cache.py to build it.")

    frame = read_image_rgb(frame_path)
    tile_size = (int(tiles["width"].iloc[0]), int(tiles["height"].iloc[0]))
    pipeline.warmup((frame.shape[1], frame.shape[0]), tile_size, args.warmup)

    print(f"Searching {len(tiles)} of {len(all_tiles)} tiles with {method} on {matcher.device} ...")
    search = TileSearch(pipeline, TILES_DIR, (W, H), gate)
    result = search.search(frame, tiles, progress_every=100)

    # --- Вивід --------------------------------------------------------------
    top = result.candidates[:args.top_k]
    print(f"\nTop-{len(top)} candidates ({len(result.passed)} of {len(tiles)} passed the gate):")
    print(f"{'rank':>4} {'tile':>14} {'passed':>6} {'score':>8} {'inl':>5} {'match':>5} "
          f"{'ratio':>5} {'E,px':>5}  reasons")
    for rank, c in enumerate(top, 1):
        m, ev = c.metrics, c.evaluation
        reproj = f"{m.reproj_mean:.2f}" if m.reproj_mean is not None else "  -  "
        print(f"{rank:>4} {c.tile_id:>14} {str(ev.passed):>6} {ev.score:8.2f} {m.inliers:5d} "
              f"{m.matches:5d} {m.inlier_ratio:5.2f} {reproj:>5}  {'; '.join(ev.reasons)}")

    winner, center, footprint = result.winner, result.center_map(), result.footprint_map()
    if winner is None:
        print("\nLocalization FAILED: no tile passed the gate.")
    else:
        print(f"\nWinner: {winner.tile_id} at {winner.tile_origin}, score {winner.evaluation.score:.2f}")
        print(f"Frame centre on the map: X = {center[0]:.1f}, Y = {center[1]:.1f} px "
              f"(map {W}x{H})")
    t = result.timings
    print(f"Search time: {t['search_wall']:.2f} s wall, {t['per_tile_mean'] * 1000:.1f} ms/tile "
          f"(frame extraction {t['frame_extract'] * 1000:.1f} ms, tile I/O {t['tile_io']:.2f} s)")

    # --- Збереження ---------------------------------------------------------
    rankings = RESULTS_DIR / "rankings" / f"{frame_path.stem}_{args.matcher}_candidates.csv"
    rankings.parent.mkdir(parents=True, exist_ok=True)
    result.to_frame().to_csv(rankings, index=False)

    runner_up = result.passed[1] if len(result.passed) > 1 else None
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "frame": portable_path(frame_path), "frame_size": list(result.frame_size),
        "matcher": method, "matcher_config": matcher.config(),
        "pipeline": config.to_dict(), "gate": gate.to_dict(),
        "search_space": {"roi_xyxy": args.roi, "grid_step": args.grid_step,
                         "tiles_checked": len(tiles), "tiles_total": len(all_tiles),
                         "tiles_passed": len(result.passed), "tile_ids": list(tiles["tile_id"])},
        "feature_cache": {"used": cache_used, "dir": portable_path(args.cache_dir) if cache_used
                          else None, "load_time_sec": cache_load_time,
                          "device": args.cache_device if cache_used else None},
        "timings_sec": t,
        "success": winner is not None,
        "winner": None if winner is None else {
            **candidate_summary(winner),
            "center_map": list(center),
            "footprint_map": footprint.tolist(),
            "score_margin_to_runner_up": (winner.evaluation.score / runner_up.evaluation.score
                                          if runner_up and runner_up.evaluation.score > 0
                                          else None),
        },
        "top_k": [candidate_summary(c) for c in top],
        "map_size": [W, H],
        "units": "pixels of the full map; times in seconds",
        "environment": device_info(matcher.device),
        "candidates_csv": portable_path(rankings),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}\nSaved: {rankings}")

    if args.no_figure or not Path(args.map).is_file():
        return
    map_rgb = read_image_rgb(args.map)
    rects = list(tiles[["x_origin", "y_origin", "width", "height"]].itertuples(index=False,
                                                                                 name=None))
    passed = [(*c.tile_origin, *c.metrics.tile_size) for c in result.passed]
    winner_rect = (*winner.tile_origin, *winner.metrics.tile_size) if winner else None
    title = (f"{frame_path.name} | {method} | "
             + (f"{winner.tile_id} | X={center[0]:.0f}, Y={center[1]:.0f}" if winner
                else "localization failed"))
    figures = FIGURES_DIR / "localization"
    view = tuple(args.roi) if args.roi else None
    save_figure(figures / f"{frame_path.stem}_global_loc.png",
                draw_global_localization(map_rgb, rects, winner_rect, footprint, center, passed,
                                         view_xyxy=view, title=title))
    if winner is not None:
        margin = 700
        zoom = (center[0] - margin, center[1] - margin, center[0] + margin, center[1] + margin)
        save_figure(figures / f"{frame_path.stem}_global_loc_zoom.png",
                    draw_global_localization(map_rgb, rects, winner_rect, footprint, center,
                                             passed, view_xyxy=zoom, max_side=1400, title=title))
    print(f"Saved: {figures / f'{frame_path.stem}_global_loc.png'}")


if __name__ == "__main__":
    main()
