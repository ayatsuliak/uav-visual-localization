"""Етап 2. Порівняння методів зіставлення на відомих парах «кадр — тайл».

Для кожної пари з eval_pairs.csv і кожного методу виконується єдиний pipeline
``frame + tile -> matcher -> RANSAC -> metrics`` з однаковою нормалізацією
масштабу і однаковими параметрами геометричної перевірки.

Приклади:
    python scripts/evaluate.py                                   # LightGlue і LoFTR
    python scripts/evaluate.py --methods LightGlue --repeats 5   # медіана часу з 5 запусків
    python scripts/evaluate.py --work-scale 1.5 --output experiments/results/raw/ws15.csv

Результати:
    experiments/results/raw/matching_results.csv       — один рядок на (пару, метод)
    experiments/results/raw/matching_results_info.json — параметри запуску
    experiments/figures/matching/pair_XXX/*.png        — візуалізації
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from statistics import median

import pandas as pd

from src.config import (DEFAULT_MAP_PATH, EVAL_PAIRS_PATH, FIGURES_DIR, FRAME_TO_MAP_SCALE,
                        FRAMES_DIR, MATCHING_RESULTS_PATH, MIN_INLIERS,
                        RANSAC_METHOD, RANSAC_THRESHOLD, TILES_DIR, TILES_METADATA_PATH,
                        TIMING_REPEATS, WARMUP_ITERATIONS, WORK_SCALE, portable_path)
from src.geometry.homography import RansacConfig
from src.matching import MATCHER_NAMES, create_matcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.images import read_image_rgb
from src.run_info import device_info, git_commit
from src.visualization.matches import draw_footprint, draw_matches, save_figure

TIME_COLUMNS = ("time_extract_frame", "time_extract_tile", "time_match",
                "time_matcher_total", "time_ransac")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pairs", default=str(EVAL_PAIRS_PATH))
    p.add_argument("--output", default=str(MATCHING_RESULTS_PATH))
    p.add_argument("--methods", nargs="+", default=list(MATCHER_NAMES), choices=MATCHER_NAMES)
    p.add_argument("--device", default=None, help="cuda | cpu (за замовчуванням — автоматично)")
    p.add_argument("--frame-scale", type=float, default=FRAME_TO_MAP_SCALE,
                   help="пікселів карти на піксель кадру")
    p.add_argument("--work-scale", type=float, default=WORK_SCALE)
    p.add_argument("--ransac-method", default=RANSAC_METHOD, choices=["RANSAC", "USAC_MAGSAC"])
    p.add_argument("--ransac-threshold", type=float, default=RANSAC_THRESHOLD, help="px карти")
    p.add_argument("--min-inliers", type=int, default=MIN_INLIERS)
    p.add_argument("--repeats", type=int, default=TIMING_REPEATS,
                   help="кількість запусків для вимірювання часу (записується медіана)")
    p.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    p.add_argument("--map", default=str(DEFAULT_MAP_PATH),
                   help="карта для візуалізації проєкції кадру (необов'язково)")
    p.add_argument("--figures-dir", default=str(FIGURES_DIR / "matching"))
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def run_pair(pipeline, frame, tile, tile_origin, repeats):
    """Перший запуск дає відповідності; повтори — лише для медіани часу.

    Кеш ознак тайла вимкнено: час включає екстракцію ознак обох зображень.
    """
    result = pipeline.run(frame, tile, tile_origin)
    row = result.to_row()
    if repeats > 1:
        samples = {key: [row[key]] for key in TIME_COLUMNS}
        for _ in range(repeats - 1):
            extra = pipeline.run(frame, tile, tile_origin).to_row()
            for key in TIME_COLUMNS:
                samples[key].append(extra[key])
        row.update({key: median(values) for key, values in samples.items()})
    return result, row


def save_pair_figures(result, frame, tile, map_rgb, tile_rect, out_dir, method, pair_id):
    g, m = result.homography, result.match
    status = "success" if result.success else f"fail: {result.failure_reason}"
    header = f"pair {pair_id} | {method} | matches {g.num_matches} | inliers {g.num_inliers}"
    save_figure(out_dir / f"{method.lower()}_matches.png",
                draw_matches(frame, tile, m.pts0, m.pts1, g.inlier_mask,
                             title=f"{header} | {status}"))
    save_figure(out_dir / f"{method.lower()}_inliers.png",
                draw_matches(frame, tile, m.pts0, m.pts1, g.inlier_mask, show_outliers=False,
                             title=f"{header} (inliers only)"))
    if result.diagnostics is None:
        return
    x0, y0, tw, th = tile_rect
    footprint = result.diagnostics.footprint + [x0, y0]
    center = result.frame_center_map
    if map_rgb is not None:
        margin = max(tw, th)
        cx0, cy0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
        cx1 = min(map_rgb.shape[1], int(x0 + tw + margin))
        cy1 = min(map_rgb.shape[0], int(y0 + th + margin))
        background, shift = map_rgb[cy0:cy1, cx0:cx1], (cx0, cy0)
    else:
        background, shift = tile, (x0, y0)
    image = draw_footprint(background, footprint - shift, (x0 - shift[0], y0 - shift[1], tw, th),
                           center=None if center is None else (center[0] - shift[0],
                                                               center[1] - shift[1]),
                           title=f"{method}: frame footprint (yellow), tile (blue)")
    save_figure(out_dir / f"{method.lower()}_footprint.png", image)


def main():
    args = parse_args()
    pairs = pd.read_csv(args.pairs)
    tiles = pd.read_csv(TILES_METADATA_PATH).set_index("tile_id")
    config = PipelineConfig(
        frame_to_map_scale=args.frame_scale, work_scale=args.work_scale,
        min_inliers=args.min_inliers,
        ransac=replace(RansacConfig(), method=args.ransac_method, threshold=args.ransac_threshold),
    )
    map_rgb = None
    if not args.no_figures and Path(args.map).is_file():
        map_rgb = read_image_rgb(args.map)

    # Зображення читаються один раз, щоб читання з диска не впливало на час.
    images = {}
    for _, pair in pairs.iterrows():
        for name, directory in ((pair["frame"], FRAMES_DIR), (pair["tile"], TILES_DIR)):
            if name not in images:
                images[name] = read_image_rgb(directory / name)

    rows, matcher_configs, device = [], {}, None
    for method in args.methods:
        matcher = create_matcher(method, device=args.device)
        device = matcher.device
        matcher_configs[method] = matcher.config()
        pipeline = PairMatchingPipeline(matcher, config)
        first = pairs.iloc[0]
        f, t = images[first["frame"]], images[first["tile"]]
        pipeline.warmup((f.shape[1], f.shape[0]), (t.shape[1], t.shape[0]), args.warmup)

        for _, pair in pairs.iterrows():
            pair_id = int(pair["pair_id"])
            tile_meta = tiles.loc[pair["tile"]]
            tile_rect = (float(tile_meta["x_origin"]), float(tile_meta["y_origin"]),
                         float(tile_meta["width"]), float(tile_meta["height"]))
            frame, tile = images[pair["frame"]], images[pair["tile"]]
            try:
                result, metrics = run_pair(pipeline, frame, tile, tile_rect[:2], args.repeats)
            except Exception as exc:  # одна пара не повинна зупиняти весь експеримент
                print(f"[ERROR] {method}, pair {pair_id}: {exc}")
                rows.append({"pair_id": pair_id, "method": method, "frame": pair["frame"],
                             "tile": pair["tile"], "success": False,
                             "failure_reason": f"error: {exc}"})
                continue
            rows.append({"pair_id": pair_id, "method": method, "frame": pair["frame"],
                         "tile": pair["tile"], "scene_type": pair.get("scene_type"),
                         "repeats": args.repeats, **metrics})
            print(f"{method:9s} pair {pair_id:3d}: matches {metrics['matches']:5d}, "
                  f"inliers {metrics['inliers']:5d} ({metrics['inlier_ratio']:.2f}), "
                  f"reproj {metrics['reproj_mean'] or float('nan'):.2f} px, "
                  f"scale {metrics['h_scale'] or float('nan'):.3f}, "
                  f"time {metrics['time_matcher_total']:.3f} s, "
                  f"{'OK' if metrics['success'] else 'FAIL: ' + metrics['failure_reason']}")
            if not args.no_figures:
                save_pair_figures(result, frame, tile, map_rgb, tile_rect,
                                  Path(args.figures_dir) / f"pair_{pair_id:03d}", method, pair_id)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "pairs": portable_path(args.pairs),
        "methods": args.methods, "repeats": args.repeats, "warmup": args.warmup,
        "pipeline": config.to_dict(), "matchers": matcher_configs,
        "environment": device_info(device or "cpu"),
        "units": "all distances in full-map pixels; times in seconds",
    }
    info_path = output.with_name(output.stem + "_info.json")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}\nSaved: {info_path}")


if __name__ == "__main__":
    main()
