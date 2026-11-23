"""Етап 5. Автоматизовані серії експериментів.

    tracking    — експерименти 1 і 2: трекінг усієї послідовності для
                  {LightGlue, LoFTR} × {RANSAC, USAC_MAGSAC} (4 запуски track_sequence.py);
                  з --variants — ще й для варіантів системи (покращення після Етапу 5)
    robustness  — експеримент 3: стійкість до спотворень кадру (яскравість, контраст,
                  шум, розмиття руху, поворот, похибка масштабу) на вибірці кадрів
    terrain     — експеримент 4: ознаки місцевості під кадром (рослинність, текстура)
                  vs кількість геометрично узгоджених відповідностей

Приклади:
    python scripts/run_experiments.py tracking                # ~15 хв на GTX 1650
    python scripts/run_experiments.py robustness --every 5
    python scripts/run_experiments.py all
    python scripts/run_experiments.py tracking --limit 30 --force
    python scripts/run_experiments.py tracking --variants baseline scale_prior window all \
        --runs lightglue_magsac loftr_magsac

Результати: experiments/results/experiments/{tracking/<run>/, robustness/, terrain/};
варіант, відмінний від baseline, — у tracking/<run>__<variant>/;
зведення і графіки — scripts/summarize_results.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import subprocess
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import (DEFAULT_MAP_PATH, EXPERIMENTS_DIR, FIGURES_DIR, FRAMES_DIR,
                        PROJECT_ROOT, SUPERPOINT_CACHE_DIR, TILES_DIR, TILES_METADATA_PATH,
                        WARMUP_ITERATIONS)
from src.evaluation.terrain import terrain_stats
from src.localization.search import TileSearch
from src.localization.tracker import nearest_tile, neighbor_tiles
from src.matching import create_matcher
from src.matching.pipeline import PairMatchingPipeline, PipelineConfig
from src.preprocessing.cache import load_tile_cache
from src.preprocessing.images import (add_gaussian_noise, adjust_brightness, adjust_contrast,
                                      motion_blur, read_image_rgb, rotate_expand)
from src.run_info import device_info, git_commit, map_size, tiles_run_info

TRACKING_RUNS = {
    "lightglue_ransac": ["--matcher", "lightglue", "--ransac-method", "RANSAC"],
    "lightglue_magsac": ["--matcher", "lightglue", "--ransac-method", "USAC_MAGSAC"],
    "loftr_ransac": ["--matcher", "loftr", "--ransac-method", "RANSAC"],
    "loftr_magsac": ["--matcher", "loftr", "--ransac-method", "USAC_MAGSAC"],
}
REFERENCE_RUN = "lightglue_ransac"

# Варіанти системи (покращення після Етапу 5). Кожен варіант задає ВСІ перемикачі явно,
# тому результат не залежить від поточних значень у src/config.py; baseline — система,
# якою отримано результати Етапу 5. Згладжування траєкторії (Калман / RTS) — не окремий
# прогін: summarize_results.py застосовує його до кожної траєкторії.
VARIANTS = {
    "baseline": ["--score-sigma-scale", "none", "--no-predicted-window", "--no-refine-global"],
    "scale_prior": ["--score-sigma-scale", "0.15", "--no-predicted-window", "--no-refine-global"],
    "window": ["--score-sigma-scale", "none", "--predicted-window", "--no-refine-global"],
    "refine": ["--score-sigma-scale", "none", "--no-predicted-window", "--refine-global"],
    "all": ["--score-sigma-scale", "0.15", "--predicted-window", "--refine-global"],
}


def run_name(base: str, variant: str) -> str:
    """Каталог прогону: ``<base>`` для baseline (як на Етапі 5), інакше ``<base>__<variant>``."""
    return base if variant == "baseline" else f"{base}__{variant}"

# Експеримент 3: (назва, вид, параметр). Спотворення застосовуються до оригінального
# кадру 1920×1080 (розмиття — ядро в пікселях кадру) до масштабування в pipeline.
PERTURBATIONS = [
    ("clean", "none", None),
    ("brightness_0.7", "brightness", 0.7), ("brightness_0.5", "brightness", 0.5),
    ("contrast_0.7", "contrast", 0.7), ("contrast_0.5", "contrast", 0.5),
    ("noise_10", "noise", 10), ("noise_25", "noise", 25),
    ("blur_5", "blur", 5), ("blur_9", "blur", 9),
    ("rotation_15", "rotation", 15), ("rotation_30", "rotation", 30),
    ("rotation_45", "rotation", 45), ("rotation_90", "rotation", 90),
    ("scale_x0.8", "scale", 0.8), ("scale_x1.25", "scale", 1.25),
]


def tracking_dir(run: str) -> Path:
    return EXPERIMENTS_DIR / "tracking" / run


# ---------------------------------------------------------------------------
# Експерименти 1–2
# ---------------------------------------------------------------------------
def run_tracking(args) -> None:
    jobs = [(base, variant) for base in TRACKING_RUNS for variant in args.variants
            if not args.runs or base in args.runs]
    for base, variant in jobs:
        run = run_name(base, variant)
        extra = [*TRACKING_RUNS[base], *VARIANTS[variant], "--variant", variant]
        out = tracking_dir(run)
        if (out / "trajectory.csv").is_file() and not args.force:
            print(f"[skip] {run}: {out / 'trajectory.csv'} exists (use --force)")
            continue
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "track_sequence.py"), *extra,
               "--output_dir", str(out),
               "--figure", str(FIGURES_DIR / "analysis" / f"trajectory_{run}.png")]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n=== {run}: {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        start = time.perf_counter()
        log = out / "run.log"
        out.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as fh:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        status = "OK" if result.returncode == 0 else f"FAILED ({result.returncode}), see {log}"
        print(f"    {status} in {time.perf_counter() - start:.0f} s", flush=True)


# ---------------------------------------------------------------------------
# Експеримент 3
# ---------------------------------------------------------------------------
def perturb(frame: np.ndarray, kind: str, value):
    if kind == "brightness":
        return adjust_brightness(frame, value)
    if kind == "contrast":
        return adjust_contrast(frame, value)
    if kind == "noise":
        return add_gaussian_noise(frame, value, seed=0)
    if kind == "blur":
        return motion_blur(frame, value, angle_deg=0.0)
    if kind == "rotation":
        # Поворот навколо центра з розширенням полотна: центр кадру лишається тією
        # самою точкою сцени, тому еталонне положення не змінюється.
        return rotate_expand(frame, value)[0]
    return frame


def run_robustness(args) -> None:
    ref = pd.read_csv(tracking_dir(REFERENCE_RUN) / "trajectory.csv")
    ref = ref[ref["status"].isin(["TRACKING", "INITIALIZED"])].iloc[::args.every]
    tiles = pd.read_csv(TILES_METADATA_PATH)
    W, H = map_size(tiles)
    out = EXPERIMENTS_DIR / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for matcher_name in args.matchers:
        matcher = create_matcher(matcher_name)
        pipelines = {}
        for scale_factor in (1.0, 0.8, 1.25):
            base = PipelineConfig()
            pipelines[scale_factor] = PairMatchingPipeline(
                matcher, replace(base, frame_to_map_scale=base.frame_to_map_scale * scale_factor))
            if matcher.supports_feature_cache:
                load_tile_cache(pipelines[scale_factor], SUPERPOINT_CACHE_DIR, tiles["tile_id"],
                                TILES_METADATA_PATH, tiles_run_info().get("map_sha256"))
        pipelines[1.0].warmup((1920, 1080), (512, 512), WARMUP_ITERATIONS)
        print(f"\n=== robustness: {matcher.name}, {len(ref)} frames x {len(PERTURBATIONS)} "
              f"perturbations", flush=True)
        for r in ref.itertuples(index=False):
            frame = read_image_rgb(FRAMES_DIR / r.frame_id)
            # Пул і курс — як у трекері на цьому кадрі (еталонний прогін LightGlue+RANSAC).
            pool = neighbor_tiles(tiles, nearest_tile(tiles, (r.x_map, r.y_map)), 1)
            yaw = float(r.yaw_used) if pd.notna(r.yaw_used) else 0.0
            clean_center = None
            for name, kind, value in PERTURBATIONS:
                pipeline = pipelines[value if kind == "scale" else 1.0]
                image = perturb(frame, kind, value)
                search = TileSearch(pipeline, TILES_DIR, (W, H))
                t0 = time.perf_counter()
                result = search.search(image, pool, rotation_deg=yaw)
                elapsed = time.perf_counter() - t0
                w = result.winner
                center = w.center_map if w else None
                if name == "clean":
                    clean_center = center
                shift = (float(np.hypot(center[0] - clean_center[0], center[1] - clean_center[1]))
                         if center and clean_center else np.nan)
                rows.append({
                    "matcher": matcher.name, "frame_id": r.frame_id, "perturbation": name,
                    "kind": kind, "value": value, "localized": w is not None,
                    "tile_id": w.tile_id if w else None,
                    "inliers": w.metrics.inliers if w else 0,
                    "matches": w.metrics.matches if w else 0,
                    "inlier_ratio": w.metrics.inlier_ratio if w else 0.0,
                    "reproj_mean": w.metrics.reproj_mean if w else None,
                    "x_map": center[0] if center else None, "y_map": center[1] if center else None,
                    "shift_from_clean_px": shift, "search_time": elapsed,
                    "tiles_passed": len(result.passed)})
            done = rows[-len(PERTURBATIONS):]
            print(f"  {r.frame_id}: " + " ".join(
                f"{d['perturbation']}={d['inliers']}" for d in done), flush=True)
        del matcher, pipelines
    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness.csv", index=False)
    info = {"reference_run": REFERENCE_RUN, "every": args.every, "frames": list(ref["frame_id"]),
            "perturbations": PERTURBATIONS, "git_commit": git_commit(),
            "environment": device_info("cuda" if __import__("torch").cuda.is_available()
                                       else "cpu"),
            "notes": "pool = 3x3 tiles around the reference position; frame rotated by the "
                     "reference yaw (as in tracking); rotation perturbation is NOT compensated"}
    (out / "robustness_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"Saved: {out / 'robustness.csv'}")


# ---------------------------------------------------------------------------
# Експеримент 4
# ---------------------------------------------------------------------------
def run_terrain(args) -> None:
    map_rgb = read_image_rgb(DEFAULT_MAP_PATH)
    out = EXPERIMENTS_DIR / "terrain"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in TRACKING_RUNS:
        path = tracking_dir(run)
        if not (path / "trajectory.csv").is_file():
            continue
        traj = pd.read_csv(path / "trajectory.csv")
        footprints = json.loads((path / "trajectory_info.json").read_text(encoding="utf-8")
                                ).get("footprints", {})
        for r in traj.itertuples(index=False):
            fp = footprints.get(r.frame_id)
            stats = terrain_stats(map_rgb, fp) if fp else {"vegetation_fraction": np.nan,
                                                           "texture": np.nan, "area_px": 0}
            rows.append({"run": run, "frame_id": r.frame_id, "timestamp": r.timestamp,
                         "status": r.status, "inliers": r.inliers,
                         "inlier_ratio": r.inlier_ratio, "x_map": r.x_map, "y_map": r.y_map,
                         **stats})
    df = pd.DataFrame(rows)
    df.to_csv(out / "terrain.csv", index=False)
    print(f"Saved: {out / 'terrain.csv'} ({df['run'].nunique()} runs)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("experiments", nargs="+", choices=["tracking", "robustness", "terrain", "all"])
    p.add_argument("--limit", type=int, default=None, help="кадрів для трекінгу (усі — None)")
    p.add_argument("--runs", nargs="+", choices=list(TRACKING_RUNS), default=None)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["baseline"],
                   help="варіанти системи для трекінгу (типово — лише baseline, як на Етапі 5)")
    p.add_argument("--force", action="store_true", help="перезапустити наявні прогони трекінгу")
    p.add_argument("--dry-run", action="store_true",
                   help="лише показати команди трекінгу, нічого не запускати")
    p.add_argument("--every", type=int, default=5, help="кожен n-й кадр для експерименту 3")
    p.add_argument("--matchers", nargs="+", default=["LightGlue", "LoFTR"])
    args = p.parse_args()
    todo = {"tracking", "robustness", "terrain"} if "all" in args.experiments \
        else set(args.experiments)
    if "tracking" in todo:
        run_tracking(args)
    if args.dry_run:
        return
    if "robustness" in todo:
        run_robustness(args)
    if "terrain" in todo:
        run_terrain(args)


if __name__ == "__main__":
    main()
