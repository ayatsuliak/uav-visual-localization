"""Етап 5. Зведені таблиці та графіки за результатами scripts/run_experiments.py.

Якщо існує data/metadata/ground_truth.csv, рахуються похибки відносно еталону
(окремо для опорних кадрів — основна метрика — і для всіх кадрів з
інтерпольованим еталоном). Без еталону таблиці містять лише показники, що
від нього не залежать (покриття, час, відповідності, взаємні відхилення
траєкторій), а графік похибки замінюється графіком відхилення між методами.

Приклад:
    python scripts/summarize_results.py

Результати:
    experiments/results/summary/benchmark_table.{csv,md}
    experiments/results/summary/{trajectory_deviation,robustness_table,terrain_table}.{csv,md}
    experiments/results/summary/smoothing_table.{csv,md}  — сира траєкторія vs Калман / RTS
    experiments/results/summary/variants_table.{csv,md}   — варіанти системи vs baseline

Прогони варіантів (каталоги ``<run>__<variant>``, scripts/run_experiments.py --variants)
потрапляють у benchmark/smoothing/variants-таблиці; графіки, взаємні відхилення і
зв'язок з місцевістю будуються лише за baseline-прогонами (як на Етапі 5).
    experiments/figures/analysis/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (DEFAULT_MAP_PATH, EXPERIMENTS_DIR, FIGURES_DIR, GROUND_TRUTH_PATH,
                        MAP_GSD_M, RESULTS_DIR)
from src.evaluation.ground_truth import load_ground_truth
from src.evaluation.metrics import SUCCESS_THRESHOLDS_PX, error_statistics, trajectory_errors
from src.localization.smoothing import SmoothingConfig, smooth_trajectory, with_position
from src.preprocessing.images import read_image_rgb

# Палітра (перевірена валідатором): колір — метод, тип лінії — оцінювач геометрії.
COLORS = {"LightGlue": "#2a78d6", "LoFTR": "#eb6834"}
LINESTYLES = {"RANSAC": "-", "USAC_MAGSAC": "--"}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
PROBLEM_FRAMES = (81, 84)          # ділянка над деревами (Етап 4)
SUCCESS_THRESHOLDS_M = (5, 10, 25)  # м — лише для карт з прив'язкою (MAP_GSD_M)
ACCEPTED = ("TRACKING", "INITIALIZED")
LOCAL_MODES = ("local", "window")  # трекінг: 3×3 тайли або вікно за прогнозом

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "text.color": INK,
})


# ---------------------------------------------------------------------------
# Завантаження
# ---------------------------------------------------------------------------
def load_runs() -> dict:
    runs = {}
    for path in sorted((EXPERIMENTS_DIR / "tracking").glob("*/trajectory.csv")):
        info = json.loads((path.parent / "trajectory_info.json").read_text(encoding="utf-8"))
        traj = pd.read_csv(path)
        traj["frame_no"] = traj["frame_id"].str.extract(r"(\d+)").astype(int)
        base, _, suffix = path.parent.name.partition("__")
        runs[path.parent.name] = {"traj": traj, "info": info, "matcher": info["matcher"],
                                  "ransac": info["pipeline"]["ransac"]["method"], "base": base,
                                  "variant": info.get("variant") or suffix or "baseline"}
    return runs


def baseline_runs(runs: dict) -> dict:
    return {name: run for name, run in runs.items() if run["variant"] == "baseline"}


def reliable_keyframes(df: pd.DataFrame) -> pd.Series:
    """Опорні кадри з надійною розміткою (критерій — src.evaluation.ground_truth, без
    використання результатів системи); без колонки ``reliable`` — усі опорні кадри."""
    mask = df["is_keyframe"].astype(bool)
    if "reliable" in df:
        mask &= df["reliable"].fillna(False).astype(bool)
    return mask


def label(run: dict) -> str:
    text = f"{run['matcher']} + {'MAGSAC++' if run['ransac'] == 'USAC_MAGSAC' else 'RANSAC'}"
    return text if run.get("variant", "baseline") == "baseline" else f"{text} [{run['variant']}]"


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else floatfmt.format(v)
        return "—" if v is None else str(v)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def save_table(df: pd.DataFrame, name: str, out: Path, note: str = "",
               md_drop_prefixes: tuple[str, ...] = ()) -> None:
    """CSV — повна таблиця; Markdown (для тексту роботи) — без колонок з ``md_drop_prefixes``."""
    df.to_csv(out / f"{name}.csv", index=False)
    md = df[[c for c in df.columns if not c.startswith(md_drop_prefixes)]]
    (out / f"{name}.md").write_text((note + "\n\n" if note else "") + to_markdown(md),
                                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Таблиці
# ---------------------------------------------------------------------------
def benchmark_table(runs: dict, gt: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for name, run in runs.items():
        t = run["traj"]
        local = t[t["search_mode"].isin(LOCAL_MODES)]
        counts = t["status"].value_counts()
        row = {"run": name, "method": label(run), "variant": run["variant"], "frames": len(t),
               "tracking": int(counts.get("TRACKING", 0)),
               "initialized": int(counts.get("INITIALIZED", 0)),
               "interpolated": int(counts.get("INTERPOLATED", 0)),
               "lost": int(counts.get("LOST", 0)),
               "outliers_rejected": int((t["event"] == "OUTLIER_REJECTED").sum()),
               "window_frames": int((t["search_mode"] == "window").sum()),
               "time_global_s": t.loc[t["search_mode"] == "global", "process_time"].mean(),
               "time_local_mean_s": local["process_time"].mean(),
               "time_local_p90_s": local["process_time"].quantile(0.9),
               "inliers_median": t["inliers"].median(),
               "inlier_ratio_mean": t["inlier_ratio"].mean(),
               "reproj_mean_px": t["reproj_error"].mean()}
        if gt is not None:
            errors = trajectory_errors(t, gt)
            for subset, mask in (("key", reliable_keyframes(errors)), ("all", slice(None))):
                s = error_statistics(errors.loc[mask, "error"])
                row.update({f"{subset}_n": s["n_frames"], f"{subset}_mean_px": s["mean"],
                            f"{subset}_median_px": s["median"], f"{subset}_rmse_px": s["rmse"],
                            f"{subset}_p90_px": s["p90"], f"{subset}_max_px": s["max"],
                            **{f"{subset}_success@{k}": s[f"success@{k:g}"]
                               for k in SUCCESS_THRESHOLDS_PX}})
            if MAP_GSD_M:
                row.update(metres_row(errors, reliable_keyframes(errors)))
        rows.append(row)
    return pd.DataFrame(rows)


def metres_row(errors: pd.DataFrame, mask) -> dict:
    """Похибки в метрах (карта з прив'язкою): усі кадри з еталоном і лише кадри з
    прийнятою оцінкою (TRACKING / INITIALIZED, без екстраполяції)."""
    e = errors.loc[mask]
    row = {}
    for subset, sel in (("m", slice(None)), ("fix_m", e["status"].isin(ACCEPTED))):
        s = error_statistics(e.loc[sel, "error"] * MAP_GSD_M,
                             thresholds=SUCCESS_THRESHOLDS_M if subset == "m" else ())
        row.update({f"{subset}_n": s["n_frames"], f"{subset}_mean": s["mean"],
                    f"{subset}_median": s["median"], f"{subset}_rmse": s["rmse"],
                    f"{subset}_p90": s["p90"], f"{subset}_max": s["max"]})
        if subset == "m":
            row.update({f"m_success@{k}m": s[f"success@{k:g}"] for k in SUCCESS_THRESHOLDS_M})
    return row


def variants_table(bench: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """Варіанти системи поруч із baseline того самого прогону; ``d_*`` — різниця з baseline."""
    base_of = {name: run["base"] for name, run in runs.items()}
    cols = ["key_median_px", "key_rmse_px", "key_p90_px", "key_success@10",
            "time_local_mean_s", "interpolated", "lost"]
    cols = [c for c in cols if c in bench]
    table = bench.assign(base=bench["run"].map(base_of))[
        ["base", "method", "variant", *cols, "window_frames"]]
    rows = []
    for base, g in table.groupby("base", sort=False):
        ref = g[g["variant"] == "baseline"]
        for row in g.to_dict("records"):
            if not ref.empty:
                for c in ("key_median_px", "key_rmse_px", "time_local_mean_s"):
                    if c in row:
                        row[f"d_{c}"] = row[c] - ref[c].iloc[0]
            rows.append(row)
    return pd.DataFrame(rows)


def smoothing_table(runs: dict, gt: pd.DataFrame, config: SmoothingConfig) -> pd.DataFrame:
    """Похибка на надійних опорних кадрах: сира траєкторія, фільтр Калмана, RTS."""
    rows = []
    for name, run in runs.items():
        smoothed = smooth_trajectory(run["traj"], config)
        raw_errors = None
        for variant, title in (("raw", "без згладжування"), ("kf", "Калман (онлайн)"),
                               ("rts", "RTS (офлайн)")):
            errors = trajectory_errors(with_position(smoothed, variant), gt)
            errors = errors.loc[reliable_keyframes(errors)]
            s = error_statistics(errors["error"])
            row = {"run": name, "method": label(run), "smoothing": title,
                   "key_n": s["n_frames"], "key_mean_px": s["mean"],
                   "key_median_px": s["median"], "key_rmse_px": s["rmse"],
                   "key_p90_px": s["p90"], "key_max_px": s["max"],
                   **{f"key_success@{k}": s[f"success@{k:g}"] for k in SUCCESS_THRESHOLDS_PX}}
            if variant == "raw":
                raw_errors = errors["error"].to_numpy()
                row["better_frames"] = None
            else:
                e = errors["error"].to_numpy()
                ok = np.isfinite(e) & np.isfinite(raw_errors)
                row["better_frames"] = f"{int((e[ok] < raw_errors[ok]).sum())}/{int(ok.sum())}"
            rows.append(row)
    return pd.DataFrame(rows)


def deviation_table(runs: dict) -> pd.DataFrame:
    """Відхилення позицій між прогонами на спільних кадрах з оцінкою (без еталону)."""
    rows = []
    for a, b in itertools.combinations(runs, 2):
        ta, tb = runs[a]["traj"], runs[b]["traj"]
        m = ta.merge(tb, on="frame_id", suffixes=("_a", "_b")).dropna(
            subset=["x_map_a", "y_map_a", "x_map_b", "y_map_b"])
        d = np.hypot(m["x_map_a"] - m["x_map_b"], m["y_map_a"] - m["y_map_b"])
        rows.append({"run_a": a, "run_b": b, "frames": len(d), "mean_px": d.mean(),
                     "median_px": d.median(), "p90_px": d.quantile(0.9), "max_px": d.max()})
    return pd.DataFrame(rows)


def robustness_table(rob: pd.DataFrame) -> pd.DataFrame:
    clean = rob[rob["perturbation"] == "clean"][["matcher", "frame_id", "inliers"]]
    rob = rob.merge(clean.rename(columns={"inliers": "clean_inliers"}), on=["matcher", "frame_id"])
    rob["retention"] = np.where(rob["clean_inliers"] > 0, rob["inliers"] / rob["clean_inliers"],
                                np.nan)
    order = {p: i for i, p in enumerate(dict.fromkeys(rob["perturbation"]))}
    table = rob.groupby(["matcher", "perturbation"], sort=False).agg(
        frames=("frame_id", "count"), localized_rate=("localized", "mean"),
        inliers_median=("inliers", "median"), retention_median=("retention", "median"),
        inlier_ratio_median=("inlier_ratio", "median"),
        shift_median_px=("shift_from_clean_px", "median"),
        shift_p90_px=("shift_from_clean_px", lambda s: s.quantile(0.9)),
        time_mean_s=("search_time", "mean")).reset_index()
    table["_o"] = table["perturbation"].map(order)
    return table.sort_values(["matcher", "_o"]).drop(columns="_o")


def terrain_table(ter: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in ter.groupby("run"):
        g = g.dropna(subset=["vegetation_fraction", "inliers"])
        if len(g) < 3:
            continue
        rank = lambda s: s.rank().to_numpy()
        bins = pd.cut(g["vegetation_fraction"], [0, 0.2, 0.5, 1.0], include_lowest=True)
        by_bin = g.groupby(bins, observed=False)["inliers"].median()
        fno = g["frame_id"].str.extract(r"(\d+)")[0].astype(int)
        problem = g[(fno >= PROBLEM_FRAMES[0]) & (fno <= PROBLEM_FRAMES[1])]
        rows.append({
            "run": run, "frames": len(g),
            "spearman_inliers_vs_vegetation": np.corrcoef(rank(g["inliers"]),
                                                          rank(g["vegetation_fraction"]))[0, 1],
            "spearman_inliers_vs_texture": np.corrcoef(rank(g["inliers"]),
                                                       rank(g["texture"]))[0, 1],
            "inliers_median_veg<0.2": by_bin.iloc[0], "inliers_median_veg0.2-0.5": by_bin.iloc[1],
            "inliers_median_veg>0.5": by_bin.iloc[2],
            "inliers_median_frames81-84": problem["inliers"].median(),
            "vegetation_frames81-84": problem["vegetation_fraction"].mean(),
            "inliers_median_all": g["inliers"].median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Графіки
# ---------------------------------------------------------------------------
def style_line(run):
    return dict(color=COLORS.get(run["matcher"], INK_2), linestyle=LINESTYLES.get(run["ransac"], "-"),
                linewidth=1.6)


def shade_problem(ax):
    ax.axvspan(PROBLEM_FRAMES[0] - 0.5, PROBLEM_FRAMES[1] + 0.5, color=INK_3, alpha=0.12, lw=0)


def fig_error_over_time(runs, gt, out):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    if gt is not None:
        for name, run in runs.items():
            e = trajectory_errors(run["traj"], gt)
            e["frame_no"] = e["frame_id"].str.extract(r"(\d+)")[0].astype(int)
            # Лише опорні кадри: інтерпольований еталон зрізає повороти, і похибка
            # на проміжних кадрах відображала б похибку інтерполяції, а не системи.
            key = e[reliable_keyframes(e)]
            ax.plot(key["frame_no"], key["error"], marker="o", ms=4, label=label(run),
                    **style_line(run))
        ax.set_ylabel("Похибка відносно еталону, px карти")
        ax.set_title("Похибка локалізації на опорних кадрах (ручний еталон)", loc="left")
    else:
        ref = runs.get("lightglue_ransac")
        for name, run in runs.items():
            if run is ref or ref is None:
                continue
            m = ref["traj"].merge(run["traj"], on="frame_id", suffixes=("_r", ""))
            d = np.hypot(m["x_map"] - m["x_map_r"], m["y_map"] - m["y_map_r"])
            ax.plot(m["frame_no"], d, label=label(run), **style_line(run))
        ax.set_ylabel("Відхилення між траєкторіями, px карти")
        ax.set_title("Відхилення від LightGlue + RANSAC (еталон ще не розмічено)", loc="left")
    shade_problem(ax)
    ax.set_xlabel("Номер кадру (1 кадр/с)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=4, fontsize=8)
    ax.annotate("кадри 81–84", xy=(np.mean(PROBLEM_FRAMES), 1.0), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "error_over_time.png")
    plt.close(fig)


def fig_trajectories(runs, gt, map_rgb, out):
    pts = [r["traj"][["x_map", "y_map"]].dropna().to_numpy() for r in runs.values()]
    if gt is not None:
        pts.append(gt[["x_true", "y_true"]].to_numpy())
    allp = np.vstack(pts)
    x0, y0 = np.maximum(allp.min(0) - 250, 0).astype(int)
    x1, y1 = np.minimum(allp.max(0) + 250, map_rgb.shape[1::-1]).astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 7.5 * (y1 - y0) / (x1 - x0)))
    ax.imshow(map_rgb[y0:y1, x0:x1], extent=(x0, x1, y1, y0), alpha=0.55)
    for run in runs.values():
        t = run["traj"].dropna(subset=["x_map"])
        ax.plot(t["x_map"], t["y_map"], label=label(run), **style_line(run))
    if gt is not None:
        ax.plot(gt["x_true"], gt["y_true"], color=INK, lw=1.0, ls=":", label="Еталон (інтерп.)")
        key = gt[reliable_keyframes(gt)]
        bad = gt[gt["is_keyframe"] & ~reliable_keyframes(gt)]
        if len(bad):
            ax.plot(bad["x_true"], bad["y_true"], "x", ms=6, mew=1.5, color=INK_3,
                    label="Опорний кадр, ненадійна розмітка")
        ax.plot(key["x_true"], key["y_true"], "o", ms=5, mfc="white", mec=INK, mew=1.4,
                label="Еталон (опорні кадри)")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xlabel("X, px карти")
    ax.set_ylabel("Y, px карти")
    ax.grid(False)
    ax.set_title("Траєкторії: еталон і оцінки" if gt is not None else "Оцінені траєкторії",
                 loc="left")
    ax.legend(loc="lower left", fontsize=8, facecolor="white", framealpha=0.85, frameon=True)
    fig.tight_layout()
    fig.savefig(out / "trajectories_vs_gt.png")
    plt.close(fig)


def fig_terrain(ter, runs, out):
    ransac_runs = [n for n, r in runs.items() if r["ransac"] == "RANSAC"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for n in ransac_runs:
        g = ter[ter["run"] == n].copy()
        g["frame_no"] = g["frame_id"].str.extract(r"(\d+)")[0].astype(int)
        ax1.plot(g["frame_no"], g["inliers"], label=runs[n]["matcher"], **style_line(runs[n]))
    ax1.set_ylabel("Геометрично узгоджені\nвідповідності")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left")
    ax1.set_title("Кількість узгоджених відповідностей і рослинність під кадром", loc="left")
    ref = ter[ter["run"] == ransac_runs[0]].copy() if ransac_runs else ter.iloc[:0]
    ref["frame_no"] = ref["frame_id"].str.extract(r"(\d+)")[0].astype(int)
    ax2.fill_between(ref["frame_no"], ref["vegetation_fraction"], color="#008300", alpha=0.35,
                     lw=0, step="mid")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Частка рослинності\n(ExG, карта)")
    ax2.set_xlabel("Номер кадру")
    for ax in (ax1, ax2):
        shade_problem(ax)
    ax1.annotate("кадри 81–84", xy=(np.mean(PROBLEM_FRAMES), ax1.get_ylim()[1] * 0.92),
                 ha="center", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "inliers_vs_terrain.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for n in ransac_runs:
        g = ter[ter["run"] == n]
        ax.scatter(g["vegetation_fraction"], g["inliers"], s=16, alpha=0.8,
                   color=COLORS.get(runs[n]["matcher"], INK_2), edgecolor="white", lw=0.5,
                   label=runs[n]["matcher"])
    ax.set_xlabel("Частка рослинності під кадром (ExG, карта)")
    ax.set_ylabel("Узгоджені відповідності")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title("Відповідності vs тип місцевості", loc="left")
    fig.tight_layout()
    fig.savefig(out / "inliers_vs_vegetation_scatter.png")
    plt.close(fig)


def fig_robustness(table, out):
    perts = list(dict.fromkeys(table["perturbation"]))
    matchers = list(dict.fromkeys(table["matcher"]))
    y = np.arange(len(perts))
    h = 0.8 / len(matchers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 0.32 * len(perts) + 1.4), sharey=True)
    for i, m in enumerate(matchers):
        t = table[table["matcher"] == m].set_index("perturbation").reindex(perts)
        offset = (i - (len(matchers) - 1) / 2) * h
        ax1.barh(y + offset, t["retention_median"], height=h * 0.9, color=COLORS.get(m, INK_2),
                 label=m)
        ax2.barh(y + offset, t["localized_rate"] * 100, height=h * 0.9, color=COLORS.get(m, INK_2))
        # Нульові значення не мають видимого стовпчика — підписуємо їх явно.
        for ax, values in ((ax1, t["retention_median"]), (ax2, t["localized_rate"] * 100)):
            for yi, v in zip(y + offset, values):
                if not v > 0:
                    ax.text(0.01, yi, f"{m}: 0", va="center", fontsize=7, color=INK_2,
                            transform=ax.get_yaxis_transform())
    ax1.set_yticks(y, perts)
    ax1.invert_yaxis()
    ax1.axvline(1.0, color=INK_3, lw=0.8)
    ax1.set_xlabel("Медіана N_inliers / N_inliers(clean)")
    ax2.set_xlabel("Кадри локалізовано, %")
    ax2.set_xlim(0, 100)
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    fig.suptitle("Стійкість до спотворень кадру (19 кадрів, вікно 3×3, без компенсації повороту)",
                 x=0.01, ha="left", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "robustness.png")
    plt.close(fig)


def fig_timing(runs, out):
    names = list(runs)
    data = [runs[n]["traj"].loc[runs[n]["traj"]["search_mode"].isin(LOCAL_MODES), "process_time"]
            for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(names) + 1.2))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "lw": 1.2}, flierprops={"ms": 3})
    for patch, n in zip(bp["boxes"], names):
        patch.set(facecolor=COLORS.get(runs[n]["matcher"], INK_2), alpha=0.55,
                  hatch="//" if runs[n]["ransac"] == "USAC_MAGSAC" else "",
                  edgecolor=COLORS.get(runs[n]["matcher"], INK_2))
    ax.set_yticks(range(1, len(names) + 1), [label(runs[n]) for n in names])
    ax.set_xlabel("Час обробки кадру в режимі TRACKING, с")
    ax.set_title("Час локального кроку (вікно 3×3)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "timing_local.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ground-truth", default=str(GROUND_TRUTH_PATH))
    args = p.parse_args()

    tables_out = RESULTS_DIR / "summary"
    figs_out = FIGURES_DIR / "analysis"
    tables_out.mkdir(parents=True, exist_ok=True)
    figs_out.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(args.ground_truth) if Path(args.ground_truth).is_file() else None
    if gt is None:
        print("Ground truth: not found — error metrics skipped")
    else:
        reliable = reliable_keyframes(gt)
        extrapolated = (reliable & ~gt["center_in_hull"].astype(bool)
                        if "center_in_hull" in gt else reliable & False)
        print(f"Ground truth: {int(gt['is_keyframe'].sum())} keyframes ({int(reliable.sum())} "
              f"reliable, used for key_* metrics), {len(gt)} rows")
        if extrapolated.any():
            print("  centre extrapolated outside the clicked points (check if possible): "
                  + ", ".join(gt.loc[extrapolated, "frame_id"]))

    runs = load_runs()
    all_runs, runs = runs, baseline_runs(runs)
    if all_runs:
        bench = benchmark_table(all_runs, gt)
        note = ("Похибки key_* — за опорними кадрами з надійною ручною розміткою (reliable), "
                "px карти. Метрики all_* за інтерпольованим еталоном є лише в CSV: на поворотах "
                "між опорними кадрами вони вимірюють похибку інтерполяції, а не системи.") \
            if gt is not None else \
            "Еталон (ground_truth.csv) відсутній: похибки не обчислено."
        # all_* (інтерпольований еталон) — лише в CSV: на поворотах між опорними кадрами
        # вони вимірюють похибку інтерполяції, а не системи.
        save_table(bench, "benchmark_table", tables_out, note, md_drop_prefixes=("all_",))
        if len(all_runs) > len(runs):
            save_table(variants_table(bench, all_runs), "variants_table", tables_out,
                       "Варіанти системи порівняно з baseline того самого методу; d_* — "
                       "різниця з baseline (від'ємна похибка чи час — покращення). Похибки — "
                       "на надійних опорних кадрах, px карти.")
        if gt is not None:
            sm = SmoothingConfig()
            save_table(smoothing_table(all_runs, gt, sm), "smoothing_table", tables_out,
                       f"Похибка на надійних опорних кадрах, px карти. Фільтр Калмана зі сталою "
                       f"швидкістю: σ_z = {sm.meas_std_px:g} px, σ_a = {sm.accel_std_px_s2:g} px/с². "
                       f"better_frames — на скількох кадрах похибка менша, ніж без згладжування.")
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print(bench.round(3).T)
    if runs:
        save_table(deviation_table(runs), "trajectory_deviation", tables_out,
                   "Відхилення позицій між прогонами на спільних кадрах, px карти.")
        fig_error_over_time(runs, gt, figs_out)
        fig_trajectories(runs, gt, read_image_rgb(DEFAULT_MAP_PATH), figs_out)
        fig_timing(runs, figs_out)

    rob_path = EXPERIMENTS_DIR / "robustness" / "robustness.csv"
    if rob_path.is_file():
        rob = robustness_table(pd.read_csv(rob_path))
        save_table(rob, "robustness_table", tables_out,
                   "retention = N_inliers / N_inliers(clean) для того самого кадру; shift — "
                   "зміщення центру відносно неспотвореного кадру, px карти.")
        fig_robustness(rob, figs_out)

    ter_path = EXPERIMENTS_DIR / "terrain" / "terrain.csv"
    if ter_path.is_file() and runs:
        ter = pd.read_csv(ter_path)
        save_table(terrain_table(ter), "terrain_table", tables_out,
                   "Кореляція Спірмена та медіана N_inliers за частками рослинності під кадром.")
        fig_terrain(ter, runs, figs_out)
    print(f"Tables: {tables_out}\nFigures: {figs_out}")


if __name__ == "__main__":
    main()
