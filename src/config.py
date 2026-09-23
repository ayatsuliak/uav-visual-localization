"""Централізована конфігурація проєкту.

Усі шляхи будуються від кореня проєкту, тому скрипти можна запускати
з будь-якої робочої директорії.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Каталоги даних --------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

VIDEO_DIR = RAW_DIR / "video"
MAPS_DIR = RAW_DIR / "maps"
FRAMES_DIR = PROCESSED_DIR / "frames"
TILES_DIR = PROCESSED_DIR / "tiles"

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"

# --- Вхідні файли ----------------------------------------------------------
DEFAULT_MAP_PATH = MAPS_DIR / "university_map.png"
DEFAULT_VIDEO_PATH = VIDEO_DIR / "DJI_0331.mp4"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# --- Метадані --------------------------------------------------------------
TILES_METADATA_PATH = METADATA_DIR / "tiles.csv"
TILES_RUN_INFO_PATH = METADATA_DIR / "tiles_info.json"
FRAMES_METADATA_PATH = METADATA_DIR / "frames.csv"
FRAMES_RUN_INFO_PATH = METADATA_DIR / "frames_info.json"
EVAL_PAIRS_PATH = METADATA_DIR / "eval_pairs.csv"
MATCHING_RESULTS_PATH = RESULTS_DIR / "raw" / "matching_results.csv"
# --- Етап 1: тайлування ортофотоплану --------------------------------------
TILE_SIZE = 512                 # S_tile, px (квадратний тайл)
TILE_OVERLAP = 0.5              # частка перекриття сусідніх тайлів
ALPHA_BACKGROUND_RGB = (255, 255, 255)  # тло для альфа-блендингу RGBA -> RGB
TILE_NAME_TEMPLATE = "tile_{index:04d}.png"   # нумерація з 1

# --- Етап 1: вилучення кадрів ----------------------------------------------
FRAME_SAMPLING_FPS = 1.0        # цільова частота вибірки кадрів
FRAME_NAME_TEMPLATE = "frame_{index:04d}.png"  # нумерація з 1

# --- Етап 2: нормалізація масштабу ------------------------------------------
# Скільки пікселів карти припадає на один піксель кадру (s_f = GSD_frame / GSD_map).
# Калібрування (етап 2): frame_0002 зіставлено з фрагментом карти 1024×1024,
# що повністю містить кадр; LightGlue і LoFTR (RANSAC/MAGSAC, work_scale 1.0/1.5)
# узгоджено дають масштаб H 0.267–0.277. Отже, кадр 1920×1080 покриває ≈518×292 px
# карти. Перевіряється за колонкою h_scale у результатах scripts/evaluate.py.
FRAME_TO_MAP_SCALE = 0.27
# Спільний множник роздільності для кадру і тайла відносно роздільності карти:
# 1.0 -> тайл подається у matcher як є (512 px), кадр -> ≈480×270 px.
WORK_SCALE = 1.0

# --- Етап 2: зіставлення ----------------------------------------------------
MAX_KEYPOINTS = 2048            # SuperPoint (LightGlue)
LOFTR_PRETRAINED = "outdoor"    # ваги Kornia LoFTR
TIMING_REPEATS = 1              # повторів для вимірювання часу (медіана)
WARMUP_ITERATIONS = 2           # «прогрівання» моделі перед вимірюванням часу

# --- Етап 2: геометрична перевірка (RANSAC + гомографія) --------------------
# Усі пороги — у пікселях повної карти (гомографія: кадр -> тайл в оригінальних px).
RANSAC_METHOD = "RANSAC"        # RANSAC | USAC_MAGSAC
RANSAC_THRESHOLD = 3.0          # px карти
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000
RANSAC_SEED = 0                 # фіксоване зерно -> відтворювані результати
MIN_INLIERS = 15                # мінімум геометрично узгоджених відповідностей
# Перевірка правдоподібності гомографії (надира, приблизно плоска сцена):
SCALE_TOLERANCE = 2.0           # масштаб H у [s_f / 2, s_f * 2]
MAX_ANISOTROPY = 1.5            # відношення сингулярних чисел якобіана H у центрі кадру


def portable_path(path) -> str:
    """Шлях відносно кореня проєкту (якщо можливо) у POSIX-форматі — для переносних метаданих."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
