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

# --- Зіставлення та геометрія (етапи 2+) ------------------------------------
IMAGE_SIZE = (512, 512)         # робочий розмір зображень для matcher (W, H)
MAX_KEYPOINTS = 2048
RANSAC_THRESHOLD = 5.0
MIN_INLIERS = 10


def portable_path(path) -> str:
    """Шлях відносно кореня проєкту (якщо можливо) у POSIX-форматі — для переносних метаданих."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
