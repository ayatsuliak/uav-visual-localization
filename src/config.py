"""Централізована конфігурація проєкту.

Усі шляхи будуються від кореня проєкту, тому скрипти можна запускати
з будь-якої робочої директорії.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Каталоги даних --------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

# --- Набір даних (інший політ / інша карта) ----------------------------------
# UAV_DATASET=<name> -> data/datasets/<name>/dataset.json задає карту, відео, журнал польоту
# і параметри (розділ "overrides"); кадри, тайли, кеш, метадані й еталон — у
# data/datasets/<name>/{processed,metadata}/, результати — в experiments/datasets/<name>/.
# Без змінної використовується основний набір (перший політ), шляхи не змінюються.
DATASET = os.environ.get("UAV_DATASET") or None
DATASET_DIR = DATA_DIR / "datasets" / DATASET if DATASET else None
DATASET_INFO: dict = {}
if DATASET_DIR is not None:
    _info_path = DATASET_DIR / "dataset.json"
    if not _info_path.is_file():
        raise FileNotFoundError(f"UAV_DATASET={DATASET}: {_info_path} not found")
    DATASET_INFO = json.loads(_info_path.read_text(encoding="utf-8"))
    PROCESSED_DIR = DATASET_DIR / "processed"
    METADATA_DIR = DATASET_DIR / "metadata"
    EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "datasets" / DATASET

VIDEO_DIR = RAW_DIR / "video"
MAPS_DIR = RAW_DIR / "maps"
FRAMES_DIR = PROCESSED_DIR / "frames"
TILES_DIR = PROCESSED_DIR / "tiles"

RESULTS_DIR = EXPERIMENTS_ROOT / "results"
FIGURES_DIR = EXPERIMENTS_ROOT / "figures"

# --- Окремі серії експериментів (основні дані й результати не змінюються) ---
# UAV_TILESET=<name>    -> тайли, tiles.csv і кеш ознак у data/processed/tilesets/<name>/;
# UAV_EXPERIMENT=<name> -> результати й графіки в experiments/series/<name>/{results,figures}/.
# Якщо задано лише UAV_TILESET, серія результатів має ту саму назву: прогін на іншому
# наборі тайлів ніколи не пише в основні experiments/results.
TILESET = os.environ.get("UAV_TILESET") or None
EXPERIMENT_SERIES = os.environ.get("UAV_EXPERIMENT") or TILESET
TILESET_DIR = PROCESSED_DIR / "tilesets" / TILESET if TILESET else None
if TILESET_DIR is not None:
    TILES_DIR = TILESET_DIR / "tiles"
if EXPERIMENT_SERIES:
    RESULTS_DIR = EXPERIMENTS_ROOT / "series" / EXPERIMENT_SERIES / "results"
    FIGURES_DIR = EXPERIMENTS_ROOT / "series" / EXPERIMENT_SERIES / "figures"

# --- Вхідні файли ----------------------------------------------------------
DEFAULT_MAP_PATH = MAPS_DIR / "university_map.png"
DEFAULT_VIDEO_PATH = VIDEO_DIR / "DJI_0331.mp4"
if DATASET_INFO:
    DEFAULT_MAP_PATH = PROJECT_ROOT / DATASET_INFO["map"]
    DEFAULT_VIDEO_PATH = PROJECT_ROOT / DATASET_INFO["video"]
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# --- Метадані --------------------------------------------------------------
TILES_METADATA_PATH = (TILESET_DIR or METADATA_DIR) / "tiles.csv"
TILES_RUN_INFO_PATH = (TILESET_DIR or METADATA_DIR) / "tiles_info.json"
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

# --- Етап 3: вибір тайла (src/evaluation/scoring.py) ------------------------
MIN_INLIER_RATIO = 0.15         # мінімальна частка геометрично узгоджених відповідностей
MAX_REPROJ_ERROR = 3.5          # px карти, середня похибка репроєкції (запобіжник)
CENTER_MARGIN = 30.0            # px: допуск для «центр кадру всередині тайла»
SCORE_SIGMA_R = 2.0             # px: регуляризація похибки в балі ранжування
# Штраф за відхилення масштабу H від очікуваного s_f: exp(-ln²(s / s_f) / (2·σ_s²)).
# None — вимкнено (так отримано результати Етапу 5); рекомендоване значення 0.15 (≈ ±15 %).
SCORE_SIGMA_SCALE = None
TOP_K = 5                       # скільки найкращих кандидатів виводити

# --- Етап 4: трекінг послідовності кадрів (src/localization/tracker.py) -------
NEIGHBORHOOD_RADIUS = 1         # 1 -> вікно 3×3 тайли навколо базового тайла
MAX_MISSES = 3                  # кадрів поспіль без прийнятої оцінки -> стан LOST
# D_max = V_max · Δt + D_margin; за Δt = 1 с -> 150 px карти.
MAX_SPEED_PX_PER_S = 120.0      # px карти / с (≈15 м/с — оцінка, масштаб карти в метрах невідомий)
JUMP_MARGIN_PX = 30.0           # px карти: запас на похибку оцінки положення
INIT_MIN_SCORE = 5.0            # мінімальний бал для (ре)ініціалізації з глобального пошуку
# Компенсація курсу: кадр повертається на прогнозований кут «кадр -> карта».
MAX_YAW_RATE_DEG_PER_S = 45.0   # обмеження кутової швидкості під час екстраполяції курсу
# Якщо з прогнозованим курсом тайл не знайдено — повтор з цими зсувами курсу (°).
YAW_RETRY_OFFSETS_DEG = (45.0, -45.0, 90.0, -90.0)
# У стані LOST: зсув курсу для глобального пошуку циклічно змінюється від кадру до кадру.
GLOBAL_YAW_CYCLE_DEG = (0.0, 90.0, 180.0, -90.0)
ROTATION_BORDER_PX = 4          # звуження маски коректної області після повороту, px
# Вікно за прогнозом (TRACKING): замість 3×3 тайлів — одна ділянка карти навколо очікуваного
# контуру огляду (контур останнього прийнятого кадру, зсунутий у прогнозовану позицію).
# Не вдалося — звичайний пошук у 3×3 тайлах. False — вимкнено (так отримано результати Етапу 5).
PREDICTED_WINDOW = False
WINDOW_MARGIN_PX = 128.0        # запас навколо контуру, px карти (≥ прогноз + зміна bbox при повороті)
WINDOW_MAX_SIDE_PX = 1024       # більше вікно -> пошук у тайлах (пам'ять LoFTR, час)
# Уточнення після глобального пошуку: повторне зіставлення з ділянкою карти навколо контуру
# огляду переможця (кадр потрапляє в неї повністю, а не обрізається краєм тайла).
REFINE_AFTER_GLOBAL = False
REFINE_MARGIN_PX = 64.0         # запас навколо контуру переможця, px карти
REFINE_MAX_SHIFT_PX = 30.0      # уточнення не може зсунути центр далі, ніж на стільки, px карти
TRACKING_DIR = RESULTS_DIR / "tracking"
# Згладжування траєкторії (src/localization/smoothing.py): фільтр Калмана, стала швидкість.
# Значення задано апріорі, а не підібрано за еталоном (інакше оцінка на опорних кадрах завищена).
KALMAN_MEAS_STD_PX = 4.0        # σ_z: похибка одиничної оцінки положення, px карти
# σ_a: «маневреність». Крок польоту ≈ 44 px/с (Етап 4); розворот на 90° за ~3 с змінює
# швидкість на ≈ 62 px/с, тобто ≈ 20 px/с².
KALMAN_ACCEL_STD_PX_S2 = 20.0

# --- Етап 5: ground truth і експерименти -------------------------------------
GROUND_TRUTH_PATH = METADATA_DIR / "ground_truth.csv"
GROUND_TRUTH_POINTS_PATH = METADATA_DIR / "ground_truth_points.json"
# Показ компаса для кожного кадру (scripts/make_compass.py); використовується лише з
# track_sequence.py --compass.
COMPASS_PATH = METADATA_DIR / "compass.csv"
# Δ у θ = ψ + Δ: магнітне схилення + кут зближення меридіанів сітки карти, °.
COMPASS_OFFSET_DEG = 0.0
# Розбіжність курсу з компаса й прогнозу (з гомографії), за якої компас додається
# як окрема гіпотеза курсу в стані TRACKING, °.
COMPASS_MIN_DIFF_DEG = 10.0
# Опорні кадри обрано візуально (experiments/figures/analysis/frames_contact_sheet.png) так,
# щоб на кадрі були чіткі наземні орієнтири (дороги, розмітка, паркінги, перехрестя), а кадри
# рівномірно покривали політ, включно з розворотами (19, 51, 72, 94) і ділянкою біля
# дерев (82 — там видно паркінг). Кадри 33–40 (пустир) і 66–69 (канал, дерева) пропущено:
# там немає надійних точок, тож еталон для них — лише інтерпольований.
GT_KEYFRAMES = (1, 6, 12, 19, 25, 31, 44, 51, 60, 64, 72, 76, 80, 82, 90, 94)
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

# --- Етап 3: кеш ознак тайлів ------------------------------------------------
SUPERPOINT_CACHE_DIR = (TILESET_DIR or PROCESSED_DIR) / "cache_superpoint"
LOCALIZATION_RESULT_PATH = RESULTS_DIR / "localization_single_frame.json"


def portable_path(path) -> str:
    """Шлях відносно кореня проєкту (якщо можливо) у POSIX-форматі — для переносних метаданих."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --- Параметри набору даних ----------------------------------------------------
# Розмір пікселя карти, м (None — карта без прив'язки, похибки лише в px).
MAP_GSD_M = None
# Скалярні параметри з dataset.json["overrides"] (FRAME_TO_MAP_SCALE, TILE_SIZE, ...).
# Шляхи тут не змінюються — вони вже задані вище.
for _key, _value in DATASET_INFO.get("overrides", {}).items():
    if _key not in globals() or isinstance(globals()[_key], Path):
        raise KeyError(f"dataset.json: unknown or non-scalar parameter {_key}")
    globals()[_key] = tuple(_value) if isinstance(_value, list) else _value
