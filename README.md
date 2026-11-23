# UAV Visual Localization

GNSS-free visual localization of a UAV by matching onboard video frames against a georeferenced-style orthophoto map, using the Transformer-based matchers **LightGlue** and **LoFTR**.

This repository contains the software prototype for the Master's thesis *"Modeling Navigation of Aerial Platforms Using Computer Vision Methods Based on Transformer Architectures"* (Ivan Franko National University of Lviv, Faculty of Applied Mathematics and Informatics). It continues the author's term paper, which compared LightGlue and LoFTR on known frame–tile pairs.

## Goal

The system should not only match two images, but answer the question *"where on the map is this frame?"* and, for a sequence of frames, reconstruct the flight trajectory:

```
UAV video → frames → candidate map tiles → LightGlue / LoFTR → RANSAC + homography
          → best tile → position on the full map → temporal filtering → trajectory
```

Out of scope: training new networks, re-implementing LightGlue/LoFTR, full SLAM, autopilot integration (PX4/ArduPilot), sensor fusion with GPS/IMU/LiDAR.

## Project status

| Stage | Description | Status |
|---|---|---|
| 1 | Data preparation: map tiling, frame extraction, metadata, integrity checks | ✅ done |
| 2 | Unified pipeline `frame + tile → matcher → RANSAC → metrics` | ✅ done |
| 3 | Automatic best-tile search and ranking | ✅ done |
| 4 | Frame position on the full map (centre + footprint) | ✅ done (single frame, Stage 3) |
| 5–6 | Frame sequences, temporal consistency, neighbour-tile search (Stage 4) | ✅ done |
| 7–10 | Experiments, ground truth, error statistics, figures (Stage 5) | 🧪 experiments done; ground truth to annotate |

## Repository structure

```text
uav-visual-localization/
├── data/
│   ├── raw/
│   │   ├── maps/          # university_map.png            (local only, not in Git)
│   │   └── video/         # DJI_0331.mp4                  (local only, not in Git)
│   ├── processed/
│   │   ├── frames/        # frame_0001.png, ...           (generated)
│   │   ├── tiles/         # tile_0001.png, ...            (generated)
│   │   └── cache_superpoint/  # tile_XXXX.npz + manifest.json (generated)
│   └── metadata/
│       ├── tiles.csv                  # tile registry (generated, committed)
│       ├── frames.csv                 # frame registry (generated, committed)
│       ├── eval_pairs.csv             # verified frame–tile pairs (manual)
│       ├── eval_pairs_legacy_256.csv  # old pairs, marked INVALID (kept for traceability)
│       ├── golden_pair.csv            # reference pair from the term paper
│       ├── ground_truth.csv           # reference frame centres (manual, Stage 5)
│       └── ground_truth_points.json   # manual frame↔map correspondences
├── src/
│   ├── config.py          # paths and default parameters
│   ├── run_info.py        # git commit, device, map size for result metadata
│   ├── preprocessing/     # images.py, tiles.py, frames.py, validation.py, cache.py
│   ├── matching/          # base.py (ImageMatcher), lightglue.py, loftr.py, pipeline.py
│   ├── geometry/          # homography.py (RANSAC, reprojection error, plausibility)
│   ├── evaluation/        # metrics.py, scoring.py, ground_truth.py, terrain.py
│   ├── localization/      # coordinates.py (frame → map), search.py (tile search), tracker.py
│   └── visualization/     # matches.py, localization.py, trajectory.py
├── scripts/               # command-line entry points (see below)
├── tests/                 # unit tests on synthetic data
├── experiments/
│   ├── results/{raw,summary,rankings,tracking,experiments}/
│   └── figures/{matching,localization,analysis}/
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ is required.

```bash
git clone https://github.com/<your-username>/uav-visual-localization.git
cd uav-visual-localization

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate

# For GPU, install a CUDA build of PyTorch first (the cu124 index stops at torch 2.6).
# Used for the thesis: torch 2.14.0 + CUDA 12.6 on a GTX 1650 (sm_75):
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Stage 1 (data preparation and tests) only needs `opencv-python`, `numpy` and `pandas`; PyTorch, Kornia and LightGlue are required from Stage 2 on. Pretrained weights are downloaded automatically on first use.

## Data

Raw data is **not** stored in the repository (size and third-party licensing of the imagery). Place the files locally:

| File | Location | Properties |
|---|---|---|
| Orthophoto map | `data/raw/maps/university_map.png` | 6200 × 4320 px, RGBA (alpha fully opaque) |
| UAV video | `data/raw/video/DJI_0331.mp4` | 1920 × 1080, 30 FPS, ~94 s, near-nadir |

All map coordinates are **pixel** coordinates. The map has no georeference or known ground sampling distance yet, so errors are reported in map pixels, not metres.

Observed properties relevant to the method (from the reference pair):

- the video was recorded in a different season than the map (leafless winter vs. summer imagery);
- one frame pixel corresponds to ≈0.27 map pixels (calibrated in Stage 2, see below), so a full 1920 × 1080 frame covers about **518 × 292 map pixels**, which is wider than a 512 px tile;
- the scene is not planar: tall buildings cause parallax between roofs and the ground.

## Conventions (mandatory for all modules)

- **Points:** `(x, y) = (column, row)`, origin at the top-left corner, `x ∈ [0, W−1]`, `y ∈ [0, H−1]`; arrays are indexed as `image[y, x]`.
- **Map coordinates:** always in pixels of the full original map, including tiles produced from an ROI.
- **Images in memory:** `uint8`, shape `(H, W, 3)`, **RGB**. BGR↔RGB conversion happens only in `src/preprocessing/images.py`.
- **Model tensors:** `(B, C, H, W)`, `float32`, values in `[0, 1]` (`image_to_tensor`).
- **Naming:** `tile_0001.png`, `frame_0001.png`; numbering starts at 1; tiles are ordered row by row (row-major).
- **I/O:** images are read and written with `imdecode`/`imencode`, so paths with non-ASCII characters work on Windows.

## Stage 1: data preparation

All scripts can be run from any directory as `python scripts/<name>.py`.

### 1. Map tiling

```bash
python scripts/prepare_map.py                                # 512 px, 50 % overlap → 384 tiles
python scripts/prepare_map.py --tile-size 768 --overlap 0.25 # parameter experiments
python scripts/prepare_map.py --roi 912 1409 768 1024        # tile only a region of interest
```

- stride: `stride = floor(S · (1 − overlap))` (256 px for the defaults);
- if the regular grid does not reach the right/bottom border, one extra edge tile is added at `W − S` / `H − S`, so every tile has the same size and every map pixel is covered;
- tile count: `(ceil((W−S)/stride) + 1) · (ceil((H−S)/stride) + 1)` = 24 × 16 = **384** for the 6200 × 4320 map;
- alpha is blended onto a configurable background (`--background R G B`, white by default); for a fully opaque map this is bit-identical to dropping the alpha channel;
- old `tile_*.png` files are deleted before writing (`--keep-existing` disables this).

Outputs:

- `data/processed/tiles/tile_XXXX.png`: RGB, lossless PNG;
- `data/metadata/tiles.csv`: `tile_id, x_origin, y_origin, width, height`;
- `data/metadata/tiles_info.json`: run parameters, map SHA-256, grid origins (useful for neighbour search).

### 2. Frame extraction

```bash
python scripts/prepare_frames.py                  # 1 FPS (every 30th frame) → ~94 frames
python scripts/prepare_frames.py --limit 30       # first 30 sampled frames only (debugging)
python scripts/prepare_frames.py --fps 2          # a different sampling rate
python scripts/prepare_frames.py --every-n 15     # explicit step in raw frames
python scripts/prepare_frames.py --start-sec 10   # skip the beginning of the video
```

Frames are decoded sequentially (seeking with `CAP_PROP_POS_FRAMES` is unreliable for H.264/H.265), saved at full resolution as lossless PNG, and timestamped as `timestamp_sec = source_frame_idx / fps`. If `--video` is omitted, `DJI_0331.mp4` or the only video in `data/raw/video/` is used.

Outputs:

- `data/processed/frames/frame_XXXX.png`;
- `data/metadata/frames.csv`: `frame_id, source_frame_idx, timestamp_sec, width, height`;
- `data/metadata/frames_info.json`: video properties and sampling parameters.

### 3. Integrity check

```bash
python scripts/check_data_integrity.py            # tiles + frames; exit code 0 = OK, 1 = problems
python scripts/check_data_integrity.py --sample 50
python scripts/check_data_integrity.py --skip-frames
```

The check verifies:

- CSV schemas without NaN, unique identifiers, and tiles within the map bounds;
- full map coverage and the expected tile count;
- no missing or stale files, and three channels without alpha;
- **pixel identity of every tile with the map crop at `(x_origin, y_origin)`**;
- frame sequences: strictly increasing indices with a constant step, `timestamp = idx / fps`, and frame size.

## Stage 2: unified matching pipeline

```bash
python scripts/evaluate.py                        # LightGlue and LoFTR on data/metadata/eval_pairs.csv
python scripts/evaluate.py --repeats 5            # median time over 5 runs
python scripts/evaluate.py --methods LoFTR --work-scale 1.5 --ransac-method USAC_MAGSAC \
    --output experiments/results/raw/loftr_ws15_magsac.csv
python scripts/summarize.py                       # aggregates into experiments/results/summary/
```

Both methods run through the same `PairMatchingPipeline` (`src/matching/pipeline.py`), with the same preprocessing, geometric verification and success criterion:

```
frame (1920×1080) ──resize × s_f·k──┐
                                    ├─ extract / match_features ─ points → original px ─ RANSAC ─ plausibility ─ metrics
tile  (512×512)   ──resize × k──────┘
```

- **Scale normalization.** `s_f = FRAME_TO_MAP_SCALE = 0.27` map pixels per frame pixel; `k = WORK_SCALE = 1.0`. Frame and tile are brought to the same resolution, and the whole frame is used (no centre crop).
- **One coordinate system.** Matchers work at working resolution, but the pipeline maps all correspondences back to *original* frame and tile pixels (pixel-centre aware). The homography `H` therefore maps frame pixels → tile pixels. The RANSAC threshold, reprojection error and frame footprint are in **map pixels**, whatever the method or `WORK_SCALE`.
- **Common matcher interface** (`src/matching/base.py`): `extract(image) → Features` and `match_features(f0, f1) → MatchResult`, plus `warmup()`.
  - LightGlue: SuperPoint keypoints + LightGlue.
  - LoFTR: Kornia, `outdoor` weights, input sizes are multiples of 8.
  - Tile features can be cached (`tile_cache_key`), and frame features can be reused across tiles (used in Stage 3).
- **Fair timing.**
  - Warm-up on images of the experiment size, and `torch.cuda.synchronize()` around every timed block.
  - Time is split into frame extraction, tile extraction, matching and RANSAC. Images are read from disk before timing.
  - LoFTR has no separate detection step, so its whole cost is in `time_match`.

### Metrics (`experiments/results/raw/matching_results.csv`)

| Column | Definition |
|---|---|
| `matches` | correspondences returned by the method |
| `inliers` | geometrically consistent correspondences: RANSAC reprojection error ≤ `RANSAC_THRESHOLD` = 3 map px |
| `inlier_ratio` | `inliers / matches` |
| `reproj_mean/median/rmse/max` | forward error `‖H·p_frame − p_tile‖` over inliers, in map px (bounded by the RANSAC threshold by construction) |
| `h_scale`, `h_rotation_deg`, `h_anisotropy` | local similarity parameters of `H` at the frame centre (Jacobian) |
| `h_plausible` | orientation preserved, footprint convex, no sign change of the projective denominator, anisotropy ≤ 1.5, scale within `[s_f/2, 2·s_f]` |
| `success` | `H` found **and** `inliers ≥ MIN_INLIERS` (15) **and** `h_plausible`; otherwise `failure_reason` says why |
| `center_x_map`, `center_y_map` | projection of the frame centre onto the full map (px) |
| `time_*` | seconds; `time_matcher_total = time_extract_frame + time_extract_tile + time_match` |

Each run also writes `*_info.json` with the full pipeline and matcher configuration, the git commit and the device. Figures go to `experiments/figures/matching/pair_XXX/`:

- `*_matches.png`: all correspondences, inliers in green and the rest in red;
- `*_inliers.png`: inliers only;
- `*_footprint.png`: the projected frame outline and centre on the map, with the tile outline.

### Stage 2 findings (frame_0002)

- **The old prototype compared the methods at different resolutions.** `SuperPoint.extract()` rescales its input to 1024 px by default, so LightGlue ran at 1024 × 1024 while LoFTR ran at 512 × 512, and both saw only a centre crop of the frame. The pipeline now disables this implicit resize (`resize=None`).
- **Scale calibration.** Matching frame_0002 against a 1024 × 1024 map crop that contains the whole frame gives a consistent `h_scale` of 0.267–0.277. This holds for both methods, with RANSAC or MAGSAC and `WORK_SCALE` 1.0 or 1.5, and the frame centre estimates agree within ≈16 map px. Hence `FRAME_TO_MAP_SCALE = 0.27`.
- **512 px tiles do not contain a whole frame footprint (≈518 × 292 px).** On the reference tile `tile_0196`, RANSAC fits the plane of the tall building roof: LoFTR gives `h_scale` ≈ 0.20, LightGlue ≈ 0.25. The frame centre still lands within ≈15 px of the full-crop estimate, because it lies on that roof. This motivates the tile-size experiment and a larger search window in Stage 3.
- **Negative control** (frame_0002 against 33 tiles):
  - Tiles that do not overlap the frame yield at most 15 inliers (LightGlue) and 14 (LoFTR), and all of them are rejected, one of them by the plausibility check.
  - Every overlapping tile succeeds with 18–240 inliers.
  - The margin around `MIN_INLIERS = 15` is small; the ranking criterion must be settled in Stage 3.

## Stage 3: automatic tile search and single-frame localization

```bash
python scripts/build_tile_cache.py                 # once: SuperPoint features of all 384 tiles
python scripts/localize_frame.py --frame data/processed/frames/frame_0002.png --matcher lightglue
python scripts/localize_frame.py --matcher loftr --roi 256 1536 2048 3072   # search only inside a ROI
python scripts/localize_frame.py --matcher loftr --grid-step 2              # every 2nd tile row/column
```

No information about the correct tile is used:

```
frame → frame features (once) → for every tile in the pool: matcher → RANSAC → plausibility
      → gate → ranking score → winner tile → (X_map, Y_map) + footprint on the 6200×4320 map
```

### Tile feature cache (`src/preprocessing/cache.py`)

- `build_tile_cache.py` stores SuperPoint keypoints, scores and descriptors of every tile as float32 (no loss) in `data/processed/cache_superpoint/tile_XXXX.npz`.
  - For the current map this is 384 tiles with 523–1609 keypoints each, 347 MiB in total, built in ≈14 s on the GTX 1650.
- `manifest.json` records what the cache depends on: the matcher configuration, `WORK_SCALE`, and the SHA-256 of `tiles.csv` and of the map. If any of them change, loading fails instead of silently mixing features. Rebuild with `--overwrite`.
- At search time the whole cache is loaded into VRAM (`--cache-device model`, the default) or RAM (`--cache-device cpu`).
  - Cached and on-the-fly features give identical results.
- LoFTR is detector-free, so there is nothing to cache. For LoFTR, use `--roi` and/or `--grid-step` to limit the search space.

### Tile selection (`src/evaluation/scoring.py`)

**Step 1, gate.** A candidate is rejected if any condition fails:

| Condition | Default |
|---|---|
| `H` found and plausible (see Stage 2) | — |
| `N_inliers ≥ MIN_INLIERS` | 15 |
| `N_inliers / N_all ≥ MIN_INLIER_RATIO` | 0.15 |
| mean inlier reprojection error `E ≤ MAX_REPROJ_ERROR` | 3.5 map px |
| projected frame centre inside the tile: `−m ≤ x_local ≤ S−1+m`, the same for `y` | `m = CENTER_MARGIN` = 30 px |
| projected frame centre inside the map | 6200 × 4320 |

**Step 2, ranking.** Among the candidates that pass the gate:

```
Score = N_inliers · (N_inliers / N_all) · exp(−E / σ_r),   σ_r = 2 px
```

The highest score wins; ties go to more inliers. With `RANSAC_THRESHOLD = 3` px, `E` is always ≤ 3 px, so the reprojection-error gate never fires. It is kept as a safeguard in case the RANSAC threshold is raised.

**Optional scale prior.** With `SCORE_SIGMA_SCALE = σ_s` (in `src/config.py`, or `--score-sigma-scale` for `localize_frame.py` / `track_sequence.py`), the score is multiplied by `exp(−ln²(s / s_f) / (2σ_s²))`, where `s` is the scale of `H` at the frame centre and `s_f = FRAME_TO_MAP_SCALE`. It favours candidates whose `H` describes the ground plane over those fitted to a tall roof (e.g. frame 2: tile 0220 with `s = 0.207` loses to tile 0196 with `s = 0.267` at `σ_s = 0.15`). The default is `None` (off), which is how the Stage 5 results were produced.

### Global coordinates (`src/localization/coordinates.py`)

`H` maps original frame pixels to original tile pixels. The homogeneous result is always divided by its third component; a point with `w ≈ 0` becomes `NaN` and is rejected. Then:

```
X_map = X_origin + x_local,   Y_map = Y_origin + y_local
```

The project uses pixel-centre coordinates, so the frame centre is `((W−1)/2, (H−1)/2)`, and the footprint is the projection of the four corner pixels `(0,0), (W−1,0), (W−1,H−1), (0,H−1)`.

### Outputs

- `experiments/results/localization_single_frame.json` contains:
  - the frame, matcher and all parameters;
  - the search space (ROI, grid step, checked tile ids) and whether the cache was used;
  - timings;
  - the winner: tile, score, inliers, `H`, centre in local and map coordinates, footprint polygon, and the score ratio to the runner-up;
  - the top-K candidates with gate reasons.
- `experiments/results/rankings/<frame>_<matcher>_candidates.csv`: every checked tile with its metrics, gate outcome and score.
- `experiments/figures/localization/<frame>_global_loc.png` and `_zoom.png`:
  - checked tiles as pale grid lines, and tiles that passed the gate in orange;
  - the winner tile in blue;
  - the camera footprint as a semi-transparent green polygon;
  - the frame centre as a red marker.

### Stage 3 results (frame_0002, GTX 1650)

| Run | Tiles checked | Passed gate | Winner | Centre (X, Y), map px | Time |
|---|---|---|---|---|---|
| LightGlue, whole map, cache | 384 | 5 | `tile_0220` | (1121.4, 2344.7) | 33 s (85 ms/tile) |
| LoFTR, ROI `256 1536 2048 3072` | 56 | 5 | `tile_0197` | (1108.6, 2327.5) | 18 s (313 ms/tile) |
| LoFTR, `--grid-step 2` | 96 | 1 | `tile_0197` | (1108.6, 2327.5) | 32 s (324 ms/tile) |

- Both winners are tiles adjacent to `tile_0196` that contain the frame centre.
- The reference estimate from Stage 2, using a 1024 px crop that holds the whole frame, is ≈(1110, 2330). The LoFTR result is within 3 px of it, and the LightGlue result is within ≈19 px.
- All five tiles that passed the gate overlap the frame footprint. Every tile far from the true position was rejected.
- The winning tile covers only part of the frame, because the footprint (≈518 × 292 px) is wider than a 512 px tile. This is the main limitation to address in the tile-size experiment.

## Stage 4: frame sequence, temporal consistency and trajectory

```bash
python scripts/build_tile_cache.py                 # once, for LightGlue
python scripts/track_sequence.py --limit 20        # first 20 frames
python scripts/track_sequence.py                   # all frames from frames.csv
python scripts/track_sequence.py --ransac-method USAC_MAGSAC        # faster local step
python scripts/track_sequence.py --no-yaw-compensation              # baseline without heading
python scripts/track_sequence.py --matcher loftr --limit 15 --stride_search 2
```

### Tracker (`src/localization/tracker.py`)

| State | What happens |
|---|---|
| `LOST` (start, or after losing the target) | global search, as in Stage 3, over the whole map, a `--roi` or the `--stride_search` grid. If the winner scores at least `INIT_MIN_SCORE` = 5, the frame is `INITIALIZED` and the state becomes `TRACKING`. |
| `TRACKING` | search only in the (2r+1)×(2r+1) = **3×3 tile window** (`NEIGHBORHOOD_RADIUS` = 1, at most 9 tiles). After `MAX_MISSES` = 3 frames in a row without an accepted position, the state goes back to `LOST`. |

- **Window centre.** The window is centred on the tile whose centre is nearest to the *predicted* position (constant-velocity extrapolation), not on the previous winner. The previous winner may lie up to 30 px outside its own tile (the centre margin), so a window around it could be off-centre.
- **Kinematic gate.** `‖P_k − P_last‖ ≤ D_max = V_max·Δt + D_margin`, with `V_max` = 120 map px/s and `D_margin` = 30 px, which gives 150 px at 1 FPS.
  - The comparison is made against the last **accepted** position, with Δt measured from it, so the allowed radius grows while frames are being extrapolated.
  - Among the candidates that pass the Stage 3 gate, the best-scoring one that also passes the kinematic gate is chosen.
  - If no candidate passes, the frame is `INTERPOLATED` at `P̂_k = P_last + v·(t_k − t_last)`, where `v` comes from the last two accepted positions and is capped at `V_max`. At constant Δt this equals `P_{k−1} + (P_{k−1} − P_{k−2})`. The rejected estimate is stored in `x_raw, y_raw` with `event = OUTLIER_REJECTED`.
- **Heading (yaw) compensation.** SuperPoint/LightGlue and LoFTR are not invariant to large rotations, and the UAV turns during the flight. The frame-to-map angle is known from `H` of the accepted frames (`h_rotation_deg`).
  - Before matching, the frame is rotated to the predicted heading. The heading is extrapolated with the angular rate, capped at 45°/s. The canvas is expanded, and matches on the black border are masked out.
  - `H` is still estimated in original frame pixels.
  - If nothing is found in `TRACKING`, the heading offsets ±45° and ±90° are tried.
  - In `LOST`, each successive global search uses the next offset from (0°, 90°, 180°, −90°), so a global search costs no more than before.
- **RANSAC.** It is skipped when `matches < MIN_INLIERS`, because the success criterion is then unreachable. `cv2.RANSAC` on pure outliers runs all 5000 iterations, which takes ≈0.1 s.
- **Predicted window (`PREDICTED_WINDOW`, `--predicted-window`; off by default).** In `TRACKING` the frame is first matched against one map crop instead of the 3×3 tiles.
  - The crop is the bounding box of the last accepted footprint, shifted to the predicted position, plus `WINDOW_MARGIN_PX` (128 px). Its sides are multiples of 8 (LoFTR) and it is clipped to the map; above `WINDOW_MAX_SIDE_PX` (1024) the tiles are used.
  - The frame falls into the crop completely, and one match replaces up to 9. Crop features are extracted on the fly (no cache).
  - The same Stage 3 gate and kinematic gate apply. If nothing is accepted, the usual 3×3 search with all heading hypotheses follows. Accepted frames have `search_mode = window`.
- **Refinement after a global search (`REFINE_AFTER_GLOBAL`, `--refine-global`; off by default).** The global winner is matched again against a crop around its footprint (`REFINE_MARGIN_PX` = 64 px). The refined fix is kept only if it passes the gate, has at least as many inliers and moves the centre by ≤ `REFINE_MAX_SHIFT_PX` (30 px). Such frames have `refined = True`.
- **Trajectory smoothing (`src/localization/smoothing.py`).** A constant-velocity Kalman filter is applied to a finished `trajectory.csv`, so it does not change the tracker's decisions.
  - Measurements are `TRACKING` / `INITIALIZED` positions with `σ_z = KALMAN_MEAS_STD_PX` (4 px); `INTERPOLATED` frames use the filter prediction, `LOST` frames stay without a position, and every `INITIALIZED` restarts the filter.
  - Process noise is white acceleration `σ_a = KALMAN_ACCEL_STD_PX_S2` (20 px/s²: a 90° turn at ≈44 px/s over ≈3 s). Both values are set a priori, not tuned on the ground truth.
  - Two outputs: `x_kf, y_kf` (causal filter, usable online) and `x_rts, y_rts` (Rauch–Tung–Striebel smoother, offline only).
  - `summarize_results.py` writes `smoothing_table.{csv,md}`: errors on the reliable keyframes for raw / Kalman / RTS, and on how many keyframes each one is better than raw.

### Outputs (`--output_dir`, default `experiments/results/tracking/`)

- `trajectory.csv` has one row per frame, with no gaps.
  - Main columns: `frame_id, timestamp, status (INITIALIZED | TRACKING | INTERPOLATED | LOST), tile_id, x_map, y_map, inliers, inlier_ratio, reproj_error, process_time`.
  - Diagnostics: `event (OK | OUTLIER_REJECTED | NO_MATCH | LOW_SCORE), search_mode, tiles_checked, tiles_passed, score, jump_px, max_jump_px, x_raw, y_raw, h_rotation_deg, h_scale, yaw_used, attempts, search_time, refined`. `search_mode` is `global`, `local` (3×3 tiles) or `window`.
- `trajectory_info.json` records all parameters, summary statistics and the footprints of accepted frames.
- `experiments/figures/localization/trajectory_plot.png` shows the flight zone, or the whole map with `--full-map`:
  - the trajectory line with direction arrows;
  - green = TRACKING, blue = INITIALIZED, yellow = INTERPOLATED, red crosses = rejected outliers;
  - the footprint of every 10th frame.

### Stage 4 results (all 95 frames of DJI_0331, LightGlue + cache, GTX 1650)

| | Heading compensation (default) | Baseline `--no-yaw-compensation` |
|---|---|---|
| INITIALIZED / TRACKING / INTERPOLATED / LOST | 1 / 94 / 0 / 0 | 1 / 17 / 3 / 74 |
| Target lost | never | at the first turn (frame 19) and never recovered |
| Mean time, global / local frame | 32.9 s / 1.07 s (median 1.04, P90 1.40) | 31.7 s / 1.17 s |
| Total run time | 133 s | ≈41 min |

- The flight is a loop around a block of about 1000 × 1200 map px: north, then east, south and west. The four turns change the heading 0° → 90° → 175° → −90° → −40°.
- Every turn was tracked using the predicted heading alone. The retry offsets were needed for only one frame (frame 72, two attempts).
- The path length is 4151 map px, and the step is 44 px per frame on average (max 95 px), below `D_max` = 150 px. The last frame ends 375 px from the first one.
- `h_scale` ranges from 0.20 to 0.32 (median 0.269), which is consistent with the calibrated `FRAME_TO_MAP_SCALE` = 0.27.
- The local step takes 1.07 s on average: LightGlue on 9 pairs ≈ 0.78 s, and RANSAC ≈ 0.25 s, most of it on tiles without a counterpart.
  - With `--ransac-method USAC_MAGSAC`, the step takes 0.82 s, and positions differ from RANSAC by 1.7 px on average (max 5.1 px; 16 frames checked). The default stays `RANSAC`, so that Stage 2–4 results remain comparable.
- Frames 81–84 over trees have only 16–33 inliers, and the trajectory wiggles there by ≈20–30 px. The next stage should add smoothing and ground truth to quantify this.

## Stage 5: ground truth, error metrics and experiments

```bash
python scripts/annotate_ground_truth.py            # manual annotation of keyframes (GUI)
python scripts/annotate_ground_truth.py --check    # check figures in experiments/figures/analysis/gt_check/
python scripts/run_experiments.py all              # experiments 1–4 (~40 min on GTX 1650)
# system variants (improvements after Stage 5), baseline = Stage 5 system:
python scripts/run_experiments.py tracking --variants baseline scale_prior window refine all --dry-run
python scripts/summarize_results.py                # tables + figures
```

### Ground truth (`src/evaluation/ground_truth.py`, `scripts/annotate_ground_truth.py`)

The reference must not depend on the system being evaluated. It is therefore **not** taken from the tracker output. It is built from **manual** correspondences:

1. For each keyframe, click ≥ 4 pairs of points "frame ↔ map". Use ground-level points (road markings, curb corners, bases of poles), spread over the frame; roofs shift because of parallax.
2. A least-squares homography is fitted to all pairs (no RANSAC; a bad point shows up in `residual_px`).
3. The frame centre `((W−1)/2, (H−1)/2)` is projected through this homography.

**Annotation quality.** With 4 pairs the homography fits exactly, so the residual is always 0 and checks nothing. A keyframe is `reliable` only if it has ≥ 5 pairs, no single pair disagrees by more than 3 map px, and the centre is stable under click noise: the 95th percentile of the centre shift under random click errors (σ = 2 px on the frame, 1 px on the map) must be ≤ 5 map px. Clustered points, or a centre far outside them, make this value large. The criterion does not use the system's output. `key_*` metrics use only reliable keyframes, and `annotate_ground_truth.py --recompute` lists the frames to fix. The tool shows these checks live and does not save a frame with fewer than 5 pairs. A separate warning, which does not affect `reliable`, appears when the frame centre lies outside the convex hull of the clicked points. In that case the centre is extrapolated, and systematic errors (lens distortion, parallax) are not covered by the click-noise estimate. `benchmark_table.md` omits the `all_*` columns (interpolated reference); they remain in the CSV.

The tool shows the frame and the map side by side. Once 4 pairs are entered it also shows the projected frame outline, the centre and the residual of each pair. The tracker trajectory is used only to centre the map view initially.

The keyframes are listed in `GT_KEYFRAMES` (16 frames: 1, 6, 12, 19, 25, 31, 44, 51, 60, 64, 72, 76, 80, 82, 90, 94). They were picked visually from `experiments/figures/analysis/frames_contact_sheet.png`:

- each keyframe shows clear ground-level landmarks: roads, markings, parking lots, junctions;
- together they cover the whole flight evenly, including the turns (19, 51, 72, 94) and the stretch near trees (82, where a parking lot is visible);
- frames 33–40 (waste ground) and 66–69 (canal, trees) are skipped because they have no reliable points, so their reference is interpolated only.

Note for the thesis: keyframes were selected for how easy they are to annotate, not for how well the system matches them. Even so, urban frames are over-represented, and error over featureless terrain is estimated less precisely.

Outputs:

- `data/metadata/ground_truth_points.json` stores the clicked correspondences, so results can be recomputed (`--recompute`);
- `data/metadata/ground_truth.csv` has the columns `frame_id, timestamp, x_true, y_true, is_keyframe, n_points, residual_px, source`.
  - Frames between keyframes get linearly interpolated positions with `is_keyframe = False`.
  - The **main metrics use keyframes only**, because interpolation is inaccurate across turns.

### Localization metrics (`src/evaluation/metrics.py`)

The error is `e_i = ‖P_pred,i − P_gt,i‖₂` in map px, computed over the frames that have a reference. Frames without an estimate (`LOST`) are left out of mean, median, RMSE, P90, P95 and max. They are counted as failures in:

- `coverage = N_pred / N`;
- `success@K = |{e_i ≤ K}| / N` for K = 10, 25, 50 px.

### Experiments (`scripts/run_experiments.py`) and results

Setup: all 95 frames of DJI_0331 at 1 FPS, GTX 1650, heading compensation enabled.

**Experiments 1–2: LightGlue vs LoFTR, RANSAC vs USAC_MAGSAC** (`experiments/results/summary/benchmark_table.md`)

| Method | TRACKING / INIT / INTERP / LOST | Global search, s | Local step, s (mean / P90) | Median inliers | Mean inlier ratio | Mean reprojection error, px |
|---|---|---|---|---|---|---|
| LightGlue + RANSAC | 94 / 1 / 0 / 0 | 38.3 | 1.08 / 1.40 | 81 | 0.48 | 1.65 |
| LightGlue + MAGSAC++ | 93 / 1 / 1 / 0 | 20.1 | **0.78** / 1.02 | 87 | 0.50 | 1.73 |
| LoFTR + RANSAC | 94 / 1 / 0 / 0 | 127.9 | 2.72 / 2.93 | 316 | 0.59 | 1.27 |
| LoFTR + MAGSAC++ | 94 / 1 / 0 / 0 | 87.5 | 2.27 / 2.49 | 353 | 0.61 | 1.33 |

Positions differ between runs by a median of 0.8–1.7 px and a P90 of 2.3–10.2 px (`trajectory_deviation.md`). The largest differences, 19–43 px, are at frames 79–83 over trees.

- **Method.** Both methods track the whole flight.
  - LoFTR yields ≈4× more geometrically consistent correspondences and a lower reprojection error, but its local step is ≈2.5× slower.
  - Which method is more accurate cannot be claimed without the ground truth.
- **MAGSAC++ vs RANSAC.** MAGSAC++ speeds up the local step by 28% for LightGlue and 17% for LoFTR, and the global search by 1.5–1.9×.
  - The gain comes from tiles with no counterpart: there, `cv2.RANSAC` runs all 5000 iterations.
  - With MAGSAC++, LightGlue positions differ from RANSAC by a median of 1.2 px.
  - With LightGlue + MAGSAC++, one frame was interpolated.
  - Only LightGlue + MAGSAC++ meets the ≤ 1.0 s target: mean 0.78 s, P90 1.02 s.

**Experiment 3: robustness to distortions** (19 frames, 3×3 window, `robustness_table.md`, `robustness.png`)

The distortions are applied to the original 1920×1080 frame. `retention` is the median of `N_inliers / N_inliers(clean)` for the same frame, and `shift` is the centre displacement relative to the undistorted frame.

| Distortion | LightGlue: localized / retention | LoFTR: localized / retention |
|---|---|---|
| brightness ×0.7 / ×0.5 | 100% / 1.00; 100% / 1.01 | 100% / 0.97; 100% / 0.97 |
| contrast ×0.7 / ×0.5 | 100% / 1.02; 100% / 0.99 | 100% / 1.00; 100% / 0.97 |
| noise σ = 10 / 25 | 95% / 0.99; 95% / 0.92 | 100% / 1.02; 95% / 0.95 |
| motion blur 5 / 9 px | 100% / 1.02; 100% / 1.02 | 100% / 1.02; 100% / 1.04 |
| rotation 15° / 30° | 95% / 1.03; 95% / 0.99 | 95% / 0.82; 95% / 0.52 |
| rotation 45° / 90° | **53% / 0.35; 0%** | **11% / 0; 0%** |
| scale error ×0.8 / ×1.25 | 100% / 0.68; 95% / 1.29 | 100% / 0.71; 100% / 1.05 |

- Brightness, contrast, noise and blur at these levels have almost no effect. The median centre shift is ≤ 1.5 px.
- The critical factor is **rotation**:
  - LightGlue holds up to 30°, and LoFTR degrades from 15° onwards;
  - at 45° both methods mostly fail, and at 90° they fail completely.
  - This quantitatively justifies the heading compensation added in Stage 4.
- Blur is applied at the original resolution. After the ×0.27 downscale, a 9 px kernel shrinks to ≈2.4 px.

**Experiment 4: problem areas** (`terrain_table.md`, `inliers_vs_terrain.png`)

The terrain features are measured on the map under the frame footprint: the vegetation fraction (Excess Green index) and the texture (mean Sobel gradient).

- Frames 81–84: LightGlue finds a median of **27** inliers against 81 over the whole flight (LoFTR: 138–160 against 316–353).
  - All methods disagree most with each other there.
- The vegetation fraction by itself **does not explain** the drop.
  - For LightGlue, the Spearman correlation between inliers and vegetation is +0.37, because allotment gardens with houses count as "green" yet are rich in detail.
  - For LoFTR the correlation is −0.29 to −0.33.
- Texture is the better predictor for LightGlue (ρ = 0.65); for LoFTR ρ = 0.29.
- A plausible cause of the drop at 81–84 is tree canopy that changes with the season (winter video, summer map). Confirming this requires visual analysis of the frames.

### Status of the Stage 5 acceptance criteria

- ✅ Metrics module, ground-truth tool, experiments 1–4, tables and figures.
- ⏳ **`ground_truth.csv` still has to be annotated by hand** (16 keyframes, ≈15–20 min):
  1. run `annotate_ground_truth.py`;
  2. then run `summarize_results.py`.

  This fills in `key_*` / `all_*` (mean, median, RMSE, P90, max, success@10/25/50) in `benchmark_table`. It also replaces the inter-run deviation plot in `error_over_time.png` with the error-to-reference plot, and adds the reference to `trajectories_vs_gt.png`.

## Tests

```bash
python -m pytest tests -q
# or, without pytest:
python tests/test_stage1_preprocessing.py
python tests/test_stage2_pipeline.py
python tests/test_stage3_localization.py
python tests/test_stage4_tracking.py
python tests/test_stage5_evaluation.py
```

Stage 1 tests use synthetic maps and videos. The frame-extraction test embeds the frame index into each frame and checks that the saved frames are exactly the ones recorded in `frames.csv`.

Stage 2 tests use an oracle matcher that returns correspondences generated from a known homography, plus outliers. They check:

- the coordinate round trip through resizing, including rounding sizes to multiples of 8;
- that `H` is recovered in original pixels for several working scales;
- reproducible RANSAC with a fixed seed;
- rejection of implausible homographies;
- frame-centre projection onto the map;
- tile-feature caching.

Stage 3 tests check:

- `coordinates.py`: pixel-centre convention, dehomogenization including points at infinity, the tile → map shift, the footprint of a rotated frame, and map bounds;
- `scoring.py`: the score formula and its monotonicity, every gate condition and its boundary values, and the ranking order;
- ROI and grid-step tile selection;
- search with a synthetic matcher, which finds the true tile and reproduces the known global centre and footprint;
- the disk cache: a round trip gives the same result, and a cache built with other parameters is refused.

Stage 4 tests check:

- the 3×3 neighbourhood inside the grid, at corners and edges, the irregular last column, radius 2, and the nearest tile;
- `D_max`, the kinematic gate at its boundary, and position extrapolation with the speed cap;
- tracker states with a scripted search: initialization and the local 3×3 search, outlier rejection with extrapolation, choosing the second candidate when the best one is an outlier, `LOST` after 3 misses followed by global re-initialization, the minimum initialization score, and one CSV row per frame;
- heading: angle wrapping, heading extrapolation across ±180°, following a turn, retry offsets, the global heading cycle, and disabling compensation;
- the pipeline with a rotated frame: `H` in original pixels is unchanged, the work image is aligned with the map, and matches on the black border are filtered out.

Stage 5 tests check:

- the reference from manual points: exact points, click noise, too few or mismatched points, the JSON round trip, and interpolation only between keyframes;
- `trajectory_errors` and `error_statistics`: missing estimates, coverage, RMSE, and success@K counted over all frames with the threshold inclusive;
- the distortions (brightness, contrast, reproducible noise, blur direction) and the terrain features.

## Known issues

- The homography model assumes an approximately planar scene and a near-nadir camera. With tall buildings, RANSAC can lock onto a roof plane instead of the ground; see the Stage 2 findings.
- The frame footprint (≈518 × 292 map px) is wider than a 512 px tile, so the winning tile holds only part of the frame. The tile-size experiment (768/1024) should address this.
- Heading compensation assumes that the heading changes by less than ≈45° between frames at 1 FPS, beyond what the angular-rate prediction covers; larger jumps rely on the retry offsets (±45°, ±90°).
- The first global search runs at a heading of 0°. If the video starts with the camera rotated relative to the map, initialization takes up to 4 frames (heading cycle).
- The single-frame result is saved to one file (`localization_single_frame.json`) and is overwritten by the next run; pass `--output` to keep several.
- `FRAME_TO_MAP_SCALE` was calibrated on a single frame. If the flight altitude changes, the scale changes as well. The plausibility check tolerates a factor of 2.
- The evaluation pairs used before Stage 1 did not show the same place (see `eval_pairs_legacy_256.csv`). The old `tiles.csv` also lacked the ROI offset (912, 1409). Results obtained with them should not be reused, and neither should the timing and inlier numbers from the term paper, which came from the unequal resolutions described above.
- Results are not bit-identical across devices. On frame_0002, LightGlue returned 309 matches on CPU and 310 on GPU, and `h_scale` changed from 0.250 to 0.267. Compare results only within one device, and `*_info.json` records which device was used.
- Timing on frame_0002 (median of 5 runs, `time_matcher_total`):
  - GTX 1650: LightGlue 0.25 s, LoFTR 0.27 s;
  - CPU: 1.85 s and 3.24 s.

## Roadmap

1. ~~Unified matching pipeline with scale normalization, cached tile features and GPU warm-up for fair timing.~~ Done in Stage 2.
2. ~~Automatic tile search with a clearly defined ranking criterion and homography sanity checks.~~ Done in Stage 3.
3. ~~Frame-to-map position estimation~~ (done in Stage 3) and a manually annotated `ground_truth.csv`.
4. ~~Sequence processing with temporal consistency and neighbour-tile search.~~ Done in Stage 4 (plus heading compensation).
5. Experiments (Stage 5; the robustness set also includes scale error and rotation):
   - LightGlue vs LoFTR;
   - tile size 256/512/768/1024;
   - overlap;
   - robustness to brightness, contrast, blur, noise, scale and rotation.
6. Error statistics (mean, median, max, P90, success rate) and figures for the thesis.

## Author

Andrii Yatsuliak, Master's student, Department of Programming, Ivan Franko National University of Lviv.
