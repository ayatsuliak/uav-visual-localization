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
| 3 | Automatic best-tile search and ranking | ⏳ next (prototype: `localize_frame.py`) |
| 4 | Frame position on the full map | 🧪 prototype |
| 5–6 | Frame sequences, temporal consistency, neighbour-tile search | planned |
| 7–10 | Experiments, ground truth, error statistics, figures | planned |

## Repository structure

```text
uav-visual-localization/
├── data/
│   ├── raw/
│   │   ├── maps/          # university_map.png            (local only, not in Git)
│   │   └── video/         # DJI_0331.mp4                  (local only, not in Git)
│   ├── processed/
│   │   ├── frames/        # frame_0001.png, ...           (generated)
│   │   └── tiles/         # tile_0001.png, ...            (generated)
│   └── metadata/
│       ├── tiles.csv                  # tile registry (generated, committed)
│       ├── frames.csv                 # frame registry (generated, committed)
│       ├── eval_pairs.csv             # verified frame–tile pairs (manual)
│       ├── eval_pairs_legacy_256.csv  # old pairs, marked INVALID (kept for traceability)
│       └── golden_pair.csv            # reference pair from the term paper
├── src/
│   ├── config.py          # paths and default parameters
│   ├── preprocessing/     # images.py, tiles.py, frames.py, validation.py
│   ├── matching/          # base.py (ImageMatcher), lightglue.py, loftr.py, pipeline.py
│   ├── geometry/          # homography.py (RANSAC, reprojection error, plausibility)
│   ├── evaluation/        # metrics.py, scoring.py
│   ├── localization/      # estimate.py (tile → map coordinates)
│   └── visualization/     # matches.py
├── scripts/               # command-line entry points (see below)
├── tests/                 # unit tests on synthetic data
├── experiments/
│   ├── results/{raw,summary,rankings}/
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

## Stage 3 prototype

```bash
python scripts/localize_frame.py --frame frame_0002.png --method LightGlue --max-tiles 20
```

Exhaustive search over tiles with cached tile features. The candidate order (success, then inliers) is provisional. `src/evaluation/scoring.py` is not used yet and will be replaced by the Stage 3 ranking criterion.

## Tests

```bash
python -m pytest tests -q
# or, without pytest:
python tests/test_stage1_preprocessing.py
python tests/test_stage2_pipeline.py
```

Stage 1 tests use synthetic maps and videos. The frame-extraction test embeds the frame index into each frame and checks that the saved frames are exactly the ones recorded in `frames.csv`.

Stage 2 tests use an oracle matcher that returns correspondences generated from a known homography, plus outliers. They check:

- the coordinate round trip through resizing, including rounding sizes to multiples of 8;
- that `H` is recovered in original pixels for several working scales;
- reproducible RANSAC with a fixed seed;
- rejection of implausible homographies;
- frame-centre projection onto the map;
- tile-feature caching.

## Known issues

- The homography model assumes an approximately planar scene and a near-nadir camera. With tall buildings, RANSAC can lock onto a roof plane instead of the ground; see the Stage 2 findings.
- `FRAME_TO_MAP_SCALE` was calibrated on a single frame. If the flight altitude changes, the scale changes as well. The plausibility check tolerates a factor of 2.
- The evaluation pairs used before Stage 1 did not show the same place (see `eval_pairs_legacy_256.csv`). The old `tiles.csv` also lacked the ROI offset (912, 1409). Results obtained with them should not be reused, and neither should the timing and inlier numbers from the term paper, which came from the unequal resolutions described above.
- Results are not bit-identical across devices. On frame_0002, LightGlue returned 309 matches on CPU and 310 on GPU, and `h_scale` changed from 0.250 to 0.267. Compare results only within one device, and `*_info.json` records which device was used.
- Timing on frame_0002 (median of 5 runs, `time_matcher_total`):
  - GTX 1650: LightGlue 0.25 s, LoFTR 0.27 s;
  - CPU: 1.85 s and 3.24 s.

## Roadmap

1. ~~Unified matching pipeline with scale normalization, cached tile features and GPU warm-up for fair timing.~~ Done in Stage 2.
2. Automatic tile search with a clearly defined ranking criterion and homography sanity checks.
3. Frame-to-map position estimation and a manually annotated `ground_truth.csv`.
4. Sequence processing with temporal consistency and neighbour-tile search.
5. Experiments:
   - LightGlue vs LoFTR;
   - tile size 256/512/768/1024;
   - overlap;
   - robustness to brightness, contrast, blur, noise, scale and rotation.
6. Error statistics (mean, median, max, P90, success rate) and figures for the thesis.

## Author

Andrii Yatsuliak, Master's student, Department of Programming, Ivan Franko National University of Lviv.
