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
| 2 | Unified pipeline `frame + tile → matcher → RANSAC → metrics` | ⏳ next |
| 3 | Automatic best-tile search and ranking | 🧪 prototype (`localize_frame.py`) |
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
│   ├── matching/          # base.py (ImageMatcher), lightglue.py, loftr.py
│   ├── geometry/          # homography.py (RANSAC, reprojection error)
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

# For GPU, install a CUDA build of PyTorch first, e.g. CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

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
- one frame pixel corresponds to roughly 0.25 map pixels, so a full 1920 × 1080 frame covers about 480 × 270 map pixels (rough manual estimate, to be calibrated in Stage 2).

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

### Tests

```bash
python -m pytest tests -q
# or, without pytest:
python tests/test_stage1_preprocessing.py
```

The tests use synthetic maps and videos. The frame-extraction test embeds the frame index into each frame and checks that the saved frames are exactly the ones recorded in `frames.csv`.

## Matching and localization (prototype, revised in Stage 2)

```bash
python scripts/evaluate.py          # LightGlue vs LoFTR on data/metadata/eval_pairs.csv
python scripts/summarize.py         # aggregates into experiments/results/summary/
python scripts/localize_frame.py --frame frame_0002.png --method LightGlue --max-tiles 20
```

Reported metrics:

- processing time;
- number of matches;
- number and share of geometrically consistent matches after RANSAC;
- mean reprojection error;
- success rate.

Tile ranking uses a separate, documented technical score (`src/evaluation/scoring.py`) that is not treated as a research metric.

## Known issues

- `scripts/evaluate_golden_pair.py` and `scripts/visualize_golden_pair.py` are out of date with the current matcher/geometry API. They will be replaced in Stage 2.
- Matchers currently center-crop the frame to a square and resize it to 512 px. With 512 px tiles this leaves a ≈1.9× scale gap between frame and tile and discards ~44 % of the frame width. Scale normalization is planned for Stage 2.
- The evaluation pairs used before Stage 1 did not show the same place (see `eval_pairs_legacy_256.csv`). The old `tiles.csv` also lacked the ROI offset (912, 1409). Results obtained with them should not be reused.
- The homography model assumes an approximately planar scene and a near-nadir camera.

## Roadmap

1. Unified matching pipeline with scale normalization, cached tile features and GPU warm-up for fair timing.
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
