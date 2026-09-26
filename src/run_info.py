"""Метадані запуску експерименту: git-коміт, пристрій, параметри карти."""
from __future__ import annotations

import json
import platform
import subprocess

import pandas as pd

from src.config import PROJECT_ROOT, TILES_RUN_INFO_PATH


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def device_info(device: str) -> dict:
    import torch
    info = {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "platform": platform.platform()}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["cpu"] = platform.processor()
    return info


def tiles_run_info() -> dict:
    """Параметри останнього тайлування (``tiles_info.json``) або порожній словник."""
    if TILES_RUN_INFO_PATH.is_file():
        return json.loads(TILES_RUN_INFO_PATH.read_text(encoding="utf-8"))
    return {}


def map_size(tiles: pd.DataFrame) -> tuple[int, int]:
    """Розмір повної карти ``(W, H)``: з ``tiles_info.json`` або за межами тайлів."""
    info = tiles_run_info()
    if "map_width" in info and "map_height" in info:
        return int(info["map_width"]), int(info["map_height"])
    return (int((tiles["x_origin"] + tiles["width"]).max()),
            int((tiles["y_origin"] + tiles["height"]).max()))
