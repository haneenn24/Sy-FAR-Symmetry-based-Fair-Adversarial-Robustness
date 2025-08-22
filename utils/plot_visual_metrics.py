# evaluation/plot_visual_metrics.py
# -*- coding: utf-8 -*-

"""
Plot confusion-matrix heatmaps with an appended strip of images aligned to columns.

Outputs:
  - <title_prefix>_full_heatmap.png
  - <title_prefix>_upper_triangle_diff.png
  - <title_prefix>_upper_triangle_absdiff.png

Examples
--------
# Minimal (matrix in the file, class names 0..N-1)
python evaluation/plot_visual_metrics.py \
  --cm-file ./cm.npy \
  --title "Robust Accuracy PubFig +Siblings + VGG16 FAAL (targeted)"

# With images and custom class names
python evaluation/plot_visual_metrics.py \
  --cm-file ./cm.csv \
  --images datasets/mixds_pubfig2siblings/candidates/00_Dakota_Fanning.jpg \
           datasets/mixds_pubfig2siblings/candidates/01_Elle_Fanning.jpg \
           datasets/mixds_pubfig2siblings/candidates/AntonioBanderas.jpg \
           datasets/mixds_pubfig2siblings/candidates/ColinPowell.jpg \
           datasets/mixds_pubfig2siblings/candidates/HughGrant.jpg \
           datasets/mixds_pubfig2siblings/candidates/JenniferLopez.jpg \
           datasets/mixds_pubfig2siblings/candidates/JohnTravolta.jpg \
           datasets/mixds_pubfig2siblings/candidates/ReeseWitherspoon.jpg \
           datasets/mixds_pubfig2siblings/candidates/TyraBanks.jpg \
           datasets/mixds_pubfig2siblings/candidates/WillSmith.jpg \
  --classnames "Dakota_Fanning,Elle_Fanning,AntonioBanderas,ColinPowell,HughGrant,JenniferLopez,JohnTravolta,ReeseWitherspoon,TyraBanks,WillSmith" \
  --title "Robust Accuracy PubFig +Siblings + VGG16 FAAL (targeted)"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# ---------------------- helpers ----------------------

def _safe_open_image(path_or_arr):
    """Open an image path or return PIL.Image if already provided. Returns None if fails."""
    try:
        if isinstance(path_or_arr, Image.Image):
            return path_or_arr
        if isinstance(path_or_arr, np.ndarray):
            return Image.fromarray(path_or_arr)
        return Image.open(path_or_arr).convert("RGB")
    except Exception:
        return None


def _add_column_image_strip(
    fig: plt.Figure,
    ax_heat: plt.Axes,
    image_paths: Optional[List[str | np.ndarray | Image.Image]],
    height_frac: float = 0.25,
    vpad: float = 0.0,
) -> None:
    """
    Adds a horizontal strip of images under the heatmap, aligned to columns.

    Args:
        fig: matplotlib Figure
        ax_heat: axes of the heatmap
        image_paths: list of length N (column count). Each can be path/ndarray/PIL.Image.
        height_frac: fraction of available vertical space allocated to the strip
        vpad: additional vertical padding below heatmap axes
    """
    if not image_paths:
        return
    N = len(image_paths)

    fig.canvas.draw()
    bbox = ax_heat.get_position()
    x0, y0, x1, _ = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    col_w = (x1 - x0) / N

    img_h = max(0.01, min(0.35, height_frac)) * y0
    bottom = max(0.01, y0 - img_h - vpad)

    for j in range(N):
        ax_img = plt.axes([x0 + j * col_w, bottom, col_w, img_h])
        img = _safe_open_image(image_paths[j])
        if img is None:
            # draw empty placeholder if image missing
            ax_img.set_facecolor((0.95, 0.95, 0.95))
            ax_img.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
        else:
            ax_img.imshow(img)
        ax_img.axis("off")


def compute_directional_difference(conf_matrix: np.ndarray) -> np.ndarray:
    """Upper-triangle directional difference: cm[i,j] - cm[j,i] (else 0)."""
    cm = np.array(conf_matrix, dtype=float)
    N = cm.shape[0]
    diff = np.zeros_like(cm, dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            diff[i, j] = cm[i, j] - cm[j, i]
    return diff


def compute_absolute_difference(conf_matrix: np.ndarray) -> np.ndarray:
    """Upper-triangle absolute difference: |cm[i,j] - cm[j,i]| (else 0)."""
    cm = np.array(conf_matrix, dtype=float)
    N = cm.shape[0]
    abs_diff = np.zeros_like(cm, dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            abs_diff[i, j] = abs(cm[i, j] - cm[j, i])
    return abs_diff


# ---------------------- core plotting ----------------------

def plot_confusion_matrix_heatmaps_with_images(
    confusion_matrix: np.ndarray,
    image_paths: Optional[List[str | np.ndarray | Image.Image]] = None,
    class_names: Optional[List[str]] = None,
    title_prefix: str = "metrics",
    fixed_vmin: Optional[float] = 0.0,
    fixed_vmax: Optional[float] = 1.0,
    fig_size: tuple = (10, 9),
    cmap_main: str = "Blues",
    cmap_diff: str = "coolwarm",
    cmap_abs: str = "Oranges",
) -> List[Path]:
    """
    Renders three heatmaps and saves PNGs. Returns saved file paths.

    Args:
        confusion_matrix: (N, N) array; can be normalized or raw.
        image_paths: list of N images aligned to predicted columns (0..N-1).
        class_names: list of N names; if None, uses "0..N-1".
        title_prefix: prefix for output filenames.
        fixed_vmin/fixed_vmax: color scaling for the main heatmap (None -> auto).
        fig_size: figure size for all plots.
        cmap_main/cmap_diff/cmap_abs: colormaps.

    Returns:
        List of Path objects for the three saved figures.
    """
    cm = np.array(confusion_matrix, dtype=float)
    N = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(N)]
    if image_paths is not None and len(image_paths) != N:
        raise ValueError(f"image_paths length ({len(image_paths)}) must match cm columns ({N})")

    saved: List[Path] = []

    # --- Full Confusion Matrix ---
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes([0.10, 0.28, 0.80, 0.68])
    sns.heatmap(
        cm, ax=ax, annot=True, fmt=".2f", cmap=cmap_main,
        vmin=fixed_vmin, vmax=fixed_vmax,
        cbar_kws={'ticks': np.linspace(0, 1, 11)} if fixed_vmin == 0.0 and fixed_vmax == 1.0 else None
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)

    _add_column_image_strip(fig, ax, image_paths, height_frac=0.25, vpad=0.0)
    out_full = Path(f"{title_prefix}_full_heatmap.png")
    plt.savefig(out_full.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_full)

    # --- Directional Difference (upper triangle) ---
    diff_matrix = compute_directional_difference(cm)
    mask = np.tril(np.ones_like(diff_matrix, dtype=bool))
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes([0.10, 0.28, 0.80, 0.68])
    sns.heatmap(
        diff_matrix, ax=ax, annot=True, fmt=".2f", cmap=cmap_diff,
        mask=mask, vmin=-1.0, vmax=1.0,
        cbar_kws={'ticks': np.linspace(-1, 1, 11)}
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)

    _add_column_image_strip(fig, ax, image_paths, height_frac=0.25, vpad=0.0)
    out_diff = Path(f"{title_prefix}_upper_triangle_diff.png")
    plt.savefig(out_diff.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_diff)

    # --- Absolute Difference (upper triangle) ---
    abs_diff_matrix = compute_absolute_difference(cm)
    mask = np.tril(np.ones_like(abs_diff_matrix, dtype=bool))
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes([0.10, 0.28, 0.80, 0.68])
    sns.heatmap(
        abs_diff_matrix, ax=ax, annot=True, fmt=".2f", cmap=cmap_abs,
        mask=mask, vmin=0.0, vmax=1.0,
        cbar_kws={'ticks': np.linspace(0, 1, 11)}
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)

    _add_column_image_strip(fig, ax, image_paths, height_frac=0.25, vpad=0.0)
    out_abs = Path(f"{title_prefix}_upper_triangle_absdiff.png")
    plt.savefig(out_abs.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(out_abs)

    return saved


# ---------------------- I/O & CLI ----------------------

def _load_cm(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    # default: CSV
    return np.loadtxt(path, delimiter=",")


def _parse_classnames(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    # split on comma and trim
    return [t.strip() for t in s.split(",") if t.strip() != ""]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot confusion-matrix heatmaps with appended column-aligned images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cm-file", type=Path, required=True, help=".npy or .csv confusion matrix")
    p.add_argument("--images", type=str, nargs="*", default=None,
                  help="List of N image paths aligned to columns (predicted classes).")
    p.add_argument("--classnames", type=str, default=None,
                  help="Comma-separated class names (N entries) for axes labels.")
    p.add_argument("--title", type=str, default="metrics",
                  help="Prefix for output filenames (spaces allowed).")
    p.add_argument("--fig-w", type=float, default=10.0)
    p.add_argument("--fig-h", type=float, default=9.0)
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cm = _load_cm(args.cm_file)
    class_names = _parse_classnames(args.classnames)
    images = args.images if args.images else None

    saved = plot_confusion_matrix_heatmaps_with_images(
        cm,
        image_paths=images,
        class_names=class_names,
        title_prefix=args.title,
        fixed_vmin=args.vmin,
        fixed_vmax=args.vmax,
        fig_size=(args.fig_w, args.fig_h),
    )
    for p in saved:
        print(f"[saved] {p}")


if __name__ == "__main__":
    main()
