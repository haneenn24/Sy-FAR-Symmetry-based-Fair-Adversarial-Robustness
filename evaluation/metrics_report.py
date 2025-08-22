#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics Report for Robustness Disparity & Symmetry

Use a provided confusion matrix OR compute it from a model on the test set.

Examples
--------
# 1) From a saved confusion matrix (.npy or .csv):
python evaluation/metrics_report.py --cm-file ./runs/cm.npy

# 2) Compute confusion matrix from a trained model:
python evaluation/metrics_report.py \
  --model-checkpoint ./glass/donemodel/model.pt \
  --batch-size 64 \
  --rgb-to-bgr

# 3) Save a JSON report:
python evaluation/metrics_report.py --cm-file ./cm.csv --out-json ./metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Project-local imports
from utils import data_process
from models.vgg16 import VGG_16


# -----------------------------
# Label normalization & gender map
# -----------------------------

def _norm_name(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]', '', s.lower())

# 0 = Female, 1 = Male
_KNOWN_GENDER_MAP = {
    # female
    _norm_name("00_Dakota_Fanning"): 0,
    _norm_name("01_Elle_Fanning"): 0,
    _norm_name("jenniferlopez"): 0,
    _norm_name("reesewitherspoon"): 0,
    _norm_name("tyrabanks"): 0,
    # male
    _norm_name("antoniobanderas"): 1,
    _norm_name("colinpowell"): 1,
    _norm_name("hughgrant"): 1,
    _norm_name("johntravolta"): 1,
    _norm_name("willsmith"): 1,
}


def make_gender_groups_from_classnames(class_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Map class_names to Female/Male via known map; error if any are missing."""
    K = len(class_names)
    ctg = np.full(K, -1, dtype=int)
    missing = []
    for idx, name in enumerate(class_names):
        key = _norm_name(name)
        if key in _KNOWN_GENDER_MAP:
            ctg[idx] = _KNOWN_GENDER_MAP[key]
        else:
            missing.append(name)
    if np.any(ctg < 0):
        raise ValueError(f"Gender mapping incomplete. Missing mappings for: {missing}")
    return ctg, ["Female", "Male"]


# -----------------------------
# Core metric helpers
# -----------------------------

def normalize_by_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = np.sum(cm, axis=1, keepdims=True)
    return cm / (row_sums + 1e-8)

def overall_accuracy(cm_row_norm: np.ndarray) -> float:
    # mean of diagonal when rows are normalized (i.e., average per-class recall)
    K = cm_row_norm.shape[0]
    return float(np.trace(cm_row_norm) / K)

def source_class_accuracy(cm_row_norm: np.ndarray) -> np.ndarray:
    return np.diag(cm_row_norm)

def calculate_min_max_gap(values: np.ndarray) -> Tuple[float, float, float, int, int]:
    min_acc = float(np.min(values))
    max_acc = float(np.max(values))
    gap = max_acc - min_acc
    return min_acc, max_acc, gap, int(np.argmin(values)), int(np.argmax(values))

def find_min_max_with_class_off_diagonal(cm_row_norm: np.ndarray) -> Tuple[float, float, float, int, int]:
    """Target-perspective average off-diagonal by column (exclude diagonal)."""
    n = cm_row_norm.shape[0]
    off = cm_row_norm.copy()
    np.fill_diagonal(off, np.nan)
    col_means = np.nanmean(off, axis=0)
    min_val = float(np.nanmin(col_means))
    max_val = float(np.nanmax(col_means))
    return min_val, max_val, max_val - min_val, int(np.nanargmin(col_means)), int(np.nanargmax(col_means))

def find_extreme_off_diagonal_values(cm: np.ndarray) -> Tuple[float, Tuple[int,int], float, Tuple[int,int]]:
    n = cm.shape[0]
    min_val, max_val = float("inf"), float("-inf")
    min_pair, max_pair = (-1, -1), (-1, -1)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            v = float(cm[i, j])
            if v < min_val:
                min_val, min_pair = v, (i, j)
            if v > max_val:
                max_val, max_pair = v, (i, j)
    return min_val, min_pair, max_val, max_pair

def compute_symmetry_score(cm: np.ndarray) -> float:
    """Asymmetry penalty (pairwise absolute diff normalized by pair mass). Lower is better."""
    N = cm.shape[0]
    eps = 1.0 / N
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            a, b = float(cm[i, j]), float(cm[j, i])
            penalty += (abs(a - b) / (a + b + eps)) * (a + b)
    return float(penalty)

def max_asymmetry_gap(cm: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """max |cm[i,j] - cm[j,i]| over i<j."""
    n = cm.shape[0]
    max_gap, max_pair = -1.0, (-1, -1)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(float(cm[i, j]) - float(cm[j, i]))
            if diff > max_gap:
                max_gap, max_pair = diff, (i, j)
    return float(max_gap), max_pair

def compute_symmetry_score_improved(cm: np.ndarray, threshold: float = 0.05) -> float:
    """Log-space asymmetry with significance threshold on pair mass."""
    N = cm.shape[0]
    eps = 1e-6
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            a, b = float(cm[i, j]), float(cm[j, i])
            if (a + b) > threshold:
                diff = np.log1p(a + eps) - np.log1p(b + eps)
                penalty += (diff * diff) * (a + b)
    return float(penalty)

def target_fairness_normalized_gap(cm: np.ndarray) -> Tuple[float, float, float, List[float]]:
    """Share of total off-diagonal that lands in each target column."""
    K = cm.shape[0]
    total_mis = float(np.sum(cm) - np.trace(cm))
    shares = []
    for j in range(K):
        col = cm[:, j].copy()
        col[j] = 0.0
        mis_to_j = float(np.sum(col))
        shares.append(mis_to_j / total_mis if total_mis > 0 else 0.0)
    min_s, max_s = float(np.min(shares)), float(np.max(shares))
    return min_s, max_s, max_s - min_s, shares

def subgroup_robust_accuracy(cm_row_norm: np.ndarray, class_to_group: np.ndarray, G: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recall per subgroup (source perspective)."""
    K = cm_row_norm.shape[0]
    sra = np.zeros(G, dtype=float)
    counts = np.zeros(G, dtype=float)
    for i in range(K):
        g = int(class_to_group[i])
        row_sum = float(np.sum(cm_row_norm[i, :]))  # should be ~1
        counts[g] += row_sum
        sra[g] += float(cm_row_norm[i, i])
    with np.errstate(divide='ignore', invalid='ignore'):
        sra = np.divide(sra, counts, out=np.zeros_like(sra), where=counts > 0)
    return sra, counts

def robustness_disparity(values: np.ndarray) -> Tuple[float, Tuple[int,int]]:
    if values.size == 0:
        return 0.0, (-1, -1)
    vmin_idx = int(np.argmin(values))
    vmax_idx = int(np.argmax(values))
    return float(values[vmax_idx] - values[vmin_idx]), (vmin_idx, vmax_idx)


# -----------------------------
# Confusion matrix I/O / compute
# -----------------------------

def load_confusion_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    # assume CSV
    return np.loadtxt(path, delimiter=",")

def rgb_to_bgr(x: torch.Tensor) -> torch.Tensor:
    return x[:, [2, 1, 0], :, :]

@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    rgb2bgr: bool = False,
    device: torch.device | None = None,
) -> np.ndarray:
    """Raw counts confusion matrix (KxK). Rows = true, Cols = predicted."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if rgb2bgr:
            images = rgb_to_bgr(images)
        logits = model(images)
        preds = logits.argmax(dim=1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[int(t.item()), int(p.item())] += 1

    return cm


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and report robustness disparity metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Path inputs
    parser.add_argument("--cm-file", type=Path, default=None,
                        help="Confusion matrix file (.npy or .csv). If omitted, the CM is computed.")
    parser.add_argument("--model-checkpoint", type=Path, default=None,
                        help="Model checkpoint to compute confusion matrix if --cm-file is not given.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rgb-to-bgr", action="store_true",
                        help="Swap channels (RGB->BGR) before evaluating the model.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save metrics as JSON.")

    # Behavior
    parser.add_argument("--normalize-rows", action="store_true", default=True,
                        help="Normalize confusion matrix rows before metric computation.")
    parser.add_argument("--no-normalize-rows", dest="normalize_rows", action="store_false",
                        help="Use raw counts (not recommended for class-imbalanced sets).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Load or compute confusion matrix
    if args.cm_file is not None:
        cm = load_confusion_matrix(args.cm_file)
        # For class names, try to use dataloader if model path given; else fallback to generic class ids.
        class_names = [str(i) for i in range(cm.shape[0])]
    else:
        if args.model_checkpoint is None:
            raise ValueError("Provide --cm-file OR --model-checkpoint to compute the confusion matrix.")
        # Build model + data
        model = VGG_16()
        state = torch.load(str(args.model_checkpoint), map_location="cpu")
        model.load_state_dict(state)
        dataloaders, dataset_sizes, class_names = data_process(batch_size=args.batch_size)
        if "test" not in dataloaders:
            raise KeyError("data_process must return a 'test' dataloader.")
        cm = compute_confusion_matrix(model, dataloaders["test"], num_classes=len(class_names),
                                      rgb2bgr=args.rgb_to_bgr)

    # Normalize rows if requested
    cm_used = normalize_by_rows(cm) if args.normalize_rows else cm.astype(np.float64)

    # ---- Metrics ----
    overall_acc = overall_accuracy(cm_used)
    src_acc = source_class_accuracy(cm_used)
    min_acc, max_acc, acc_gap, min_c, max_c = calculate_min_max_gap(src_acc)

    min_t, max_t, gap_t, min_c_t, max_c_t = find_min_max_with_class_off_diagonal(cm_used)
    min_off, min_pair, max_off, max_pair = find_extreme_off_diagonal_values(cm_used)

    sym_score = compute_symmetry_score(cm_used)
    sym_score_impr = compute_symmetry_score_improved(cm_used, threshold=0.05)
    max_gap_val, (i_asym, j_asym) = max_asymmetry_gap(cm_used)

    min_share, max_share, share_gap, shares = target_fairness_normalized_gap(cm_used)

    # Subgroup metrics (Gender) if class names exist and mapping is complete
    subgroup_metrics = None
    try:
        class_to_group, group_names = make_gender_groups_from_classnames(class_names)
        G = len(group_names)
        sra, subgroup_counts = subgroup_robust_accuracy(cm_used, class_to_group, G)
        rd_val, (rd_min_idx, rd_max_idx) = robustness_disparity(sra)
        subgroup_metrics = {
            "group_names": group_names,
            "SRA": sra.tolist(),
            "counts": [int(c) for c in subgroup_counts.tolist()],
            "RD": rd_val,
            "RD_pair": [int(rd_min_idx), int(rd_max_idx)],
        }
    except Exception as e:
        subgroup_metrics = {"warning": f"Skipping subgroup metrics: {e}"}

    # ---- Print summary ----
    print("\n=== Metrics Results ===")
    print(f"Robust Accuracy (mean per-class recall): {overall_acc:.4f}")
    print(f"Diagonal min/max/gap: min={min_acc:.4f} (class {min_c}), "
          f"max={max_acc:.4f} (class {max_c}), gap={acc_gap:.4f}")

    print("\n-- Target-perspective (off-diagonal) --")
    print(f"Avg off-diagonal per target: min={min_t:.4f} (class {min_c_t}), "
          f"max={max_t:.4f} (class {max_c_t}), gap={gap_t:.4f}")
    print(f"Extreme off-diagonals: min={min_off:.4f} @ {min_pair}, max={max_off:.4f} @ {max_pair}")

    print("\n-- Symmetry --")
    print(f"Asymmetry penalty: {sym_score:.4f}")
    print(f"Improved (log-space) asymmetry: {sym_score_impr:.4f}")
    print(f"Max asymmetry gap: {max_gap_val:.4f} (between {class_names[i_asym] if i_asym>=0 and i_asym<len(class_names) else i_asym} "
          f"and {class_names[j_asym] if j_asym>=0 and j_asym<len(class_names) else j_asym})")

    print("\n-- Target misclassification share --")
    print(f"Shares: {[round(s, 4) for s in shares]}")
    print(f"Min={min_share:.4f} Max={max_share:.4f} Gap={share_gap:.4f}")

    print("\n-- Subgroup (Gender) --")
    if "warning" in subgroup_metrics:
        print(subgroup_metrics["warning"])
    else:
        gnames = subgroup_metrics["group_names"]
        sra = subgroup_metrics["SRA"]
        counts = subgroup_metrics["counts"]
        print("SRA per subgroup: " + ", ".join([f"{gnames[g]}={sra[g]:.4f} (n={counts[g]})" for g in range(len(gnames))]))
        print(f"RD (gap): {subgroup_metrics['RD']:.4f} "
              f"({gnames[subgroup_metrics['RD_pair'][0]]} vs {gnames[subgroup_metrics['RD_pair'][1]]})")

    # ---- Optional JSON output ----
    if args.out_json:
        out = {
            "overall_acc": overall_acc,
            "diag_min": min_acc,
            "diag_max": max_acc,
            "diag_gap": acc_gap,
            "diag_min_class": int(min_c),
            "diag_max_class": int(max_c),
            "target_avg_offdiag_min": min_t,
            "target_avg_offdiag_max": max_t,
            "target_avg_offdiag_gap": gap_t,
            "extreme_offdiag_min": {"value": min_off, "pair": list(map(int, min_pair))},
            "extreme_offdiag_max": {"value": max_off, "pair": list(map(int, max_pair))},
            "asymmetry_penalty": sym_score,
            "asymmetry_penalty_improved": sym_score_impr,
            "max_asymmetry_gap": {"value": max_gap_val, "pair": [int(i_asym), int(j_asym)]},
            "target_mis_share": {
                "shares": shares,
                "min": min_share,
                "max": max_share,
                "gap": share_gap,
            },
            "subgroup_metrics": subgroup_metrics,
            "class_names": class_names,
            "normalized_rows": args.normalize_rows,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved JSON report to: {args.out_json}")


if __name__ == "__main__":
    main()
