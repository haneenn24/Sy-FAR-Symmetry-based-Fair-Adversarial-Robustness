#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rectangular Occlusion Attack (ROA)
----------------------------------
Runs ROA against a classifier using either:
  - exhaustive search over rectangle locations, or
  - gradient-based search to select promising rectangles, then cPGD refinement.

Usage:
python sticker_attack.py \
  --model-checkpoint ../donemodel/model.pt \
  --alpha 4 \
  --iters 50 \
  --search 1 \
  --stride 10 \
  --width 70 \
  --height 70 \
  --batch-size 8

Notes:
- Attack simulates a "physical" sticker style occlusion; mask is rectangular here,
  but conceptually the occluder can be any shape.
- Pipeline assumes VGG-Face preprocessing with BGR mean and RGB->BGR channel flip.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import random
from pathlib import Path
from time import time
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Project-local imports
from utils import data_process
from models.vgg16 import VGG_16


# ----------------------------
# Utilities
# ----------------------------

def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), mode="w"))
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def set_seed(seed: int = 123456) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rgb_to_bgr(x: torch.Tensor) -> torch.Tensor:
    """Swap channels: RGB -> BGR."""
    return x[:, [2, 1, 0], :, :]


# ----------------------------
# ROA Implementation
# ----------------------------

class ROA:
    """
    Rectangular Occlusion Attack (ROA).

    Parameters
    ----------
    base_classifier : torch.nn.Module
        The victim classifier.
    alpha : float
        cPGD step size.
    iters : int
        cPGD iterations.
    bgr_mean : tuple[float, float, float]
        Per-channel mean in BGR ordering (VGG-Face defaults).
    """

    def __init__(self, base_classifier: torch.nn.Module, alpha: float, iters: int,
                 bgr_mean: Tuple[float, float, float] = (129.1863, 104.7624, 93.5940)) -> None:
        self.base_classifier = base_classifier
        self.alpha = float(alpha)
        self.iters = int(iters)
        mean = torch.tensor(bgr_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = mean

    @torch.no_grad()
    def exhaustive_search(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        width: int,
        height: int,
        xskip: int,
        yskip: int,
    ) -> torch.Tensor:
        """
        Brute-force search over rectangle positions; choose the one with highest CE loss.
        """
        model = self.base_classifier
        model.eval()

        device = X.device
        mean = self.mean.to(device)

        max_loss = torch.full((y.shape[0],), -100.0, device=device)
        xtimes = (224 - width) // xskip
        ytimes = (224 - height) // yskip

        best_j = torch.zeros(y.shape[0], device=device)
        best_i = torch.zeros(y.shape[0], device=device)

        loss_fn = nn.CrossEntropyLoss(reduction="none")

        for i in range(xtimes):
            for j in range(ytimes):
                sticker = X + mean
                sticker[:, :, yskip * j : (yskip * j + height), xskip * i : (xskip * i + width)] = 255.0 / 2.0
                sticker1 = sticker - mean

                losses = loss_fn(model(sticker1), y)  # (N,)
                take = losses > max_loss
                best_j[take] = float(j)
                best_i[take] = float(i)
                max_loss = torch.maximum(max_loss, losses)

        # If the max loss is zero for some samples, choose a random position
        zeros = (max_loss == 0).nonzero(as_tuple=True)[0]
        if zeros.numel() > 0:
            best_j[zeros] = torch.randint(0, ytimes, (zeros.numel(),), device=device).float()
            best_i[zeros] = torch.randint(0, xtimes, (zeros.numel(),), device=device).float()

        return self._cpgd(X, y, width, height, xskip, yskip, best_j, best_i)

    def gradient_based_search(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        width: int,
        height: int,
        xskip: int,
        yskip: int,
        topk: int = 30,
    ) -> torch.Tensor:
        """
        Use input gradient to score rectangle locations; evaluate the best K; pick the top-scoring one per sample.
        Then run cPGD refinement on that rectangle.
        """
        device = X.device
        model = self.base_classifier
        model.eval()

        X1 = X.clone().detach().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(X1), y)
        loss.backward()

        grad = X1.grad.detach()  # (N,3,H,W)
        # Normalize per sample by max |grad|
        flat = grad.view(grad.shape[0], -1).abs()
        max_val, _ = flat.max(dim=1)
        denom = torch.where(max_val > 0, max_val, torch.ones_like(max_val))
        grad = grad / denom.view(-1, 1, 1, 1)

        X1.grad.zero_()

        mean = self.mean.to(device)
        xtimes = 224 // xskip
        ytimes = 224 // yskip

        # Score each (i,j) window by local grad energy within the rectangle
        # Build matrix (N, ytimes*xtimes) of scores
        matrix = torch.zeros((y.shape[0], ytimes * xtimes), device=device)

        for i in range(xtimes):
            for j in range(ytimes):
                region = grad[:, :, yskip * j : (yskip * j + height), xskip * i : (xskip * i + width)]
                # Sum of squared grads over the rectangle region
                score = (region * region).sum(dim=(1, 2, 3))  # (N,)
                idx = j * xtimes + i  # NOTE: fixed indexing bug (was j*ytimes + i)
                matrix[:, idx] = score

        # Take top-k indices per sample
        topk_vals, topk_idx = torch.topk(matrix, k=min(topk, matrix.shape[1]), dim=1)
        cand_js = topk_idx // xtimes
        cand_is = topk_idx % xtimes

        # Evaluate the top-k rectangles using CE loss; select best per-sample
        best_j = cand_js[:, 0].float()
        best_i = cand_is[:, 0].float()
        max_loss = torch.zeros(y.shape[0], device=device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for l in range(cand_js.shape[1]):
                j_sel = cand_js[:, l]
                i_sel = cand_is[:, l]

                sticker = X + mean
                for m in range(sticker.shape[0]):
                    j = int(j_sel[m].item())
                    i = int(i_sel[m].item())
                    sticker[m, :, yskip * j : (yskip * j + height), xskip * i : (xskip * i + width)] = 255.0 / 2.0
                sticker1 = sticker - mean

                losses = loss_fn(model(sticker1), y)
                take = losses > max_loss
                best_j[take] = j_sel[take].float()
                best_i[take] = i_sel[take].float()
                max_loss = torch.maximum(max_loss, losses)

        return self._cpgd(X, y, width, height, xskip, yskip, best_j, best_i)

    def _cpgd(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        width: int,
        height: int,
        xskip: int,
        yskip: int,
        out_j: torch.Tensor,
        out_i: torch.Tensor,
    ) -> torch.Tensor:
        """
        Constrained PGD that updates only within the selected rectangular mask per sample.
        """
        model = self.base_classifier
        model.eval()
        device = X.device
        mean = self.mean.to(device)

        # Build the rectangle mask ("sticker") per sample
        sticker = torch.zeros_like(X, requires_grad=False)
        for n, ii in enumerate(out_i):
            j = int(out_j[n].item())
            i = int(ii.item())
            sticker[n, :, yskip * j : (yskip * j + height), xskip * i : (xskip * i + width)] = 1.0
        sticker = sticker.to(device)

        # Initialize masked region to mid-gray (255/2) in pixel space (with mean subtracted)
        delta = torch.zeros_like(X, requires_grad=False) + 255.0 / 2.0
        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach() * (1 - sticker) + ((delta.detach() - mean) * sticker)

        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.iters):
            loss = loss_fn(model(X1), y)
            loss.backward()

            # Sign gradient ascent within the sticker mask
            X1.data = X1.detach() + self.alpha * X1.grad.detach().sign() * sticker
            # Clamp to valid pixel range in digit space: (X1 + mean) in [0,255]
            X1.data = (X1.detach() + mean).clamp(0, 255) - mean

            X1.grad.zero_()

        return X1.detach()


# ----------------------------
# Evaluation loop
# ----------------------------

def evaluate_roa(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    alpha: float,
    iters: int,
    search: int,
    stride: int,
    width: int,
    height: int,
    device: torch.device,
) -> None:
    """
    Runs ROA on the test set and logs streaming accuracy + time per pass.
    """
    set_seed(123456)
    total = 0
    correct = 0

    for images, labels in dataloader:
        images = rgb_to_bgr(images.to(device))
        labels = labels.to(device)

        roa = ROA(model, alpha=alpha, iters=iters)
        start = time()

        if search == 0:
            adv = roa.exhaustive_search(images, labels, width, height, stride, stride)
        else:
            adv = roa.gradient_based_search(images, labels, width, height, stride, stride)

        with torch.no_grad():
            outputs = model(adv)
            preds = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        elapsed = str(datetime.timedelta(seconds=(time() - start)))
        acc = correct / max(1, total)
        logging.info(f"Running Acc: {acc:.4f}  |  Seen: {total}  |  Elapsed: {elapsed}")

    logging.info(f"Final ROA Robust Accuracy: {100.0 * correct / max(1, total):.2f}% ({correct}/{total})")


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectangular Occlusion Attack (ROA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / data
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pt/.pth).")
    parser.add_argument("--batch-size", type=int, default=8, help="Test batch size.")

    # Attack
    parser.add_argument("--alpha", type=float, required=True, help="cPGD step size.")
    parser.add_argument("--iters", type=int, required=True, help="cPGD iterations.")
    parser.add_argument("--search", type=int, default=1,
                        help="0 = exhaustive_search, 1 = gradient_based_search.")
    parser.add_argument("--stride", type=int, default=10, help="Search stride (pixels).")
    parser.add_argument("--width", type=int, default=70, help="Rectangle width.")
    parser.add_argument("--height", type=int, default=70, help="Rectangle height.")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=123456, help="Random seed.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path.")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    if device.type == "cpu" and "cuda" in args.device:
        logging.warning("CUDA requested but not available. Falling back to CPU.")

    # Load model
    if not args.model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    model = VGG_16()
    state = torch.load(str(args.model_checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Data (expects utils.data_process to return dict with 'test')
    dataloaders, dataset_sizes, _ = data_process(args.batch_size)
    if "test" not in dataloaders:
        raise KeyError("Expected a 'test' dataloader from utils.data_process().")

    evaluate_roa(
        model=model,
        dataloader=dataloaders["test"],
        alpha=args.alpha,
        iters=args.iters,
        search=args.search,
        stride=args.stride,
        width=args.width,
        height=args.height,
        device=device,
    )


if __name__ == "__main__":
    main()
