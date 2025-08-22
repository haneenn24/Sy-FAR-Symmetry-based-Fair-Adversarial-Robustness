#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rectangular Occlusion Attack (ROA)

Library:
- ROA: exhaustive_search, gradient_based_search, inside_pgd

CLI:
- Evaluate a classifier on a dataset under ROA using either exhaustive or gradient-based location search.
- Assumes inputs are already in the model’s expected space. If your model expects BGR (e.g., VGG-Face style),
  pass --rgb-to-bgr to swap channels before attack/eval.

Example
-------
python roa.py \
  --model-checkpoint ../donemodel/model.pt \
  --img-size 224 \
  --search gradient \
  --alpha 4 \
  --iters 50 \
  --width 70 --height 70 \
  --xskip 10 --yskip 10 \
  --batch-size 8 \
  --rgb-to-bgr
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from time import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Project-local imports (updated per your repo layout)
from utils import data_process
from models.vgg16 import VGG_16


# =========================
# Utility helpers
# =========================

def setup_logging(level: str = "INFO", log_file: Path = None) -> None:
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


def set_seed(seed: int = 12345) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def rgb_to_bgr(x: torch.Tensor) -> torch.Tensor:
    """Swap channels: RGB -> BGR."""
    return x[:, [2, 1, 0], :, :]


# =========================
# ROA library
# =========================

class ROA:
    """
    Rectangular Occlusion Attack (ROA).

    Parameters
    ----------
    base_classifier : torch.nn.Module
        Victim classifier mapping [B, C, H, W] -> [B, num_classes].
    size : int
        Assumed square image size (H = W = size) for grid computations.
        If your images are not square, adapt xtimes/ytimes logic before use.
    """

    def __init__(self, base_classifier: torch.nn.Module, size: int) -> None:
        self.base_classifier = base_classifier
        self.img_size = int(size)

    @torch.no_grad()
    def exhaustive_search(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
        num_iter: int,
        width: int,
        height: int,
        xskip: int,
        yskip: int,
        random: bool = False,
    ) -> torch.Tensor:
        """
        Brute-force search over rectangle positions; choose the one with highest CE loss,
        then run constrained PGD within that rectangle.
        """
        model = self.base_classifier
        model.eval()

        device = X.device
        B = y.shape[0]

        size = self.img_size
        width, height, xskip, yskip = int(width), int(height), int(xskip), int(yskip)
        xtimes = max((size - width) // xskip, 1)
        ytimes = max((size - height) // yskip, 1)

        max_loss = torch.full((B,), -100.0, device=device)
        output_j = torch.zeros(B, device=device)
        output_i = torch.zeros(B, device=device)
        tie_count = torch.zeros(B, device=device)

        loss_fn = nn.CrossEntropyLoss(reduction="none")

        for i in range(xtimes):
            for j in range(ytimes):
                sticker = X.clone()
                sticker[:, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width] = 0.5
                losses = loss_fn(model(sticker), y)  # (B,)

                take = losses > max_loss
                output_j = torch.where(take, torch.tensor(float(j), device=device), output_j)
                output_i = torch.where(take, torch.tensor(float(i), device=device), output_i)
                tie_count += (losses == max_loss).float()
                max_loss = torch.maximum(max_loss, losses)

        # Heuristic: if many ties or zero loss, pick random positions
        tie_threshold = xtimes * ytimes * 0.9
        tie_idx = (tie_count >= tie_threshold).nonzero(as_tuple=True)[0]
        zero_idx = (max_loss == 0).nonzero(as_tuple=True)[0]

        if tie_idx.numel() > 0:
            output_j[tie_idx] = torch.randint(ytimes, (tie_idx.numel(),), device=device).float()
            output_i[tie_idx] = torch.randint(xtimes, (tie_idx.numel(),), device=device).float()
        if zero_idx.numel() > 0:
            output_j[zero_idx] = torch.randint(ytimes, (zero_idx.numel(),), device=device).float()
            output_i[zero_idx] = torch.randint(xtimes, (zero_idx.numel(),), device=device).float()

        return self.inside_pgd(
            X, y, width, height, alpha, num_iter, xskip, yskip, output_j, output_i, random
        )

    def gradient_based_search(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
        num_iter: int,
        width: int,
        height: int,
        xskip: int,
        yskip: int,
        potential_nums: int,
        random: bool = False,
    ) -> torch.Tensor:
        """
        Use input gradients to score rectangle positions (sum of squared grads in the window),
        keep top-k candidates, evaluate them by CE, choose the best, then run constrained PGD.
        """
        model = self.base_classifier
        model.eval()

        device = X.device
        size = self.img_size
        B = y.shape[0]

        width, height, xskip, yskip = int(width), int(height), int(xskip), int(yskip)
        xtimes = max((size - width) // xskip, 1)
        ytimes = max((size - height) // yskip, 1)

        X1 = X.clone().detach().to(device).requires_grad_(True)
        y = y.to(device)

        loss = nn.CrossEntropyLoss()(model(X1), y)
        loss.backward()

        grad = X1.grad.detach()  # (B, C, H, W)
        flat = grad.view(B, -1).abs()
        max_val = flat.max(dim=1).values.clamp_min(1e-12)
        grad = grad / max_val.view(-1, 1, 1, 1)
        X1.grad.zero_()

        scores = torch.zeros((B, ytimes * xtimes), device=device)
        for i in range(xtimes):
            for j in range(ytimes):
                region = grad[:, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width]
                idx = j * xtimes + i  # row-major indexing
                scores[:, idx] = (region * region).sum(dim=(1, 2, 3))

        k = int(min(max(1, potential_nums), scores.shape[1]))
        _, topk_idx = torch.topk(scores, k=k, dim=1)
        cand_js = topk_idx // xtimes
        cand_is = topk_idx % xtimes

        best_j = cand_js[:, 0].float()
        best_i = cand_is[:, 0].float()
        max_loss = torch.zeros(B, device=device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for l in range(k):
                j_sel = cand_js[:, l]
                i_sel = cand_is[:, l]

                sticker = X.clone()
                for b in range(B):
                    j = int(j_sel[b].item())
                    i = int(i_sel[b].item())
                    sticker[b, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width] = 0.5

                losses = loss_fn(model(sticker), y)  # (B,)
                take = losses > max_loss
                best_j = torch.where(take, j_sel.float(), best_j)
                best_i = torch.where(take, i_sel.float(), best_i)
                max_loss = torch.maximum(max_loss, losses)

        return self.inside_pgd(
            X, y, width, height, alpha, num_iter, xskip, yskip, best_j, best_i, random
        )

    def inside_pgd(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        width: int,
        height: int,
        alpha: float,
        num_iter: int,
        xskip: int,
        yskip: int,
        out_j: torch.Tensor,
        out_i: torch.Tensor,
        random: bool = False,
    ) -> torch.Tensor:
        """
        Constrained PGD that updates only inside the selected rectangle per sample.

        Assumes inputs in [0,1] (or already in model’s normalized space). The rectangle
        values are clamped to [0,1]. If your model expects a different scale, adapt upstream.
        """
        model = self.base_classifier
        model.eval()

        device = X.device

        # Build per-sample binary mask for the rectangle
        sticker = torch.zeros_like(X, requires_grad=False, device=device)
        for b, ii in enumerate(out_i):
            j = int(out_j[b].item())
            i = int(ii.item())
            sticker[b, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width] = 1.0

        # Initialize rectangle values
        delta = torch.rand_like(X, device=device) if random else (torch.zeros_like(X, device=device) + 0.5)

        X1 = X.detach() * (1 - sticker) + delta * sticker
        X1 = X1.clone().detach().requires_grad_(True)

        loss_fn = nn.CrossEntropyLoss()
        step = float(alpha)

        for _ in range(int(num_iter)):
            loss = loss_fn(model(X1), y)
            loss.backward()

            X1.data = X1.detach() + step * X1.grad.detach().sign() * sticker
            X1.data = X1.data.clamp(0.0, 1.0)
            X1.grad.zero_()

        return X1.detach()


# =========================
# CLI
# =========================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectangular Occlusion Attack (ROA) CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / data
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pt/.pth).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--img-size", type=int, default=224, help="Assumed square image size used by ROA grid.")
    parser.add_argument("--rgb-to-bgr", action="store_true",
                        help="Swap channels RGB->BGR before attack/eval (for models expecting BGR).")

    # Attack config
    parser.add_argument("--search", type=str, default="gradient", choices=["exhaustive", "gradient"],
                        help="Location search strategy.")
    parser.add_argument("--alpha", type=float, default=4.0, help="cPGD step size.")
    parser.add_argument("--iters", type=int, default=50, help="cPGD iterations.")
    parser.add_argument("--width", type=int, default=70, help="Rectangle width.")
    parser.add_argument("--height", type=int, default=70, help="Rectangle height.")
    parser.add_argument("--xskip", type=int, default=10, help="Stride on x during location search.")
    parser.add_argument("--yskip", type=int, default=10, help="Stride on y during location search.")
    parser.add_argument("--potential-nums", type=int, default=30,
                        help="Top-K candidates kept in gradient-based search.")
    parser.add_argument("--random-init", action="store_true",
                        help="Randomly initialize rectangle values (else mid-gray 0.5).")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path.")

    return parser.parse_args(argv)


def evaluate_with_roa(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    img_size: int,
    search: str,
    alpha: float,
    iters: int,
    width: int,
    height: int,
    xskip: int,
    yskip: int,
    potential_nums: int,
    random_init: bool,
    rgb2bgr: bool,
) -> None:
    """
    Run ROA on the test set and report running & final robust accuracy.
    """
    attacker = ROA(model, size=img_size)
    total = 0
    correct = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if rgb2bgr:
            images = rgb_to_bgr(images)

        start = time()
        if search == "exhaustive":
            adv = attacker.exhaustive_search(
                images, labels, alpha, iters, width, height, xskip, yskip, random=random_init
            )
        else:
            adv = attacker.gradient_based_search(
                images, labels, alpha, iters, width, height, xskip, yskip,
                potential_nums=potential_nums, random=random_init
            )
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        elapsed = time() - start
        logging.info(f"Running Acc: {correct/max(1,total):.4f} | Seen: {total} | Batch time: {elapsed:.2f}s")

    logging.info(f"Final ROA Robust Accuracy: {100.0 * correct / max(1, total):.2f}% ({correct}/{total})")


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    if device.type == "cpu" and "cuda" in args.device:
        logging.warning("CUDA requested but not available. Falling back to CPU.")

    # Build / load model
    model = VGG_16()
    if not args.model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    state = torch.load(str(args.model_checkpoint), map_workdir := "cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    # Data (expects utils.data_process to provide a dict with 'test')
    dataloaders, dataset_sizes, _ = data_process(args.batch_size)
    if "test" not in dataloaders:
        raise KeyError("Expected a 'test' dataloader from utils.data_process().")

    evaluate_with_roa(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        img_size=args.img_size,
        search=args.search,
        alpha=args.alpha,
        iters=args.iters,
        width=args.width,
        height=args.height,
        xskip=args.xskip,
        yskip=args.yskip,
        potential_nums=args.potential_nums,
        random_init=args.random_init,
        rgb2bgr=args.rgb_to_bgr,
    )


if __name__ == "__main__":
    main()
