#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomized Smoothing under Learned Adversarial Patch (circle/square)

The attack is learned from a small set of images (e.g., 27 images, 3 per class).
Evaluates smoothed classifier accuracy against the patched inputs.

Original scaffold based on prior patch-attack scripts; refactored for clarity and modularity.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from core import Smooth
from make_patch_utils import (  # utility funcs expected in your repo
    init_patch_circle,
    init_patch_square,
    circle_transform,
    square_transform,
    submatrix,
    progress_bar,
)
from models.vgg16 import VGG_16


# ----------------------------
# Configuration structures
# ----------------------------

@dataclass
class NetConfig:
    input_size: Tuple[int, int, int] = (3, 224, 224)
    input_range: Tuple[int, int] = (0, 255)
    mean: Tuple[float, float, float] = (0.367035294117647, 0.41083294117647057, 0.5066129411764705)
    std: Tuple[float, float, float] = (1 / 255, 1 / 255, 1 / 255)
    num_classes: int = 10
    input_space: str = "RGB"


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True  # keep as original behavior


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


def compute_out_bounds(net_cfg: NetConfig) -> Tuple[float, float]:
    """Compute normalized output min/max given input_range, mean, std."""
    min_in, max_in = net_cfg.input_range
    min_in = np.array([min_in] * 3, dtype=np.float32)
    max_in = np.array([max_in] * 3, dtype=np.float32)
    mean = np.array(net_cfg.mean, dtype=np.float32)
    std = np.array(net_cfg.std, dtype=np.float32)
    min_out = np.min((min_in - mean) / std)
    max_out = np.max((max_in - mean) / std)
    return float(min_out), float(max_out)


def rgb_to_bgr(x: torch.Tensor) -> torch.Tensor:
    """Swap channels RGB->BGR."""
    return x[:, [2, 1, 0], :, :]


# ----------------------------
# Attack primitives
# ----------------------------

@torch.no_grad()
def _build_adv_from_patch(x: torch.Tensor, patch: torch.Tensor, mask: torch.Tensor,
                          min_out: float, max_out: float) -> torch.Tensor:
    # adv_x = (1 - mask) * x + mask * patch
    adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)
    adv_x = torch.clamp(adv_x, min_out, max_out)
    return adv_x


def attack_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    patch: torch.Tensor,
    mask: torch.Tensor,
    target_cls: int,
    min_out: float,
    max_out: float,
    step_size: float,
    max_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimize the patch (masked region) to increase target class log-probability.

    Returns:
        adv_x, mask, patch
    """
    model.eval()

    # Initial target prob
    with torch.no_grad():
        x_out = F.softmax(model(x), dim=1)
        target_prob = x_out.data[0, target_cls]

    adv_x = _build_adv_from_patch(x, patch, mask, min_out, max_out)

    count = 0
    while target_prob < 1.0 and count < max_count:
        count += 1

        adv_x = Variable(adv_x.data, requires_grad=True)
        adv_out = F.log_softmax(model(adv_x), dim=1)
        # maximize log-prob of target -> minimize negative
        loss = -adv_out[0, target_cls]
        loss.backward()

        adv_grad = adv_x.grad.clone()
        adv_x.grad.data.zero_()

        # Normalize by max absolute gradient to keep step scale stable
        denom = torch.max(adv_grad.abs())
        denom = denom if denom > 0 else torch.tensor(1.0, device=adv_grad.device)
        patch = patch - step_size * (adv_grad / denom)

        adv_x = _build_adv_from_patch(x, patch, mask, min_out, max_out)

        with torch.no_grad():
            out = F.softmax(model(adv_x), dim=1)
            target_prob = out.data[0, target_cls]

        if count >= max_count:
            break

    return adv_x, mask, patch


# ----------------------------
# Train / Test epochs
# ----------------------------

def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    patch: np.ndarray,
    patch_shape: Tuple[int, int, int],
    patch_type: str,
    image_size: int,
    target_cls: int,
    min_out: float,
    max_out: float,
    step_size: float,
    max_count: int,
    device: torch.device,
) -> np.ndarray:
    """
    Learns/updates the patch over the training set (one pass), only on samples that
    are originally correctly classified and not belonging to the target class.
    """
    model.eval()
    success, total = 0, 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        if labels.item() == target_cls:
            continue

        data = data.to(device)
        labels = labels.to(device)
        data = rgb_to_bgr(data)  # RGB -> BGR to match historical preprocessing

        with torch.no_grad():
            pred = model(data)
            pred_labels = pred.argmax(dim=1)

        # Attack only examples originally classified correctly
        if pred_labels[0] != labels[0]:
            continue

        total += 1

        data_shape = data.shape  # (1, 3, H, W)
        patch_copy = np.copy(patch)

        # Transform patch & build mask for current sample
        if patch_type == "circle":
            patch_t, mask_t, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == "square":
            patch_t, mask_t = square_transform(patch, data_shape, patch_shape, image_size)
        else:
            raise ValueError("patch_type must be 'circle' or 'square'")

        patch_t = torch.FloatTensor(patch_t).to(device)
        mask_t = torch.FloatTensor(mask_t).to(device)

        adv_x, mask_t, patch_t = attack_step(
            model=model,
            x=data,
            patch=patch_t,
            mask=mask_t,
            target_cls=target_cls,
            min_out=min_out,
            max_out=max_out,
            step_size=5.0,          # matches legacy scale; override via --step-size if you expose it
            max_count=max_count,
        )

        with torch.no_grad():
            adv_label = model(adv_x).argmax(dim=1)[0]
        if int(adv_label.item()) == int(target_cls):
            success += 1

        # Extract learned (masked) patch back to canonical shape
        masked_patch = torch.mul(mask_t, patch_t)
        patch_arr = masked_patch.data.detach().cpu().numpy()
        new_patch = np.zeros(patch_shape, dtype=patch_arr.dtype)

        # Validate shape compatibility (legacy guard)
        if submatrix(patch_arr[0][0]).shape != new_patch[0][0].shape:
            patch = patch_copy
            progress_bar(batch_idx, len(train_loader), f"Train Patch Success: {success/max(1,total):.3f} (skipped due to shape)")
            continue

        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch_arr[i][j])

        patch = new_patch
        progress_bar(batch_idx, len(train_loader), f"Train Patch Success: {success/max(1,total):.3f}")

    return patch


def test_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    test_loader: DataLoader,
    patch: np.ndarray,
    patch_shape: Tuple[int, int, int],
    patch_type: str,
    image_size: int,
    target_cls: int,
    min_out: float,
    max_out: float,
    sigma: float,
    N: int,
    alpha: float,
    batch_mc: int,
    device: torch.device,
    final_epoch: int,
) -> np.ndarray:
    """
    Applies the current patch on test samples and, on the final epoch, evaluates
    randomized smoothing accuracy (smoothed prediction equals true label).
    """
    model.eval()
    cor, tot = 0, 0
    smoothed_classifier = Smooth(model, num_classes=10, sigma=sigma)

    for batch_idx, (data, labels) in enumerate(test_loader):
        if labels.item() == target_cls:
            continue

        data = data.to(device)
        labels = labels.to(device)
        data = rgb_to_bgr(data)

        data_shape = data.shape
        if patch_type == "circle":
            patch_t, mask_t, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == "square":
            patch_t, mask_t = square_transform(patch, data_shape, patch_shape, image_size)
        else:
            raise ValueError("patch_type must be 'circle' or 'square'")

        patch_t = torch.FloatTensor(patch_t).to(device)
        mask_t = torch.FloatTensor(mask_t).to(device)

        adv_x = _build_adv_from_patch(data, patch_t, mask_t, min_out=min_out, max_out=max_out)

        if epoch == final_epoch:
            before_time = time()
            prediction = smoothed_classifier.predict(adv_x, N, alpha, batch_mc)
            _ = time() - before_time  # time is available if you want to log per-sample
            cor += int(prediction == int(labels.item()))
            tot += 1

        # extract masked patch back to canonical shape
        masked_patch = torch.mul(mask_t, patch_t)
        patch_arr = masked_patch.data.detach().cpu().numpy()
        new_patch = np.zeros(patch_shape, dtype=patch_arr.dtype)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch_arr[i][j])
        patch = new_patch

    if epoch == final_epoch:
        acc = 100.0 * cor / max(1, tot)
        logging.info(f"[Final] Smoothed accuracy with patch: {cor}/{tot} = {acc:.2f}%")
    else:
        logging.info("Continuing to next epoch...")

    return patch


# ----------------------------
# Dataloaders
# ----------------------------

def build_loaders(
    data_root: Path,
    net_cfg: NetConfig,
    workers: int,
    train_size: int,
    test_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test loaders from ImageFolder structure:
        data_root/Train_patch
        data_root/val
    """
    normalize = transforms.Normalize(mean=list(net_cfg.mean), std=list(net_cfg.std))
    resize_to = round(max(net_cfg.input_size) * 1.050)
    crop_to = max(net_cfg.input_size)

    train_ds = dset.ImageFolder(
        str(data_root / "Train_patch"),
        transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(crop_to),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    test_ds = dset.ImageFolder(
        str(data_root / "val"),
        transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(crop_to),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    # The original script uses batch_size=1 and shuffle=True
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learned adversarial patch + randomized smoothing evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / paths
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pt/.pth).")
    parser.add_argument("--data-root", type=Path, default=Path(".."),
                        help="Root folder containing 'Train_patch' and 'val' subfolders.")
    parser.add_argument("--out-dir", type=Path, default=Path("./logs"),
                        help="Output folder for logs/artifacts.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])

    # Smoothing
    parser.add_argument("--sigma", type=float, required=True, help="Noise sigma for randomized smoothing.")
    parser.add_argument("--N", type=int, default=1000, help="Number of MC samples for smoothing.")
    parser.add_argument("--alpha", type=float, default=0.001, help="Failure probability.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size used inside Smooth.predict.")

    # Data / training
    parser.add_argument("--workers", type=int, default=1, help="Number of data loader workers.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to learn the patch.")
    parser.add_argument("--target", type=int, default=0, help="Target class for the patch.")
    parser.add_argument("--max-count", type=int, default=100, help="Max iterations per sample in patch optimization.")
    parser.add_argument("--patch-type", type=str, default="square", choices=["circle", "square"],
                        help="Patch geometry.")
    parser.add_argument("--patch-sizes", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.25],
                        help="List of relative patch sizes to iterate over.")
    parser.add_argument("--image-size", type=int, default=224, help="Input square size for transforms.")
    parser.add_argument("--seed", type=int, default=1338, help="Random seed.")

    return parser.parse_args(argv)


# ----------------------------
# Main
# ----------------------------

def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    set_seed(args.seed, use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Model
    logging.info("Creating model...")
    model = VGG_16()
    if not args.model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    state = torch.load(str(args.model_checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Data config
    net_cfg = NetConfig(input_size=(3, args.image_size, args.image_size))
    min_out, max_out = compute_out_bounds(net_cfg)

    # Data loaders (batch_size=1 to match original logic)
    train_loader, test_loader = build_loaders(
        data_root=args.data_root,
        net_cfg=net_cfg,
        workers=args.workers,
        train_size=2000,  # kept for parity; not explicitly used in original snippet
        test_size=2000,   # kept for parity; not explicitly used in original snippet
    )

    # Iterate over patch sizes
    for patch_size in args.patch_sizes:
        logging.info(f"=== Running patch size {patch_size:.2f} ===")

        # Initialize canonical patch
        if args.patch_type == "circle":
            patch, patch_shape = init_patch_circle(args.image_size, patch_size)
        elif args.patch_type == "square":
            patch, patch_shape = init_patch_square(args.image_size, patch_size)
        else:
            raise ValueError("patch_type must be 'circle' or 'square'")

        # Train the patch across epochs
        for epoch in range(1, args.epochs + 1):
            patch = train_one_epoch(
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                patch=patch,
                patch_shape=patch_shape,
                patch_type=args.patch_type,
                image_size=args.image_size,
                target_cls=args.target,
                min_out=min_out,
                max_out=max_out,
                step_size=5.0,
                max_count=args.max_count,
                device=device,
            )

            patch = test_one_epoch(
                epoch=epoch,
                model=model,
                test_loader=test_loader,
                patch=patch,
                patch_shape=patch_shape,
                patch_type=args.patch_type,
                image_size=args.image_size,
                target_cls=args.target,
                min_out=min_out,
                max_out=max_out,
                sigma=args.sigma,
                N=args.N,
                alpha=args.alpha,
                batch_mc=args.batch,
                device=device,
                final_epoch=args.epochs,
            )

    logging.info("Done.")


if __name__ == "__main__":
    main()
