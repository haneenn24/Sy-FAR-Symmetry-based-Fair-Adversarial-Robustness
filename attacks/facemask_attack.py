#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sy-FAR / Face Mask Attack (targeted + untargeted, digit-space, fixed mask)

Usage
-----
UNTARGETED:
python attacks/facemask_attack.py \
  --model-checkpoint /path/to/your_model.pt \
  --mask-path attacks/mask/facemask.png \
  --data-dir /path_to_dataset/ \
  --batch-size 64 \
  --alpha 20 \
  --iters 1 10 50 100 \
  --restarts 1 \
  --num-classes 10


TARGETED:
python attacks/facemask_attack.py \
  --model-checkpoint /path/to/your_model.pt \
  --mask-path attacks/mask/facemask.png \
  --data-dir /path_to_dataset/ \
  --batch-size 64 \
  --alpha 20 \
  --iters 200 \
  --restarts 1 \
  --num-classes 10 \
  --targeted \
  --target-class 5

"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# ----------------------------
# Local imports
# ----------------------------
from utils.data_process import data_process
from models.vgg16 import VGG_16


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 12345) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(level: str = "INFO", log_file: str = "") -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def rgb_to_bgr(x: torch.Tensor) -> torch.Tensor:
    return x[:, [2,1,0], :, :]


def update_confusion_matrix(cm: np.ndarray, labels: torch.Tensor, preds: torch.Tensor):
    for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
        cm[t, p] += 1


def compute_acc(cm: np.ndarray):
    acc = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            acc[i] = cm[i,i] / cm[i].sum()
        else:
            acc[i] = np.nan
    return acc


# ============================================================
# Face Mask Grid-Level Attack
# ============================================================

class FaceMaskAttack:

    def __init__(self,
        model,
        mask_M: torch.Tensor,
        device,
        alpha: float = 20.0,
        momentum: float = 0.4,
        grid_size: Tuple[int,int]=(8,16),
        targeted: bool=False,
        target_class: int=None,
        bgr_mean=(129.1863,104.7624,93.5940)
    ):
        self.model = model
        self.M = mask_M.to(device)
        self.device = device
        self.alpha = alpha
        self.mu = momentum
        self.targeted = targeted
        self.target_class = target_class
        self.Gh, self.Gw = grid_size

        self.loss_fn = nn.CrossEntropyLoss()
        self.mean = torch.tensor(bgr_mean).view(1,3,1,1).float().to(device)

    def _T(self, delta_grid, H, W):
        up = torch.nn.functional.interpolate(
            delta_grid, size=(H,W),
            mode="bilinear", align_corners=False
        )
        return up * self.M

    def __call__(self, base, labels, iters):
        N, C, H, W = base.shape

        if self.targeted:
            y = torch.zeros_like(labels) + self.target_class
        else:
            y = labels

        delta = torch.zeros((N,3,self.Gh,self.Gw), device=self.device, requires_grad=True)
        momentum = torch.zeros_like(delta)

        for _ in range(iters):
            delta_img = self._T(delta, H, W)
            X_adv = torch.round(base + delta_img).clamp(0,255) - self.mean

            logits = self.model(X_adv)
            loss = self.loss_fn(logits, y)

            # targeted → minimize CE → take negative
            if self.targeted:
                loss = -loss

            loss.backward()

            g = delta.grad
            g_norm = g.abs().mean() + 1e-8

            momentum = self.mu * momentum + g / g_norm
            delta = (delta + self.alpha * momentum.sign()).detach().requires_grad_(True)

            delta = delta.clamp(0, 1).detach().requires_grad_(True)


        delta_img = self._T(delta, H, W)
        X_adv = torch.round(base + delta_img).clamp(0,255) - self.mean
        return X_adv.detach()


# ============================================================
# Evaluation
# ============================================================

def evaluate_attack(
    model,
    dataloader,
    device,
    mask,
    iters_list,
    alpha,
    restarts,
    num_classes,
    targeted,
    target_class,
):
    attacker = FaceMaskAttack(
        model=model,
        mask_M=mask,
        device=device,
        alpha=alpha,
        targeted=targeted,
        target_class=target_class
    )

    for iters in iters_list:
        logging.info(f"\n=== FaceMask Attack | {'TARGETED' if targeted else 'UNTARGETED'} | iters={iters} ===")

        cm = np.zeros((num_classes, num_classes), dtype=np.int32)

        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            xb_bgr = rgb_to_bgr(xb)

            base = (xb_bgr + attacker.mean) * (1 - mask)

            preds_final = None

            for _ in range(restarts):
                X_adv = attacker(base, yb, iters)
                with torch.no_grad():
                    preds_final = model(X_adv).argmax(1)

            update_confusion_matrix(cm, yb, preds_final)

        logging.info("\nConfusion Matrix:")
        logging.info(cm)

        acc = compute_acc(cm)
        logging.info("Per-class accuracy:")
        for i,a in enumerate(acc):
            logging.info(f"  class {i}: {0 if np.isnan(a) else a:.4f}")


# ============================================================
# CLI and Main
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Grid-level FaceMask Attack")

    # I/O
    p.add_argument("--model-checkpoint", type=Path, required=True)
    p.add_argument("--mask-path", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)   # <<< ADDED!

    # attack
    p.add_argument("--targeted", action="store_true")
    p.add_argument("--target-class", type=int)
    p.add_argument("--alpha", type=float, default=20)
    p.add_argument("--iters", nargs="+", type=int, default=[100])
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=8)

    # system
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--log-level", type=str, default="INFO")

    return p.parse_args(argv)


def load_mask(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = transforms.ToTensor()(img)          # (1,H,W)
    mask = (mask > 0.1).float()                # binarize
    return mask.repeat(3,1,1)                  # (3,H,W)


def main(argv=None):
    args = parse_args(argv)

    setup_logging(args.log_level)
    set_seed(args.seed)

    device = torch.device(args.device)

    # mask
    mask = load_mask(args.mask_path).to(device)

    # model
    model = VGG_16()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location="cpu"))
    model.to(device).eval()

    # data
    dataloaders, _, _ = data_process(batch_size=args.batch_size, data_dir=args.data_dir)

    test_loader = dataloaders["test"]

    evaluate_attack(
        model=model,
        dataloader=test_loader,
        device=device,
        mask=mask,
        iters_list=args.iters,
        alpha=args.alpha,
        restarts=args.restarts,
        num_classes=args.num_classes,
        targeted=args.targeted,
        target_class=args.target_class,
    )


if __name__ == "__main__":
    main()
