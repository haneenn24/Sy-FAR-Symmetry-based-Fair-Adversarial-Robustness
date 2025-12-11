#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sy-FAR / Face Mask Attack (targeted + untargeted, digit-space, fixed mask)

Usage
-----
UNTARGETED:
python mask_attack.py \
  --model-checkpoint /path/to/model.pt \
  --mask-path attacks/mask/facemask.png \
  --batch-size 64 \
  --alpha 20 \
  --iters 1 10 50 100 \
  --num-classes 8

TARGETED:
python mask_attack.py \
  --model-checkpoint /path/to/model.pt \
  --mask-path attacks/mask/facemask.png \
  --targeted \
  --target-class 5 \
  --alpha 20 \
  --iters 200

Notes
-----
- This is the *grid-level* face mask attack inspired by FACESEC (δ-grid).
- Untargeted: maximize CE(f(X), y)
- Targeted: minimize CE(f(X), y_target)
- δ is a learnable (8×16×3) grid → upsampled → masked → added in digit space.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# ----------------------------
# Local project imports
# ----------------------------
from utils import data_process
from models.vgg16 import VGG_16


# ============================================================
# Utility Functions
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


def save_image_tensor(image: torch.Tensor, filename: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(image, str(directory / filename))


def rgb_to_bgr(images: torch.Tensor) -> torch.Tensor:
    return images[:, [2, 1, 0], :, :]


def update_confusion_matrix(cm: np.ndarray, labels: torch.Tensor, preds: torch.Tensor) -> None:
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1


def compute_class_accuracy(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1)
        diag = np.diag(cm)
        out = diag / row_sum
        out[row_sum == 0] = np.nan
        return out


# ============================================================
# FACE MASK ATTACK (GRID-LEVEL)
# ============================================================

class FaceMaskAttack:
    """
    Implements grid-level δ attack used on the face mask region.

    δ-grid: shape (N,3,Gh,Gw)
    T(δ): bilinear upsample to image → multiply by binary mask M
    """

    def __init__(
        self,
        model,
        mask_M: torch.Tensor,
        device,
        alpha: float = 20.0,
        momentum: float = 0.4,
        grid_size: Tuple[int,int] = (8,16),
        targeted: bool = False,
        target_class: int = None,
        bgr_mean: Tuple[float,float,float] = (129.1863,104.7624,93.5940),
    ):
        self.model = model
        self.M = mask_M.to(device)               # (3,H,W)
        self.device = device
        self.alpha = alpha
        self.mu = momentum
        self.targeted = targeted
        self.target_class = target_class
        self.Gh, self.Gw = grid_size

        self.loss_red = nn.CrossEntropyLoss()

        mean = torch.tensor(bgr_mean).view(1,3,1,1).float()
        self.mean = mean.to(device)

    # ---------------------------
    # TRANSFORM T(δ): upsample + mask
    # ---------------------------
    def _apply_T(self, delta_grid, H, W):
        up = torch.nn.functional.interpolate(delta_grid, size=(H,W),
                    mode="bilinear", align_corners=False)
        return up * self.M

    # ---------------------------
    # MAIN ATTACK CALL
    # ---------------------------
    def __call__(self, base, labels, iters: int):
        """
        base = (X_bgr + mean) * (1 - mask)
        labels: true labels (untargeted)
        """
        N, C, H, W = base.shape

        if self.targeted:
            y = torch.zeros_like(labels) + self.target_class
        else:
            y = labels

        # Initialise δ-grid (zero start)
        delta = torch.zeros((N,3,self.Gh,self.Gw), device=self.device, requires_grad=True)
        mom = torch.zeros_like(delta)

        for _ in range(iters):
            delta_img = self._apply_T(delta, H, W)
            X_adv = torch.clamp(
                torch.round(base + delta_img),
                0, 255
            ) - self.mean

            logits = self.model(X_adv)
            loss = self.loss_red(logits, y)

            # Targeted: minimize CE → maximize negative CE
            if self.targeted:
                loss = -loss

            loss.backward()

            grad = delta.grad
            grad_norm = grad.abs().mean() + 1e-8

            mom = self.mu * mom + grad / grad_norm
            delta = (delta + self.alpha * mom.sign()).detach().requires_grad_(True)

            delta.clamp_(0, 1.0)   # safe normalized grid range

        # Final attack image
        delta_img = self._apply_T(delta, H, W)
        X_adv = torch.round(base + delta_img) - self.mean
        return X_adv.detach()


# ============================================================
# EVALUATION LOOP
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
    save_images,
    save_dir,
    targeted,
    target_class,
    log_images_every=0,
):

    attacker = FaceMaskAttack(
        model=model,
        mask_M=mask,
        device=device,
        alpha=alpha,
        targeted=targeted,
        target_class=target_class
    )

    for n_iter in iters_list:
        logging.info(f"\n=== Face Mask Attack | {'TARGETED' if targeted else 'UNTARGETED'} "
                     f"| iters={n_iter} | alpha={alpha} ===")

        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        total = 0

        success_loose = 0
        success_strict = 0

        for images_rgb, labels in dataloader:
            images_rgb = images_rgb.to(device)
            labels = labels.to(device)

            images_bgr = rgb_to_bgr(images_rgb)

            for r in range(restarts):
                # Build base X0 = (X + mean)*(1-M)
                base = (images_bgr + attacker.mean) * (1 - mask)

                X_adv = attacker(base, labels, n_iter)

                with torch.no_grad():
                    preds = model(X_adv).argmax(1)

            # Update confusion matrix
            update_confusion_matrix(confusion, labels, preds)

            if targeted:
                success_loose += (preds == target_class).sum().item()
                mask_no_diag = labels != target_class
                success_strict += ((preds == target_class) & mask_no_diag).sum().item()

            total += labels.size(0)

        # Log Results
        logging.info("\nConfusion Matrix:")
        logging.info(confusion)

        if targeted:
            logging.info(f"Loose Success Rate: {success_loose/total:.4f}")
            strict_total = total - confusion[target_class,target_class]
            logging.info(f"Strict Success Rate: {success_strict/strict_total:.4f}")

        per_class = compute_class_accuracy(confusion)
        logging.info("Per-class accuracy:")
        for c in range(num_classes):
            logging.info(f"Class {c}: {0 if np.isnan(per_class[c]) else per_class[c]:.4f}")


# ============================================================
# CLI + Main
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Grid-level Face Mask Attack"
    )

    # Paths / Model
    p.add_argument("--model-checkpoint", type=Path, required=True)
    p.add_argument("--mask-path", type=Path, required=True,
                   help="Path to facemask.png mask (white region = attackable).")
    p.add_argument("--save-dir", type=Path, default=Path("./mask_attack_outputs"))

    # Attack
    p.add_argument("--targeted", action="store_true")
    p.add_argument("--target-class", type=int, default=None)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--iters", type=int, nargs="+", default=[100])
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=8)

    # Data / system
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default="cuda:0")

    # Logging
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--log-images-every", type=int, default=0)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-file", type=str, default="")

    return p.parse_args(argv)


def load_mask(mask_path: Path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = transforms.ToTensor()(img)  # (1,H,W)
    mask = (mask > 0.1).float()
    return mask.repeat(3,1,1)          # → (3,H,W)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    set_seed(args.seed)

    if args.targeted and args.target_class is None:
        raise ValueError("You must provide --target-class in targeted mode.")

    device = torch.device(args.device)

    mask = load_mask(args.mask_path)

    # Load model
    model = VGG_16()
    state = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    # Data
    dataloaders, _, _ = data_process(args.batch_size)
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
        save_images=args.save_images,
        save_dir=args.save_dir,
        targeted=args.targeted,
        target_class=args.target_class,
        log_images_every=args.log_images_every,
    )


if __name__ == "__main__":
    main()
