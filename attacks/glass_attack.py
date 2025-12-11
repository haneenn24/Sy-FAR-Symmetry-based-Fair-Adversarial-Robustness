#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sy-FAR / Eyeglass Frame Attack (targeted + untargeted, digit-space, fixed mask)

Usage
-----
UNTARGETED:
python glass_attack.py \
  --model-checkpoint /path/to/model.pt \
  --glass-mask-path ./attacks/mask/eyeglass.png \
  --batch-size 64 \
  --alpha 20 \
  --iters 1 10 50 100 300 \
  --restarts 1 \
  --num-classes 8

TARGETED:
python glass_attack.py \
  --model-checkpoint /path/to/model.pt \
  --glass-mask-path ./attacks/mask/eyeglass.png \
  --targeted \
  --target-class 5 \
  --batch-size 64 \
  --alpha 20 \
  --iters 100 \
  --restarts 1

Notes
-----
- UNTARGETED: maximizes CE(f(x), y)
- TARGETED: minimizes CE(f(x), y_target)
- Mask is fixed; digit-space update; VGG-Face BGR mean is used.
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
# Local imports
# ----------------------------
from utils import data_process
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


def save_image_tensor(image: torch.Tensor, filename: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(image, str(directory / filename))


def rgb_to_bgr(images: torch.Tensor) -> torch.Tensor:
    return images[:, [2, 1, 0], :, :]


def update_confusion_matrix(cm: np.ndarray, labels: torch.Tensor, preds: torch.Tensor) -> None:
    y_true = labels.detach().cpu().numpy()
    y_pred = preds.detach().cpu().numpy()
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1


def compute_class_accuracy(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1)
        diag = np.diag(cm)
        acc = diag / row_sum
        acc[row_sum == 0] = np.nan
        return acc


# ============================================================
# Eyeglass Attack: unified targeted/untargeted
# ============================================================

class EyeglassAttack:
    """
    Unified attack class supporting:
    - untargeted: maximize CE(f(x), true_y)
    - targeted: minimize CE(f(x), target_class)

    Args:
        targeted (bool): attack mode
        target_class (int): only used when targeted=True
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mask: torch.Tensor,
        device: torch.device,
        alpha: float = 20.0,
        momentum: float = 0.4,
        targeted: bool = False,
        target_class: int = None,
        bgr_mean: Tuple[float, float, float] = (129.1863, 104.7624, 93.5940),
    ):
        self.model = model
        self.mask = mask.to(device)
        self.device = device
        self.alpha = alpha
        self.momentum = momentum
        self.targeted = targeted
        self.target_class = target_class

        mean = torch.tensor(bgr_mean).view(1, 3, 1, 1).float()
        self.mean = mean.to(device)

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # color candidates
        self._c0 = [128, 220, 160, 200, 220]
        self._c1 = [128, 130, 105, 175, 210]
        self._c2 = [128, 0, 55, 30, 50]

    @torch.no_grad()
    def _choose_color(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        For untargeted: choose color maximizing CE(f(X), y)
        For targeted: choose color minimizing CE(f(X), target)
        """
        N = labels.shape[0]
        delta_best = torch.zeros_like(X)
        if self.targeted:
            # targeted: minimize CE
            best_loss = torch.full((N,), float("inf"), device=self.device)
        else:
            # untargeted: maximize CE
            best_loss = torch.full((N,), -float("inf"), device=self.device)

        targets = None
        if self.targeted:
            targets = torch.zeros_like(labels) + self.target_class

        for i in range(len(self._c0)):
            d = torch.zeros_like(X)
            d[:, 0] = self.mask[0] * self._c2[i]
            d[:, 1] = self.mask[1] * self._c1[i]
            d[:, 2] = self.mask[2] * self._c0[i]

            logits = self.model(X + d - self.mean)

            loss_vec = nn.CrossEntropyLoss(reduction="none")(
                logits,
                labels if not self.targeted else targets
            )

            if self.targeted:
                take = loss_vec <= best_loss
            else:
                take = loss_vec >= best_loss

            delta_best[take] = d[take]
            best_loss[take] = loss_vec[take]

        return delta_best

    def __call__(self, base: torch.Tensor, labels: torch.Tensor, num_iter: int) -> torch.Tensor:
        """
        base = (original_bgr + mean) * (1 - mask)
        labels = true labels (for untargeted)
        """
        # determine label objective
        if self.targeted:
            y = torch.zeros_like(labels) + self.target_class
        else:
            y = labels

        with torch.no_grad():
            color = self._choose_color(base, labels)

        X = base.clone().detach().requires_grad_(True)
        X.data = X.data + color - self.mean

        delta = torch.zeros_like(X)

        for _ in range(num_iter):
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            loss.backward()

            grad = X.grad.detach() * self.mask
            flat = grad.view(grad.size(0), -1).abs()
            max_val, _ = flat.max(1)
            scale = torch.where(max_val > 0, max_val, torch.ones_like(max_val))

            r = self.alpha * grad / scale.view(-1, 1, 1, 1)
            delta = self.momentum * delta.detach() + r

            over = (delta + X + self.mean) > 255
            under = (delta + X + self.mean) < 0
            delta[over] = 0
            delta[under] = 0

            X.data = X.detach() + delta
            X.data = torch.round(X.detach() + self.mean) - self.mean

            X.grad.zero_()

        return X.detach()


# ============================================================
# Combined Evaluation Loop
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

    attacker = EyeglassAttack(
        model=model,
        mask=mask,
        device=device,
        alpha=alpha,
        targeted=targeted,
        target_class=target_class,
    )

    for n_iter in iters_list:
        logging.info(f"\n=== Attack mode: {'TARGETED' if targeted else 'UNTARGETED'} "
                     f"| iters={n_iter} | alpha={alpha} | restarts={restarts} ===")

        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        total = 0

        success_targeted = 0
        success_strict = 0

        for images_rgb, labels in dataloader:
            images_rgb = images_rgb.to(device)
            labels = labels.to(device)

            images_bgr = rgb_to_bgr(images_rgb)
            targets = torch.zeros_like(labels) + target_class if targeted else labels

            per_restart_success = torch.zeros(labels.size(0), dtype=torch.bool, device=device)
            last_preds = None

            for r in range(restarts):
                base = (images_bgr + attacker.mean) * (1 - attacker.mask)
                Xadv = attacker(base, labels, n_iter)

                with torch.no_grad():
                    preds = model(Xadv).argmax(1)
                    last_preds = preds

                    if targeted:
                        per_restart_success |= (preds == target_class)

                if save_images:
                    pass  # saving logic optional

            update_confusion_matrix(confusion, labels, last_preds)

            if targeted:
                success_targeted += (last_preds == target_class).sum().item()
                mask_no_diag = labels != target_class
                success_strict += ((last_preds == target_class) & mask_no_diag).sum().item()

            total += labels.size(0)

        logging.info("\nConfusion Matrix:")
        logging.info(confusion)

        if targeted:
            logging.info(f"Success Rate (loose): {success_targeted / total:.4f}")
            strict_total = total - confusion[target_class, target_class]
            logging.info(f"Modified Success Rate (strict): {success_strict / strict_total:.4f}")

        per_class = compute_class_accuracy(confusion)
        logging.info("Per-class accuracy:")
        for c in range(num_classes):
            val = per_class[c]
            logging.info(f"Class {c}: {0 if np.isnan(val) else val:.4f}")


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser()

    p.add_argument("--model-checkpoint", type=Path, required=True)
    p.add_argument("--glass-mask-path", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=Path("./attack_outputs"))

    # attack mode
    p.add_argument("--targeted", action="store_true", help="Use targeted attack")
    p.add_argument("--target-class", type=int, default=None, help="Target class for targeted attack")

    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--iters", type=int, nargs="+", default=[100])
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--save-images", action="store_true")
    p.add_argument("--log-images-every", type=int, default=0)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-file", type=str, default="")

    return p.parse_args(argv)


# ============================================================
# Main
# ============================================================

def load_mask_as_tensor(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(img_rgb)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    set_seed(args.seed)

    if args.targeted and args.target_class is None:
        raise ValueError("You must specify --target-class when using --targeted")

    device = torch.device(args.device)

    mask = load_mask_as_tensor(args.glass_mask_path)

    model = VGG_16()
    state = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

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
