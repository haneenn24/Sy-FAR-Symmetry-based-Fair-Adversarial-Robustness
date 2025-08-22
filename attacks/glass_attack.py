#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sy-FAR / Eyeglass Frame Attack (untargeted, digit-space, fixed mask)

Original reference (pytorch port): https://github.com/mahmoods01/accessorize-to-a-crime
This script performs an eyeglass-frame adversarial attack and evaluates robustness,
including confusion matrix and per-class accuracy, with optional adversarial image saving.

Usage
-----
python glass_attack.py \
  --model-checkpoint /path/to/model.pt \
  --glass-mask-path ./glass/Experiment/dataprepare/silhouette.png \
  --batch-size 64 \
  --alpha 20 \
  --iters 1 10 50 100 300 \
  --restarts 1 \
  --num-classes 8 \
  --save-images \
  --save-dir ./attack_outputs

Notes
-----
- Attack is untargeted: maximizes CE loss of (f(x), y).
- Attack operates in "digit space" (no geometric transforms); mask is fixed.
- The mean is VGG-Face (BGR) and inputs are converted RGB->BGR to match the original pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
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



# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 12345) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If you need deterministic/cuDNN behavior, uncomment the following:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_logging(level: str = "INFO", log_file: str = "") -> None:
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def save_image_tensor(image: torch.Tensor, filename: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    torchvision.utils.save_image(image, str(path))


# ----------------------------
# Eyeglass Attack Core
# ----------------------------

class EyeglassAttack:
    """
    Implements the eyeglass-frame attack with momentum on a fixed mask in digit space.

    Args:
        model: Torch model f(x).
        mask: Eyeglass mask tensor, shape (3, H, W), values in [0, 1] selecting pixels to edit.
        device: Torch device string or torch.device.
        alpha: Step size (scaled by max(abs(grad over mask)) per-iteration).
        momentum: Momentum factor in [0,1].
        bgr_mean: VGG-Face BGR mean used by the pipeline (broadcasted later).
    """
    def __init__(
        self,
        model: torch.nn.Module,
        mask: torch.Tensor,
        device: torch.device,
        alpha: float = 20.0,
        momentum: float = 0.4,
        bgr_mean: Tuple[float, float, float] = (129.1863, 104.7624, 93.5940),
    ):
        self.model = model
        self.mask = mask.to(device)  # (3, H, W) in [0,1]
        self.device = device
        self.alpha = float(alpha)
        self.momentum = float(momentum)

        mean = torch.tensor(bgr_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = mean.to(device)

        self._loss_ce = nn.CrossEntropyLoss(reduction="none")
        self._loss_ce_red = nn.CrossEntropyLoss(reduction="mean")

        # Predefined color candidates (per the original approach)
        self._color_c0 = [128, 220, 160, 200, 220]
        self._color_c1 = [128, 130, 105, 175, 210]
        self._color_c2 = [128,   0,  55,  30,  50]

    @torch.no_grad()
    def _choose_color(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Selects an initial colorization on the mask that maximizes loss(f(X+delta) , y)
        over a small set of RGB candidates (ordered BGR in this pipeline).

        Args:
            X: Base image-like tensor already containing background digits (BGR-mean subtracted), shape (N,3,H,W).
            y: Ground-truth labels, shape (N,).

        Returns:
            delta_color: Tensor of shape (N,3,H,W) with the chosen colorization applied only on the mask.
        """
        self.model.eval()

        max_loss = torch.zeros(y.shape[0], device=self.device)
        max_delta = torch.zeros_like(X)

        # Apply candidate colors over the masked region; evaluate CE loss; keep the best per-sample
        for i in range(len(self._color_c0)):
            delta1 = torch.zeros_like(X)
            # Note: the pipeline uses BGR ordering for historical reasons (VGG-Face preprocessing).
            delta1[:, 0, :, :] = self.mask[0, :, :] * self._color_c2[i]  # B
            delta1[:, 1, :, :] = self.mask[1, :, :] * self._color_c1[i]  # G
            delta1[:, 2, :, :] = self.mask[2, :, :] * self._color_c0[i]  # R

            logits = self.model(X + delta1 - self.mean)
            all_loss = self._loss_ce(logits, y)  # (N,)

            take = all_loss >= max_loss
            if take.any():
                max_delta[take] = delta1.detach()[take]
                max_loss = torch.maximum(max_loss, all_loss)

        return max_delta

    def __call__(self, images_bgr_submean: torch.Tensor, labels: torch.Tensor, num_iter: int) -> torch.Tensor:
        """
        Run the eyeglass attack.

        Args:
            images_bgr_submean: Tensor (N,3,H,W), BGR-ordered and already shifted by -mean as per pipeline:
                                X_bgr_submean = (X_bgr + mean) * (1 - mask)  then we add colored mask.
            labels: Tensor (N,)
            num_iter: Number of gradient steps.

        Returns:
            X_adv: Adversarial tensor (N,3,H,W), same space as input (BGR-mean subtracted).
        """
        self.model.eval()

        # Choose an initial color on the mask that increases CE
        with torch.no_grad():
            color_glass = self._choose_color(images_bgr_submean, labels)

        # Initialize X1 = base + color - mean, and momentum delta
        X1 = images_bgr_submean.clone().detach().requires_grad_(True)
        X1.data = X1.data + color_glass - self.mean

        delta = torch.zeros_like(X1)

        for _ in range(int(num_iter)):
            # Compute gradient of CE wrt X1
            loss = self._loss_ce_red(self.model(X1), labels)
            loss.backward()

            # Only update on masked region; normalize by per-sample max abs grad (avoid div by 0)
            grad = (X1.grad.detach() * self.mask)
            flat = grad.view(grad.shape[0], -1).abs()
            max_val, _ = flat.max(dim=1)
            # Avoid divide-by-zero: if max_val == 0, skip update for that sample
            scale = torch.where(max_val > 0, max_val, torch.ones_like(max_val))
            r = self.alpha * grad / scale.view(-1, 1, 1, 1)

            # Momentum update
            delta = self.momentum * delta.detach() + r

            # Keep pixel range valid in digit space: (X1 + mean) must be in [0, 255]
            over = (delta.detach() + X1.detach() + self.mean) > 255
            under = (delta.detach() + X1.detach() + self.mean) < 0
            delta = delta.detach()
            delta[over] = 0
            delta[under] = 0

            # Apply update
            X1.data = X1.detach() + delta

            # Round to pixel grid then subtract mean again (digit space rounding)
            X1.data = torch.round(X1.detach() + self.mean) - self.mean

            # Clear grad for next step
            X1.grad.zero_()

        return X1.detach()


# ----------------------------
# Data & Evaluation helpers
# ----------------------------

def rgb_to_bgr(images: torch.Tensor) -> torch.Tensor:
    """Swap channels RGB -> BGR."""
    return images[:, [2, 1, 0], :, :]


def update_confusion_matrix(
    cm: np.ndarray, labels: torch.Tensor, preds: torch.Tensor
) -> None:
    """
    In-place update of a confusion matrix.

    Args:
        cm: (K,K) int matrix
        labels: (N,) tensor
        preds: (N,) tensor
    """
    y_true = labels.detach().cpu().numpy().astype(int)
    y_pred = preds.detach().cpu().numpy().astype(int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1


def compute_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """
    Per-class accuracy from confusion matrix (diagonal / row sum).

    Args:
        cm: (K,K) int matrix

    Returns:
        per_class_acc: (K,) float array in [0,1], NaN when class has 0 support.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1)
        diag = np.diag(cm)
        acc = diag / row_sum
        acc[row_sum == 0] = np.nan
        return acc


# ----------------------------
# Main Evaluation Loop
# ----------------------------

def evaluate_attack(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask: torch.Tensor,
    iters_list: Iterable[int],
    alpha: float,
    restarts: int,
    num_classes: int,
    save_images: bool,
    save_dir: Path,
    log_images_every: int = 0,
) -> None:
    """
    Runs the eyeglass attack across iteration counts and evaluates accuracy, confusion, and per-class accuracy.

    Args:
        model: Classifier.
        dataloader: Test dataloader.
        device: Device.
        mask: (3,H,W) tensor in [0,1].
        iters_list: Iterable of iteration counts to try (e.g., [1, 10, 50, 100, 300]).
        alpha: Step size.
        restarts: Number of attack restarts (re-run attack; mark sample robust if correctly classified in all restarts).
        num_classes: Number of classes for reporting.
        save_images: If True, save original and adversarial images.
        save_dir: Base directory to store images.
        log_images_every: If >0, only save every N-th image to reduce I/O.
    """
    attacker = EyeglassAttack(model=model, mask=mask, device=device, alpha=alpha)

    for n_iter in iters_list:
        logging.info(f"=== Attack iterations: {n_iter} | alpha: {alpha} | restarts: {restarts} ===")

        total = 0
        robust_correct = 0  # count of samples still correct after 'restarts' attempts
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

        sample_index = 0  # for naming saved images

        for batch in dataloader:
            images_rgb, labels = batch
            images_rgb = images_rgb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Convert to BGR as per the original pipeline, and compute base X1 input:
            # X_bgr_submean = (X_bgr + mean) * (1 - mask)
            images_bgr = rgb_to_bgr(images_rgb)
            # Broadcast mask and mean inside the attack object

            # Bookkeeping per-sample correctness across restarts
            # We mark a sample robustly correct if it is correct in ALL restarts under the attack
            per_sample_correct_across_restarts = torch.ones(labels.size(0), dtype=torch.bool, device=device)
            final_preds_for_cm = None  # last restart's predictions (for CM/logging)

            for r in range(restarts):
                # Build base X1
                base = (images_bgr + attacker.mean) * (1 - attacker.mask)  # (N,3,H,W)
                # Run attack for n_iter steps
                X_adv = attacker(base, labels, num_iter=n_iter)
                with torch.no_grad():
                    logits = model(X_adv)
                    preds = torch.argmax(logits, dim=1)
                    correct = (preds == labels)
                    per_sample_correct_across_restarts &= correct
                    final_preds_for_cm = preds  # keep last for CM

                # Optionally save images
                if save_images:
                    for i in range(images_rgb.size(0)):
                        # Skip saving unless we want all or a specific cadence
                        if (log_images_every > 0) and (sample_index % log_images_every != 0):
                            sample_index += 1
                            continue

                        cls = int(labels[i].item())
                        subdir = "correct" if preds[i] == labels[i] else "misclassified"
                        out_dir = save_dir / f"iter_{n_iter}" / subdir / f"class_{cls}"
                        save_image_tensor(images_bgr[i], f"original_{sample_index}.png", out_dir)
                        save_image_tensor(X_adv[i],     f"adversarial_{sample_index}.png", out_dir)
                        sample_index += 1

            # Update robust accuracy (correct in all restarts)
            robust_correct += per_sample_correct_across_restarts.sum().item()
            total += labels.size(0)

            # Update confusion matrix from (last) restart predictions
            if final_preds_for_cm is not None:
                update_confusion_matrix(confusion, labels, final_preds_for_cm)

        robust_acc = 100.0 * robust_correct / max(1, total)
        per_class_acc = compute_class_accuracy(confusion)  # in [0,1] or nan

        logging.info(f"[RESULT] alpha={alpha} iters={n_iter} restarts={restarts} | Robust Accuracy: {robust_acc:.4f}%")
        logging.info("Per-class accuracy (diagonal / row-sum):")
        for c in range(num_classes):
            val = per_class_acc[c]
            if np.isnan(val):
                logging.info(f"  Class {c}: NaN (no samples)")
            else:
                logging.info(f"  Class {c}: {100.0 * val:.2f}%")

        logging.debug(f"Confusion matrix (rows=true, cols=pred):\n{confusion}")


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eyeglass frame attack evaluation (untargeted, digit-space, fixed mask).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths / Model
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pt / .pth).")
    parser.add_argument("--glass-mask-path", type=Path, required=True,
                        help="Path to eyeglass mask image (e.g., silhouette.png).")
    parser.add_argument("--save-dir", type=Path, default=Path("./attack_outputs"),
                        help="Directory to save adversarial/original images if --save-images is set.")

    # Attack / Eval
    parser.add_argument("--alpha", type=float, default=20.0,
                        help="Attack step size (scaled by per-sample max |grad| over mask).")
    parser.add_argument("--iters", type=int, nargs="+", default=[1, 10, 50, 100, 300],
                        help="Iteration counts to evaluate.")
    parser.add_argument("--restarts", type=int, default=1,
                        help="Number of attack restarts (AND across restarts for robust correctness).")
    parser.add_argument("--num-classes", type=int, default=8,
                        help="Number of classes (for confusion/per-class reporting).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for the test loader.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device, e.g., 'cuda:0' or 'cpu'.")

    # Saving / Logging
    parser.add_argument("--save-images", action="store_true",
                        help="If set, save original and adversarial images.")
    parser.add_argument("--log-images-every", type=int, default=0,
                        help="Save every N-th image only (0 = save all).")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help="Logging verbosity.")
    parser.add_argument("--log-file", type=Path, default=Path(""),
                        help="Optional log file path.")

    return parser.parse_args(argv)


def load_mask_as_tensor(mask_path: Path) -> torch.Tensor:
    """
    Load an eyeglass mask image into a float tensor in [0,1], shape (3,H,W).
    Non-zero pixels are considered as the mask region (already continuous in original code).
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise RuntimeError(f"Failed to read mask image: {mask_path}")
    # Convert to RGB for ToTensor semantics, then ToTensor -> (C,H,W) in [0,1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_tensor = transforms.ToTensor()(img_rgb)  # (3,H,W), float in [0,1]
    return mask_tensor


def main(argv: List[str] = None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level, log_file=str(args.log_file) if args.log_file else "")

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    if device.type == "cpu" and "cuda" in args.device:
        logging.warning("CUDA requested but not available. Falling back to CPU.")

    # Load mask
    mask = load_mask_as_tensor(args.glass_mask_path)

    # Load model
    model = VGG_16()
    if not args.model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    state = torch.load(str(args.model_checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Data
    # Expect origin_train.data_process(batch_size) -> (dataloaders, dataset_sizes, class_names)
    dataloaders, dataset_sizes, _ = data_process(args.batch_size)
    if "test" not in dataloaders:
        raise KeyError("Expected a 'test' dataloader from origin_train.data_process().")

    # Run attack + evaluation
    evaluate_attack(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        mask=mask,
        iters_list=args.iters,
        alpha=args.alpha,
        restarts=args.restarts,
        num_classes=args.num_classes,
        save_images=bool(args.save_images),
        save_dir=args.save_dir,
        log_images_every=int(args.log_images_every),
    )


if __name__ == "__main__":
    main()
