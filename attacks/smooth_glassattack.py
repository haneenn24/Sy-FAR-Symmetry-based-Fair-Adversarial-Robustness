#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomized Smoothing against Eyeglass Attack

Based on: https://github.com/locuslab/smoothing
Original authors: Jeremy Cohen, Elan Rosenfeld, Zico Kolter

This script evaluates randomized smoothing certification under the eyeglass-frame attack.

Usage
-----
python smooth_glassattack.py \
  --model-checkpoint ./checkpoints/model.pt \
  --glass-mask-path ./glass/Experiment/dataprepare/silhouette.png \
  --sigma 1.0 \
  --outfile output.tsv \
  --batch 32 \
  --N 1000 \
  --alpha 0.001

Notes
-----
- Model should be trained with Gaussian noise of variance sigma² (via gaussian_train.py).
- Attack is untargeted, digit-space, fixed mask.
- Output is a tab-separated log file with predictions, correctness, and timing.
"""

import argparse
import datetime
import logging
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
from torchvision import transforms

# Local project imports
from core import Smooth
from utils import data_process
from models.vgg16 import VGG_16
from glass_attack import glass_attack


# ----------------------------
# Utilities
# ----------------------------

def setup_logging(level: str = "INFO") -> None:
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_mask_as_tensor(mask_path: Path) -> torch.Tensor:
    """
    Load an eyeglass mask image into a float tensor in [0,1], shape (3,H,W).
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read mask image: {mask_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(img_rgb)


# ----------------------------
# Main
# ----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Randomized smoothing certification under eyeglass attack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="Path to Gaussian-trained model checkpoint.")
    parser.add_argument("--glass-mask-path", type=Path, required=True,
                        help="Path to eyeglass mask image (e.g., silhouette.png).")
    parser.add_argument("--outfile", type=Path, required=True,
                        help="Path to write tab-separated results.")

    # Randomized smoothing params
    parser.add_argument("--sigma", type=float, required=True,
                        help="Noise sigma used in training.")
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of Monte Carlo samples for smoothing.")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Failure probability for smoothing certification.")

    # Data / Attack params
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size for randomized smoothing prediction.")
    parser.add_argument("--attack-iters", type=int, nargs="+",
                        default=[1, 2, 3, 5, 7, 10, 20, 50, 100, 300],
                        help="List of iteration counts for the eyeglass attack.")
    parser.add_argument("--attack-alpha", type=float, default=20.0,
                        help="Attack step size.")
    parser.add_argument("--attack-momentum", type=float, default=0.4,
                        help="Attack momentum.")

    args = parser.parse_args(argv)
    setup_logging()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    dataloaders, dataset_sizes, _ = data_process(batch_size=1)
    if "test" not in dataloaders:
        raise KeyError("Expected a 'test' dataloader from data_process().")

    # Load mask
    mask = load_mask_as_tensor(args.glass_mask_path).to(device)

    # Load model
    model = VGG_16()
    if not args.model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    state = torch.load(str(args.model_checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Wrap with smoothing
    smoothed_classifier = Smooth(model, num_classes=10, sigma=args.sigma)

    # Prepare output file
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w") as f:
        print("iters\tlabel\tprediction\tcorrect\ttime", file=f, flush=True)

        for n_iter in args.attack_iters:
            logging.info(f"Running eyeglass attack with {n_iter} iterations")
            cor, tot = 0, 0

            for x, labels in dataloaders["test"]:
                x = x[:, [2, 1, 0], :, :].to(device)  # RGB -> BGR
                labels = labels.to(device)

                before_time = time()
                x_adv = glass_attack(model, x, labels, mask,
                                     alpha=args.attack_alpha,
                                     num_iter=n_iter,
                                     momentum=args.attack_momentum)
                prediction = smoothed_classifier.predict(x_adv, args.N, args.alpha, args.batch)
                after_time = time()

                is_correct = int(prediction == int(labels.item()))
                cor += is_correct
                tot += 1
                time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

                print(f"{n_iter}\t{labels.item()}\t{prediction}\t{is_correct}\t{time_elapsed}",
                      file=f, flush=True)

            logging.info(f"[{n_iter} iters] Accuracy: {cor}/{tot} = {100.0 * cor / max(1, tot):.2f}%")


if __name__ == "__main__":
    main()
