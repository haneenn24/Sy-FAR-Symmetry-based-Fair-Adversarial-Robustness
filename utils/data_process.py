# utils/data_process.py
# -*- coding: utf-8 -*-

"""
Dataset utilities for PubFig-style directory layouts.

- Exposes `data_process(...)` to build train/val/test dataloaders
- Configurable transforms (image size, mean/std, augment, num_workers, etc.)
- No model imports; this module is model-agnostic
- Includes a small CLI to preview class names and dataset sizes

Expected directory layout:
    data_dir/
        train/
            class_a/
            class_b/
            ...
        val/
            class_a/
            class_b/
            ...
        test/
            class_a/
            class_b/
            ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Default stats (VGGFace-style; your previous code used these)
_DEFAULT_MEAN = (0.367035294117647, 0.41083294117647057, 0.5066129411764705)
_DEFAULT_STD = (1 / 255.0, 1 / 255.0, 1 / 255.0)


def build_transforms(
    image_size: int | Tuple[int, int] = 224,
    mean: Tuple[float, float, float] = _DEFAULT_MEAN,
    std: Tuple[float, float, float] = _DEFAULT_STD,
    use_train_aug: bool = False,
) -> Dict[str, transforms.Compose]:
    """
    Construct torchvision transforms for train/val/test.

    Args:
        image_size: int or (H, W)
        mean, std: normalization stats (RGB order)
        use_train_aug: if True, apply light augmentation to train

    Returns:
        Dict of {'train','val','test'} -> transforms.Compose
    """
    size = (image_size, image_size) if isinstance(image_size, int) else image_size

    common = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if use_train_aug:
        train_tf = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose(common)

    eval_tf = transforms.Compose(common)

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def data_process(
    batch_size: int = 64,
    data_dir: str | Path = "./data",
    image_size: int | Tuple[int, int] = 224,
    mean: Tuple[float, float, float] = _DEFAULT_MEAN,
    std: Tuple[float, float, float] = _DEFAULT_STD,
    num_workers: int = 8,
    pin_memory: bool = True,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    test_shuffle: bool = False,
    use_train_aug: bool = False,
    persistent_workers: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """
    Build dataloaders and dataset metadata.

    Args:
        batch_size: per-loader batch size
        data_dir: root directory with train/val/test subfolders
        image_size: int or (H, W)
        mean, std: normalization stats
        num_workers: DataLoader workers
        pin_memory: pin host memory (speeds GPU transfers)
        train_shuffle, val_shuffle, test_shuffle: per-split shuffling
        use_train_aug: enable light augmentation for train split
        persistent_workers: keep workers alive across epochs

    Returns:
        dataloaders: dict with keys 'train','val','test'
        dataset_sizes: dict split -> size
        class_names: list of class names (from train set)
    """
    data_dir = Path(data_dir)
    splits = ["train", "val", "test"]
    for s in splits:
        if not (data_dir / s).exists():
            raise FileNotFoundError(f"Missing split directory: {data_dir / s}")

    tfs = build_transforms(image_size=image_size, mean=mean, std=std, use_train_aug=use_train_aug)

    image_datasets = {
        split: datasets.ImageFolder(str(data_dir / split), transform=tfs[split])
        for split in splits
    }

    class_names = image_datasets["train"].classes
    dataset_sizes = {split: len(image_datasets[split]) for split in splits}

    loader_args = dict(
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    dataloaders: Dict[str, DataLoader] = {
        "train": DataLoader(image_datasets["train"], shuffle=train_shuffle, **loader_args),
        "val":   DataLoader(image_datasets["val"],   shuffle=val_shuffle,   **loader_args),
        "test":  DataLoader(image_datasets["test"],  shuffle=test_shuffle,  **loader_args),
    }

    print(f"[data_process] Classes ({len(class_names)}): {class_names}")
    print(f"[data_process] Sizes: {dataset_sizes}")

    return dataloaders, dataset_sizes, class_names


# ---------------- CLI for quick inspection ----------------

def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Build dataloaders for a PubFig-style dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root directory containing train/val/test subfolders")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", default=True)
    parser.add_argument("--train-shuffle", action="store_true", default=True)
    parser.add_argument("--no-train-shuffle", dest="train_shuffle", action="store_false")
    parser.add_argument("--val-shuffle", action="store_true", default=False)
    parser.add_argument("--test-shuffle", action="store_true", default=False)
    parser.add_argument("--use-train-aug", action="store_true", help="Enable light train augmentations")
    return parser


def _main():
    args = _build_argparser().parse_args()
    dataloaders, dataset_sizes, class_names = data_process(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        image_size=args.image_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_shuffle=args.train_shuffle,
        val_shuffle=args.val_shuffle,
        test_shuffle=args.test_shuffle,
        use_train_aug=args.use_train_aug,
    )
    # Simple sanity check: one batch
    dl = dataloaders["train"]
    xb, yb = next(iter(dl))
    print(f"[sanity] Train batch: images={tuple(xb.shape)} labels={tuple(yb.shape)}")


if __name__ == "__main__":
    _main()
