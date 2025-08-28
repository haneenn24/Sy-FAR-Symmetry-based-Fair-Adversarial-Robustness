#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate robustness with AutoAttack on CIFAR-10.

Notes
-----
- Normalization is handled via a lightweight `Normalize` module placed in front of the model.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

# Local project utils
from utils import in_class  # must exist in your repo

# Optional backbones
from preact_resnet import PreActResNet18
from wideresnet import WideResNet as WideResNet_MART  # your local WRN impl (for MART)
# For robustbench WRN variants, we'll import lazily if selected


# ----------------------------
# Constants
# ----------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


# ----------------------------
# Utilities
# ----------------------------

def setup_logging(level: str = "INFO") -> None:
    level_map = {
        "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR,
        "WARNING": logging.WARNING, "INFO": logging.INFO, "DEBUG": logging.DEBUG
    }
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_test_loader(data_dir: Path, batch_size: int, num_workers: int = 2) -> torch.utils.data.DataLoader:
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    return torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )


class Normalize(nn.Module):
    """Channel-wise input normalization as a layer."""
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def resolve_norm(norm_key: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if norm_key == "std":
        return CIFAR10_MEAN, CIFAR10_STD
    if norm_key == "01":
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    if norm_key == "+-1":
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    raise ValueError(f"Unknown normalization key: {norm_key}")


def load_checkpoint_state(model_path: Path) -> Dict[str, Any]:
    """Loads a checkpoint (state_dict or full checkpoint) from disk onto CPU."""
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    ckpt = torch.load(str(model_path), map_location="cpu")
    # Many training scripts save {'state_dict': ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(k.startswith(("module.", "layer", "conv", "fc", "bn")) or True for k in ckpt.keys()):
        return ckpt  # looks like a raw state_dict
    # Fallback: if top-level is not a dict state, try attribute
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    raise RuntimeError("Unsupported checkpoint format (expected state_dict or dict with 'state_dict').")


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'module.' prefix from DataParallel checkpoints."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd


def build_model(
    arch: str,
    pre_trained: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    device: torch.device
) -> nn.Module:
    """
    Builds the requested model architecture and wraps it with Normalize(mean,std).
    - arch: 'WRN' or 'PRN'
    - pre_trained: one of ['MART', 'AWP', 'TRADES', 'PGD'] (for WRN varieties).
    """
    if arch == "WRN":
        if pre_trained == "MART":
            net = WideResNet_MART().to(device)
            net = torch.nn.DataParallel(net).to(device)
        elif pre_trained in ("AWP", "TRADES", "PGD"):
            # robustbench WRN variants
            from robustbench.model_zoo.architectures.wide_resnet import WideResNet as WideResNet_RB
            if pre_trained == "TRADES":
                net = WideResNet_RB(depth=34, widen_factor=10, sub_block1=True).to(device)
            else:
                net = WideResNet_RB(depth=34, widen_factor=10).to(device)
            net = torch.nn.DataParallel(net).to(device)
        else:
            raise ValueError(f"Unknown pre_trained variant for WRN: {pre_trained}")
    elif arch == "PRN":
        net = PreActResNet18().to(device)
    else:
        raise ValueError(f"Unknown model architecture: {arch}")

    model = nn.Sequential(Normalize(mean=mean, std=std), net).to(device)
    return model


def resolve_model_path(out_dir: Path, model_name: str, model_tag: str) -> Path:
    """
    Resolves checkpoint path based on flags.
    - If model_name in {'best','last','both','worst'} -> uses '{tag}_{suffix}.pth' under out_dir.
    - Else model_name is treated as a file path (relative to out_dir).
    """
    if model_name == "best":
        return out_dir / f"{model_tag}_best.pth"
    if model_name == "last":
        return out_dir / f"{model_tag}_last.pth"
    if model_name == "both":
        return out_dir / f"{model_tag}_both_best.pth"
    if model_name == "worst":
        return out_dir / f"{model_tag}_worst_best.pth"
    # explicit path (relative or absolute)
    p = Path(model_name)
    return p if p.is_absolute() else (out_dir / p)


def get_autoattack(log_path: Path):
    """
    Import AutoAttack from the PyPI package safely, even if this file is named 'autoattack.py'.
    """
    try:
        # Avoid from autoattack import AutoAttack (which would import THIS file if misnamed)
        import importlib
        aa_pkg = importlib.import_module("autoattack")
        AutoAttack = getattr(aa_pkg, "AutoAttack")
        return AutoAttack
    except Exception as e:
        raise ImportError(
            "Could not import 'AutoAttack' from the 'autoattack' package. "
            "Ensure the PyPI package is installed and this script is not named 'autoattack.py'."
        ) from e


def evaluate_autoattack(
    test_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    batch_size: int,
    eps: int,
    log: Path | None
) -> torch.Tensor:
    """
    Runs AutoAttack (standard Linf suite) and returns per-sample predicted labels on adversarial inputs,
    then computes robust accuracy via `utils.in_class`.
    """
    epsilon = eps / 255.0
    AutoAttack = get_autoattack(log)

    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=epsilon,
        verbose=False,
        log_path=str(log) if log else None,
        version="standard",
    )

    model.eval()
    all_pred_adv = []
    all_label = []

    for X, y in test_loader:
        X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        x_adv, y_adv = adversary.run_standard_evaluation(X, y, bs=batch_size, return_labels=True)
        all_pred_adv.append(y_adv)
        all_label.append(y)

    all_pred_adv = torch.cat(all_pred_adv).flatten()
    all_label = torch.cat(all_label).flatten()

    acc_adv = in_class(all_pred_adv, all_label)  # your util: expect tensor/array
    return acc_adv


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10 model robustness with AutoAttack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for AA eval.")
    parser.add_argument("--normalization", type=str, default="01", choices=["std", "01", "+-1"],
                        help="Input normalization scheme.")
    parser.add_argument("--data-dir", type=Path, default=Path("./cifar-data"), help="CIFAR-10 data dir.")
    parser.add_argument("--out-dir", type=Path, default=Path("mdeat_out"), help="Output dir / checkpoints dir.")
    parser.add_argument("--model-name", type=str, default="model_pre",
                        help="Checkpoint selector: ['best','last','both','worst'] or a file path.")
    parser.add_argument("--epsilon", type=int, default=8, help="Linf epsilon (in pixel units).")
    parser.add_argument("--log-name", type=str, default="aa_score", help="Log filename stem (under out-dir).")
    parser.add_argument("--model", type=str, default="PRN", choices=["WRN", "PRN"], help="Backbone architecture.")
    parser.add_argument("--pre-trained", type=str, default="MART", choices=["MART", "AWP", "TRADES", "PGD"],
                        help="Pretrained variant for WRN.")
    parser.add_argument("--gpuid", type=int, default=0, help="GPU id to use if CUDA available.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    # Warn about filename collision
    this_file = Path(__file__).name.lower()
    if this_file == "autoattack.py":
        logging.warning(
            "This script is named 'autoattack.py'. It may shadow the PyPI package 'autoattack'. "
            "Consider renaming it to 'eval_autoattack.py'."
        )

    device = get_device(args.gpuid)

    mean, std = resolve_norm(args.normalization)

    # Data
    test_loader = get_test_loader(args.data_dir, args.batch_size)

    # Model path resolution
    model_path = resolve_model_path(args.out_dir, args.model_name, args.model)
    log_path = args.out_dir / f"{args.log_name}.log"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build model & load weights
    model = build_model(args.model, args.pre_trained, mean, std, device)

    state = load_checkpoint_state(model_path)
    state = strip_module_prefix(state)
    # Load into net (second module in Sequential)
    net = model[1]
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.warning(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    model.float().eval()

    logging.info(f"Evaluating checkpoint: {model_path}")
    acc_adv = evaluate_autoattack(test_loader, model, args.batch_size, args.epsilon, log_path)

    # acc_adv could be a tensor/array. Print summary similarly to your original script.
    try:
        acc_tensor = torch.as_tensor(acc_adv)
        print(acc_tensor)             # full per-class / vector (if provided by in_class)
        print(acc_tensor.mean().item())
        print(acc_tensor.min().item())
    except Exception:
        # Fallback if in_class returns a scalar
        print(acc_adv)


if __name__ == "__main__":
    main()
