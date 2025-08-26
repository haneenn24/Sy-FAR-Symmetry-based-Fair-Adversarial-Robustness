#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PGD & Evaluation Utilities (CIFAR-10)

Example
-------
# Standard accuracy
python pgd.py --mode standard --data-dir ./cifar-data --batch-size 256

# PGD evaluation (Linf eps=8/255, 10 iters, step=2, 1 restart)
python pgd.py --mode pgd --data-dir ./cifar-data --attack-iters 10 --eps 8 --step 2

# Fair PGD evaluation (per-class metrics)
python pgd.py --mode pgd_fair --data-dir ./cifar-data --attack-iters 20 --eps 8 --step 2
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------
# Constants
# ----------------------------

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2471, 0.2435, 0.2616)

UPPER_LIMIT = 1.0
LOWER_LIMIT = 0.0


# ----------------------------
# Logging / Seed
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


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Init / helpers
# ----------------------------

def weights_init(m: nn.Module) -> None:
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.1)


def split_feature(tensor: torch.Tensor, type: str = "split") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    type in {"split", "cross"}
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
    raise ValueError("type must be 'split' or 'cross'")


def build_norm_tensors(
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = torch.tensor(mean, dtype=torch.float32, device=device).view(3, 1, 1)
    sd = torch.tensor(std, dtype=torch.float32, device=device).view(3, 1, 1)
    return mu, sd


def normalize(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sd


def unnormalize(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return x * sd + mu


def clamp(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, upper), lower)


def l2_norm_batch(v: torch.Tensor) -> torch.Tensor:
    return (v ** 2).sum(dim=[1, 2, 3]).sqrt()


def l2_norm_batch2(v: torch.Tensor) -> torch.Tensor:
    return (v ** 2).sum(dim=[1]).sqrt()


# ----------------------------
# CIFAR-10 loaders
# ----------------------------

def get_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()

    train_ds = datasets.CIFAR10(str(data_dir), train=True, transform=train_tf, download=True)
    test_ds = datasets.CIFAR10(str(data_dir), train=False, transform=test_tf, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader


# ----------------------------
# Augment (legacy helper)
# ----------------------------

def atta_aug(input_tensor: torch.Tensor, rst: torch.Tensor):
    """
    Random crop + optional horizontal flip (legacy helper).
    Returns (augmented, meta-info)
    """
    B = input_tensor.shape[0]
    x = torch.zeros(B, dtype=torch.long)
    y = torch.zeros(B, dtype=torch.long)
    flip = [False] * B

    for i in range(B):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 8)
        y_t = random.randint(0, 8)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], dims=[2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {"x": x, "y": y}, "flipped": flip}


# ----------------------------
# Losses
# ----------------------------

def CW_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Untargeted Carlini-Wagner margin loss on logits.
    """
    x_sorted, ind_sorted = logits.sort(dim=1)
    top_is_label = (ind_sorted[:, -1] == y).float()
    u = torch.arange(logits.shape[0], device=logits.device)
    loss_value = -(logits[u, y] - x_sorted[:, -2] * top_is_label - x_sorted[:, -1] * (1.0 - top_is_label))
    return loss_value.mean()


def dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sorted, ind_sorted = logits.sort(dim=1)
    u = torch.arange(logits.shape[0], device=logits.device)
    top_is_label = (ind_sorted[:, -1] == y).float()
    denom = (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    return (-(logits[u, y] - x_sorted[:, -2] * top_is_label - x_sorted[:, -1] * (1. - top_is_label)) / denom).mean()


def dlr_loss_targeted(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sorted, _ = logits.sort(dim=1)
    u = torch.arange(logits.shape[0], device=logits.device)

    loss_value = 0.0
    for target_rank in range(2, 11):
        y_target = logits.sort(dim=1)[1][:, -target_rank]
        denom = (x_sorted[:, -1] - 0.5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)
        loss_value += (-(logits[u, y] - logits[u, y_target]) / denom).mean()
    return loss_value / 9.0


# ----------------------------
# Attacks
# ----------------------------

@torch.no_grad()
def attack_trade(
    model: nn.Module,
    x_natural: torch.Tensor,
    epsilon: float,
    step_size: float,
    attack_iters: int,
    mu: torch.Tensor,
    sd: torch.Tensor,
) -> torch.Tensor:
    """
    TRADES-style attack (KL to clean prediction).
    Returns the perturbation (delta).
    """
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
    for _ in range(attack_iters):
        x_adv.requires_grad_(True)
        loss_kl = F.kl_div(
            F.log_softmax(model(normalize(x_adv, mu, sd)), dim=1),
            F.softmax(model(normalize(x_natural, mu, sd)), dim=1),
            reduction="sum",
        )
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv - x_natural


def attack_pgd(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    attack_iters: int,
    restarts: int,
    mu: torch.Tensor,
    sd: torch.Tensor,
    use_CWloss: bool = False,
) -> torch.Tensor:
    """
    PGD with early escape for already-misclassified samples (common variant).
    Returns the worst-case delta across restarts per sample.
    """
    device = X.device
    max_loss = torch.zeros(y.shape[0], device=device)
    max_delta = torch.zeros_like(X, device=device)

    for _ in range(restarts):
        delta = torch.empty_like(X, device=device).uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, LOWER_LIMIT - X, UPPER_LIMIT - X)
        delta.requires_grad_(True)

        for _ in range(attack_iters):
            logits = model(normalize(X + delta, mu, sd))
            idx = torch.where(logits.max(1)[1] == y)[0]
            if idx.numel() == 0:
                break
            loss = CW_loss(logits, y) if use_CWloss else F.cross_entropy(logits, y)
            loss.backward()
            grad = delta.grad.detach()

            d = delta[idx]
            g = grad[idx]
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, LOWER_LIMIT - X[idx], UPPER_LIMIT - X[idx])
            delta.data[idx] = d
            delta.grad.zero_()

        losses = F.cross_entropy(model(normalize(X + delta, mu, sd)), y, reduction="none").detach()
        take = losses >= max_loss
        max_delta[take] = delta.detach()[take]
        max_loss = torch.maximum(max_loss, losses)

    return max_delta


def pgd_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    attack_iters: int,
    restarts: int,
    mu: torch.Tensor,
    sd: torch.Tensor,
    use_CWloss: bool = False,
) -> torch.Tensor:
    """
    PGD variant that keeps updating all samples (no early exclusion).
    Returns the worst-case delta across restarts per sample.
    """
    device = X.device
    max_loss = torch.zeros(y.shape[0], device=device)
    max_delta = torch.zeros_like(X, device=device)

    for _ in range(restarts):
        delta = torch.empty_like(X, device=device).uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, LOWER_LIMIT - X, UPPER_LIMIT - X)
        delta.requires_grad_(True)

        for _ in range(attack_iters):
            logits = model(normalize(X + delta, mu, sd))
            loss = CW_loss(logits, y) if use_CWloss else F.cross_entropy(logits, y)
            loss.backward()
            grad = delta.grad.detach()

            delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, LOWER_LIMIT - X, UPPER_LIMIT - X)
            delta.grad.zero_()

        losses = F.cross_entropy(model(normalize(X + delta, mu, sd)), y, reduction="none").detach()
        take = losses >= max_loss
        max_delta[take] = delta.detach()[take]
        max_loss = torch.maximum(max_loss, losses)

    return max_delta


# ----------------------------
# Metrics / Evaluation
# ----------------------------

def in_class(predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Per-class accuracy: P(pred == label | label = c), for c=0..9 (CIFAR-10).
    """
    probs = torch.zeros(10, device=predict.device if predict.is_cuda else "cpu")
    for c in range(10):
        mask = (label == c)
        if mask.sum() > 0:
            acc = (predict[mask] == label[mask]).float().mean()
            probs[c] = acc
    return probs


@torch.no_grad()
def evaluate_standard(
    test_loader: DataLoader,
    model: nn.Module,
    mu: torch.Tensor,
    sd: torch.Tensor,
    val: int = None,
) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc, n = 0.0, 0.0, 0

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        logits = model(normalize(X, mu, sd))
        loss = F.cross_entropy(logits, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (logits.max(1)[1] == y).sum().item()
        n += y.size(0)
        if val and i == val - 1:
            break

    return test_loss / max(1, n), test_acc / max(1, n)


@torch.no_grad()
def evaluate_pgd(
    test_loader: DataLoader,
    model: nn.Module,
    mu: torch.Tensor,
    sd: torch.Tensor,
    attack_iters: int,
    restarts: int = 1,
    eps: int = 8,
    step: int = 2,
    val: int = None,
    use_CWloss: bool = False,
) -> Tuple[float, float, torch.Tensor]:
    """
    Standard PGD evaluation on CIFAR-10.
    Returns (avg loss, acc, per-class correct counts scaled as in legacy code).
    """
    epsilon = eps / 255.0
    alpha = epsilon if attack_iters == 1 else (step / 255.0)

    model.eval()
    pgd_loss, pgd_acc, n = 0.0, 0.0, 0
    results = torch.zeros(10)

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, mu, sd, use_CWloss=use_CWloss)
        logits = model(normalize(X + delta, mu, sd))
        loss = F.cross_entropy(logits, y)

        pgd_loss += loss.item() * y.size(0)
        pgd_acc += (logits.max(1)[1] == y).sum().item()

        for cc in range(10):
            yy = (y == cc)
            if yy.sum() > 0:
                # Legacy scaling to approximate fraction over 1000 samples
                results[cc] += ((logits[yy]).max(1)[1] == (y[yy])).sum().item() * 0.001

        n += y.size(0)
        if val and i == val - 1:
            break

    return pgd_loss / max(1, n), pgd_acc / max(1, n), results


def evaluate_pgd_fair(
    test_loader: DataLoader,
    model: nn.Module,
    mu: torch.Tensor,
    sd: torch.Tensor,
    attack_iters: int,
    restarts: int = 1,
    eps: int = 8,
    step: int = 2,
    val: int = None,
    use_CWloss: bool = False,
):
    """
    Fairness-oriented PGD evaluation.
    Returns:
      class_clean_error, class_bndy_error, total_clean_error, total_bndy_error, avg_loss, acc, results
    """
    epsilon = eps / 255.0
    alpha = epsilon if attack_iters == 1 else (step / 255.0)

    model.eval()
    pgd_loss, pgd_acc, n = 0.0, 0.0, 0
    correct, correct_adv = 0, 0

    all_label, all_pred, all_pred_adv = [], [], []

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        all_label.append(y)

        logits_clean = model(normalize(X, mu, sd))
        pred_clean = logits_clean.argmax(dim=1)
        correct += (pred_clean == y).sum().item()
        all_pred.append(pred_clean)

        delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, mu, sd, use_CWloss=use_CWloss)
        logits_adv = model(normalize(X + delta, mu, sd))
        loss = F.cross_entropy(logits_adv, y)

        pgd_loss += loss.item() * y.size(0)
        pgd_acc += (logits_adv.max(1)[1] == y).sum().item()
        n += y.size(0)

        pred_adv = logits_adv.argmax(dim=1)
        correct_adv += (pred_adv == y).sum().item()
        all_pred_adv.append(pred_adv)

        if val and i == val - 1:
            break

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc_clean = in_class(all_pred, all_label)      # per-class clean accuracy
    acc_adv = in_class(all_pred_adv, all_label)    # per-class robust accuracy

    total_clean_error = 1.0 - correct / max(1, len(test_loader.dataset))
    total_bndy_error = (correct - correct_adv) / max(1, len(test_loader.dataset))

    class_clean_error = 1.0 - acc_clean
    class_bndy_error = acc_clean - acc_adv

    return (
        class_clean_error,
        class_bndy_error,
        total_clean_error,
        total_bndy_error,
        pgd_loss / max(1, n),
        pgd_acc / max(1, n),
        acc_adv,  # report robust per-class accuracy instead of legacy "results"
    )


# ----------------------------
# Weight averaging
# ----------------------------

def weight_average(model: nn.Module, new_model: nn.Module, decay_rate: float, init: bool = False) -> nn.Module:
    model.eval()
    new_model.eval()
    sd = model.state_dict()
    nd = new_model.state_dict()
    if init:
        decay_rate = 0.0
    for k in sd:
        nd[k] = (sd[k] * decay_rate + nd[k] * (1.0 - decay_rate)).clone().detach()
    model.load_state_dict(nd)
    return model


# ----------------------------
# CLI
# ----------------------------

def build_dummy_model(num_classes: int = 10) -> nn.Module:
    """
    Placeholder simple model for quick CLI demos.
    Replace with your own architecture import when integrating (e.g., from models.*).
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 3, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes),
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PGD utilities and evaluations for CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="pgd", choices=["standard", "pgd", "pgd_fair"],
                        help="Which evaluation to run.")
    parser.add_argument("--data-dir", type=Path, default=Path("./cifar-data"),
                        help="CIFAR-10 data directory.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eps", type=int, default=8, help="Linf epsilon (in pixel units).")
    parser.add_argument("--step", type=int, default=2, help="PGD step size (in pixel units).")
    parser.add_argument("--attack-iters", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--use-cwloss", action="store_true", help="Use CW loss instead of CE for attack.")
    parser.add_argument("--val-batches", type=int, default=0,
                        help="If >0, limit evaluation to the first N batches.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data
    _, test_loader = get_loaders(args.data_dir, args.batch_size, num_workers=args.num_workers)

    # Model (replace with your real model)
    model = build_dummy_model().to(device)

    # Norm tensors on correct device
    mu, sd = build_norm_tensors(CIFAR10_MEAN, CIFAR10_STD, device=device)

    # Evaluate
    val = args.val_batches if args.val_batches > 0 else None

    if args.mode == "standard":
        loss, acc = evaluate_standard(test_loader, model, mu, sd, val=val)
        logging.info(f"Standard: loss={loss:.4f}  acc={acc:.4f}")
        print(f"{acc:.6f}")
    elif args.mode == "pgd":
        loss, acc, results = evaluate_pgd(
            test_loader, model, mu, sd,
            attack_iters=args.attack_iters, restarts=args.restarts,
            eps=args.eps, step=args.step, val=val, use_CWloss=args.use_cwloss
        )
        logging.info(f"PGD: loss={loss:.4f}  acc={acc:.4f}")
        logging.debug(f"Per-class (legacy scaled) results: {results.tolist()}")
        print(f"{acc:.6f}")
    else:  # pgd_fair
        class_clean_error, class_bndy_error, total_clean_error, total_bndy_error, loss, acc, acc_adv = evaluate_pgd_fair(
            test_loader, model, mu, sd,
            attack_iters=args.attack_iters, restarts=args.restarts,
            eps=args.eps, step=args.step, val=val, use_CWloss=args.use_cwloss
        )
        logging.info(f"PGD Fair: loss={loss:.4f}  acc={acc:.4f}  "
                     f"clean_err={total_clean_error:.4f}  bndy_err={total_bndy_error:.4f}")
        logging.debug(f"Per-class clean error: {class_clean_error.tolist()}")
        logging.debug(f"Per-class boundary error: {class_bndy_error.tolist()}")
        logging.debug(f"Per-class robust acc: {acc_adv.tolist()}")
        print(f"{acc:.6f}")


if __name__ == "__main__":
    main()
