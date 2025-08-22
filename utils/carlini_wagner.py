# utils/carlini_wagner.py
# -*- coding: utf-8 -*-

"""
Carlini & Wagner margin-style loss.

This implements the common logit-margin objective used in C&W-style attacks:

Untargeted (default):
    f(x, y) = ReLU( max_{i != y} z_i(x) - z_y(x) + kappa )

Targeted:
    f(x, t) = ReLU( max_{i != t} z_i(x) - z_t(x) + kappa )

where z(x) are logits and kappa >= 0 is the "confidence" margin.

Notes
-----
- Inputs are expected to be **logits**, not probabilities.
- `labels` are ground-truth for untargeted mode, or **target labels** for targeted mode.
"""

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn.functional as F


def _max_except_class(logits: torch.Tensor, cls: torch.Tensor, large_const: float = 1e6) -> torch.Tensor:
    """
    For each sample, returns max logit among classes != cls.
    Achieved by subtracting a large constant from the 'cls' position, then taking max.
    """
    if cls.dim() != 1:
        raise ValueError("`cls` must be a 1D tensor of class indices.")
    one_hot = F.one_hot(cls, num_classes=logits.shape[1]).to(dtype=logits.dtype)
    masked = logits - large_const * one_hot
    return masked.max(dim=1).values


def carlini_wagner_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    targeted: bool = False,
    kappa: float = 0.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
    large_const: float = 1e6,
) -> torch.Tensor:
    """
    Compute C&W margin loss from logits.

    Args:
        logits: Tensor of shape (N, C) – model outputs **before** softmax.
        labels: Tensor of shape (N,) – class indices.
                - Untargeted: true labels.
                - Targeted:   desired target labels.
        targeted: If True, compute targeted version.
        kappa: Nonnegative confidence margin (>= 0). Larger values enforce bigger gaps.
        reduction: 'none' | 'mean' | 'sum'.
        large_const: Large masking constant to exclude a specific class from max.

    Returns:
        Tensor: scalar if reduction != 'none', else per-sample losses (N,).
    """
    if kappa < 0:
        raise ValueError("kappa must be nonnegative.")
    if logits.dim() != 2:
        raise ValueError("logits must have shape (N, C).")
    if labels.dim() != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must be (N,) and match logits batch size.")

    # z_y (or z_t in targeted mode)
    y_onehot = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
    z_y = (logits * y_onehot).sum(dim=1)

    # max_{i != y} z_i  (or max_{i != t} z_i for targeted)
    z_max_not_y = _max_except_class(logits, labels, large_const=large_const)

    # Core margin: max_non_label - label
    # Untargeted:   push max_non_label > label  (increase loss when model is correct)
    # Targeted:     push label (target) > max_non_label (so same form works with labels=target)
    margin = z_max_not_y - z_y + float(kappa)

    # ReLU for hinge-like behavior
    loss = torch.clamp(margin, min=0.0)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


__all__ = ["carlini_wagner_loss"]
