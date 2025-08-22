# utils/specnorm.py
# -*- coding: utf-8 -*-
"""
Spectral norm utilities.
Compute the top singular vectors and a normalized weight matrix (gm) from
a confusion-like matrix.
"""

import torch


def compute_spectral_weights(confusion_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute spectral weights from a confusion-like matrix using SVD.

    Args:
        confusion_matrix (torch.Tensor): Square (C x C) matrix, e.g., misclassification counts.
        eps (float): Small value to avoid division by zero in normalization.

    Returns:
        gm (torch.Tensor): Normalized outer product of top singular vectors, shape (C, C).
                           Values scaled to [0.01, 2.01].
    """
    if confusion_matrix.dim() != 2 or confusion_matrix.size(0) != confusion_matrix.size(1):
        raise ValueError("confusion_matrix must be square")

    # Compute SVD
    u, s, v = torch.svd(confusion_matrix)

    # Outer product of first singular vectors
    gm = torch.outer(u[:, 0], v[:, 0])

    # Normalize to [0.01, 2.01]
    min_val, max_val = gm.min(), gm.max()
    gm = 2.0 * (gm - min_val) / (max_val - min_val + eps) + 0.01

    return gm
