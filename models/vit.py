# models/vit.py
# -*- coding: utf-8 -*-

"""
Vision Transformer (ViT) wrapper built on top of TIMM.

Defaults to: vit_base_patch16_224 (pretrained)
- Configure num_classes, in_chans, drop rates, drop path
- Choose training mode: 'end2end' or 'head_only' (freeze backbone, train classifier head)
- Re-initialize classification head with multiple schemes
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

try:
    import timm
except Exception as e:
    raise ImportError("This module requires `timm`. Install with `pip install timm`.") from e


InitMode = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "normal",
    "uniform",
]

TrainMode = Literal["end2end", "head_only"]


def _init_linear(module: nn.Linear, mode: InitMode, bias_init: float = 0.0, seed: Optional[int] = None) -> None:
    if seed is not None:
        torch.manual_seed(int(seed))

    if mode == "xavier_uniform":
        nn.init.xavier_uniform_(module.weight)
    elif mode == "xavier_normal":
        nn.init.xavier_normal_(module.weight)
    elif mode == "kaiming_uniform":
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    elif mode == "kaiming_normal":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif mode == "orthogonal":
        nn.init.orthogonal_(module.weight)
    elif mode == "normal":
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif mode == "uniform":
        nn.init.uniform_(module.weight, a=-0.01, b=0.01)
    else:
        raise ValueError(f"Unsupported init mode: {mode}")

    if module.bias is not None:
        nn.init.constant_(module.bias, bias_init)


class ViT(nn.Module):
    """
    Thin wrapper around timm Vision Transformers.

    Attributes
    ----------
    backbone : nn.Module
        The TIMM model instance.
    head : nn.Module
        The classifier head (reference to backbone.head or classifier attribute depending on arch).
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1000,          # will be overridden by training script as needed
        in_chans: int = 3,
        img_size: int | tuple[int, int] = 224,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        train_mode: TrainMode = "end2end",
        reinit_head: bool = False,
        head_init_mode: InitMode = "xavier_uniform",
        head_bias_init: float = 0.0,
        seed: Optional[int] = None,
        global_pool: str | None = None,   # e.g., "token", "avg", None -> use timm default
    ) -> None:
        super().__init__()

        # Build TIMM model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            img_size=img_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            global_pool=global_pool,
        )

        # Locate head for consistent control across ViT variants
        # Common: head (Linear), some have classifier
        head = None
        if hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Module):
            head = self.backbone.head
        elif hasattr(self.backbone, "classifier") and isinstance(self.backbone.classifier, nn.Module):
            head = self.backbone.classifier
        else:
            # Fallback: try to find a single linear at top
            for name, module in reversed(list(self.backbone.named_modules())):
                if isinstance(module, nn.Linear):
                    head = module
                    break

        if head is None or not isinstance(head, nn.Module):
            raise RuntimeError("Could not locate classifier head in the TIMM model.")

        self.head = head  # keep a reference for re-init and freezing logic

        # Optionally re-init head
        if reinit_head:
            self.reinit_head(mode=head_init_mode, bias_init=head_bias_init, seed=seed)

        # Apply training mode
        self.apply_train_mode(train_mode)

    # ---------------- Controls ----------------

    def apply_train_mode(self, mode: TrainMode) -> None:
        """
        'head_only': freeze all parameters except the classification head.
        'end2end'  : unfreeze everything.
        """
        if mode not in ("end2end", "head_only"):
            raise ValueError(f"Unsupported train_mode: {mode}")

        if mode == "head_only":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def reinit_head(self, mode: InitMode = "xavier_uniform", bias_init: float = 0.0, seed: Optional[int] = None) -> None:
        """
        Reinitialize the classifier head (Linear) with a chosen scheme.
        """
        if isinstance(self.head, nn.Linear):
            _init_linear(self.head, mode=mode, bias_init=bias_init, seed=seed)
        else:
            # Some TIMM heads are composite; try to re-init the last Linear inside.
            last_linear = None
            for m in reversed(list(self.head.modules())):
                if isinstance(m, nn.Linear):
                    last_linear = m
                    break
            if last_linear is None:
                raise RuntimeError("Unable to find a Linear layer to re-initialize in the head.")
            _init_linear(last_linear, mode=mode, bias_init=bias_init, seed=seed)

    # ---------------- Forward ----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------- Minimal CLI for sanity checks ----------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="TIMM ViT wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--in-chans", type=int, default=3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    p.add_argument("--attn-drop-rate", type=float, default=0.0)
    p.add_argument("--train-mode", type=str, default="end2end", choices=["end2end", "head_only"])
    p.add_argument("--reinit-head", action="store_true")
    p.add_argument("--head-init", type=str, default="xavier_uniform",
                   choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal", "normal", "uniform"])
    p.add_argument("--head-bias", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--global-pool", type=str, default=None, choices=[None, "token", "avg"], nargs='?')
    return p


def _main():
    args = _build_argparser().parse_args()
    model = ViT(
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        img_size=args.img_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        attn_drop_rate=args.attn_drop_rate,
        train_mode=args.train_mode,
        reinit_head=args.reinit_head,
        head_init_mode=args.head_init,
        head_bias_init=args.head_bias,
        seed=args.seed,
        global_pool=args.global_pool,
    )
    x = torch.randn(2, 3, args.img_size, args.img_size)
    with torch.no_grad():
        y = model(x)
    print(f"Built {args.model_name}: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    # Print which params are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total")


if __name__ == "__main__":
    _main()
