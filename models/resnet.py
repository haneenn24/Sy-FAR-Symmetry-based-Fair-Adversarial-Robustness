# models/resnet.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Type, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config & helpers
# ----------------------------

ActType = Literal["relu", "silu"]
NormType = Literal["batch", "group", "layer"]
InitMode = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "normal",
    "uniform",
]


def make_activation(kind: ActType) -> nn.Module:
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {kind}")


def make_norm(kind: NormType, num_features: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(num_features)
    if kind == "group":
        # 32 groups if divisible, else fall back to 1 (LayerNorm-like across channels)
        groups = 32 if num_features % 32 == 0 else max(1, num_features // 2)
        return nn.GroupNorm(groups, num_features)
    if kind == "layer":
        # LayerNorm over CxHxW -> use GroupNorm with 1 group to emulate channel-wise LN
        return nn.GroupNorm(1, num_features)
    raise ValueError(f"Unsupported norm: {kind}")


def init_linear(module: nn.Linear, mode: InitMode, bias_init: float = 0.0, seed: Optional[int] = None) -> None:
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


# ----------------------------
# Blocks
# ----------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm: NormType = "batch",
        act: ActType = "relu",
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = make_norm(norm, planes)
        self.act = make_activation(act)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = make_norm(norm, planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm: NormType = "batch",
        act: ActType = "relu",
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = make_norm(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = make_norm(norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = make_norm(norm, planes * self.expansion)
        self.act = make_activation(act)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


# ----------------------------
# ResNet
# ----------------------------

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 10,
        norm: NormType = "batch",
        act: ActType = "relu",
        imagenet_stem: bool = False,
        zero_init_residual: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args
        ----
        block: BasicBlock or Bottleneck
        layers: list with 4 stage depths, e.g. [2,2,2,2]
        num_classes: output classes
        norm: normalization type ('batch'|'group'|'layer')
        act: activation type ('relu'|'silu')
        imagenet_stem: if True, use 7x7/stride2 stem + 3x3 maxpool; else CIFAR stem (3x3 stride1)
        zero_init_residual: if True, zero-init the last BN in each residual branch (He et al. 2016)
        dropout: dropout before final FC
        """
        super().__init__()
        self.in_planes = 64
        self.norm_type = norm
        self.act_type = act

        if imagenet_stem:
            # 7x7 conv stem + maxpool
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = make_norm(norm, 64)
            self.act = make_activation(act)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            stem_stride = None  # handled via maxpool
        else:
            # CIFAR stem: keep resolution (32x32 -> 32x32)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = make_norm(norm, 64)
            self.act = make_activation(act)
            self.maxpool = None

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Init: Kaiming for conv, ones/zeros for norm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Zero-init residual branch last BN: improves optimization (per He et al.)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0.0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.0)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                make_norm(self.norm_type, planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, norm=self.norm_type, act=self.act_type, downsample=downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, norm=self.norm_type, act=self.act_type))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        if self.maxpool is not None:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    # -------- convenience: re-init last layer --------
    def reinit_classifier(self, mode: InitMode = "xavier_uniform", bias_init: float = 0.0, seed: Optional[int] = None) -> None:
        init_linear(self.fc, mode, bias_init=bias_init, seed=seed)


# ----------------------------
# Factories
# ----------------------------

def ResNet18(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def ResNet34(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ResNet50(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ResNet101(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def ResNet152(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


# ----------------------------
# Minimal CLI for quick checks
# ----------------------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="ResNet (CIFAR-friendly) with flexible options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--norm", type=str, default="batch", choices=["batch", "group", "layer"])
    p.add_argument("--act", type=str, default="relu", choices=["relu", "silu"])
    p.add_argument("--imagenet-stem", action="store_true", help="Use 7x7/stride2 stem + maxpool")
    p.add_argument("--zero-init-residual", action="store_true", default=True)
    p.add_argument("--no-zero-init-residual", dest="zero_init_residual", action="store_false")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--reinit-fc", action="store_true", help="Reinitialize classifier layer")
    p.add_argument("--fc-init", type=str, default="xavier_uniform",
                   choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal", "normal", "uniform"])
    p.add_argument("--fc-bias", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    return p


def _main():
    args = _build_argparser().parse_args()

    arch_map = {
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
    }
    model = arch_map[args.arch](
        num_classes=args.num_classes,
        norm=args.norm,
        act=args.act,
        imagenet_stem=args.imagenet_stem,
        zero_init_residual=args.zero_init_residual,
        dropout=args.dropout,
    )

    if args.reinit_fc:
        model.reinit_classifier(mode=args.fc_init, bias_init=args.fc_bias, seed=args.seed)
        print(f"Reinitialized classifier with {args.fc_init} (bias={args.fc_bias}, seed={args.seed})")

    # Quick forward pass test
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    print(f"Built {args.arch}: input {tuple(x.shape)} -> output {tuple(y.shape)}")


if __name__ == "__main__":
    _main()
