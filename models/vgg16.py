# models/vgg16.py
# -*- coding: utf-8 -*-
"""
VGG-16 (VGG-Face backbone) with flexible training modes and last-layer initialization.

- Convolutional blocks are compatible with VGG-Face. You can load Lua Torch (.t7) weights.
- Fully-connected head: 512*7*7 -> 1024 -> 1024 -> num_classes
- Training modes:
    * 'end2end'     : train all layers
    * 'last_layer'  : freeze backbone (all conv + fc6/fc7), train only fc8
- Last-layer (fc8) initialization modes:
    * 'xavier_uniform', 'xavier_normal',
      'kaiming_uniform', 'kaiming_normal',
      'orthogonal', 'normal', 'uniform'
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchfile  # for loading VGG-Face .t7
except Exception:
    torchfile = None  # optional dependency


TrainMode = Literal["end2end", "last_layer"]
InitMode = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "normal",
    "uniform",
]


class VGG_16(nn.Module):
    """
    VGG-16 with VGG-Face-style convs and a compact classification head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        train_mode: TrainMode = "end2end",
        init_mode: InitMode = "xavier_uniform",
        bias_init: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            num_classes: output classes.
            train_mode: 'end2end' or 'last_layer' (freeze all but fc8).
            init_mode: initialization method for fc8.
            bias_init: bias constant for fc8.
            seed: optional RNG seed for deterministic fc8 init.
        """
        super().__init__()

        self.block_size = [2, 2, 3, 3, 3]
        # Conv blocks
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        # Classification head
        self.fc6 = nn.Linear(512 * 7 * 7, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, num_classes)

        # Store config for later (e.g., when loading weights)
        self._train_mode = train_mode
        self._init_mode = init_mode
        self._bias_init = bias_init
        self._seed = seed

        # Initialize the last layer now (head can be re-initialized later as well)
        self.init_last_layer(init_mode=self._init_mode, bias_init=self._bias_init, seed=self._seed)

        # Apply training mode (freeze/unfreeze)
        self.apply_train_mode(self._train_mode)

    # --------- Public controls ---------

    def apply_train_mode(self, mode: TrainMode) -> None:
        """
        Apply training mode by freezing/unfreezing parameters.
        - 'last_layer': freeze all except fc8
        - 'end2end'   : unfreeze all
        """
        if mode not in ("end2end", "last_layer"):
            raise ValueError(f"Unsupported train_mode: {mode}")

        if mode == "last_layer":
            for name, p in self.named_parameters():
                p.requires_grad = name.startswith("fc8")
        else:  # end2end
            for _, p in self.named_parameters():
                p.requires_grad = True

        self._train_mode = mode

    def init_last_layer(
        self,
        init_mode: InitMode = "xavier_uniform",
        bias_init: float = 0.01,
        seed: Optional[int] = None,
        mean: float = 0.0,
        std: float = 0.02,
        a: float = -0.01,
        b: float = 0.01,
        gain: float = 1.0,
        nonlinearity: str = "relu",
    ) -> None:
        """
        Re-initialize fc8 with a chosen scheme.

        Args:
            init_mode: one of
                ['xavier_uniform','xavier_normal','kaiming_uniform','kaiming_normal',
                 'orthogonal','normal','uniform']
            bias_init: constant bias value.
            seed: optional RNG seed for deterministic init.
            mean,std: for 'normal'
            a,b: for 'uniform' (range [a,b])
            gain: gain for xavier/orthogonal
            nonlinearity: for kaiming / orthogonal when relevant
        """
        if seed is not None:
            g = torch.Generator(device=self.fc8.weight.device)
            g.manual_seed(int(seed))
        else:
            g = None

        w = self.fc8.weight
        if init_mode == "xavier_uniform":
            nn.init.xavier_uniform_(w, gain=gain)
        elif init_mode == "xavier_normal":
            nn.init.xavier_normal_(w, gain=gain)
        elif init_mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, nonlinearity=nonlinearity)
        elif init_mode == "kaiming_normal":
            nn.init.kaiming_normal_(w, nonlinearity=nonlinearity)
        elif init_mode == "orthogonal":
            nn.init.orthogonal_(w, gain=gain)
        elif init_mode == "normal":
            if g is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(seed))
                    nn.init.normal_(w, mean=mean, std=std)
            else:
                nn.init.normal_(w, mean=mean, std=std)
        elif init_mode == "uniform":
            if g is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(seed))
                    nn.init.uniform_(w, a=a, b=b)
            else:
                nn.init.uniform_(w, a=a, b=b)
        else:
            raise ValueError(f"Unsupported init_mode: {init_mode}")

        nn.init.constant_(self.fc8.bias, bias_init)

        # Remember current settings
        self._init_mode = init_mode
        self._bias_init = bias_init
        self._seed = seed

    # --------- Weights loading ---------

    def load_vggface_convs(self, path: str) -> None:
        """
        Load pretrained convolution weights from Torch7 (.t7) VGG-Face file.
        Only conv layers are copied; fc6/fc7/fc8 stay as defined here.

        Args:
            path: path to VGG_FACE.t7
        """
        if torchfile is None:
            raise ImportError("torchfile is required to load .t7 weights. Install `torchfile`.")

        model = torchfile.load(path)
        counter = 1
        block = 1
        for layer in model.modules:
            if getattr(layer, "weight", None) is not None:
                if block <= 5:
                    self_layer = getattr(self, f"conv_{block}_{counter}")
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    # copy weights/bias
                    self_layer.weight.data.copy_(torch.tensor(layer.weight).view_as(self_layer.weight))
                    self_layer.bias.data.copy_(torch.tensor(layer.bias).view_as(self_layer.bias))
                else:
                    break  # stop after conv5_3

    # --------- Forward ---------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_1_1(x)); x = F.relu(self.conv_1_2(x)); x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x)); x = F.relu(self.conv_2_2(x)); x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x)); x = F.relu(self.conv_3_2(x)); x = F.relu(self.conv_3_3(x)); x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x)); x = F.relu(self.conv_4_2(x)); x = F.relu(self.conv_4_3(x)); x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x)); x = F.relu(self.conv_5_2(x)); x = F.relu(self.conv_5_3(x)); x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x)); x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x)); x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)


# ---------------- CLI (optional) ----------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="VGG-16 (VGG-Face backbone) with flexible training and init options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--train-mode", type=str, default="end2end", choices=["end2end", "last_layer"])
    p.add_argument("--init-mode", type=str, default="xavier_uniform",
                   choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal", "normal", "uniform"])
    p.add_argument("--bias-init", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--vggface-t7", type=str, default=None, help="Path to VGG_FACE.t7 (optional; loads conv weights).")
    p.add_argument("--reinit-last-layer", action="store_true",
                   help="Reinitialize fc8 after loading conv weights (uses init-mode).")
    return p


def _main():
    args = _build_argparser().parse_args()

    model = VGG_16(
        num_classes=args.num_classes,
        train_mode=args.train_mode,
        init_mode=args.init_mode,
        bias_init=args.bias_init,
        seed=args.seed,
    )

    if args.vggface_t7:
        model.load_vggface_convs(args.vggface_t7)
        print(f"Loaded VGG-Face conv weights from: {args.vggface_t7}")
        if args.reinit_last_layer:
            model.init_last_layer(init_mode=args.init_mode, bias_init=args.bias_init, seed=args.seed)
            print(f"Reinitialized fc8 with: {args.init_mode}")

    # Print a brief summary of what’s trainable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Train mode: {args.train_mode}")
    print(f"Init mode (fc8): {args.init_mode}, bias={args.bias_init}, seed={args.seed}")
    print(f"Trainable params ({len(trainable)} layers):")
    for n in trainable:
        print(f"  - {n}")


if __name__ == "__main__":
    _main()
