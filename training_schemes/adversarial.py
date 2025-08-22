# defenses/ROA.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm

# Project imports
from utils.data_process import data_process
from models.vgg16 import VGG_16
from models.resnet import ResNet18
from models.vit import ViT
from utils.carlini_wagner import carlini_wagner_loss
from utils.faal import FAAL  # KL-DRO reweighting over classes


# --------------------- ROA Attack ---------------------

class ROA:
    """
    Rectangular Occlusion Attack (ROA): choose the most damaging rectangle position
    (via exhaustive search) and refine the pixel values inside the rectangle using
    sign-PGD while keeping the sticker region constrained to [0, 1].

    This version operates directly in normalized image space [0, 1].
    """

    def __init__(self, base_classifier: nn.Module, alpha: float, iters: int, targeted_label: int | None = None):
        """
        Args:
            base_classifier: model with logits output.
            alpha: PGD step size in [0,1] scale.
            iters: number of PGD steps.
            targeted_label: if not None, run targeted loss toward this label; else untargeted.
        """
        self.base_classifier = base_classifier
        self.alpha = float(alpha)
        self.iters = int(iters)
        self.targeted_label = targeted_label

    @torch.no_grad()
    def _choose_position(
        self, X: torch.Tensor, y: torch.Tensor, width: int, height: int, xskip: int, yskip: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exhaustive search over rectangle positions. Returns (row_indices, col_indices)
        per example (floats for later indexing arithmetic).
        """
        model = self.base_classifier
        model.eval()

        B, C, H, W = X.shape
        device = X.device

        xtimes = max(1, (W - width) // max(1, xskip))
        ytimes = max(1, (H - height) // max(1, yskip))

        max_loss = torch.full((B,), -1e9, device=device, dtype=torch.float)
        out_i = torch.zeros(B, device=device, dtype=torch.float)
        out_j = torch.zeros(B, device=device, dtype=torch.float)

        for i in range(xtimes):
            for j in range(ytimes):
                sticker = X.clone()
                # Gray(0.5) rectangle:
                sticker[:, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width] = 0.5
                logits = model(sticker)

                if self.targeted_label is None:
                    # Untargeted: maximize CE loss to true labels
                    loss = F.cross_entropy(logits, y, reduction="none")
                else:
                    # Targeted: encourage target label
                    target = torch.full_like(y, self.targeted_label)
                    loss = -F.cross_entropy(logits, target, reduction="none")

                better = loss > max_loss
                out_j[better] = float(j)
                out_i[better] = float(i)
                max_loss = torch.maximum(max_loss, loss)

        # handle degenerate (all equal) by randomizing a position
        equal_mask = (max_loss == max_loss.min())
        if equal_mask.any():
            out_j[equal_mask] = torch.randint(low=0, high=max(1, ytimes), size=(int(equal_mask.sum()),), device=device).float()
            out_i[equal_mask] = torch.randint(low=0, high=max(1, xtimes), size=(int(equal_mask.sum()),), device=device).float()

        return out_j, out_i

    def _refine_pgd(
        self, X: torch.Tensor, y: torch.Tensor, width: int, height: int, xskip: int, yskip: int,
        out_j: torch.Tensor, out_i: torch.Tensor
    ) -> torch.Tensor:
        """
        Constrained PGD on the selected rectangle. Keeps rectangle values in [0,1].
        """
        model = self.base_classifier
        model.eval()
        device = X.device

        B, C, H, W = X.shape
        sticker_mask = torch.zeros_like(X)
        for b in range(B):
            j = int(out_j[b].item())
            i = int(out_i[b].item())
            sticker_mask[b, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width] = 1.0

        # Initialize with a neutral sticker
        X1 = X.clone()
        X1 = torch.where(sticker_mask.bool(), torch.full_like(X1, 0.5), X1).detach()
        X1.requires_grad_(True)

        target = None
        if self.targeted_label is not None:
            target = torch.full_like(y, self.targeted_label)

        for _ in range(self.iters):
            logits = model(X1)
            if target is None:
                loss = F.cross_entropy(logits, y)
                grad_sign = torch.sign(X1.grad) if X1.grad is not None else 0.0
                # standard (increase loss)
                loss.backward()
                grad = X1.grad.detach()
                X1.grad.zero_()
                step = self.alpha * torch.sign(grad)
                X1 = (X1 + step * sticker_mask).clamp(0.0, 1.0).detach().requires_grad_(True)
            else:
                # targeted (decrease CE toward target)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                grad = X1.grad.detach()
                X1.grad.zero_()
                step = -self.alpha * torch.sign(grad)  # gradient descent toward target
                X1 = (X1 + step * sticker_mask).clamp(0.0, 1.0).detach().requires_grad_(True)

        return X1.detach()

    def generate(
        self, X: torch.Tensor, y: torch.Tensor, width: int, height: int, xskip: int, yskip: int
    ) -> torch.Tensor:
        """
        Full ROA pipeline: choose position then refine with PGD.
        """
        out_j, out_i = self._choose_position(X, y, width, height, xskip, yskip)
        return self._refine_pgd(X, y, width, height, xskip, yskip, out_j, out_i)


# --------------------- Training ---------------------

def build_model(name: str, num_classes: int, vggface_t7: str | None = None) -> nn.Module:
    name = name.lower()
    if name in ("vgg", "vgg16", "vgg_face"):
        m = VGG_16(num_classes=num_classes)
        if vggface_t7:
            m.load_weights(vggface_t7)
        else:
            # default: trainable end-to-end
            for p in m.parameters():
                p.requires_grad = True
        return m
    if name in ("resnet18", "resnet"):
        return ResNet18(num_classes=num_classes)
    if name in ("vit", "vit_base_patch16_224"):
        return ViT("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def select_loss(outputs: torch.Tensor, labels: torch.Tensor, loss_name: str) -> torch.Tensor:
    loss_name = loss_name.lower()
    if loss_name in ("ce", "cross_entropy"):
        return F.cross_entropy(outputs, labels)
    if loss_name in ("cw", "carlini_wagner"):
        return carlini_wagner_loss(outputs, labels)
    raise ValueError(f"Unknown loss: {loss_name}")


def train_one_epoch_roa(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    sizes: Dict[str, int],
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler | None,
    device: torch.device,
    epoch: int,
    *,
    roa_params: Dict,
    targeted_label: int | None,
    clean_weight: float,
    adv_weight: float,
    class_names: List[str],
    loss_name: str,
    use_faal: bool,
    faal_radius: float,
) -> Dict[str, float]:
    """
    Train for one epoch with ROA adversarial examples.
    Optionally apply FAAL/KL-DRO weighting over per-class adversarial losses.
    """
    model.train()
    running_loss = 0.0
    running_clean_correct = 0
    running_adv_correct = 0
    total = 0

    attacker = ROA(
        base_classifier=model,
        alpha=float(roa_params["alpha"]),
        iters=int(roa_params["iters"]),
        targeted_label=targeted_label,
    )

    # per-class stats
    K = len(class_names)
    class_clean_total = torch.zeros(K, dtype=torch.long)
    class_clean_correct = torch.zeros(K, dtype=torch.long)
    class_adv_total = torch.zeros(K, dtype=torch.long)
    class_adv_correct = torch.zeros(K, dtype=torch.long)

    # FAAL helper (over classes)
    faal = FAAL(train_batch_size=K, r_choice=float(faal_radius)) if use_faal else None

    for inputs, labels in tqdm(loaders["train"], desc=f"Epoch {epoch} [train]"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Clean pass
        clean_logits = model(inputs)
        clean_loss = select_loss(clean_logits, labels, loss_name)
        clean_preds = clean_logits.argmax(1)

        # ROA adversarial examples
        adv_inputs = attacker.generate(
            inputs, labels,
            width=int(roa_params["width"]), height=int(roa_params["height"]),
            xskip=int(roa_params["xskip"]), yskip=int(roa_params["yskip"])
        )
        adv_logits = model(adv_inputs)
        adv_preds = adv_logits.argmax(1)

        # Per-class adversarial losses (for FAAL)
        if use_faal:
            class_losses = []
            for c in range(K):
                mask = (labels == c)
                if mask.any():
                    class_losses.append(select_loss(adv_logits[mask], labels[mask], loss_name))
                else:
                    class_losses.append(torch.tensor(0.0, device=device))
            class_losses = torch.stack(class_losses)  # (K,)

            weights = faal.compute_weights(class_losses.detach())  # (K,)
            adv_loss = (weights * class_losses).sum()
        else:
            adv_loss = select_loss(adv_logits, labels, loss_name)

        loss = clean_weight * clean_loss + adv_weight * adv_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # stats
        running_loss += float(loss.item()) * inputs.size(0)
        total += inputs.size(0)

        running_clean_correct += int((clean_preds == labels).sum().item())
        running_adv_correct += int((adv_preds == labels).sum().item())

        # per-class stats
        for c in range(K):
            cmask = (labels == c)
            n_c = int(cmask.sum().item())
            if n_c > 0:
                class_clean_total[c] += n_c
                class_clean_correct[c] += int((clean_preds[cmask] == labels[cmask]).sum().item())
                class_adv_total[c] += n_c
                class_adv_correct[c] += int((adv_preds[cmask] == labels[cmask]).sum().item())

    if scheduler is not None:
        scheduler.step()

    train_loss = running_loss / max(1, total)
    clean_acc = running_clean_correct / max(1, total)
    adv_acc = running_adv_correct / max(1, total)

    # log per-class
    logging.info(f"[Per-class stats @ epoch {epoch}]")
    for c, name in enumerate(class_names):
        ct = int(class_clean_total[c])
        ca = int(class_adv_total[c])
        cc = (class_clean_correct[c].item() / max(1, ct)) if ct > 0 else 0.0
        ac = (class_adv_correct[c].item() / max(1, ca)) if ca > 0 else 0.0
        logging.info(f"  {name:>20s} | clean n={ct:4d} acc={cc:.3f} | adv n={ca:4d} acc={ac:.3f}")

    return {"train_loss": train_loss, "clean_acc": clean_acc, "adv_acc": adv_acc}


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        tot_loss += float(loss.item()) * x.size(0)
        tot_correct += int((logits.argmax(1) == y).sum().item())
        tot += x.size(0)
    return tot_loss / max(1, tot), tot_correct / max(1, tot)


# --------------------- CLI ---------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adversarial training with ROA (Rectangular Occlusion Attack)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data-dir", type=Path, required=True, help="Root dataset dir with train/val/test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)

    # Model
    p.add_argument("--model", type=str, default="vgg16", choices=["vgg16", "resnet18", "vit"])
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--vggface-t7", type=str, default=None, help="Path to VGG_FACE.t7 if using VGG_16")
    p.add_argument("--pretrained-ckpt", type=str, default=None, help="Path to .pt to load state_dict before training")

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="steplr", choices=["none", "steplr", "cosine"])
    p.add_argument("--step-size", type=int, default=7)
    p.add_argument("--gamma", type=float, default=0.1)

    # Loss / mixing
    p.add_argument("--loss", type=str, default="carlini_wagner", choices=["cross_entropy", "carlini_wagner"])
    p.add_argument("--clean-weight", type=float, default=0.1)
    p.add_argument("--adv-weight", type=float, default=0.9)

    # ROA params
    p.add_argument("--alpha", type=float, default=0.02, help="PGD step in [0,1]")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--width", type=int, default=70)
    p.add_argument("--height", type=int, default=70)
    p.add_argument("--xskip", type=int, default=10)
    p.add_argument("--yskip", type=int, default=10)
    p.add_argument("--targeted-label", type=int, default=None, help="If set, run targeted ROA to this label")

    # FAAL (KL-DRO) over classes
    p.add_argument("--use-faal", action="store_true")
    p.add_argument("--faal-radius", type=float, default=0.1, help="KL radius r (0 -> uniform)")

    # IO
    p.add_argument("--out-dir", type=Path, default=Path("./runs/roa_training"))
    p.add_argument("--tag", type=str, default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"{args.model}_{ts}{('_' + args.tag) if args.tag else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    logging.basicConfig(filename=log_path.as_posix(), level=logging.INFO, format="%(message)s")
    print(f"[log] {log_path}")

    # Data
    dataloaders, dataset_sizes, class_names = data_process(batch_size=args.batch_size)
    # If your data_process supports args.data_dir, you can use:
    # dataloaders, dataset_sizes, class_names = data_process(batch_size=args.batch_size, data_dir=args.data_dir)

    # Model
    model = build_model(args.model, args.num_classes, vggface_t7=args.vggface_t7).to(device)
    if args.pretrained_ckpt and Path(args.pretrained_ckpt).is_file():
        state = torch.load(args.pretrained_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[ckpt] Loaded: {args.pretrained_ckpt}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.scheduler == "steplr":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    roa_params = dict(alpha=args.alpha, iters=args.iters, width=args.width, height=args.height,
                      xskip=args.xskip, yskip=args.yskip)

    best_adv_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        stats = train_one_epoch_roa(
            model, dataloaders, dataset_sizes, optimizer, scheduler, device, epoch,
            roa_params=roa_params,
            targeted_label=args.targeted_label,
            clean_weight=args.clean_weight,
            adv_weight=args.adv_weight,
            class_names=class_names,
            loss_name=args.loss,
            use_faal=args.use_faal,
            faal_radius=args.faal_radius,
        )
        val_loss, val_acc = evaluate_clean(model, dataloaders["val"], device)

        logging.info(
            f"Epoch {epoch:03d} | train_loss={stats['train_loss']:.4f} "
            f"| clean_acc={stats['clean_acc']:.4f} | adv_acc={stats['adv_acc']:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} "
            f"| time={time.time() - t0:.1f}s"
        )
        print(
            f"Epoch {epoch:03d} | train_loss={stats['train_loss']:.4f} "
            f"| clean_acc={stats['clean_acc']:.4f} | adv_acc={stats['adv_acc']:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if stats["adv_acc"] > best_adv_acc:
            best_adv_acc = stats["adv_acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_adv.pt")
            print(f"[checkpoint] best_adv.pt (adv_acc={best_adv_acc:.4f})")

    # Save final
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "final.pt")
    print(f"[done] Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
