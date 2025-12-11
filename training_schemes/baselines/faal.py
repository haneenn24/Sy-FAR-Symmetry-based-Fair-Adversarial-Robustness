# training_schemes/faal.py
# -*- coding: utf-8 -*-
#python -m training_schemes.faal --data-dir /path/to/dataset --model vgg16 --num-classes 12 --epochs 20 --batch-size 16 --lr 0.001 --clean-weight 0.1 --adv-weight 0.9 --tau 0.5 --alpha 0.02 --iters 40 --width 70 --height 70 --xskip 10 --yskip 10 --out-dir ./runs/faal

from __future__ import annotations
import argparse
import datetime as dt
import time
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.data_process import data_process
from models.vgg16 import VGG_16
from models.resnet import ResNet18
from models.vit import ViT


# ================================================================
#                       ROA ATTACK
# ================================================================

class ROA:
    def __init__(self, model, alpha, iters):
        self.model = model
        self.alpha = alpha
        self.iters = iters

    @torch.no_grad()
    def choose_position(self, X, y, width, height, xskip, yskip):
        B, _, H, W = X.shape
        device = X.device

        xtimes = max(1, (W - width) // xskip)
        ytimes = max(1, (H - height) // yskip)

        best_loss = torch.full((B,), -1e9, device=device)
        best_i = torch.zeros(B, device=device)
        best_j = torch.zeros(B, device=device)

        self.model.eval()

        for i in range(xtimes):
            for j in range(ytimes):
                img = X.clone()
                img[:, :, j*yskip:j*yskip+height, i*xskip:i*xskip+width] = 0.5

                logits = self.model(img)
                loss = F.cross_entropy(logits, y, reduction="none")

                mask = loss > best_loss
                best_loss[mask] = loss[mask]
                best_i[mask] = float(i)
                best_j[mask] = float(j)

        return best_i, best_j

    def refine(self, X, y, best_i, best_j, width, height, xskip, yskip):
        B, C, H, W = X.shape
        device = X.device

        mask = torch.zeros_like(X, device=device)
        for b in range(B):
            i = int(best_i[b].item())
            j = int(best_j[b].item())
            mask[b, :, j*yskip:j*yskip+height, i*xskip:i*xskip+width] = 1.0

        X_adv = X.clone().detach()
        X_adv.requires_grad_(True)

        for _ in range(self.iters):
            logits = self.model(X_adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            step = self.alpha * torch.sign(X_adv.grad)
            X_adv = (X_adv + step * mask).clamp(0.0, 1.0).detach()
            X_adv.requires_grad_(True)

        return X_adv.detach()

    def generate(self, X, y, width, height, xskip, yskip):
        i, j = self.choose_position(X, y, width, height, xskip, yskip)
        return self.refine(X, y, i, j, width, height, xskip, yskip)


# ================================================================
#             CARLINI–WAGNER LOSS (used in FAAL)
# ================================================================

def cw_loss(logits, labels, margin=0.5):
    logits = F.log_softmax(logits, dim=1)
    correct = logits.gather(1, labels.view(-1, 1)).squeeze()

    temp = logits.clone()
    temp[torch.arange(logits.size(0)), labels] = -float("inf")
    max_wrong = temp.max(dim=1)[0]

    loss = F.relu(max_wrong - correct + margin)
    return loss.mean()


# ================================================================
#                       FAAL WEIGHT SOLVER
# ================================================================

def solve_kl_dro(class_losses: torch.Tensor, tau: float):
    K = class_losses.size(0)
    max_loss = class_losses.max()

    w = torch.exp((class_losses - max_loss) / tau)
    w = w / w.sum()

    prior = torch.ones_like(w) / K
    kl = torch.sum(w * torch.log(w / prior + 1e-8))

    w = w * torch.exp(-tau * kl)
    w = w / w.sum()
    w = torch.clamp(w, min=0.005)

    return w


# ================================================================
#                     MODEL BUILDER
# ================================================================

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name in ("vgg", "vgg16"):
        return VGG_16(num_classes=num_classes)
    if name in ("resnet", "resnet18"):
        return ResNet18(num_classes=num_classes)
    if name in ("vit", "vit_base_patch16_224"):
        return ViT("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")


# ================================================================
#                  TRAINING LOOP (FAAL)
# ================================================================

def train_one_epoch_faal(
    model, loader, device, optimizer, roa, roa_params,
    clean_weight, adv_weight, tau, epoch
):
    model.train()

    running_loss = 0.0
    running_clean_acc = 0
    running_adv_acc = 0
    total = 0

    num_classes = model.fc8.out_features if hasattr(model, "fc8") else None

    for X, y in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        clean_logits = model(X)
        clean_loss = cw_loss(clean_logits, y)
        clean_preds = clean_logits.argmax(1)

        # ROA adversarial examples
        X_adv = roa.generate(X, y, roa_params["width"], roa_params["height"],
                             roa_params["xskip"], roa_params["yskip"])
        adv_logits = model(X_adv)
        adv_preds = adv_logits.argmax(1)

        # per-class DRO
        K = adv_logits.shape[1]
        class_losses = torch.zeros(K, device=device)
        for c in range(K):
            mask = (y == c)
            if mask.any():
                class_losses[c] = cw_loss(adv_logits[mask], y[mask])
            else:
                class_losses[c] = 0.01

        w = solve_kl_dro(class_losses, tau)
        faal_loss = (w * class_losses).sum()

        loss = clean_weight * clean_loss + adv_weight * faal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        running_clean_acc += (clean_preds == y).sum().item()
        running_adv_acc += (adv_preds == y).sum().item()
        total += X.size(0)

    return (
        running_loss / total,
        running_clean_acc / total,
        running_adv_acc / total,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_accum = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        loss_accum += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)

    return loss_accum / total, correct / total


# ================================================================
#                           CLI
# ================================================================

def build_argparser():
    p = argparse.ArgumentParser(description="FAAL adversarial training with ROA")

    # data
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)

    # model
    p.add_argument("--model", type=str, default="vgg16")
    p.add_argument("--num-classes", type=int, required=True)

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)

    # FAAL weights
    p.add_argument("--clean-weight", type=float, default=0.1)
    p.add_argument("--adv-weight", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.5)

    # ROA params
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--width", type=int, default=70)
    p.add_argument("--height", type=int, default=70)
    p.add_argument("--xskip", type=int, default=10)
    p.add_argument("--yskip", type=int, default=10)

    p.add_argument("--out-dir", type=Path, default=Path("./runs/faal"))
    return p


# ================================================================
#                        MAIN FUNCTION
# ================================================================

def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"{args.model}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataloaders, dataset_sizes, class_names = data_process(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        image_size=224,
        pin_memory=True,
    )

    model = build_model(args.model, args.num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    roa_params = dict(
        alpha=args.alpha,
        iters=args.iters,
        width=args.width,
        height=args.height,
        xskip=args.xskip,
        yskip=args.yskip,
    )
    roa = ROA(model, args.alpha, args.iters)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, clean_acc, adv_acc = train_one_epoch_faal(
            model, dataloaders["train"], device, optimizer,
            roa, roa_params, args.clean_weight, args.adv_weight, args.tau, epoch
        )
        val_loss, val_acc = evaluate(model, dataloaders["val"], device)

        print(
            f"Epoch {epoch:02d} | "
            f"clean_acc={clean_acc:.4f} | adv_acc={adv_acc:.4f} | val_acc={val_acc:.4f}"
        )

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best.pt")

    torch.save(best_state, out_dir / "final.pt")
    print(f"[done] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
