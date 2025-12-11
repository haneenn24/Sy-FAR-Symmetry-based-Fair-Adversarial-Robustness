# training_schemes/adversarial.py
# -*- coding: utf-8 -*-

#python -m training_schemes.adversarial --data-dir /path/to/dataset --model vgg16 --num-classes 12 --epochs 20 --batch-size 16 --lr 0.001 --clean-weight 0.1 --adv-weight 0.9 --alpha 0.02 --iters 40 --width 70 --height 70 --xskip 10 --yskip 10 --out-dir ./runs/adversarial_roa


from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.data_process import data_process
from models.vgg16 import VGG_16
from models.resnet import ResNet18
from models.vit import ViT


# ============================================================
#                    ROA ATTACK (Simplified)
# ============================================================

class ROA:
    def __init__(self, model, alpha, iters):
        self.model = model
        self.alpha = float(alpha)
        self.iters = int(iters)

    @torch.no_grad()
    def choose_position(self, X, y, width, height, xskip, yskip):
        """
        Exhaustive search for the most harmful rectangle position.
        """
        B, C, H, W = X.shape
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
        """
        PGD refinement over the chosen rectangle.
        """
        self.model.eval()
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
        best_i, best_j = self.choose_position(X, y, width, height, xskip, yskip)
        return self.refine(X, y, best_i, best_j, width, height, xskip, yskip)


# ============================================================
#                         MODEL BUILDER
# ============================================================

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name in ("vgg", "vgg16"):
        return VGG_16(num_classes=num_classes)
    if name in ("resnet", "resnet18"):
        return ResNet18(num_classes=num_classes)
    if name in ("vit", "vit_base_patch16_224"):
        return ViT("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


# ============================================================
#                  TRAINING LOOP (CLEAN + ADV)
# ============================================================

def train_one_epoch(model, loader, optimizer, device, roa, roa_params, epoch, clean_w, adv_w):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for X, y in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # clean pass
        clean_logits = model(X)
        clean_loss = F.cross_entropy(clean_logits, y)

        # adversarial generation (ROA)
        X_adv = roa.generate(
            X, y,
            width=roa_params["width"],
            height=roa_params["height"],
            xskip=roa_params["xskip"],
            yskip=roa_params["yskip"],
        )
        adv_logits = model(X_adv)
        adv_loss = F.cross_entropy(adv_logits, y)

        # weighted sum
        loss = clean_w * clean_loss + adv_w * adv_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_correct += (clean_logits.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, total_correct / total


# ============================================================
#                             MAIN
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Adversarial Training with ROA")

    # data
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-classes", type=int, required=True)

    # model
    p.add_argument("--model", type=str, default="vgg16")

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)

    # loss weights
    p.add_argument("--clean-weight", type=float, default=0.1)
    p.add_argument("--adv-weight", type=float, default=0.9)

    # ROA hyperparameters
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--width", type=int, default=70)
    p.add_argument("--height", type=int, default=70)
    p.add_argument("--xskip", type=int, default=10)
    p.add_argument("--yskip", type=int, default=10)

    # output
    p.add_argument("--out-dir", type=Path, default=Path("./runs/adversarial_roa"))

    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output directory
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"{args.model}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # load data
    dataloaders, dataset_sizes, class_names = data_process(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        image_size=224,
        pin_memory=True,
    )

    # model
    model = build_model(args.model, args.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    roa_params = dict(
        width=args.width,
        height=args.height,
        xskip=args.xskip,
        yskip=args.yskip,
    )

    roa = ROA(model, alpha=args.alpha, iters=args.iters)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            roa,
            roa_params,
            epoch,
            args.clean_weight,
            args.adv_weight,
        )

        val_loss, val_acc = evaluate(model, dataloaders["val"], device)

        print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best.pt")

    torch.save(best_state, run_dir / "final.pt")
    print(f"[done] Saved results to {run_dir}")


if __name__ == "__main__":
    main()
