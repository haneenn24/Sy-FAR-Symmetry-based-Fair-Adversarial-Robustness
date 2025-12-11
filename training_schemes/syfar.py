# training_schemes/syfar.py
# -*- coding: utf-8 -*-
#python -m training_schemes.syfar --data-dir /path/to/dataset --model vgg16 --num-classes 12 --epochs 20 --batch-size 16 --lr 0.001 --w-clean 0.1 --w-adv 1.0 --w-sym 10.0 --eps 0.1 --alpha 0.02 --iters 40 --width 70 --height 70 --xskip 10 --yskip 10 --out-dir ./runs/syfar

from __future__ import annotations

import argparse
import time
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


# ============================
#   ROA ATTACK (same as adv)
# ============================

class ROA:
    def __init__(self, model, alpha, iters):
        self.model = model
        self.alpha = alpha
        self.iters = iters

    @torch.no_grad()
    def search(self, X, y, width, height, xskip, yskip):
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
        self.model.eval()
        B, _, _, _ = X.shape
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
        bi, bj = self.search(X, y, width, height, xskip, yskip)
        return self.refine(X, y, bi, bj, width, height, xskip, yskip)


# ============================
#   MODEL BUILDER
# ============================

def build_model(name, num_classes):
    name = name.lower()
    if name == "vgg16":
        return VGG_16(num_classes=num_classes)
    if name == "resnet18":
        return ResNet18(num_classes=num_classes)
    if name == "vit":
        return ViT("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


# ============================
#   SYMMETRY PENALTY
# ============================

def symmetry_penalty(adv_outputs, labels, num_classes, eps=0.1):
    probs = F.softmax(adv_outputs, dim=1)
    conf = torch.zeros(num_classes, num_classes, device=labels.device)
    counts = torch.zeros(num_classes, device=labels.device)

    for i in range(len(labels)):
        conf[labels[i]] += probs[i]
        counts[labels[i]] += 1

    for c in range(num_classes):
        if counts[c] > 0:
            conf[c] /= counts[c]

    penalty = 0.0
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            a = conf[i, j]
            b = conf[j, i]
            term = torch.abs(a - b) / (a + b + eps) * (a + b)
            penalty += term

    return penalty


# ============================
#   TRAIN ONE EPOCH (SyFAR)
# ============================

def train_epoch(model, loader, optimizer, roa, params, device, num_classes):
    model.train()
    total_loss = correct = total = 0

    for X, y in tqdm(loader, desc="Train"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        clean_logits = model(X)
        clean_loss = F.cross_entropy(clean_logits, y)

        X_adv = roa.generate(
            X, y,
            width=params["width"],
            height=params["height"],
            xskip=params["xskip"],
            yskip=params["yskip"]
        )
        adv_logits = model(X_adv)
        adv_loss = F.cross_entropy(adv_logits, y)

        sym_loss = symmetry_penalty(adv_logits, y, num_classes, eps=params["eps"])

        loss = (
            params["w_clean"] * clean_loss +
            params["w_adv"] * adv_loss +
            params["w_sym"] * sym_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total += X.size(0)
        correct += (clean_logits.argmax(1) == y).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * X.size(0)
        total += X.size(0)
        correct += (logits.argmax(1) == y).sum().item()

    return total_loss / total, correct / total


# ============================
#            MAIN
# ============================

def build_parser():
    p = argparse.ArgumentParser(description="SyFAR training (clean + adv + symmetry loss)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="vgg16")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    # weights for losses
    p.add_argument("--w-clean", type=float, default=0.1)
    p.add_argument("--w-adv", type=float, default=1.0)
    p.add_argument("--w-sym", type=float, default=10.0)
    p.add_argument("--eps", type=float, default=0.1)

    # ROA params
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--width", type=int, default=70)
    p.add_argument("--height", type=int, default=70)
    p.add_argument("--xskip", type=int, default=10)
    p.add_argument("--yskip", type=int, default=10)

    p.add_argument("--out-dir", type=Path, default=Path("./runs/syfar"))
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"{args.model}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataloaders, dataset_sizes, class_names = data_process(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        image_size=224,
        pin_memory=True,
    )

    model = build_model(args.model, args.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    roa = ROA(model, args.alpha, args.iters)

    params = dict(
        width=args.width,
        height=args.height,
        xskip=args.xskip,
        yskip=args.yskip,
        w_clean=args.w_clean,
        w_adv=args.w_adv,
        w_sym=args.w_sym,
        eps=args.eps,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, dataloaders["train"], optimizer, roa, params, device, args.num_classes)
        val_loss, val_acc = evaluate(model, dataloaders["val"], device)

        print(f"Epoch {epoch:02d} | train {train_acc:.4f} | val {val_acc:.4f}")
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best.pt")

    torch.save(best_state, run_dir / "final.pt")
    print(f"[done] Saved in: {run_dir}")


if __name__ == "__main__":
    main()
