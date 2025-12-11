# training/standard.py
# -*- coding: utf-8 -*-

#python training/standard.py --data-dir /path/to/dataset --model vgg16 --num-classes 8 --batch-size 32 --epochs 30 --optimizer sgd --lr 0.001 --momentum 0.9 --weight-decay 0.0005 --scheduler steplr --step-size 7 --gamma 0.1 --amp --out-dir ./runs/standard_vgg

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler

from utils.data_process import data_process

# Models
from models.vgg16 import VGG_16
from models.resnet import ResNet18
from models.vit import ViT


# ---------------- Utilities ----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(
    model_name: str,
    num_classes: int,
    *,
    vggface_weights: str | None = None,
    vgg_train_mode: str = "end2end",
    vgg_head_init: str = "xavier_uniform",
    vgg_reinit_head: bool = False,
    vit_pretrained: bool = True,
    vit_train_mode: str = "end2end",
    vit_head_init: str = "xavier_uniform",
    vit_reinit_head: bool = False,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name in ("vgg16", "vgg_face", "vggface"):
        model = VGG_16(num_classes=num_classes)
        if vggface_weights:
            model.load_weights(vggface_weights)
        # Optionally freeze or head-only controlled in VGG_16 via flags?
        # Our VGG_16 exposes only load_weights; training mode is controlled by requires_grad.
        if vgg_train_mode == "head_only":
            for n, p in model.named_parameters():
                p.requires_grad = ("fc" in n)
        elif vgg_train_mode == "end2end":
            for p in model.parameters():
                p.requires_grad = True
        # Re-init head if requested
        if vgg_reinit_head:
            if vgg_head_init == "xavier_uniform":
                nn.init.xavier_uniform_(model.fc8.weight)
            elif vgg_head_init == "xavier_normal":
                nn.init.xavier_normal_(model.fc8.weight)
            elif vgg_head_init == "kaiming_uniform":
                nn.init.kaiming_uniform_(model.fc8.weight, nonlinearity="relu")
            elif vgg_head_init == "kaiming_normal":
                nn.init.kaiming_normal_(model.fc8.weight, nonlinearity="relu")
            elif vgg_head_init == "orthogonal":
                nn.init.orthogonal_(model.fc8.weight)
            elif vgg_head_init == "normal":
                nn.init.normal_(model.fc8.weight, mean=0.0, std=0.02)
            elif vgg_head_init == "uniform":
                nn.init.uniform_(model.fc8.weight, a=-0.01, b=0.01)
            else:
                raise ValueError(f"Unknown init: {vgg_head_init}")
            if model.fc8.bias is not None:
                nn.init.constant_(model.fc8.bias, 0.0)
        return model

    if model_name in ("resnet18", "resnet"):
        return ResNet18(num_classes=num_classes)

    if model_name in ("vit", "vit_base_patch16_224"):
        return ViT(
            model_name="vit_base_patch16_224",
            pretrained=vit_pretrained,
            num_classes=num_classes,
            train_mode=vit_train_mode,
            reinit_head=vit_reinit_head,
            head_init_mode=vit_head_init,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def build_optimizer(name: str, params, lr: float, momentum: float, weight_decay: float) -> optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name: str, optimizer: optim.Optimizer, step_size: int, gamma: float, tmax: int) -> lr_scheduler._LRScheduler | None:
    name = (name or "").lower()
    if name in ("", "none"):
        return None
    if name == "steplr":
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
    if name == "multistep":
        return lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=gamma)
    raise ValueError(f"Unknown scheduler: {name}")


# ---------------- Training / Eval ----------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += float(loss.item()) * inputs.size(0)
        preds = outputs.argmax(1)
        total_correct += int((preds == labels).sum().item())
        total += inputs.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    sizes: Dict[str, int],
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    scheduler: lr_scheduler._LRScheduler | None = None,
) -> Dict[str, float]:
    scaler = GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in loaders["train"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * inputs.size(0)
        preds = outputs.argmax(1)
        running_corrects += int((preds == labels).sum().item())
        total += inputs.size(0)

    if scheduler is not None:
        scheduler.step()

    train_loss = running_loss / max(1, total)
    train_acc = running_corrects / max(1, total)

    # validation
    val_loss, val_acc = evaluate(model, loaders["val"], device)

    return {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    dataset_sizes: Dict[str, int],
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler | None,
    num_epochs: int,
    device: torch.device,
    use_amp: bool = True,
    save_best_to: Path | None = None,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    since = time.time()
    best_acc = 0.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        stats = train_one_epoch(
            model, dataloaders, dataset_sizes, optimizer, device, epoch, use_amp=use_amp, scheduler=scheduler
        )
        history.append({"epoch": epoch, **stats})
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train: loss {stats['train_loss']:.4f}, acc {stats['train_acc']:.4f} | "
            f"val: loss {stats['val_loss']:.4f}, acc {stats['val_acc']:.4f}"
        )

        if stats["val_acc"] > best_acc:
            best_acc = stats["val_acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if save_best_to is not None:
                save_best_to.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_best_to)
                print(f"[checkpoint] Saved best to: {save_best_to} (acc={best_acc:.4f})")

    elapsed = time.time() - since
    print(f"Training complete in {int(elapsed // 60)}m {int(elapsed % 60)}s | Best val acc: {best_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ---------------- CLI ----------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standard training loop (clean, modular, reproducible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data-dir", type=Path, required=True, help="Root with train/val/test subdirs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--use-train-aug", action="store_true")

    # Model
    p.add_argument("--model", type=str, default="vgg16", choices=["vgg16", "resnet18", "vit"])
    p.add_argument("--num-classes", type=int, default=10)

    # VGG options
    p.add_argument("--vggface-weights", type=str, default=None, help="Path to VGG_FACE.t7 (Torch7) if using VGG_16")
    p.add_argument("--vgg-train-mode", type=str, default="end2end", choices=["end2end", "head_only"])
    p.add_argument("--vgg-reinit-head", action="store_true")
    p.add_argument("--vgg-head-init", type=str, default="xavier_uniform",
                   choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal", "normal", "uniform"])

    # ViT options
    p.add_argument("--vit-pretrained", action="store_true", default=True)
    p.add_argument("--no-vit-pretrained", dest="vit_pretrained", action="store_false")
    p.add_argument("--vit-train-mode", type=str, default="end2end", choices=["end2end", "head_only"])
    p.add_argument("--vit-reinit-head", action="store_true")
    p.add_argument("--vit-head-init", type=str, default="xavier_uniform",
                   choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal", "normal", "uniform"])

    # Optim / sched
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="steplr", choices=["none", "steplr", "cosine", "multistep"])
    p.add_argument("--step-size", type=int, default=7)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--tmax", type=int, default=25)

    # Runtime
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--repetitions", type=int, default=1, help="Repeat training with different seeds")
    p.add_argument("--base-seed", type=int, default=1337)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    p.add_argument("--out-dir", type=Path, default=Path("./runs/standard"))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataloaders, dataset_sizes, class_names = data_process(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        image_size=args.image_size,
        num_workers=args.num_workers,
        pin_memory=True,
        use_train_aug=args.use_train_aug,
    )

    # Output setup
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "log.csv"

    # CSV log header
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_ckpt_path"])

    # Repetitions
    for rep in range(args.repetitions):
        seed = args.base_seed + rep
        set_seed(seed)
        print(f"\n=== Repetition {rep+1}/{args.repetitions} | seed={seed} ===")

        # Build model
        model = build_model(
            args.model,
            args.num_classes,
            vggface_weights=args.vggface_weights,
            vgg_train_mode=args.vgg_train_mode,
            vgg_head_init=args.vgg_head_init,
            vgg_reinit_head=args.vgg_reinit_head,
            vit_pretrained=args.vit_pretrained,
            vit_train_mode=args.vit_train_mode,
            vit_head_init=args.vit_head_init,
            vit_reinit_head=args.vit_reinit_head,
        ).to(device)

        # Optim & sched
        optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.weight_decay)
        scheduler = build_scheduler(args.scheduler, optimizer, step_size=args.step_size, gamma=args.gamma, tmax=args.tmax)

        # Train
        best_ckpt = run_dir / f"best_rep{rep+1}.pt"
        model, history = train_model(
            model,
            dataloaders,
            dataset_sizes,
            optimizer,
            scheduler,
            num_epochs=args.epochs,
            device=device,
            use_amp=args.amp,
            save_best_to=best_ckpt,
        )

        # Append CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for h in history:
                w.writerow([
                    rep + 1,
                    h["epoch"],
                    f'{h["train_loss"]:.6f}',
                    f'{h["train_acc"]:.6f}',
                    f'{h["val_loss"]:.6f}',
                    f'{h["val_acc"]:.6f}',
                    best_ckpt.as_posix(),
                ])
        print(f"[log] CSV appended at {csv_path}")

        # Optionally save final model
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / f"final_rep{rep+1}.pt")

    print(f"\nDone. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
