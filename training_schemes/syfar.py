# defenses/syfar_train.py
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

# --- Project imports (aligned with your structure) ---
from utils.data_process import data_process
from models.vgg16 import VGG_16
# Optional (enable with --use-faal if present in your repo)
try:
    from utils.faal import FAAL  # expects .compute_weights(class_losses)
    _HAS_FAAL = True
except Exception:
    _HAS_FAAL = False


# ----------------------- Losses -----------------------

def carlini_wagner_loss(outputs: torch.Tensor, y: torch.Tensor, large_const: float = 1e6) -> torch.Tensor:
    """Untargeted CW-style margin loss (logit form)."""
    y_onehot = F.one_hot(y, num_classes=outputs.shape[1]).float()
    logits_y = torch.sum(outputs * y_onehot, dim=1)
    logits_max_non_y, _ = torch.max(outputs - large_const * y_onehot, dim=1)
    return torch.mean(logits_max_non_y - logits_y)


def select_loss(outputs: torch.Tensor, y: torch.Tensor, name: str) -> torch.Tensor:
    name = name.lower()
    if name in ("ce", "cross_entropy"):
        return F.cross_entropy(outputs, y)
    if name in ("cw", "carlini_wagner"):
        return carlini_wagner_loss(outputs, y)
    raise ValueError(f"Unknown loss '{name}'")


# -------------- Symmetry (asymmetry) penalty ----------

def compute_asymmetry_penalty(conf_matrix: torch.Tensor, num_classes: int, epsilon: float) -> torch.Tensor:
    """
    penalty_{i,j} = (|a-b| / (a+b+ε)) * (a+b) for i<j, where
    a = conf[i,j], b = conf[j,i]. Confusion is row-normalized (source perspective).
    """
    device = conf_matrix.device
    pen = torch.tensor(0.0, device=device)
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            a = conf_matrix[i, j]
            b = conf_matrix[j, i]
            denom = a + b + epsilon
            pen += (torch.abs(a - b) / denom) * (a + b)
    return pen


# --------------------------- ROA ----------------------

class ROA:
    """
    Rectangular Occlusion Attack working in *pixel space* with the VGG-Face mean,
    matching your earlier scripts (X is mean-centered; X+mean => [0..255] pixels).
    """
    def __init__(self, base_classifier: nn.Module, alpha: float, iters: int):
        self.base_classifier = base_classifier
        self.alpha = float(alpha)
        self.iters = int(iters)
        # VGG-Face pixel means (BGR order in original; here we keep channel order matching your VGG_16)
        self.pixel_mean = torch.tensor([129.1863, 104.7624, 93.5940]).view(1, 3, 1, 1)

    @torch.no_grad()
    def _search(self, X: torch.Tensor, y: torch.Tensor, width: int, height: int, xskip: int, yskip: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.base_classifier
        model.eval()
        device = X.device

        mean = self.pixel_mean.to(device)

        B, _, H, W = X.shape
        xtimes = max(1, (W - width) // max(1, xskip))
        ytimes = max(1, (H - height) // max(1, yskip))

        max_loss = torch.full((B,), -1e9, device=device)
        out_i = torch.zeros(B, device=device)
        out_j = torch.zeros(B, device=device)

        for i in range(xtimes):
            for j in range(ytimes):
                sticker = X + mean
                sticker[:, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width] = 255.0 / 2.0
                sticker1 = sticker - mean
                logits = model(sticker1)
                loss = F.cross_entropy(logits, y, reduction="none")
                better = loss > max_loss
                out_i[better] = float(i)
                out_j[better] = float(j)
                max_loss = torch.maximum(max_loss, loss)

        # If flat losses, randomize a position to avoid degenerate mask
        flat = (max_loss == max_loss.min())
        if flat.any():
            out_i[flat] = torch.randint(low=0, high=max(1, xtimes), size=(int(flat.sum()),), device=device).float()
            out_j[flat] = torch.randint(low=0, high=max(1, ytimes), size=(int(flat.sum()),), device=device).float()

        return out_j, out_i

    def _refine(self, X: torch.Tensor, y: torch.Tensor, width: int, height: int, xskip: int, yskip: int,
                out_j: torch.Tensor, out_i: torch.Tensor) -> torch.Tensor:
        model = self.base_classifier
        model.eval()
        device = X.device
        mean = self.pixel_mean.to(device)

        B = X.shape[0]
        mask = torch.zeros_like(X)
        for b in range(B):
            j = int(out_j[b].item())
            i = int(out_i[b].item())
            mask[b, :, yskip * j : yskip * j + height, xskip * i : xskip * i + width] = 1.0

        delta = torch.zeros_like(X, requires_grad=True) + 255.0 / 2.0
        X1 = torch.rand_like(X, requires_grad=True)
        X1.data = X.detach() * (1 - mask) + ((delta.detach() - mean) * mask)

        for _ in range(self.iters):
            logits = model(X1)
            loss = F.cross_entropy(logits, y)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # untargeted: ascend loss inside sticker region
            X1.data = (X1.detach() + self.alpha * X1.grad.detach().sign() * mask)
            X1.data = ((X1.detach() + mean).clamp(0.0, 255.0) - mean)
            X1.grad.zero_()

        return X1.detach()

    def generate(self, X: torch.Tensor, y: torch.Tensor, width: int, height: int, xskip: int, yskip: int) -> torch.Tensor:
        out_j, out_i = self._search(X, y, width, height, xskip, yskip)
        return self._refine(X, y, width, height, xskip, yskip, out_j, out_i)


# ---------------------- Training loop ------------------

@torch.no_grad()
def _row_normalized_confusion_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Build a soft confusion matrix (row-normalized by source class):
    For each label k, average softmax(logits) over samples with label k.
    """
    device = logits.device
    probs = F.softmax(logits, dim=1)
    conf = torch.zeros(num_classes, num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)

    for i in range(labels.shape[0]):
        k = int(labels[i].item())
        conf[k] += probs[i]
        counts[k] += 1

    for k in range(num_classes):
        if counts[k] > 0:
            conf[k] /= counts[k]

    return conf


def train_one_epoch_syfar(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    sizes: Dict[str, int],
    optimizer: torch.optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler | None,
    device: torch.device,
    epoch: int,
    *,
    loss_name: str,
    clean_weight: float,
    adv_weight: float,
    sym_weight: float,
    epsilon: float,
    roa_params: Dict[str, int | float],
    class_names: List[str],
    use_faal: bool,
    faal_radius: float,
) -> Dict[str, float]:
    """
    One epoch of Sy-FAR training: clean + ROA adversarial + symmetry penalty.
    """
    model.train()
    attacker = ROA(model, alpha=float(roa_params["alpha"]), iters=int(roa_params["iters"]))

    K = len(class_names)
    faal = None
    if use_faal:
        if not _HAS_FAAL:
            raise RuntimeError("Requested --use-faal but utils.faal.FAAL not found.")
        faal = FAAL(train_batch_size=K, r_choice=float(faal_radius))

    run_loss = run_clean_correct = run_adv_correct = n_total = 0
    sym_accum = 0.0

    for inputs, labels in tqdm(loaders["train"], desc=f"Epoch {epoch} [train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Clean
        logits_clean = model(inputs)
        preds_clean = logits_clean.argmax(1)
        loss_clean = select_loss(logits_clean, labels, loss_name)

        # Adversarial via ROA
        adv_inputs = attacker.generate(
            inputs, labels,
            width=int(roa_params["width"]), height=int(roa_params["height"]),
            xskip=int(roa_params["xskip"]), yskip=int(roa_params["yskip"])
        )
        logits_adv = model(adv_inputs)
        preds_adv = logits_adv.argmax(1)

        # Adv loss (optionally FAAL class weighting)
        if use_faal:
            class_losses = []
            for c in range(K):
                mask = (labels == c)
                if mask.any():
                    class_losses.append(select_loss(logits_adv[mask], labels[mask], loss_name))
                else:
                    class_losses.append(torch.tensor(0.0, device=device))
            class_losses = torch.stack(class_losses)
            weights = faal.compute_weights(class_losses.detach())  # (K,)
            loss_adv = (weights * class_losses).sum()
        else:
            loss_adv = select_loss(logits_adv, labels, loss_name)

        # Symmetry penalty
        conf = _row_normalized_confusion_from_logits(logits_adv, labels, K)
        sym_pen = compute_asymmetry_penalty(conf, K, epsilon)

        # Total loss
        loss = clean_weight * loss_clean + adv_weight * loss_adv + sym_weight * sym_pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # stats
        bs = inputs.size(0)
        run_loss += float(loss.item()) * bs
        sym_accum += float(sym_pen.item()) * bs
        n_total += bs
        run_clean_correct += int((preds_clean == labels).sum().item())
        run_adv_correct += int((preds_adv == labels).sum().item())

    if scheduler is not None:
        scheduler.step()

    return {
        "train_loss": run_loss / max(1, n_total),
        "sym_pen": sym_accum / max(1, n_total),
        "clean_acc": run_clean_correct / max(1, n_total),
        "adv_acc": run_adv_correct / max(1, n_total),
    }


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y).item()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += x.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


# --------------------------- CLI ----------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sy-FAR: Symmetry-based Fair Adversarial Robustness (training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / batching
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)

    # Model
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--resume", type=str, default=None, help="Path to a pretrained .pt to start from")

    # Optim / schedule
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="steplr", choices=["none", "steplr", "cosine"])
    p.add_argument("--step-size", type=int, default=7)
    p.add_argument("--gamma", type=float, default=0.1)

    # Losses / mixing
    p.add_argument("--loss", type=str, default="carlini_wagner", choices=["cross_entropy", "carlini_wagner"])
    p.add_argument("--clean-weight", type=float, default=0.1)
    p.add_argument("--adv-weight", type=float, default=10.0)
    p.add_argument("--sym-weight", type=float, default=10.0)
    p.add_argument("--epsilon", type=float, default=0.1, help="ε inside symmetry penalty denominator")

    # ROA rectangle + PGD
    p.add_argument("--alpha", type=float, default=20.0, help="PGD step (pixel domain w/ mean add/removal)")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--width", type=int, default=70)
    p.add_argument("--height", type=int, default=70)
    p.add_argument("--xskip", type=int, default=10)
    p.add_argument("--yskip", type=int, default=10)

    # FAAL (optional)
    p.add_argument("--use-faal", action="store_true", help="Enable KL-DRO reweighting across classes")
    p.add_argument("--faal-radius", type=float, default=0.1, help="KL radius r for FAAL")

    # IO
    p.add_argument("--out-dir", type=Path, default=Path("./runs/syfar"))
    p.add_argument("--seed", type=int, default=1338)
    p.add_argument("--tag", type=str, default="")
    return p


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output / logging
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"vgg16_{ts}{('_' + args.tag) if args.tag else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    logging.basicConfig(filename=log_path.as_posix(), level=logging.INFO, format="%(message)s")
    print(f"[log] {log_path}")

    # Data
    # Note: current utils.data_process signature does not take data_dir; it reads from your configured path.
    dataloaders, dataset_sizes, class_names = data_process(batch_size=args.batch_size)
    assert len(class_names) == args.num_classes, f"--num-classes ({args.num_classes}) != data classes ({len(class_names)})"

    # Model
    model = VGG_16(num_classes=args.num_classes).to(device)
    if args.resume and Path(args.resume).is_file():
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[ckpt] Loaded init weights from {args.resume}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.scheduler == "steplr":
        sched = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        sched = None

    roa_params = dict(alpha=args.alpha, iters=args.iters, width=args.width, height=args.height,
                      xskip=args.xskip, yskip=args.yskip)

    best_adv = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        stats = train_one_epoch_syfar(
            model, dataloaders, dataset_sizes, optimizer, sched, device, epoch,
            loss_name=args.loss,
            clean_weight=args.clean_weight, adv_weight=args.adv_weight,
            sym_weight=args.sym_weight, epsilon=args.epsilon,
            roa_params=roa_params,
            class_names=class_names,
            use_faal=args.use_faal, faal_radius=args.faal_radius,
        )
        val_loss, val_acc = evaluate_clean(model, dataloaders["val"], device)

        logging.info(
            f"Epoch {epoch:03d} | train_loss={stats['train_loss']:.4f} "
            f"| sym_pen={stats['sym_pen']:.4f} "
            f"| clean_acc={stats['clean_acc']:.4f} | adv_acc={stats['adv_acc']:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} "
            f"| time={time.time() - t0:.1f}s"
        )
        print(
            f"Epoch {epoch:03d} | train_loss={stats['train_loss']:.4f} "
            f"| sym_pen={stats['sym_pen']:.4f} "
            f"| clean_acc={stats['clean_acc']:.4f} | adv_acc={stats['adv_acc']:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if stats["adv_acc"] > best_adv:
            best_adv = stats["adv_acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_adv.pt")
            print(f"[checkpoint] best_adv.pt (adv_acc={best_adv:.4f})")

    # Save final
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "final.pt")
    print(f"[done] Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
