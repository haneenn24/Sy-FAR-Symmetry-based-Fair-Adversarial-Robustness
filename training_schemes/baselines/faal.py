# utils/faal.py
# -*- coding: utf-8 -*-

"""
FAAL: Fair/Adversarial reweighting via KL-divergence ambiguity set.

We solve, for a batch of N samples with per-sample losses `ℓ_i`:

    maximize_p   sum_i p_i * ℓ_i
    subject to   p_i >= 0, sum_i p_i = 1,
                 KL(P_emp || p) <= r

where P_emp is the empirical (usually uniform) distribution on the batch,
and r >= 0 controls robustness (r=0 -> p = P_emp). This pushes mass toward
harder examples in a principled, distributionally robust way.

Dependencies: cvxpy (ECOS recommended). MOSEK is optional; SCS is the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import cvxpy as cp

# Optional: silence CVXPY DPP warnings etc. (uncomment if you want)
# import warnings
# warnings.filterwarnings("ignore")

# Optional: MOSEK license path if you have it (leave unset if not needed)
# import os
# os.environ.setdefault("MOSEKLM_LICENSE_FILE", "mosek.lic")


@dataclass
class FAALConfig:
    train_batch_size: int
    r_choice: float = 0.0                     # robustness radius (KL-ball size)
    learning_approach: str = "kl"             # reserved for future variants
    output_return: str = "weights"            # only 'weights' supported
    empirical: Optional[Sequence[float]] = None  # custom P_emp (len N) or None -> uniform
    solver_order: Sequence[str] = field(default_factory=lambda: ("ECOS", "MOSEK", "SCS"))
    ecos_opts: dict = field(default_factory=dict)   # e.g., {"max_iters": 5000}
    mosek_opts: dict = field(default_factory=dict)  # e.g., {"msk_ipar_log": 0}
    scs_opts: dict = field(default_factory=dict)    # e.g., {"max_iters": 10000}
    numerical_eps: float = 1e-9


class DAW:
    """
    Backward-compatible wrapper name (as in your original code).
    Use `FAAL` below for the canonical name.
    """

    def __init__(
        self,
        train_batch_size: int,
        r_choice: float,
        learning_approach: str = "kl",
        output_return: str = "weights",
        empirical: Optional[Sequence[float]] = None,
        solver_order: Sequence[str] = ("ECOS", "MOSEK", "SCS"),
    ) -> None:
        self.cfg = FAALConfig(
            train_batch_size=train_batch_size,
            r_choice=float(r_choice),
            learning_approach=learning_approach,
            output_return=output_return,
            empirical=empirical,
            solver_order=tuple(solver_order),
        )
        self._validate()

        # CVXPY state (lazy-built per call to handle variable batch sizes)
        self._p_var: Optional[cp.Variable] = None
        self._loss_param: Optional[cp.Parameter] = None
        self._prob: Optional[cp.Problem] = None
        self._N: Optional[int] = None

    # ---------------- public API ----------------

    def solve_weight(
        self,
        y: torch.Tensor,
        inf_loss: Optional[torch.Tensor] = None,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        """
        Backward-compatible entry point (signature kept).
        Uses `inf_loss` as the per-sample loss vector.

        Args:
            y: labels (unused by the KL objective; kept for compatibility)
            inf_loss: tensor of shape (N,) with per-sample losses
            device: device for returned tensor ('cuda'/'cpu'/torch.device)

        Returns:
            weights: tensor of shape (N,), sums to 1
        """
        if inf_loss is None:
            raise ValueError("`inf_loss` must be provided (shape: (N,)).")
        return self.compute_weights(inf_loss, device=device)

    def compute_weights(
        self,
        losses: torch.Tensor,
        device: str | torch.device = None,
    ) -> torch.Tensor:
        """
        Preferred API. Solve for adversarial weights given per-sample losses.

        Args:
            losses: tensor (N,) of per-sample losses (logits-based losses recommended).
            device: device for returned tensor. If None, use losses.device.

        Returns:
            weights: tensor (N,), nonnegative, sums to 1.
        """
        if losses.dim() != 1:
            raise ValueError(f"`losses` must be 1D, got shape {tuple(losses.shape)}")
        N = int(losses.shape[0])

        if self.cfg.r_choice <= 0:
            # No robustness -> uniform weights
            w = torch.full((N,), 1.0 / N, device=losses.device, dtype=losses.dtype)
            return w

        # Build / rebuild CVXPY problem if N changed
        if self._prob is None or self._N != N:
            self._build_problem(N)

        # Set loss parameter
        loss_np = losses.detach().to(dtype=torch.float64, device="cpu").numpy()
        self._loss_param.value = loss_np

        # Set empirical distribution
        if self.cfg.empirical is None:
            Pemp = np.full(N, 1.0 / N, dtype=np.float64)
        else:
            Pemp = np.asarray(self.cfg.empirical, dtype=np.float64)
            if Pemp.shape != (N,):
                raise ValueError("`empirical` must have shape (N,).")
            if np.any(Pemp < 0) or not np.isclose(Pemp.sum(), 1.0, atol=1e-6):
                raise ValueError("`empirical` must be a valid probability vector.")

        self._Pemp.value = Pemp

        # Solve with solver cascade
        self._solve()

        p_val = np.asarray(self._p_var.value, dtype=np.float64)
        if p_val is None or np.any(~np.isfinite(p_val)):
            raise RuntimeError("FAAL solver failed to produce a finite solution.")

        # Numerical tidy-up: project small negatives to 0, renormalize
        p_val = np.clip(p_val, 0.0, None)
        s = p_val.sum()
        if s <= self.cfg.numerical_eps:
            # fallback to uniform if degenerate
            p_val = np.full(N, 1.0 / N, dtype=np.float64)
        else:
            p_val = p_val / s

        out_device = losses.device if device is None else device
        return torch.tensor(p_val, device=out_device, dtype=losses.dtype)

    # ---------------- internal ----------------

    def _validate(self) -> None:
        if self.cfg.learning_approach != "kl":
            raise ValueError("Only 'kl' learning_approach is supported currently.")
        if self.cfg.output_return != "weights":
            raise ValueError("Only output_return='weights' is supported.")
        if self.cfg.r_choice < 0:
            raise ValueError("r_choice must be >= 0.")

    def _build_problem(self, N: int) -> None:
        """
        Build the CVXPY problem for batch size N:

            maximize   ⟨p, loss⟩
            s.t.       p >= 0, sum(p)=1, KL(Pemp || p) <= r
        """
        self._N = N
        self._p_var = cp.Variable(shape=N, nonneg=True, name="p")
        self._loss_param = cp.Parameter(shape=N, name="loss")
        self._Pemp = cp.Parameter(shape=N, name="Pemp")

        objective = cp.Maximize(cp.sum(cp.multiply(self._p_var, self._loss_param)))
        constraints = [
            cp.sum(self._p_var) == 1,
            cp.sum(cp.kl_div(self._Pemp, self._p_var)) <= float(self.cfg.r_choice),
        ]
        self._prob = cp.Problem(objective=objective, constraints=constraints)

    def _solve(self) -> None:
        """
        Attempt solvers in order until one succeeds.
        """
        assert self._prob is not None
        last_err: Optional[Exception] = None

        for solver in self.cfg.solver_order:
            try:
                if solver.upper() == "ECOS":
                    self._prob.solve(solver=cp.ECOS, **self.cfg.ecos_opts)
                elif solver.upper() == "MOSEK":
                    self._prob.solve(solver=cp.MOSEK, **self.cfg.mosek_opts)
                elif solver.upper() == "SCS":
                    self._prob.solve(solver=cp.SCS, **self.cfg.scs_opts)
                else:
                    continue

                if self._prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    return
                last_err = RuntimeError(f"Solver {solver} ended with status: {self._prob.status}")
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"All solvers failed. Last error: {last_err}")

# Canonical class name (alias for clarity in new code)
FAAL = DAW


# ---------------- CLI (quick test) ----------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="FAAL (KL-robust) batch reweighting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (N)")
    p.add_argument("--r", type=float, default=0.1, help="KL radius r >= 0")
    p.add_argument("--seed", type=int, default=0, help="Random seed for synthetic losses")
    p.add_argument("--solver-order", type=str, default="ECOS,MOSEK,SCS",
                   help="Comma-separated solver order to try")
    p.add_argument("--ecos-iters", type=int, default=5000)
    p.add_argument("--scs-iters", type=int, default=10000)
    return p


def _main():
    args = _build_argparser().parse_args()
    torch.manual_seed(args.seed)
    N = args.batch_size

    # Synthetic example: losses in [0, 1]
    losses = torch.rand(N)

    daw = FAAL(
        train_batch_size=N,
        r_choice=args.r,
        learning_approach="kl",
        output_return="weights",
        solver_order=tuple(x.strip() for x in args.solver_order.split(",") if x.strip()),
    )
    # Optional: tweak solver options
    daw.cfg.ecos_opts = {"max_iters": args.ecos_iters}
    daw.cfg.scs_opts = {"max_iters": args.scs_iters}

    w = daw.compute_weights(losses)
    print(f"losses (first 8): {losses[:8].tolist()}")
    print(f"weights (first 8): {w[:8].tolist()}")
    print(f"sum(weights) = {float(w.sum()):.6f}, min={float(w.min()):.6f}, max={float(w.max()):.6f}")


if __name__ == "__main__":
    _main()
