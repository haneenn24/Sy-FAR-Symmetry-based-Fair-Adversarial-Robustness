# utils/faal.py
# -*- coding: utf-8 -*-

"""
FAAL: Fair/Adversarial reweighting via KL-divergence ambiguity set.

We solve, for a batch of N samples with per-sample losses `ℓ_i`:

    maximize_p   sum_i p_i * ℓ_i
    subject to   p_i >= 0, sum_i p_i = 1,
                 KL(P_emp || p) <= r

where P_emp is the empirical (usually uniform) distribution on the batch,
and r >= 0 controls robustness. This pushes mass toward harder examples
in a principled distributionally robust way.

Dependencies:
    - cvxpy
    - ECOS (preferred open-source solver)
    - SCS (fallback open-source solver)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import cvxpy as cp


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class FAALConfig:
    train_batch_size: int
    r_choice: float = 0.0                     # robustness radius (KL-ball size)
    learning_approach: str = "kl"
    output_return: str = "weights"
    empirical: Optional[Sequence[float]] = None 
    solver_order: Sequence[str] = field(
        default_factory=lambda: ("ECOS", "SCS") 
    )
    ecos_opts: dict = field(default_factory=dict)   # e.g., {"max_iters": 5000}
    scs_opts: dict = field(default_factory=dict)    # e.g., {"max_iters": 10000}
    numerical_eps: float = 1e-9


# Backward-compatible alias for older code
class DAW:
    """
    Wrapper maintaining compatibility with your original API.
    The canonical class name is FAAL.
    """

    def __init__(
        self,
        train_batch_size: int,
        r_choice: float,
        learning_approach: str = "kl",
        output_return: str = "weights",
        empirical: Optional[Sequence[float]] = None,
        solver_order: Sequence[str] = ("ECOS", "SCS"),
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

        self._p_var: Optional[cp.Variable] = None
        self._loss_param: Optional[cp.Parameter] = None
        self._Pemp: Optional[cp.Parameter] = None
        self._prob: Optional[cp.Problem] = None
        self._N: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_weight(
        self,
        y: torch.Tensor,
        inf_loss: Optional[torch.Tensor] = None,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        """
        Legacy entry point (y is unused in KL objective).
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
        Main API. Solve for adversarial weights given per-sample losses.
        """
        if losses.dim() != 1:
            raise ValueError(f"`losses` must be 1D, got shape {tuple(losses.shape)}")
        N = int(losses.shape[0])

        # r = 0 → uniform weights (fast path, no CVX needed)
        if self.cfg.r_choice <= 0:
            w = torch.full((N,), 1.0 / N, device=losses.device, dtype=losses.dtype)
            return w

        # Rebuild CVX problem if needed
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
        self._Pemp.value = Pemp

        # Solve
        self._solve()

        # Retrieve solution
        p_val = np.asarray(self._p_var.value, dtype=np.float64)
        if p_val is None or np.any(~np.isfinite(p_val)):
            raise RuntimeError("FAAL solver failed to produce a finite solution.")

        # Numerical cleanup
        p_val = np.clip(p_val, 0.0, None)
        s = p_val.sum()
        if s <= self.cfg.numerical_eps:
            p_val = np.full(N, 1.0 / N, dtype=np.float64)
        else:
            p_val /= s

        out_device = losses.device if device is None else device
        return torch.tensor(p_val, device=out_device, dtype=losses.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.cfg.learning_approach != "kl":
            raise ValueError("Only 'kl' learning_approach is supported.")
        if self.cfg.output_return != "weights":
            raise ValueError("Only output_return='weights' is supported.")
        if self.cfg.r_choice < 0:
            raise ValueError("r_choice must be >= 0.")

    def _build_problem(self, N: int) -> None:
        """
        Build CVXPY problem:

            maximize   ⟨p, loss⟩
            s.t.       p >= 0, sum(p)=1,
                       KL(Pemp || p) <= r
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

        self._prob = cp.Problem(objective, constraints)

    def _solve(self) -> None:
        """
        Try ECOS, then SCS. Both are open-source.
        """
        assert self._prob is not None
        last_err: Optional[Exception] = None

        for solver in self.cfg.solver_order:
            try:
                if solver.upper() == "ECOS":
                    self._prob.solve(solver=cp.ECOS, **self.cfg.ecos_opts)
                elif solver.upper() == "SCS":
                    self._prob.solve(solver=cp.SCS, **self.cfg.scs_opts)
                else:
                    continue

                if self._prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    return

                last_err = RuntimeError(
                    f"Solver {solver} ended with status: {self._prob.status}"
                )
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"All solvers failed. Last error: {last_err}")


FAAL = DAW

# ----------------------------------------------------------------------
# CLI (quick manual test)
# ----------------------------------------------------------------------

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        description="FAAL (KL-robust) batch reweighting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--r", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver-order", type=str, default="ECOS,SCS")
    p.add_argument("--ecos-iters", type=int, default=5000)
    p.add_argument("--scs-iters", type=int, default=10000)
    return p


def _main():
    args = _build_argparser().parse_args()
    torch.manual_seed(args.seed)
    N = args.batch_size

    # Example losses
    losses = torch.rand(N)

    daw = FAAL(
        train_batch_size=N,
        r_choice=args.r,
        solver_order=tuple(x.strip() for x in args.solver_order.split(",")),
    )

    daw.cfg.ecos_opts = {"max_iters": args.ecos_iters}
    daw.cfg.scs_opts = {"max_iters": args.scs_iters}

    w = daw.compute_weights(losses)
    print(f"losses (first 8): {losses[:8].tolist()}")
    print(f"weights (first 8): {w[:8].tolist()}")
    print(f"sum(weights) = {float(w.sum()):.6f}")


if __name__ == "__main__":
    _main()
